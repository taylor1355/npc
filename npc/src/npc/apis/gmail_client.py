import ezgmail
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from ezgmail import GmailMessage, GmailThread
from functools import wraps
from google.auth.exceptions import RefreshError
from typing import Callable, Iterator, Optional, TypeVar

from npc.project_config import GMAIL_CREDENTIALS_PATH, GMAIL_TOKEN_PATH


class GmailClientError(Exception):
    """Base exception for GmailClient errors"""
    pass


class AuthenticationError(GmailClientError):
    """Raised when authentication fails"""
    pass


class OperationError(GmailClientError):
    """Raised when a Gmail operation fails"""
    pass


# TODO: instead of defining a custom retry decorator, use tenacity
@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    jitter: float = 0.1


T = TypeVar('T')
def with_retries(operation_name: str):
    """Decorator for retrying operations with exponential backoff"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self: 'GmailClient', *args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(self.retry_config.max_retries):
                try:
                    return func(self, *args, **kwargs)
                except RefreshError as e:
                    last_exception = e
                    logging.warning(
                        "Token refresh needed",
                        extra={
                            'operation': operation_name,
                            'attempt': attempt + 1,
                            'error': str(e),
                        }
                    )
                    self._ensure_authenticated()
                except Exception as e:
                    last_exception = e
                    logging.error(
                        "Operation failed",
                        extra={
                            'operation': operation_name,
                            'attempt': attempt + 1,
                            'error': str(e),
                        }
                    )
                
                if attempt < self.retry_config.max_retries - 1:
                    delay = min(
                        self.retry_config.base_delay * (2 ** attempt) +
                        random.uniform(0, self.retry_config.jitter),
                        self.retry_config.max_delay
                    )
                    time.sleep(delay)
            
            raise OperationError(
                f"{operation_name} failed after {self.retry_config.max_retries} attempts"
            ) from last_exception
            
        return wrapper
    return decorator


@dataclass
class GmailSearchOptions:
    """Options for filtering email search results"""
    labels: Optional[list[str]] = None
    unread: Optional[bool] = None
    
    def to_query(self) -> str:
        """Convert search options to Gmail query string"""
        query_parts = []
        if self.unread is not None:
            query_parts.append("is:unread" if self.unread else "is:read")
        if self.labels:
            query_parts.extend(f"label:{label}" for label in self.labels)
        return " ".join(query_parts) if query_parts else "in:inbox"


@dataclass
class Email:
    """Represents a Gmail email message"""
    id: str
    thread_id: str
    subject: str
    sender: str
    recipient: str
    timestamp: datetime
    body: str
    snippet: str
    
    @classmethod
    def from_message(cls, message: GmailMessage, thread_id: str) -> 'Email':
        """Create an Email from an ezgmail message"""
        return cls(
            id=message.id,
            thread_id=thread_id,
            subject=message.subject,
            sender=message.sender,
            recipient=message.recipient,
            timestamp=message.timestamp,
            body=cls._extract_content(message),
            snippet=message.snippet
        )
    
    @staticmethod
    def _extract_content(message: GmailMessage) -> str:
        """Safely extract content from a message"""
        try:
            return message.body
        except (UnicodeDecodeError, AttributeError):
            try:
                return message._rawMessage['snippet']
            except Exception:
                logging.warning(f"Could not decode body for message: {message.id}")
                return "Could not decode message body"


@dataclass
class GmailClient:
    """Client for interacting with Gmail API"""
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    
    def __post_init__(self):
        self._ensure_authenticated()
    
    def _ensure_authenticated(self, _allow_retry=True) -> None:
        """Ensures authentication is valid"""
        if not ezgmail.LOGGED_IN:
            try:
                ezgmail.init(
                    credentialsFile=GMAIL_CREDENTIALS_PATH,
                    tokenFile=GMAIL_TOKEN_PATH
                )
                # Test authentication with actual API call
                ezgmail.SERVICE_GMAIL.users().getProfile(userId='me').execute()
            except Exception as e:
                if _allow_retry:
                    if os.path.exists(GMAIL_TOKEN_PATH):
                        os.remove(GMAIL_TOKEN_PATH) 
                        logging.info("Removed invalid token file")
                    self._ensure_authenticated(_allow_retry=False)
                else:
                    raise AuthenticationError("Failed to authenticate") from e
    
    @with_retries("retrieve_emails")
    def retrieve_emails(self, search_options: GmailSearchOptions, max_results: int = 10) -> list[Email]:
        """Retrieve emails matching search criteria"""
        emails = []
        threads = self._find_matching_threads(search_options, max_results)
        for email in self._extract_emails_from_threads(threads):
            emails.append(email)
            if len(emails) >= max_results:
                break
        return emails
    
    def _find_matching_threads(self, search_options: GmailSearchOptions, max_results: int) -> list[GmailThread]:
        """Find Gmail threads matching the search criteria"""
        try:
            query = search_options.to_query()
            return ezgmail.search(query, maxResults=max_results)
        except Exception as e:
            logging.error(f"Failed to search threads: {e}")
            return []
    
    def _extract_emails_from_threads(self, threads: list[GmailThread]) -> Iterator[Email]:
        """Extract individual emails from a list of threads"""
        for thread in threads:
            try:
                for message in thread.messages:
                    try:
                        yield Email.from_message(message, thread.id)
                    except Exception as e:
                        logging.warning(
                            f"Failed to process message in thread {thread.id}: {e}"
                        )
            except Exception as e:
                logging.warning(f"Failed to process thread {thread.id}: {e}")
    
    def _get_thread(self, thread_id: str) -> GmailThread:
        """Get a thread by ID"""
        return ezgmail.search(f'thread:{thread_id}', maxResults=1)[0]
    
    @with_retries("apply_label")
    def apply_label(self, thread_id: str, label: str) -> None:
        """Apply a label to an email thread"""
        self._get_thread(thread_id).addLabel(label)
    
    @with_retries("remove_label")
    def remove_label(self, thread_id: str, label: str) -> None:
        """Remove a label from an email thread"""
        self._get_thread(thread_id).removeLabel(label)
    
    def archive_thread(self, thread_id: str) -> None:
        """Archive an email thread"""
        self.remove_label(thread_id, "INBOX")
    
    @with_retries("mark_as_read")
    def mark_as_read(self, thread_id: str) -> None:
        """Mark an email thread as read"""
        self._get_thread(thread_id).markAsRead()
