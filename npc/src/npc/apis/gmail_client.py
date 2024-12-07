import ezgmail
import logging
import os
import portpicker
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from ezgmail import GmailMessage, GmailThread
from functools import wraps
from google.auth.exceptions import RefreshError
from typing import Callable, Optional, TypeVar

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
                    logging.error(f"Operation {operation_name} failed (attempt {attempt + 1}).\n{repr(e)}")
                
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

            logging.debug(f"Operation {operation_name} succeeded after {attempt + 1} attempts")
            
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
                # Get a random available port
                port = portpicker.pick_unused_port()
                
                ezgmail.init(
                    credentialsFile=GMAIL_CREDENTIALS_PATH,
                    tokenFile=GMAIL_TOKEN_PATH
                )
                
                if hasattr(ezgmail, '_flow'):
                    creds = ezgmail._flow.run_local_server(
                        port=port,
                        success_message='Authentication successful! You can close this window.'
                    )
                    ezgmail._flow.credentials = creds
                    
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
    
    @with_retries("retrieve_threads")
    def retrieve_threads(self, search_options: GmailSearchOptions, max_results: int = 10) -> list[GmailThread]:
        """Retrieve email threads matching search criteria"""
        try:
            query = search_options.to_query()
            return ezgmail.search(query, maxResults=max_results)
        except Exception as e:
            logging.error(f"Failed to search threads: {e}")
            return []
    
    def get_latest_email(self, thread: GmailThread) -> Optional[Email]:
        """Get the most recent email from a thread"""
        try:
            if not thread.messages:
                return None
            latest_message = thread.messages[-1]  # Last message in thread is most recent
            return Email.from_message(latest_message, thread.id)
        except Exception as e:
            logging.warning(f"Failed to get latest email from thread {thread.id}: {e}")
            return None
    
    @with_retries("apply_label")
    def apply_label(self, thread: GmailThread, label: str) -> None:
        """Apply a label to an email thread"""
        thread.addLabel(label)
    
    @with_retries("remove_label")
    def remove_label(self, thread: GmailThread, label: str) -> None:
        """Remove a label from an email thread"""
        thread.removeLabel(label)
    
    def archive_thread(self, thread: GmailThread) -> None:
        """Archive an email thread"""
        self.remove_label(thread, "INBOX")
    
    def mark_as_read(self, thread: GmailThread) -> None:
        """Mark an email thread as read"""
        thread.markAsRead()
