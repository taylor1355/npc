import aiofiles
import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

from npc.project_config import CACHE_DIR
from npc.apis.gmail_client import Email
from npc.apis.llm_client import LLMClient
from npc.prompts.ai_assistant.email_summary_template import prompt as email_summary_prompt, EmailPriority
from npc.simulators.ai_assistant.tools.tool import Tool

import pickle
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta


@dataclass
class EmailSummary:
    email_id: str
    priority: EmailPriority
    email_type: str
    summary: str
    priority_analysis: dict
    links: dict[str, str]  # Maps link URLs to their descriptions
    received_time: datetime
    _last_cache_update_time: datetime

    def __str__(self) -> str:
        return "\n".join([
            f"<email id={self.email_id}>",
            f"Priority: {self.priority}",
            f"Type: {self.email_type}",
            f"Summary: {self.summary}",
            f"Links: {self.links}",
            f"Received: {self.received_time}",
            "</email>",
        ])


class EmailSummaryCache:
    """
    Thread-safe persistent cache for email summaries that periodically removes old entries.
    Uses async I/O for non-blocking saves and batches updates for efficiency.
    """
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.cache_dir = Path(CACHE_DIR)
        self.cache_file = self.cache_dir / 'email_summaries.pkl'
        
        self._memory_cache: Dict[str, EmailSummary] = {}
        self._lock = threading.RLock()
        self._unsaved_changes = False
        self._last_cleanup = datetime.min
        
        self._load_cache()
        self._cleanup_old_entries()

    def _load_cache(self) -> None:
        try:
            if self.cache_file.exists():
                with self.cache_file.open('rb') as f:
                    self._memory_cache = pickle.load(f)
                logging.info(f"Loaded {len(self._memory_cache)} email summaries from cache")
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
            self._memory_cache = {}

    async def _save_cache_async(self) -> None:
        """Save cache to disk if there are unsaved changes."""
        with self._lock:
            if not self._unsaved_changes:
                return
            # Clear flag before I/O to properly batch updates
            self._unsaved_changes = False
            cache_data = pickle.dumps(self._memory_cache)

        try:
            temp_file = self.cache_file.with_suffix('.tmp')
            async with aiofiles.open(temp_file, 'wb') as f:
                await f.write(cache_data)
            temp_file.replace(self.cache_file)
            logging.info(f"Saved {len(self._memory_cache)} email summaries to cache")
        except Exception as e:
            logging.error(f"Error saving cache: {e}")
            # Restore flag on error to retry save
            with self._lock:
                self._unsaved_changes = True

    def _cleanup_old_entries(self) -> None:
        """Remove entries older than 24 hours if an hour has passed since last cleanup."""
        now = datetime.now()
        if now - self._last_cleanup < timedelta(hours=1):
            return

        with self._lock:
            original_size = len(self._memory_cache)
            cutoff = now - timedelta(days=1)
            
            self._memory_cache = {
                email_id: summary 
                for email_id, summary in self._memory_cache.items()
                if summary._last_cache_update_time > cutoff
            }
            
            if len(self._memory_cache) != original_size:
                logging.info(f"Cleaned up {original_size - len(self._memory_cache)} old entries")
                self._unsaved_changes = True
                asyncio.create_task(self._save_cache_async())
            
            self._last_cleanup = now

    def get_summary(self, email_id: str) -> Optional[EmailSummary]:
        """Get cached email summary and trigger cleanup if needed."""
        self._cleanup_old_entries()
        with self._lock:
            return self._memory_cache.get(email_id)

    def cache_summary(self, summary: EmailSummary) -> None:
        """Store email summary and trigger background save."""
        self._cleanup_old_entries()
        with self._lock:
            self._memory_cache[summary.email_id] = summary
            self._unsaved_changes = True
            asyncio.create_task(self._save_cache_async())


# TODO: emails on multi-message threads should have access to the previous emails in the thread. Each should summarize for the whole chain up to that point.
class EmailSummarizer(Tool):
    """Tool for generating email summaries with priority classification and content categorization."""
    
    def __init__(self, llm: any) -> None:
        super().__init__(llm)
        self.cache = EmailSummaryCache()

    def _initialize_generators(self) -> None:
        self.summary_generator = LLMClient(email_summary_prompt, self.llm)

    def summarize(self, email: Email) -> EmailSummary:
        return self.execute(email)

    def execute(self, email: Email) -> EmailSummary:
        """Generate or retrieve cached email summary."""
        cached_summary = self.cache.get_summary(email.id)
        if cached_summary:
            return cached_summary
        
        response = self.summary_generator.generate_response(
            subject=email.subject,
            sender=email.sender,
            body=email.body,
            received_datetime=email.timestamp.strftime("%Y-%m-%d %H:%M"),
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        email_summary = EmailSummary(
            email_id=email.id,
            priority=response['priority'],
            email_type=response['type'],
            summary=response['summary'],
            priority_analysis=response['priority_analysis'],
            links={link: desc for link, desc in zip(response["links"], response["link_descriptions"])},
            received_time=email.timestamp,
            _last_cache_update_time=datetime.now()
        )
        
        self.cache.cache_summary(email_summary)
        return email_summary

    @property
    def description(self) -> str:
        return "Generates summaries of emails by analyzing their content"
    
    @property
    def required_inputs(self) -> dict[str, str]:
        return {
            "email": "Email object containing email metadata and content"
        }
        