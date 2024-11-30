import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

from npc.apis.gmail_client import Email
from npc.apis.llm_client import LLMClient
from npc.prompts.ai_assistant.email_summary_template import prompt as email_summary_prompt, EmailPriority
from npc.simulators.ai_assistant.tools.tool import Tool


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


@dataclass
class EmailSummaryCache:
    def __init__(self):
        self.email_summaries: dict[str, EmailSummary] = {}
    
    def get_summary(self, email_id: str) -> Optional[EmailSummary]:
        """Retrieve a cached summary for an email."""
        return self.email_summaries.get(email_id)
    
    def cache_summary(self, summary: EmailSummary) -> None:
        """Store an email summary in the cache."""
        self.email_summaries[summary.email_id] = summary
    
    def clear_old_entries(self, current_date: date) -> None:
        """Remove cache entries from previous days."""
        self.email_summaries = {
            email_id: summary 
            for email_id, summary in self.email_summaries.items()
            if current_date == summary._last_cache_update_time.date()
        }


# TODO: emails on chains should have access to the previous email in the chain. Each should summarize for the whole chain up to that point.
class EmailSummarizer(Tool):
    """Tool for generating email summaries with priority classification and content categorization."""
    
    def __init__(self, llm: any) -> None:
        super().__init__(llm)
        self.cache = EmailSummaryCache()  # TODO: load from persistent storage if available

    def _initialize_generators(self) -> None:
        self.summary_generator = LLMClient(email_summary_prompt, self.llm)

    def summarize(self, email: Email) -> EmailSummary:
        return self.execute(email)
    
    def execute(self, email: Email) -> EmailSummary:
        """Generate email summary."""
        current_time = datetime.now()
        
        # Clear cache if it's from a previous day
        self.cache.clear_old_entries(current_time.date())
        
        cached_summary = self.cache.get_summary(email.id)
        if cached_summary:
            return cached_summary
        
        response = self.summary_generator.generate_response(
            subject=email.subject,
            sender=email.sender,
            body=email.body,
            received_datetime=email.timestamp.strftime("%Y-%m-%d %H:%M"),
            current_date=current_time.strftime("%Y-%m-%d")
        )
        
        links = {link: description for link, description in zip(response["links"], response["link_descriptions"])}
        
        email_summary = EmailSummary(
            email_id=email.id,
            priority=response['priority'],
            email_type=response['type'],
            summary=response['summary'],
            priority_analysis=response['priority_analysis'],
            links=links,
            received_time=email.timestamp,
            _last_cache_update_time=current_time
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
