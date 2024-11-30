import logging
from typing import Dict, List

from npc.apis.gmail_client import Email
from npc.apis.llm_client import LLMClient
from npc.simulators.ai_assistant.tools.tool import Tool
from npc.simulators.ai_assistant.tools.email_summarizer import EmailSummarizer, EmailSummary
from npc.prompts.ai_assistant.email_briefing_template import (
    FINAL_OUTPUT_TAG,
    prompt as briefing_prompt
)


# TODO: use cached response if there are no new emails and it is the same day as the last cache update
class EmailBriefingGenerator(Tool):
    def __init__(self, llm, email_summarizer: EmailSummarizer):
        super().__init__(llm)
        self.email_summarizer = email_summarizer

    def _initialize_generators(self):
        self.briefing_generator = LLMClient(briefing_prompt, self.llm)

    def generate_briefing(self, emails: List[Email]) -> str:
        return self.execute(emails)
   
    # TODO: Future refactor: take in a dictionary or pydantic model instead of Email objects here.
    #       Do this for each tool as well as the tool base class.
    #       This will allow LLMs to easily interact with the tools without needing to know about Python objects.
    def execute(self, emails: List[Email]) -> str:
        """
        Generate markdown-formatted briefing from provided emails
        
        Args:
            emails: List of Email objects to generate briefing from
            
        Returns:
            Markdown formatted briefing text
        """
        # Get summaries for all relevant emails
        summaries = []
        for email in emails:
            try:
                summaries.append(
                    self.email_summarizer.summarize(email)
                )
            except Exception as e:
                logging.error(f"Failed to summarize email {email.id}: {str(e)}")
                continue

        # Generate briefing using summaries
        response = self.briefing_generator.generate_response(
            emails=str([str(summary) for summary in summaries])
        )
        return response[FINAL_OUTPUT_TAG.name]
    
    @property
    def description(self) -> str:
        return "Generates a briefing from email content"
    
    @property
    def required_inputs(self) -> Dict[str, str]:
        return {
            "emails": "List of Email objects to generate briefing from"
        }
