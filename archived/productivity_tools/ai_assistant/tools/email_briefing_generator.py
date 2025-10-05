import logging

from npc.apis.gmail_client import Email
from npc.apis.llm_client import LLMFunction, Model
from npc.simulators.ai_assistant.tools.tool import Tool
from npc.simulators.ai_assistant.tools.email_summarizer import EmailSummarizer
from npc.prompts.ai_assistant.inbox_briefing_template import (
    FINAL_OUTPUT_TAG,
    prompt as briefing_prompt
)


# TODO: use cached response if there are no new emails and it is the same day as the last cache update
class EmailBriefingGenerator(Tool):
    def __init__(self, llm: Model, email_summarizer: EmailSummarizer):
        super().__init__()
        self.email_summarizer = email_summarizer
        self.briefing_generator = LLMFunction(briefing_prompt, llm)

    # TODO: Future refactor: take in a dictionary or pydantic model instead of Email objects here.
    #       Do this for each tool as well as the tool base class.
    #       This will allow LLMs to easily interact with the tools without needing to know about Python objects.
    def generate_briefing(self, emails: list[Email]) -> str:
        """
        Generate markdown-formatted briefing from provided emails
        
        Args:
            emails: List of Email objects to generate briefing from
            
        Returns:
            Markdown formatted briefing text
        """
        summaries = []
        for email in emails:
            try:
                summaries.append(
                    self.email_summarizer.summarize(email)
                )
            except Exception as e:
                logging.error(f"Failed to summarize email {email.id}: {str(e)}")
                continue

        briefing_response = self.briefing_generator.generate(
            emails=str([str(summary) for summary in summaries])
        )
        return briefing_response[FINAL_OUTPUT_TAG.name]
    
    @property
    def description(self) -> str:
        return "Generates a briefing from email content"
