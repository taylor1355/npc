from typing import Any

from npc.apis.llm_client import Model
from npc.simulators.ai_assistant.tools.email_briefing_generator import EmailBriefingGenerator
from npc.simulators.ai_assistant.tools.email_summarizer import EmailSummarizer
from npc.simulators.ai_assistant.tools.inbox_manager import InboxManager

class AIAssistant:
    def __init__(self, llm: Model):
        self.llm = llm

        self.email_summarizer = EmailSummarizer(llm)
        self.email_briefing_generator = EmailBriefingGenerator(llm, self.email_summarizer)
        self.inbox_manager = InboxManager(llm, self.email_summarizer)
        
    
    # def process_intent(self, user_intent: str) -> Any:
    #     """Process natural language intent and execute appropriate tool(s)"""
    #     # Get tool selection and parameters from LLM
    #     response = self.tool_selector.generate_response(
    #         user_intent=user_intent,
    #         available_tools={name: tool.description for name, tool in self.tools.items()}
    #     )
        
    #     results = []
    #     for tool_info in response['selected_tools']:
    #         tool = self.tools[tool_info['name']]
    #         result = tool.execute(**tool_info['parameters'])
    #         results.append(result)
        
    #     # If single result, return it directly
    #     if len(results) == 1:
    #         return results[0]
    #     return results