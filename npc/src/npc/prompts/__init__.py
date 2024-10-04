from enum import Enum
from npc.prompts.action_decision_template import prompt as action_decision_prompt
from npc.prompts.memory_queries_template import prompt as memory_query_formulation_prompt
from npc.prompts.memory_report_template import prompt as memory_report_synthesis_prompt
from npc.prompts.working_memory_template import prompt as working_memory_update_prompt

class NpcPrompt(Enum):
    ACTION_DECISION = action_decision_prompt
    MEMORY_QUERY_FORMULATION = memory_query_formulation_prompt
    MEMORY_REPORT_SYNTHESIS = memory_report_synthesis_prompt
    WORKING_MEMORY_UPDATE = working_memory_update_prompt