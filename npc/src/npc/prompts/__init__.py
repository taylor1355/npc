from enum import Enum
from npc.prompts.action_decision_template import prompt as action_decision_prompt
from npc.prompts.long_term_memory_template import prompt as long_term_memory_update_prompt
from npc.prompts.memory_queries_template import prompt as memory_query_formulation_prompt
from npc.prompts.memory_report_template import prompt as memory_report_synthesis_prompt
from npc.prompts.working_memory_template import prompt as working_memory_update_prompt

class NpcPrompt(Enum):
    ACTION_DECISION = action_decision_prompt
    LONG_TERM_MEMORY = long_term_memory_update_prompt
    MEMORY_QUERIES = memory_query_formulation_prompt
    MEMORY_REPORT = memory_report_synthesis_prompt
    WORKING_MEMORY = working_memory_update_prompt