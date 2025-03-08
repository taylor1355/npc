from .action_decision_template import prompt as action_decision_prompt
from .long_term_memory_template import prompt as long_term_memory_update_prompt
from .memory_queries_template import prompt as memory_query_formulation_prompt
from .memory_report_template import prompt as memory_report_synthesis_prompt
from .working_memory_template import prompt as working_memory_update_prompt
from .prompt_common import Prompt

class AgentPrompt:
    """Contains prompt templates for different agent functions."""
    ACTION_DECISION = action_decision_prompt
    LONG_TERM_MEMORY = long_term_memory_update_prompt
    MEMORY_QUERIES = memory_query_formulation_prompt
    MEMORY_REPORT = memory_report_synthesis_prompt
    WORKING_MEMORY = working_memory_update_prompt
    
    @classmethod
    def get_all_prompts(cls) -> dict[str, Prompt]:
        """Returns a dictionary mapping prompt names to their Prompt objects."""
        return {
            name: value for name, value in cls.__dict__.items() 
            if not name.startswith('_') and isinstance(value, Prompt)
        }