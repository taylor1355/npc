from dataclasses import dataclass
from enum import Enum
from llama_index.llms.anthropic import Anthropic

from npc.project_config import ANTHROPIC_API_KEY
from npc.prompts.prompt_common import Prompt


class Model(Enum):
    HAIKU = Anthropic(model="claude-3-haiku-20240307", api_key=ANTHROPIC_API_KEY, max_tokens=4096)
    SONNET = Anthropic(model="claude-3-5-sonnet-20240620", api_key=ANTHROPIC_API_KEY, max_tokens=4096)


def get_llm_response(text: str, model: Model) -> str:
    response = model.value.chat(text)
    return response.message.content


@dataclass
class LLMFunction:
    PROMPT_KEY = "prompt"
    
    prompt: Prompt
    model: Model

    # TODO: automatically stop generation when all output tags have been closed (can't do this for formatted tags though)
    def generate(self, **input_tag_contents) -> str:
        formatted_prompt = self.prompt.format(**input_tag_contents)
        output = get_llm_response(formatted_prompt, self.model)
        return {
            self.PROMPT_KEY: formatted_prompt,    
            **self.prompt.parse_output(output),
        }
