from enum import Enum
from llama_index.llms.anthropic import Anthropic

from npc.project_config import ANTHROPIC_API_KEY
from npc.prompts.prompt_common import Prompt


class Model(Enum):
    HAIKU = Anthropic(model="claude-3-haiku-20240307", api_key=ANTHROPIC_API_KEY, max_tokens=4096)
    SONNET = Anthropic(model="claude-3-5-sonnet-20240620", api_key=ANTHROPIC_API_KEY, max_tokens=4096)


class LLMClient:
    PROMPT_KEY = "prompt"

    def __init__(
            self,
            prompt: Prompt,
            model: Model,
    ):
        self.prompt = prompt
        self.model = model.value

    # TODO: automatically stop generation when all output tags have been closed (can't do this for formatted tags though)
    def generate_response(self, **input_tag_contents) -> str:
        formatted_prompt = self.prompt.format(**input_tag_contents)
        response = self.model.chat(formatted_prompt)
        output = response.message.content
        return {
            self.PROMPT_KEY: formatted_prompt,    
            **self.prompt.parse_output(output),
        }
