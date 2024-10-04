from llama_index.llms.anthropic import Anthropic
from npc.prompts.common import Prompt


class LLMResponseGenerator:
    PROMPT_KEY = "prompt"

    def __init__(
            self,
            prompt: Prompt,
            llm: Anthropic,
    ):
        self.prompt = prompt
        self.llm = llm

    def generate_response(self, **input_tag_contents) -> str:
        formatted_prompt = self.prompt.format(**input_tag_contents)
        response = self.llm.chat(formatted_prompt)
        output = response.message.content
        return {
            self.PROMPT_KEY: formatted_prompt,    
            **self.prompt.parse_output(output),
        }
