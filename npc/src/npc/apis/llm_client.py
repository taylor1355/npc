from dataclasses import dataclass
from enum import Enum

from llama_index.core.llms import ChatMessage, MessageRole
from openai import OpenAI

from npc.project_config import OPENROUTER_API_KEY
from npc.prompts.prompt_common import Prompt


# Create an OpenAI client configured to use OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)


class Model(Enum):
    HAIKU = "anthropic/claude-3-haiku"
    SONNET = "anthropic/claude-3-5-sonnet"
    
    def get_response(self, prompt: str | list[ChatMessage]) -> str:
        # Convert messages to OpenAI format if needed
        if isinstance(prompt, str):
            prompt = [ChatMessage(content=prompt, role=MessageRole.USER)]
        response = client.chat.completions.create(
            model=self.value,
            messages=prompt,
            max_tokens=4096,
        )
        return response.choices[0].message.content


@dataclass
class LLMFunction:
    PROMPT_KEY = "prompt"
    
    prompt: Prompt
    model: Model

    # TODO: automatically stop generation when all output tags have been closed (can't do this for formatted tags though)
    # TODO: refactor LLM responses to be instead of dics for enhanced type safety and encapsulation. Use Pydantic models to define the response structure and validate the output.
    def generate(self, **input_tag_contents) -> str:
        formatted_prompt = self.prompt.format(**input_tag_contents)
        output = self.model.get_response(formatted_prompt)
        return {
            self.PROMPT_KEY: formatted_prompt,    
            **self.prompt.parse_output(output),
        }
