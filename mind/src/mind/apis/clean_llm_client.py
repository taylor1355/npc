"""Clean LLM client without LlamaIndex dependencies"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any

from openai import OpenAI

from mind.project_config import OPENROUTER_API_KEY
from mind.prompts.prompt_common import Prompt


# Create an OpenAI client configured to use OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)


class Model(Enum):
    """Available LLM models"""
    CLAUDE_SONNET = "anthropic/claude-3.5-sonnet"
    CLAUDE_HAIKU = "anthropic/claude-3-haiku"
    GEMINI_FLASH = "google/gemini-2.0-flash-exp:free"

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the model"""
        response = client.chat.completions.create(
            model=self.value,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content


@dataclass
class LLMFunction:
    """LLM function wrapper that returns clean, serializable data"""

    prompt: Prompt
    model: Model

    def generate(self, **input_tag_contents) -> Dict[str, Any]:
        """Generate a response from the LLM"""
        # Format the prompt
        formatted_prompt = self.prompt.format(**input_tag_contents)

        # Get LLM response
        output = self.model.get_response(formatted_prompt)

        # Parse and return
        return self.prompt.parse_output(output)