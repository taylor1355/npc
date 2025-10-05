"""Type-safe LLM function wrapper"""

from dataclasses import dataclass
from typing import TypeVar, Generic, Type, Dict, Any
from pydantic import BaseModel
from mind.apis.llm_client import Model
from mind.prompts.prompt_common import Prompt

# Generic type variables for input and output
TInput = TypeVar('TInput', bound=BaseModel)
TOutput = TypeVar('TOutput', bound=BaseModel)


@dataclass
class TypedLLMFunction(Generic[TInput, TOutput]):
    """Type-safe wrapper for LLM functions with Pydantic models"""

    prompt: Prompt
    model: Model
    output_type: Type[TOutput]

    def generate(self, input_data: TInput) -> TOutput:
        """Generate output from input using LLM"""

        # Convert Pydantic model to dict for prompt formatting
        input_dict = input_data.model_dump()

        # Format the prompt
        formatted_prompt = self.prompt.format(**input_dict)

        # Get LLM response
        response = self.model.get_response(formatted_prompt)

        # Parse output
        parsed = self.prompt.parse_output(response)

        # Create typed output
        return self.output_type(**parsed)


@dataclass
class DebugInfo:
    """Debug information from LLM call"""
    prompt_text: str
    raw_response: str
    parsed_data: Dict[str, Any]


@dataclass
class TypedLLMResponse(Generic[TOutput]):
    """Response wrapper that includes debug info"""
    data: TOutput
    debug: DebugInfo