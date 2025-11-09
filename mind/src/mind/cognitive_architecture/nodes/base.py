"""Base utilities for pipeline nodes"""

import json
import time
from abc import ABC

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ValidationError

from mind.cognitive_architecture.state import PipelineState


class Node(ABC):
    """Base class for all pipeline nodes - adds automatic timing"""

    step_name: str

    async def process(self, state: PipelineState) -> PipelineState:
        """Override this - timing happens automatically"""
        raise NotImplementedError("Subclass must implement process()")

    def __init_subclass__(cls):
        """Wrap process() with timing on subclass creation"""
        original_process = cls.process

        async def timed_process(self, state: PipelineState) -> PipelineState:
            start_time = time.time()
            state = await original_process(self, state)
            state.time_ms[self.step_name] = int((time.time() - start_time) * 1000)
            return state

        cls.process = timed_process


class LLMNode(Node):
    """Base for nodes that call LLMs with automatic token tracking and optional retry

    Features:
    - Automatic token tracking via callbacks (provider-agnostic)
    - Optional retry with Pydantic validation context
    - Structured (Pydantic) or raw (str) output
    - Prompt template validation via LangChain PromptTemplate
    - Validation context is always {"state": state}
    """

    step_name: str

    def __init__(
        self,
        llm: BaseChatModel,
        prompt: PromptTemplate,
        output_model: type[BaseModel] | None = None,
        max_retries: int = 0
    ):
        """
        Args:
            llm: Language model to use
            prompt: LangChain PromptTemplate with variable validation
            output_model: Pydantic model for structured output, None for raw string
            max_retries: Number of retry attempts on validation failure (default 0)

        Raises:
            ValueError: If max_retries > 0 but output_model is None
        """
        if max_retries > 0 and output_model is None:
            raise ValueError("max_retries > 0 requires output_model (cannot retry raw string output)")

        self.llm = llm
        self.prompt = prompt
        self.output_model = output_model
        self.max_retries = max_retries
        self.parser = None

        # Set up parser if structured output
        if output_model is not None:
            self.parser = PydanticOutputParser(pydantic_object=output_model)

    async def call_llm(
        self,
        state: PipelineState,
        **prompt_vars
    ) -> BaseModel | str:
        """Call LLM with automatic token tracking and optional retry

        Args:
            state: Pipeline state (for token tracking and validation context)
            **prompt_vars: Variables to format into prompt template

        Returns:
            Parsed model if output_model set, else raw string
        """
        # Format prompt using template (validates required vars)
        prompt_text = self.prompt.format(**prompt_vars)

        # Raw string output (no retry needed)
        if self.output_model is None:
            response = await self.llm.ainvoke([HumanMessage(content=prompt_text)])
            tokens = self._extract_tokens(response)
            if tokens:
                state.tokens_used[self.step_name] = tokens
            return response.content

        # Structured output with retry
        messages = [HumanMessage(content=prompt_text)]
        last_error = None
        max_attempts = self.max_retries + 1
        total_tokens = 0

        for attempt in range(max_attempts):
            try:
                # Call LLM
                response = await self.llm.ainvoke(messages)

                # Track tokens from this attempt
                total_tokens += self._extract_tokens(response)

                # Parse JSON
                data = json.loads(response.content)

                # Validate with state context
                validated = self.parser.pydantic_object.model_validate(
                    data, context={"state": state}
                )

                # Success! Track total tokens and return
                if total_tokens:
                    state.tokens_used[self.step_name] = total_tokens
                return validated

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    # Add error feedback for retry
                    messages.append(AIMessage(content=response.content))
                    messages.append(HumanMessage(
                        content=f"Error: {str(e)}\nPlease fix the error and try again."
                    ))

        # All retries exhausted - still track tokens
        if total_tokens:
            state.tokens_used[self.step_name] = total_tokens
        raise last_error

    def _extract_tokens(self, response: AIMessage) -> int:
        """Extract token count from response.usage_metadata"""
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            return response.usage_metadata.get('total_tokens', 0)
        return 0
