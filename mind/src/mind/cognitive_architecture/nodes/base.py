"""Base utilities for pipeline nodes"""

import time
import inspect
from pathlib import Path
from abc import ABC
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from ..state import PipelineState


class Node(ABC):
    """Base class for all pipeline nodes - adds automatic timing"""

    step_name: str  # Set by subclass

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
    """Base for nodes that call LLMs with prompt templates"""

    # Subclass should define these
    step_name: str
    PROMPT_VARS: set[str] = set()  # Variable names expected in prompt

    def __init__(self, llm: BaseChatModel, output_model: type[BaseModel]):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=output_model)
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load prompt.md from node's directory (once at init)"""
        node_file = inspect.getfile(self.__class__)
        prompt_path = Path(node_file).parent / "prompt.md"
        return prompt_path.read_text()

    async def call_llm(self, **prompt_vars) -> tuple[BaseModel, int]:
        """Call LLM with template variables, return (output, tokens)

        Args:
            **prompt_vars: Variables to format into prompt template

        Returns:
            Tuple of (parsed output, total tokens used)
        """
        # Validate expected variables
        if self.PROMPT_VARS:
            missing = self.PROMPT_VARS - set(prompt_vars.keys())
            if missing:
                raise ValueError(f"{self.__class__.__name__} missing prompt vars: {missing}")

        # Format prompt
        prompt = self.prompt_template.format(
            **prompt_vars,
            format_instructions=self.parser.get_format_instructions()
        )

        # Call LLM
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        # Extract tokens
        tokens = 0
        if response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens", 0)

        return self.parser.parse(response.content), tokens

    def track_tokens(self, state: PipelineState, tokens: int):
        """Helper to track token usage"""
        if tokens:
            state.tokens_used[self.step_name] = tokens
