"""Base utilities for pipeline nodes"""

import json
import time
from abc import ABC

from json_repair import loads as json_repair_loads
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ValidationError

from mind.cognitive_architecture.state import PipelineState, StepTokenUsage
from mind.logging_config import get_logger

logger = get_logger()


def entity_tag(state: PipelineState) -> str:
    """Bracketed entity id for per-NPC log attribution.

    The simulation forwards /logs lines to an NPC's Events tab by matching the
    entity id in the message text, so every per-entity log line must carry it.
    """
    if state is not None and state.observation is not None and state.observation.entity_id:
        return f"[{state.observation.entity_id}]"
    return "[unknown]"


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

    step_name: str = "llm_node"

    def __init__(
        self,
        llm: BaseChatModel,
        prompt: PromptTemplate,
        output_model: type[BaseModel] | None = None,
        max_retries: int = 0,
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
            raise ValueError(
                "max_retries > 0 requires output_model (cannot retry raw string output)"
            )

        self.llm = llm
        self.prompt = prompt
        self.output_model = output_model
        self.max_retries = max_retries
        self.parser = None

        # Set up parser if structured output
        if output_model is not None:
            self.parser = PydanticOutputParser(pydantic_object=output_model)

    def get_format_instructions(self) -> str:
        """Get format instructions with optional enhancement for JSON-only output"""
        if not self.parser:
            return ""

        base_instructions = self.parser.get_format_instructions()

        # Add explicit instruction to output raw JSON without markdown fences
        return (
            f"{base_instructions}\n\n"
            f"IMPORTANT: Output ONLY raw JSON. Do NOT wrap in markdown code fences like ```json."
        )

    async def call_llm(self, state: PipelineState, **prompt_vars) -> BaseModel | str:
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
            start_time = time.time()
            response = await self.llm.ainvoke([HumanMessage(content=prompt_text)])
            elapsed_ms = int((time.time() - start_time) * 1000)
            usage = self._extract_usage(response) or StepTokenUsage.unreported_call()
            # Recorded unconditionally: a step that legitimately cost zero tokens
            # must still get a key, or "ran and was free" is indistinguishable
            # from "never ran".
            state.tokens_used[self.step_name] = usage
            logger.debug(
                f"{entity_tag(state)} [{self.step_name}] Completed in {elapsed_ms}ms, "
                f"{usage.total_tokens} tokens"
            )
            return response.content

        # Structured output with retry
        messages = [HumanMessage(content=prompt_text)]
        last_error = None
        max_attempts = self.max_retries + 1
        usage = StepTokenUsage()
        start_time = time.time()

        for attempt in range(max_attempts):
            try:
                # Call LLM
                response = await self.llm.ainvoke(messages)

                # Track usage from this attempt. Every round-trip is counted,
                # including the ones that failed validation -- retries are real
                # spend with no extra cognitive product.
                usage = usage.merged_with(
                    self._extract_usage(response) or StepTokenUsage.unreported_call()
                )

                # Parse JSON (json_repair handles common formatting issues)
                data = json_repair_loads(response.content)

                # Validate with state context
                validated = self.parser.pydantic_object.model_validate(
                    data, context={"state": state}
                )

                # Success! Track total tokens and return
                elapsed_ms = int((time.time() - start_time) * 1000)
                state.tokens_used[self.step_name] = usage
                logger.debug(
                    f"{entity_tag(state)} [{self.step_name}] Completed in {elapsed_ms}ms, "
                    f"{usage.total_tokens} tokens"
                )
                return validated

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    # Log retry
                    error_type = (
                        "JSONDecodeError"
                        if isinstance(e, json.JSONDecodeError)
                        else "ValidationError"
                    )
                    logger.debug(
                        f"{entity_tag(state)} [{self.step_name}] Retry {attempt + 1}/{self.max_retries}: {error_type}"
                    )

                    messages.append(AIMessage(content=response.content))

                    # Format detailed error message
                    if isinstance(e, json.JSONDecodeError):
                        # Use JSONDecodeError properties: msg, lineno, colno
                        error_msg = f"JSON error at line {e.lineno}, column {e.colno}: {e.msg}"
                    else:  # ValidationError
                        # Use Pydantic's error formatting
                        error_msg = f"Validation error:\n{e}"

                    # Add meta-instruction about response format
                    error_msg += "\n\nRespond with ONLY the corrected JSON. Do not include explanations or markdown fences."

                    messages.append(HumanMessage(content=error_msg))

        # All retries exhausted - still record what was spent. The raise below
        # escapes to a handler that cannot reach PipelineState, so this write is
        # the last chance to attribute the spend at all.
        state.tokens_used[self.step_name] = usage
        raise last_error

    def _extract_usage(self, response: AIMessage) -> StepTokenUsage | None:
        """Provider-reported usage for one call, or None if nothing was reported.

        None and a zeroed record are deliberately different answers: None means
        "the provider told us nothing", a zeroed record means "the provider told
        us zero". Callers turn None into an explicit unreported round-trip
        rather than into a free one.

        The prompt/completion split and cache_read arrive in usage_metadata
        already (langchain-core UsageMetadata / InputTokenDetails); this reads
        them through instead of collapsing everything to total_tokens.
        """
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return None

        # NotRequired on UsageMetadata: absent means the provider does not report
        # cache accounting, which is not the same fact as zero cache hits.
        details = usage.get("input_token_details")

        return StepTokenUsage(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cached_prompt_tokens=(details or {}).get("cache_read", 0),
            cache_reporting=details is not None,
            model_calls=1,
        )
