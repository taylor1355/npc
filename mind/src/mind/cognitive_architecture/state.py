"""Pipeline state for LangGraph cognitive architecture"""

from typing import Annotated

from pydantic import BaseModel, Field

from .actions import Action, AvailableAction
from .memory import Memory
from .observations import ConversationMessage, MindEvent, Observation
from .working_memory import NewMemory, WorkingMemory


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts for LangGraph state reduction"""
    return {**left, **right}


class StepTokenUsage(BaseModel):
    """Provider-reported token usage for one pipeline step.

    Records what was measured, never what was assumed. A step that ran and cost
    zero tokens is a real measurement and gets a record of zeros; a step whose
    provider returned no usage at all is counted under ``unreported_calls``, so
    "ran but the provider was silent" can never be read as "ran and was free".
    That distinction is the point of surfacing these numbers at all -- a cost
    model fit on fabricated zeros is worse than no cost model.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Subset of prompt_tokens the provider served from its own cache.
    cached_prompt_tokens: int = 0

    # Tokens the provider WROTE into its cache on this call. Anthropic bills
    # cache writes at a 1.25x premium, so without this figure the first-call
    # cost of enabling prompt caching is invisible and "caching saved money"
    # cannot be checked against what it cost to prime.
    cache_write_tokens: int = 0

    # True iff at least one response carried real provider cache accounting
    # (prompt_tokens_details.cached_tokens in the raw usage dict). OpenRouter
    # passthrough of cache accounting is provider-dependent, so a 0 in
    # cached_prompt_tokens means "no cache hits" only when this is True;
    # otherwise it means "nobody told us".
    cache_reporting: bool = False

    # Provider round-trips, retries INCLUDED. A decision burning three attempts
    # is real spend with no extra cognitive product, and must stay countable.
    model_calls: int = 0

    # Round-trips that came back with no usage_metadata at all.
    unreported_calls: int = 0

    @classmethod
    def unreported_call(cls) -> "StepTokenUsage":
        """One round-trip the provider reported nothing for."""
        return cls(model_calls=1, unreported_calls=1)

    def merged_with(self, other: "StepTokenUsage") -> "StepTokenUsage":
        """Sum two records. Counts add; cache_reporting is a logical OR."""
        return StepTokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_reporting=self.cache_reporting or other.cache_reporting,
            model_calls=self.model_calls + other.model_calls,
            unreported_calls=self.unreported_calls + other.unreported_calls,
        )

    def is_fully_unreported(self) -> bool:
        """True when every round-trip this record covers came back without usage."""
        return self.model_calls > 0 and self.unreported_calls == self.model_calls


class PipelineState(BaseModel):
    """State that flows through the cognitive pipeline"""

    # Input from simulation
    observation: Observation
    available_actions: list[AvailableAction] = Field(default_factory=list)
    personality_traits: list[str] = Field(default_factory=list)
    personality_dimensions: dict[str, float] = Field(default_factory=dict)

    # Working state
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)
    memory_queries: list[str] = Field(default_factory=list)
    retrieved_memories: list[Memory] = Field(default_factory=list)

    # Conversation histories aggregated by interaction_id
    conversation_histories: dict[str, list[ConversationMessage]] = Field(default_factory=dict)

    # Event buffer (managed by Mind, passed for node access)
    # Events are distinct from observations - they're temporal occurrences that accumulate
    recent_events: list[MindEvent] = Field(default_factory=list)

    # Pending incoming interaction bids (managed by Mind, passed for action generation)
    pending_incoming_bids: dict[str, MindEvent] = Field(default_factory=dict)

    # Daily memory buffer (cleared during sleep/consolidation)
    daily_memories: list[NewMemory] = Field(default_factory=list)

    # Output
    chosen_action: Action | None = None

    # Metadata for observability (use merge function to accumulate values)
    tokens_used: Annotated[dict[str, StepTokenUsage], merge_dicts] = Field(default_factory=dict)
    time_ms: Annotated[dict[str, int], merge_dicts] = Field(default_factory=dict)
