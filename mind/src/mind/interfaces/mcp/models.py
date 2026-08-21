"""MCP protocol models - requests, responses, configuration"""

from typing import Annotated

from pydantic import BaseModel, Field

from mind.apis.langchain_llm import LangChainModel
from mind.cognitive_architecture.state import PipelineState, StepTokenUsage
from mind.cognitive_architecture.working_memory import WorkingMemory
from mind.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_MEMORY_STORAGE_PATH

# === Configuration Models ===


class MindConfig(BaseModel):
    """Configuration for creating a new mind - traits, LLM, memory, personality only.

    The driven entity (entity_id FK) is a top-level create_mind argument, not config:
    config is pure cognitive configuration, deliberately independent of which entity
    the mind drives.
    """

    traits: list[str]

    # LLM configuration
    llm_model: str = LangChainModel.GEMINI_FLASH_LITE  # LangChain model identifier

    # Memory configuration. The defaults live in mind.constants so there is a single
    # named source for them: relink_mind/forget_mind need to name the default storage
    # path without default-constructing a MindConfig, which is the very pattern that
    # made a client-set path unreachable in the first place (NPC-1023).
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    memory_storage_path: str = DEFAULT_MEMORY_STORAGE_PATH

    # Initial state
    initial_working_memory: WorkingMemory | None = None
    initial_long_term_memories: list[str] = Field(default_factory=list)

    # Personality dimensions (numeric trait dimensions from substrate, 0.0-1.0)
    personality_dimensions: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default_factory=dict
    )


# === Protocol: Simulation → Mind ===


class DecisionTelemetry(BaseModel):
    """What one decide_action call cost, as measured on the mind side.

    Rides the success response so the simulation can build a per-decision cost
    ledger. Deliberately has no "zero means missing" field: when nothing was
    reported, `provenance` says so explicitly and the counts are to be read as
    unknown rather than as free.
    """

    # "metered"    -- at least one round-trip reported usage; counts are real.
    # "unreported" -- calls were made but every one came back silent.
    provenance: str

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int

    # Tokens written into the provider's prompt cache this decision. Anthropic
    # bills cache writes at a 1.25x premium; without this the first-call cost of
    # caching is invisible on the cost ledger.
    cache_write_tokens: int

    # False means "no provider told us about cache reads", which is a different
    # fact from cached_prompt_tokens == 0 ("told us, and there were none").
    cache_reporting: bool

    model_calls: int
    unreported_calls: int

    # Requested model slug, not necessarily the one the provider served.
    model: str

    # Mind-side pipeline wall time. Excludes transport and queueing, which only
    # the client can see.
    server_ms: int

    per_step: dict[str, StepTokenUsage] = Field(default_factory=dict)
    per_step_ms: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_pipeline_state(cls, result: PipelineState, model: str) -> "DecisionTelemetry":
        """Fold a completed pipeline's per-step usage into one decision record."""
        folded = StepTokenUsage()
        for step_usage in result.tokens_used.values():
            folded = folded.merged_with(step_usage)

        # Unreported iff calls were made and not one of them reported usage. An
        # empty tokens_used (no LLM step ran at all, or the state never reached
        # one) is also unreported: we cannot claim a measurement we never took.
        metered = folded.model_calls > 0 and not folded.is_fully_unreported()

        return cls(
            provenance="metered" if metered else "unreported",
            prompt_tokens=folded.prompt_tokens,
            completion_tokens=folded.completion_tokens,
            total_tokens=folded.total_tokens,
            cached_prompt_tokens=folded.cached_prompt_tokens,
            cache_write_tokens=folded.cache_write_tokens,
            cache_reporting=folded.cache_reporting,
            model_calls=folded.model_calls,
            unreported_calls=folded.unreported_calls,
            model=model,
            server_ms=sum(result.time_ms.values()),
            per_step=dict(result.tokens_used),
            per_step_ms=dict(result.time_ms),
        )


# === MCP Tool Response Models ===


class MindStateResponse(BaseModel):
    """Mind state for resources"""

    entity_id: str
    traits: list[str]
    working_memory: WorkingMemory
    daily_memories_count: int
    long_term_memory_count: int
    active_conversations: list[str]  # List of interaction_ids


class ConsolidationResponse(BaseModel):
    """Memory consolidation result"""

    status: str
    consolidated_count: int


class MindInfoResponse(BaseModel):
    """Create/cleanup result"""

    status: str
    mind_id: str  # PK: the mind's own identifier
    entity_id: str | None = None  # FK: the driven entity (None for lifecycle ops like cleanup)
    message: str | None = None
