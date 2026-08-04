"""MCP protocol models - requests, responses, configuration"""

from typing import Annotated

from pydantic import BaseModel, Field

from mind.apis.langchain_llm import LangChainModel
from mind.cognitive_architecture.actions import Action
from mind.cognitive_architecture.nodes.cognitive_update.models import WorkingMemory
from mind.cognitive_architecture.observations import Observation
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


class SimulationRequest(BaseModel):
    """Request from simulation to mind for action decision"""

    mind_id: str  # MCP routing key (PK); deliberately independent of the driven entity_id
    observation: Observation  # Structured observation


class MindResponse(BaseModel):
    """Response from mind to simulation with chosen action"""

    status: str  # "success" | "error"
    action: Action | None = None
    error_message: str | None = None


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
