"""MCP protocol models - requests, responses, configuration"""

from pydantic import BaseModel, Field

from mind.apis.langchain_llm import LangChainModel
from mind.cognitive_architecture.models import Action, Observation
from mind.cognitive_architecture.nodes.cognitive_update.models import WorkingMemory

# === Configuration Models ===


class MindConfig(BaseModel):
    """Configuration for creating a new mind - everything should be configurable"""

    entity_id: str  # Mind's entity ID in simulation
    traits: list[str]

    # LLM configuration
    llm_model: str = LangChainModel.GEMINI_FLASH_LITE  # LangChain model identifier

    # Memory configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    memory_storage_path: str = "./chroma_db"

    # Initial state
    initial_working_memory: WorkingMemory | None = None
    initial_long_term_memories: list[str] = Field(default_factory=list)


# === Protocol: Simulation → Mind ===


class SimulationRequest(BaseModel):
    """Request from simulation to mind for action decision"""

    mind_id: str  # MCP routing (matches entity_id typically)
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
    mind_id: str
    message: str | None = None
