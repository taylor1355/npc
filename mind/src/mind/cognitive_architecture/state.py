"""Pipeline state for LangGraph cognitive architecture"""

from typing import Annotated
from pydantic import BaseModel, Field

from .models import Memory, Action, ObservationContext
from .nodes.cognitive_update.models import NewMemory


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts for LangGraph state reduction"""
    return {**left, **right}


class PipelineState(BaseModel):
    """State that flows through the cognitive pipeline"""

    # Input from simulation
    observation_context: ObservationContext

    # Working state
    working_memory: str = ""
    memory_queries: list[str] = Field(default_factory=list)
    retrieved_memories: list[Memory] = Field(default_factory=list)

    # Daily memory buffer (cleared during sleep/consolidation)
    daily_memories: list[NewMemory] = Field(default_factory=list)

    # Flexible cognitive context
    cognitive_context: dict = Field(default_factory=dict)

    # Output
    chosen_action: Action | None = None

    # Metadata for observability (use merge function to accumulate values)
    tokens_used: Annotated[dict[str, int], merge_dicts] = Field(default_factory=dict)
    time_ms: Annotated[dict[str, int], merge_dicts] = Field(default_factory=dict)