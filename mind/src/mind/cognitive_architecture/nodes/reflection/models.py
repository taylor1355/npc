"""Models for the reflection node"""

from pydantic import BaseModel, Field

from mind.cognitive_architecture.actions import Action
from mind.cognitive_architecture.place_query import PlaceQuery
from mind.cognitive_architecture.working_memory import NewMemory, WorkingMemory


class ReflectionOutput(BaseModel):
    """Both products of one reflection.

    Field order is load-bearing: the model emits the working-memory update
    BEFORE the action, so the action is generated conditioned on the assessment
    it just wrote. Flat rather than nested -- a nested schema costs extra
    format-instruction tokens for no gain and makes partial salvage harder.
    """

    updated_working_memory: WorkingMemory = Field(
        description="Updated working memory incorporating current situation, goals, emotional state, and events"
    )
    new_memories: list[NewMemory] = Field(
        default_factory=list,
        description="New memories to store from this experience (can be empty if nothing significant)",
    )
    chosen_action: Action = Field(description="Selected action to execute")
    place_query: PlaceQuery | None = Field(
        default=None,
        description=(
            "Optional attention query over this NPC's own known places. It does not replace "
            "or modify chosen_action; matches arrive in the next cycle's place.query_result."
        ),
    )
