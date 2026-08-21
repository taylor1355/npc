"""Models for cognitive update node"""

from pydantic import BaseModel, Field

from mind.cognitive_architecture.working_memory import NewMemory, WorkingMemory


class CognitiveUpdateInput(BaseModel):
    """Input for cognitive context update"""

    working_memory: WorkingMemory = Field(description="Current working memory state")
    retrieved_memories: list[str] = Field(description="Memories retrieved from long-term storage")
    observation_text: str = Field(description="Current observation from the environment")


class CognitiveUpdateOutput(BaseModel):
    """Output from cognitive context update"""

    updated_working_memory: WorkingMemory = Field(
        description="Updated working memory incorporating current situation, goals, emotional state, and events"
    )
    new_memories: list[NewMemory] = Field(
        default_factory=list,
        description="New memories to store from this experience (can be empty if nothing significant)",
    )
