"""Models for cognitive update node"""

from pydantic import BaseModel, Field


class CognitiveUpdateInput(BaseModel):
    """Input for cognitive context update"""

    working_memory: str = Field(description="Current working memory state")
    retrieved_memories: list[str] = Field(description="Memories retrieved from long-term storage")
    observation_text: str = Field(description="Current observation from the environment")


class CognitiveUpdateOutput(BaseModel):
    """Output from cognitive context update"""

    situation_assessment: str = Field(
        description="Assessment of the current situation"
    )
    current_goals: list[str] = Field(
        default_factory=list,
        description="Goals currently active based on the situation"
    )
    emotional_state: str = Field(
        description="Current emotional state based on the situation"
    )
    updated_working_memory: str = Field(
        description="Updated working memory incorporating new information"
    )