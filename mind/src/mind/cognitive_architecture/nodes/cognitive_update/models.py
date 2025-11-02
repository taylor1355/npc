"""Models for cognitive update node"""

from pydantic import BaseModel, Field


class NewMemory(BaseModel):
    """A new memory to be stored from this decision cycle"""

    content: str = Field(description="The memory content")
    importance: float = Field(description="Importance score (1-10)", ge=1.0, le=10.0)


class WorkingMemory(BaseModel):
    """Structured working memory (flexible, extensible)"""

    model_config = {"extra": "allow"}

    situation_assessment: str = ""
    active_goals: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)
    current_plan: list[str] = Field(default_factory=list)
    emotional_state: str = ""

    def __str__(self) -> str:
        """Format working memory for LLM consumption"""
        parts = []
        if self.situation_assessment:
            parts.append(f"Situation: {self.situation_assessment}")
        if self.active_goals:
            parts.append(f"Active Goals: {', '.join(self.active_goals)}")
        if self.recent_events:
            parts.append(f"Recent Events: {', '.join(self.recent_events)}")
        if self.current_plan:
            parts.append(f"Current Plan: {', '.join(self.current_plan)}")
        if self.emotional_state:
            parts.append(f"Emotional State: {self.emotional_state}")
        return "\n".join(parts) if parts else "No working memory"


class CognitiveUpdateInput(BaseModel):
    """Input for cognitive context update"""

    working_memory: WorkingMemory = Field(description="Current working memory state")
    retrieved_memories: list[str] = Field(description="Memories retrieved from long-term storage")
    observation_text: str = Field(description="Current observation from the environment")


class CognitiveUpdateOutput(BaseModel):
    """Output from cognitive context update"""

    situation_assessment: str = Field(description="Assessment of the current situation")
    current_goals: list[str] = Field(
        default_factory=list, description="Goals currently active based on the situation"
    )
    emotional_state: str = Field(description="Current emotional state based on the situation")
    updated_working_memory: WorkingMemory = Field(
        description="Updated working memory incorporating new information"
    )
    new_memories: list[NewMemory] = Field(
        default_factory=list,
        description="New memories to store from this experience (can be empty if nothing significant)",
    )
