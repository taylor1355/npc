"""Working-memory models shared across the cognitive architecture.

WorkingMemory is the mind's persistent between-cycles state and NewMemory the
unit of memory formation. They are consumed well beyond the node that writes
them (pipeline state, the MCP wire models, the Mind runtime), so they live at
the architecture level rather than inside any single node package.
"""

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
