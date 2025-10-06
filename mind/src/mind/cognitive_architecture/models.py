"""Shared models for the cognitive pipeline"""

from enum import Enum
from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single memory with metadata"""
    id: str
    content: str
    timestamp: float | None = None
    importance: float = Field(default=1.0, ge=0.0, le=10.0)
    embedding: list[float] | None = None


class ActionType(str, Enum):
    """Available action types matching Godot's BaseAction system"""
    MOVE_TO = "move_to"
    MOVE_DIRECTION = "move_direction"
    INTERACT_WITH = "interact_with"
    WANDER = "wander"
    WAIT = "wait"
    CONTINUE = "continue"
    CANCEL_INTERACTION = "cancel_interaction"
    ACT_IN_INTERACTION = "act_in_interaction"
    RESPOND_TO_INTERACTION_BID = "respond_to_interaction_bid"


class Action(BaseModel):
    """Action to be executed, matching Godot's BaseAction format"""
    action: ActionType
    parameters: dict = Field(default_factory=dict)

    class Config:
        use_enum_values = True

    def __str__(self) -> str:
        """Format action for LLM consumption"""
        params_str = ", ".join([f"{k}={v}" for k, v in self.parameters.items()]) if self.parameters else "no parameters"
        return f"{self.action}({params_str})"


class AvailableAction(BaseModel):
    """An action that can be taken, as described by Godot"""
    name: str = Field(description="Action identifier like 'move_to'")
    description: str = Field(description="Human-readable description of what this action does")
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Parameter names mapped to their descriptions"
    )

    def __str__(self) -> str:
        """Format available action for LLM consumption"""
        if self.parameters:
            params_str = ", ".join([f"{param}: {desc}" for param, desc in self.parameters.items()])
            return f"{self.name}: {self.description} (params: {params_str})"
        return f"{self.name}: {self.description}"


class ObservationContext(BaseModel):
    """What comes from the simulation via MCP"""
    agent_id: str
    observation_text: str
    available_actions: list[AvailableAction] = Field(default_factory=list)
    personality_traits: list[str] = Field(default_factory=list)