"""Shared models for the cognitive pipeline"""

from enum import Enum

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single memory with metadata"""

    id: str
    content: str
    timestamp: int | None = None  # Simulation timestamp (game ticks/frames)
    importance: float = Field(default=1.0, ge=0.0, le=10.0)
    embedding: list[float] | None = None
    location: tuple[int, int] | None = None  # Grid coordinates (x, y)

    def __str__(self) -> str:
        """Format memory for LLM consumption"""
        parts = [f"[{self.id}"]

        if self.timestamp is not None:
            parts.append(f"T:{self.timestamp}")

        if self.location is not None:
            parts.append(f"L:{self.location}")

        header = " | ".join(parts) + "]"
        return f"{header} {self.content}"


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

    model_config = {"use_enum_values": True}

    action: ActionType
    parameters: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        """Format action for LLM consumption"""
        params_str = (
            ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
            if self.parameters
            else "no parameters"
        )
        return f"{self.action}({params_str})"


class AvailableAction(BaseModel):
    """An action that can be taken"""

    name: str = Field(description="Action identifier like 'move_to'")
    description: str = Field(description="Human-readable description of what this action does")
    parameters: dict[str, str] = Field(
        default_factory=dict, description="Parameter names mapped to their descriptions"
    )

    def __str__(self) -> str:
        """Format available action for LLM consumption"""
        if self.parameters:
            params_str = ", ".join([f"{param}: {desc}" for param, desc in self.parameters.items()])
            return f"{self.name}: {self.description} (params: {params_str})"
        return f"{self.name}: {self.description}"


# === Structured Observation Models ===


class StatusObservation(BaseModel):
    """Physical/controller state (singleton - latest overwrites)"""

    position: tuple[int, int]
    movement_locked: bool = False
    current_interaction: dict = Field(default_factory=dict)
    controller_state: dict = Field(default_factory=dict)


class NeedsObservation(BaseModel):
    """Needs state (singleton - latest overwrites)"""

    needs: dict[str, float]
    max_value: float = 100.0


class EntityData(BaseModel):
    """Visible entity with interaction affordances"""

    entity_id: str
    display_name: str
    position: tuple[int, int]
    interactions: dict[str, dict] = Field(default_factory=dict)


class VisionObservation(BaseModel):
    """What the mind sees (singleton - latest overwrites)"""

    visible_entities: list[EntityData]


class ConversationMessage(BaseModel):
    """Single conversation message"""

    speaker_id: str
    speaker_name: str
    message: str
    timestamp: int | None = None


class ConversationObservation(BaseModel):
    """Conversation update (per-interaction, mind aggregates)"""

    interaction_id: str  # Identifies which conversation
    interaction_name: str
    participants: list[str]
    conversation_history: list[ConversationMessage]  # Last K messages from simulation


class Observation(BaseModel):
    """Complete structured observation"""

    entity_id: str  # Mind's entity ID in simulation
    current_simulation_time: int

    # Singleton observations (None or latest)
    status: StatusObservation | None = None
    needs: NeedsObservation | None = None
    vision: VisionObservation | None = None

    # Per-interaction observations (mind aggregates into full history)
    conversations: list[ConversationObservation] = Field(default_factory=list)

    def __str__(self) -> str:
        """Format observation as natural language for LLM"""
        parts = []
        if self.status:
            parts.append(f"Position: {self.status.position}")
        if self.needs:
            needs_str = ", ".join([f"{k}: {v:.0f}%" for k, v in self.needs.needs.items()])
            parts.append(f"Needs: {needs_str}")
        if self.vision:
            entities_str = ", ".join([e.display_name for e in self.vision.visible_entities])
            parts.append(f"Visible: {entities_str}")
        for conv in self.conversations:
            recent = conv.conversation_history[-3:]
            msgs_str = "; ".join([f"{m.speaker_name}: {m.message}" for m in recent])
            parts.append(f"Conversation: {msgs_str}")
        return "\n".join(parts) if parts else "No observations"
