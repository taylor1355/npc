"""Observation models for the cognitive architecture"""

from pydantic import BaseModel, Field


class StatusObservation(BaseModel):
    """Physical and controller state"""

    position: tuple[int, int]
    movement_locked: bool = False
    current_interaction: dict = Field(default_factory=dict)
    controller_state: dict = Field(default_factory=dict)


class NeedsObservation(BaseModel):
    """Entity needs state"""

    needs: dict[str, float]
    max_value: float = 100.0


class EntityData(BaseModel):
    """Visible entity with interaction affordances"""

    entity_id: str
    display_name: str
    position: tuple[int, int]
    interactions: dict[str, dict] = Field(default_factory=dict)


class VisionObservation(BaseModel):
    """Visual perception data"""

    visible_entities: list[EntityData]


class ConversationMessage(BaseModel):
    """Single conversation message"""

    speaker_id: str
    speaker_name: str
    message: str
    timestamp: int | None = None


class ConversationObservation(BaseModel):
    """Conversation state for a specific interaction"""

    interaction_id: str  # Identifies which conversation
    interaction_name: str
    participants: list[str]
    conversation_history: list[ConversationMessage]  # Last K messages from simulation


class Observation(BaseModel):
    """Complete structured observation"""

    entity_id: str  # Mind's entity ID in simulation
    current_simulation_time: int

    status: StatusObservation | None = None
    needs: NeedsObservation | None = None
    vision: VisionObservation | None = None
    conversations: list[ConversationObservation] = Field(default_factory=list)

    def __str__(self) -> str:
        """Format observation as natural language for LLM"""
        parts = []

        if self.status:
            parts.append(f"Position: {self.status.position}")
            parts.append(f"Movement locked: {self.status.movement_locked}")

            # Show current interaction if active
            if self.status.current_interaction:
                parts.append(f"Current interaction: {self.status.current_interaction}")

            # Show controller state
            if self.status.controller_state:
                state_name = self.status.controller_state.get('state_name', 'unknown')
                parts.append(f"Controller state: {state_name}")

        if self.needs:
            needs_str = ", ".join([f"{k}: {v:.0f}%" for k, v in self.needs.needs.items()])
            parts.append(f"Needs: {needs_str}")

        if self.vision:
            # TODO: Show entity IDs and interactions - critical for action validation
            # See actions/validation.py for why this matters
            entities_str = ", ".join([e.display_name for e in self.vision.visible_entities])
            parts.append(f"Visible: {entities_str}")

        # TODO: Generalize interaction machinery to reduce specialized logic
        # Currently conversations have extensive specialized handling
        # while other interactions are generic. Need to find abstraction that handles
        # conversation complexity without requiring special cases everywhere.
        for conv in self.conversations:
            msgs_str = "\n".join([f"{m.speaker_name}: {m.message}" for m in conv.conversation_history])
            parts.append(f"Conversation:\n{msgs_str}")

        return "\n\n".join(parts) if parts else "No observations"
