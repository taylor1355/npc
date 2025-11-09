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

        if self.vision and self.vision.visible_entities:
            # Show entity details with IDs and interactions (critical for action selection)
            parts.append("Visible entities:")
            for entity in self.vision.visible_entities:
                entity_parts = [f"  - {entity.display_name} (ID: {entity.entity_id})"]

                if entity.interactions:
                    interaction_strs = []
                    for int_name, int_data in entity.interactions.items():
                        desc = int_data.get("description", int_name)
                        # Show needs effects if available
                        needs_filled = int_data.get("needs_filled", [])
                        needs_drained = int_data.get("needs_drained", [])
                        if needs_filled or needs_drained:
                            effects = []
                            if needs_filled:
                                effects.append(f"+{','.join(needs_filled)}")
                            if needs_drained:
                                effects.append(f"-{','.join(needs_drained)}")
                            interaction_strs.append(f"{int_name}: {desc} [{', '.join(effects)}]")
                        else:
                            interaction_strs.append(f"{int_name}: {desc}")

                    entity_parts.append(f"    Interactions: {'; '.join(interaction_strs)}")

                parts.append("\n".join(entity_parts))

        # TODO: Generalize interaction machinery to reduce specialized logic
        # Currently conversations have extensive specialized handling
        # while other interactions are generic. Need to find abstraction that handles
        # conversation complexity without requiring special cases everywhere.
        for conv in self.conversations:
            msgs_str = "\n".join([f"{m.speaker_name}: {m.message}" for m in conv.conversation_history])
            parts.append(f"Conversation:\n{msgs_str}")

        return "\n\n".join(parts) if parts else "No observations"

    def get_available_actions(self):
        """Build list of available actions from this observation."""
        # Import here to avoid circular dependency (actions imports observations for validation)
        from ..actions import ActionType, AvailableAction

        actions = []

        # General actions (always available)
        actions.append(
            AvailableAction(
                name=ActionType.MOVE_TO,
                description="Move to a specific grid position",
                parameters={"destination": "Grid coordinates as tuple (x, y)"},
            )
        )

        actions.append(
            AvailableAction(
                name=ActionType.WANDER,
                description="Wander around aimlessly",
            )
        )

        actions.append(
            AvailableAction(
                name=ActionType.WAIT,
                description="Wait and observe surroundings",
            )
        )

        # Conditional: cancel_interaction only if currently in an interaction
        if self.status and self.status.current_interaction:
            actions.append(
                AvailableAction(
                    name=ActionType.CANCEL_INTERACTION,
                    description="Cancel the current interaction",
                )
            )

        # Interaction-based actions from visible entities
        if self.vision:
            for entity in self.vision.visible_entities:
                for interaction_name, interaction_data in entity.interactions.items():
                    # Extract interaction details
                    desc = interaction_data.get("description", f"Interact with {entity.display_name}")

                    # Build parameter descriptions
                    params = {
                        "entity_id": f"Target entity ID (use: {entity.entity_id})",
                        "interaction_name": f"Interaction type (use: {interaction_name})",
                    }

                    # Add needs info if available
                    needs_filled = interaction_data.get("needs_filled", [])
                    needs_drained = interaction_data.get("needs_drained", [])
                    if needs_filled or needs_drained:
                        effects = []
                        if needs_filled:
                            effects.append(f"fills: {', '.join(needs_filled)}")
                        if needs_drained:
                            effects.append(f"drains: {', '.join(needs_drained)}")
                        desc = f"{desc} ({'; '.join(effects)})"

                    actions.append(
                        AvailableAction(
                            name=ActionType.INTERACT_WITH,
                            description=f"{entity.display_name}: {desc}",
                            parameters=params,
                        )
                    )

        return actions
