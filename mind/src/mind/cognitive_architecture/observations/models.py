"""Observation models for the cognitive architecture"""

from enum import StrEnum

from pydantic import BaseModel, Field


class MindEventType(StrEnum):
    """Event types matching Godot MindEvent.Type enum"""

    INTERACTION_BID_PENDING = "INTERACTION_BID_PENDING"
    INTERACTION_BID_REJECTED = "INTERACTION_BID_REJECTED"
    INTERACTION_BID_RECEIVED = "INTERACTION_BID_RECEIVED"
    INTERACTION_STARTED = "INTERACTION_STARTED"
    INTERACTION_CANCELED = "INTERACTION_CANCELED"
    INTERACTION_FINISHED = "INTERACTION_FINISHED"
    INTERACTION_OBSERVATION = "INTERACTION_OBSERVATION"
    MOVEMENT_COMPLETED = "MOVEMENT_COMPLETED"
    ERROR = "ERROR"
    # OBSERVATION not included - handled separately as main observation field


class MindEvent(BaseModel):
    """Mind event with typed payload matching Godot MindEvent structure"""

    timestamp: int
    event_type: MindEventType
    payload: dict  # Serialized observation data from Godot

    def __str__(self) -> str:
        """Format event as natural language for LLM"""
        event_type = self.event_type
        payload = self.payload

        if event_type == MindEventType.INTERACTION_BID_REJECTED:
            interaction_name = payload.get("interaction_name", "unknown")
            reason = payload.get("reason", "")
            if reason:
                return f"Interaction bid rejected: {interaction_name} (Reason: {reason})"
            else:
                return f"Interaction bid rejected: {interaction_name}"

        elif event_type == MindEventType.INTERACTION_STARTED:
            interaction_name = payload.get("interaction_name", "unknown")
            return f"Interaction started: {interaction_name}"

        elif event_type == MindEventType.INTERACTION_FINISHED:
            interaction_name = payload.get("interaction_name", "unknown")
            return f"Interaction finished: {interaction_name}"

        elif event_type == MindEventType.INTERACTION_CANCELED:
            interaction_name = payload.get("interaction_name", "unknown")
            return f"Interaction canceled: {interaction_name}"

        elif event_type == MindEventType.ERROR:
            message = payload.get("message", "Unknown error")
            return f"Error: {message}"

        elif event_type == MindEventType.INTERACTION_BID_PENDING:
            interaction_name = payload.get("interaction_name", "unknown")
            return f"Interaction bid pending: {interaction_name}"

        elif event_type == MindEventType.INTERACTION_BID_RECEIVED:
            interaction_name = payload.get("interaction_name", "unknown")
            return f"Interaction bid received: {interaction_name}"

        elif event_type == MindEventType.INTERACTION_OBSERVATION:
            # Interaction update - format based on payload
            return f"Interaction update: {payload}"

        elif event_type == MindEventType.MOVEMENT_COMPLETED:
            status = payload.get("status", "UNKNOWN")
            actual_dest = payload.get("actual_destination")
            intended_dest = payload.get("intended_destination")

            if status == "ARRIVED":
                return f"Arrived at ({actual_dest[0]}, {actual_dest[1]})"
            elif status == "STOPPED_SHORT":
                return f"Moved to ({actual_dest[0]}, {actual_dest[1]}), intended destination ({intended_dest[0]}, {intended_dest[1]}) was blocked"
            elif status == "BLOCKED":
                return f"Could not move to ({intended_dest[0]}, {intended_dest[1]}), no valid path"
            else:
                return f"Movement completed with status {status}"

        else:
            return f"Unknown event type: {event_type}"


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
                entity_parts = [f"  - {entity.display_name} (ID: {entity.entity_id}, Position: {entity.position})"]

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

    def get_available_actions(self, pending_incoming_bids: dict[str, "MindEvent"] = None):
        """Build list of available actions from this observation.

        Args:
            pending_incoming_bids: Optional dict of pending interaction bids (keyed by bid_id)
        """
        # Import here to avoid circular dependency (actions imports observations for validation)
        from ..actions import ActionType, AvailableAction

        actions = []

        # Bid response actions (highest priority - check first)
        if pending_incoming_bids:
            for bid_id, bid_event in pending_incoming_bids.items():
                bidder_id = bid_event.payload.get("bidder_id", "unknown")
                bidder_name = bid_event.payload.get("bidder_name", bidder_id)
                interaction_name = bid_event.payload.get("interaction_name", "unknown")

                # Accept action
                actions.append(
                    AvailableAction(
                        name=ActionType.RESPOND_TO_INTERACTION_BID,
                        description=f"Accept {interaction_name} bid from {bidder_name}",
                        parameters={
                            "bid_id": f"Bid identifier (use: {bid_id})",
                            "accept": "Accept the bid (use: true)",
                            "reason": "Optional reason (leave empty when accepting)",
                        },
                    )
                )

                # Reject action
                actions.append(
                    AvailableAction(
                        name=ActionType.RESPOND_TO_INTERACTION_BID,
                        description=f"Reject {interaction_name} bid from {bidder_name}",
                        parameters={
                            "bid_id": f"Bid identifier (use: {bid_id})",
                            "accept": "Reject the bid (use: false)",
                            "reason": "Reason for rejection (e.g., 'Currently busy', 'Not interested')",
                        },
                    )
                )

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

        # Conditional: continue action when movement or interaction is in progress
        if self.status and self.status.controller_state:
            state_name = self.status.controller_state.get('state_name', '')
            if state_name in ('moving', 'interacting'):
                actions.append(
                    AvailableAction(
                        name=ActionType.CONTINUE,
                        description="Continue current action (movement or interaction)",
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
