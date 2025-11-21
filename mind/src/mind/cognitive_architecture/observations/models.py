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
    ACTION_CHOSEN = "ACTION_CHOSEN"
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

        elif event_type == MindEventType.ACTION_CHOSEN:
            action_name = payload.get("action", "unknown")
            params = payload.get("parameters", {})
            if params:
                params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                return f"Chose action: {action_name}({params_str})"
            else:
                return f"Chose action: {action_name}"

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
            # Format needs with interpretive context (100 = fully satisfied, 0 = depleted)
            needs_parts = []
            for k, v in self.needs.needs.items():
                if v >= 70:
                    status = "satisfied"
                elif v >= 30:
                    status = "declining"
                else:
                    status = "critical"
                needs_parts.append(f"{k}: {v:.0f}% ({status})")
            parts.append(f"Needs: {', '.join(needs_parts)}")

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
            messages = []
            for m in conv.conversation_history:
                if m.speaker_id == self.entity_id:
                    # Mark own messages clearly to prevent self-responding
                    messages.append(f"[YOU] {m.speaker_name}: {m.message}")
                else:
                    messages.append(f"{m.speaker_name}: {m.message}")
            if messages:
                msgs_str = "\n".join(messages)
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
            # Add batch reject action when there are multiple bids
            if len(pending_incoming_bids) >= 2:
                bid_list = ", ".join([f"{bid_id[:8]} from {event.payload.get('bidder_name', 'unknown')}"
                                      for bid_id, event in pending_incoming_bids.items()])
                actions.append(
                    AvailableAction(
                        name=ActionType.BATCH_REJECT_INTERACTION_BIDS,
                        description=f"Reject multiple interaction bids at once ({len(pending_incoming_bids)} pending: {bid_list})",
                        parameters={
                            "ids": "'*' to reject all, or list of bid IDs like ['bid_xxx', 'bid_yyy'], or list of entity IDs to reject all bids from those entities",
                            "reason": "Reason for rejecting these bids",
                        },
                    )
                )

            # Individual bid response actions
            for bid_id, bid_event in pending_incoming_bids.items():
                bidder_id = bid_event.payload.get("bidder_id", "unknown")
                bidder_name = bid_event.payload.get("bidder_name", bidder_id)
                interaction_name = bid_event.payload.get("interaction_name", "unknown")

                # Single action that can accept or reject based on the accept parameter
                # Include bid_id in description to prevent confusion when multiple bids are present
                actions.append(
                    AvailableAction(
                        name=ActionType.RESPOND_TO_INTERACTION_BID,
                        description=f"Respond to {interaction_name} bid {bid_id} from {bidder_name}",
                        parameters={
                            "bid_id": f"{bid_id}",
                            "accept": "Boolean - true to accept the bid, false to reject the bid",
                            "reason": "Optional string - reason for accepting/rejecting (required when rejecting)",
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
            if state_name == 'moving':
                actions.append(
                    AvailableAction(
                        name=ActionType.CONTINUE,
                        description="Continue current movement without changes",
                    )
                )
            elif state_name == 'interacting' and self.status.current_interaction:
                interaction_name = self.status.current_interaction.get('interaction_name', 'interaction')
                actions.append(
                    AvailableAction(
                        name=ActionType.CONTINUE,
                        description=f"Wait/pause in the current {interaction_name} for a short moment.",
                    )
                )

        # Conditional: interaction actions only if currently in an interaction
        if self.status and self.status.current_interaction:
            interaction_name = self.status.current_interaction.get('interaction_name', 'interaction')

            # Add action to participate in the interaction (e.g., send message in conversation)
            params = {}
            if interaction_name == 'conversation':
                params = {
                    "message": "The message to send in the conversation"
                }

            actions.append(
                AvailableAction(
                    name=ActionType.ACT_IN_INTERACTION,
                    description=f"Participate in the current {interaction_name}",
                    parameters=params,
                )
            )

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
