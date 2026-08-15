"""Observation models for the cognitive architecture"""

from enum import StrEnum

from pydantic import BaseModel, Field


class MindEventType(StrEnum):
    """Event types matching Godot MindEvent.Type enum"""

    INTERACTION_BID_PENDING = "INTERACTION_BID_PENDING"
    INTERACTION_BID_REJECTED = "INTERACTION_BID_REJECTED"
    INTERACTION_BID_RECEIVED = "INTERACTION_BID_RECEIVED"
    INTERACTION_BID_CANCELED = "INTERACTION_BID_CANCELED"
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

        elif event_type == MindEventType.INTERACTION_BID_CANCELED:
            interaction_name = payload.get("interaction_name", "unknown")
            return f"Interaction bid canceled: {interaction_name}"

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
    """Physical and activity state"""

    position: tuple[int, int]
    movement_locked: bool = False
    current_interaction: dict = Field(default_factory=dict)
    activity_state: dict = Field(default_factory=dict)

    def is_interacting(self) -> bool:
        """Ground "am I in an interaction?" in BOTH authoritative observation signals.

        The simulation can clear ``current_interaction`` and the controller's
        ``activity_state`` on different frames during interaction teardown
        (documented cross-field cleanup race in the Godot
        ``entity_controller._on_interaction_ended`` handler). Requiring BOTH
        signals to agree avoids treating a half-torn-down state as "still
        interacting", which is the upstream cause of the act_in_interaction
        desync loop (NPC-688).

        Backward compatible: an older sim payload without ``activity_state``
        (or without ``state_name``) safely resolves to ``False`` rather than
        crashing, so a missing field never produces a spurious interaction
        action.
        """
        if not self.current_interaction:
            return False
        state_name = (self.activity_state or {}).get("state_name", "")
        return state_name == "interacting"


class NeedsObservation(BaseModel):
    """Entity needs state"""

    needs: dict[str, float]
    max_value: float = 100.0


class GoalDetail(BaseModel):
    """One goal produced by the simulation's substrate goal system.

    ``urgency`` is deliberately unbounded. The simulation's effective-urgency
    domain is wider than [0, 1] (preference alignment scales the template curve
    up), and the percent-of-maximum display conversion is owned by the
    simulation tier. Bounding or percent-formatting the value here would fork
    that scale, so the raw number is carried and rendered plainly.
    """

    label: str
    urgency: float
    drive_source: str = ""
    template_id: str = ""

    def urgency_clause(self) -> str:
        """The parenthesised "(urgency N, arising from your X drive)" tail.

        Shared by every surface that renders this goal, because two independent
        builds of the same clause drift: they already had differed wording before
        this was factored out. The leading phrase is deliberately NOT included --
        cognitive_update frames it as a subconscious pull and action_selection as
        a direction to move toward, and that framing difference is intentional
        where the clause itself must not be.

        Urgency stays the raw simulation value; the percent-of-maximum conversion
        belongs to the simulation tier and re-deriving it here would fork the
        display scale.
        """
        drive = f", arising from your {self.drive_source} drive" if self.drive_source else ""
        return f"(urgency {self.urgency:.2f}{drive})"


class GoalObservation(BaseModel):
    """Substrate goal state: what the NPC's drives have settled on.

    The simulation has sent this every cycle since the goal system shipped; it
    was discarded at this boundary because ``Observation`` never declared the
    field (pydantic's default ``extra="ignore"``). Declaring it is what makes
    the substrate's own answer to "why act?" visible to the LLM instead of
    leaving it to re-derive intent from raw need percentages.
    """

    active_goal: GoalDetail | None = None
    candidate_count: int = 0


class ValenceBand(StrEnum):
    """Circumplex valence band, matching Godot SubstrateState.valence_band()."""

    NEG = "neg"
    MID = "mid"
    POS = "pos"


class ArousalBand(StrEnum):
    """Circumplex arousal band, matching Godot SubstrateState.arousal_band()."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


class MoodObservation(BaseModel):
    """Valence-arousal circumplex mood.

    Structure fails loud, copy fails soft. The bands are a StrEnum because the
    3x3 grid is structure: a fourth band is a genuine breaking change and
    deserves a ValidationError. ``label`` is a free ``str`` because the nine
    mood words are copy — the simulation has already relabelled two of them, and
    a copy change must never break decisions. Never branch on ``label``; branch
    on the bands.

    ``valence``/``arousal`` are deliberately unbounded. Their nominal domains are
    [-1, 1] and [0, 1], and every stimulus path clamps to them, but the
    baseline-drift path integrates ``rate * elapsed_minutes`` without a clamp, so
    a long gap between decision cycles can overshoot. Bounding here would convert
    a cosmetic numeric excursion into a ValidationError, and in this pipeline a
    ValidationError collapses the cycle into the WAIT fallback — an NPC that
    silently stops acting. The bands stay correct either way.
    """

    valence: float
    arousal: float
    valence_band: ValenceBand
    arousal_band: ArousalBand
    label: str
    valence_baseline: float = 0.0
    arousal_baseline: float = 0.5


class RelationshipState(BaseModel):
    """Observer's relationship with one visible entity.

    Named ``RelationshipState`` rather than mirroring the simulation's
    ``RelationshipData`` so the two names cannot be confused across the boundary.
    Raw numbers only: there is no familiarity/sentiment band vocabulary in the
    simulation to reuse, and inventing one inside a serializer would put prompt
    copy in the simulation tier. The rendering below owns the words.

    Bounded, unlike mood: every registry write clamps these to their domains.
    """

    familiarity: float = Field(ge=0.0, le=1.0)
    sentiment: float = Field(ge=-1.0, le=1.0)
    interaction_count: int = 0


class EntityData(BaseModel):
    """Visible entity with interaction affordances"""

    entity_id: str
    display_name: str
    position: tuple[int, int]
    interactions: dict[str, dict] = Field(default_factory=dict)
    relationship: RelationshipState | None = None


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
    initiator_id: str = ""  # Entity who initiated this conversation
    conversation_history: list[ConversationMessage]  # Last K messages from simulation


class Observation(BaseModel):
    """Complete structured observation"""

    entity_id: str  # Mind's entity ID in simulation
    current_simulation_time: int

    status: StatusObservation | None = None
    needs: NeedsObservation | None = None
    goal: GoalObservation | None = None
    mood: MoodObservation | None = None
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

            # Show activity state
            if self.status.activity_state:
                state_name = self.status.activity_state.get("state_name", "unknown")
                parts.append(f"Currently: {state_name}")

        if self.needs:
            needs_parts = [f"{k}: {v:.0f}%" for k, v in self.needs.needs.items()]
            parts.append(f"Needs: {', '.join(needs_parts)}")

        # Rendered only when the substrate actually settled on a goal, so an
        # observation without one is byte-identical to the pre-goal rendering.
        if self.goal and self.goal.active_goal:
            active = self.goal.active_goal
            parts.append(f"Subconscious pull: {active.label} {active.urgency_clause()}")

        if self.mood:
            parts.append(
                f"Mood: {self.mood.label} "
                f"(valence {self.mood.valence:+.2f} against a resting "
                f"{self.mood.valence_baseline:+.2f}; arousal {self.mood.arousal:.2f} "
                f"against a resting {self.mood.arousal_baseline:.2f})"
            )

        if self.vision and self.vision.visible_entities:
            # Show entity details with IDs and interactions (critical for action selection)
            parts.append("Visible entities:")
            for entity in self.vision.visible_entities:
                entity_parts = [
                    f"  - {entity.display_name} (ID: {entity.entity_id}, Position: {entity.position})"
                ]

                # Omitted entirely for strangers, so the line's presence is
                # itself the signal that there is shared history.
                if entity.relationship:
                    rel = entity.relationship
                    entity_parts.append(
                        f"    Relationship: familiarity {rel.familiarity:.2f}, "
                        f"sentiment {rel.sentiment:+.2f}, "
                        f"{rel.interaction_count} shared interactions"
                    )

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
            elif conv.initiator_id:
                # No messages yet - show who initiated to provide context
                initiator_context = (
                    "you" if conv.initiator_id == self.entity_id else conv.initiator_id
                )
                parts.append(
                    f"Conversation: (just started by {initiator_context}, no messages yet)"
                )

        return "\n\n".join(parts) if parts else "No observations"

    def is_interacting(self) -> bool:
        """Authoritative "am I interacting?" grounded in the current observation.

        Defaults to ``False`` when no status is present so a malformed or
        partial observation can never advertise interaction-only actions.
        """
        return bool(self.status and self.status.is_interacting())

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
                bid_list = ", ".join(
                    [
                        f"{bid_id[:8]} from {event.payload.get('bidder_name', 'unknown')}"
                        for bid_id, event in pending_incoming_bids.items()
                    ]
                )
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

        # Wait action only available when NOT in an active interaction
        # (wait exits interactions, use cancel_interaction to explicitly end one).
        # Grounded on is_interacting() so a half-torn-down state (current_interaction
        # set but activity_state already non-interacting) still offers wait.
        if not self.is_interacting():
            actions.append(
                AvailableAction(
                    name=ActionType.WAIT,
                    description="Wait and observe surroundings",
                )
            )

        # Conditional: continue action when movement or interaction is in progress
        if self.status and self.status.activity_state:
            state_name = self.status.activity_state.get("state_name", "")
            if state_name == "moving":
                actions.append(
                    AvailableAction(
                        name=ActionType.CONTINUE,
                        description="Continue current movement without changes",
                    )
                )
            elif self.is_interacting():
                interaction_name = self.status.current_interaction.get(
                    "interaction_name", "interaction"
                )
                actions.append(
                    AvailableAction(
                        name=ActionType.CONTINUE,
                        description=f"Wait/pause in the current {interaction_name} for a short moment.",
                    )
                )

        # Conditional: interaction actions only when the observation confirms an
        # active interaction on BOTH signals (current_interaction + activity_state).
        # Grounding here (not just current_interaction presence) prevents emitting
        # act_in_interaction during interaction teardown — the NPC-688 desync loop.
        if self.is_interacting():
            interaction_name = self.status.current_interaction.get(
                "interaction_name", "interaction"
            )

            # Add action to participate in the interaction (e.g., send message in conversation)
            params = {}
            if interaction_name == "conversation":
                params = {"message": "The message to send in the conversation"}

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
                    desc = interaction_data.get(
                        "description", f"Interact with {entity.display_name}"
                    )

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
