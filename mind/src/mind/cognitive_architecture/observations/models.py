"""Observation models for the cognitive architecture"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mind.logging_config import get_logger

logger = get_logger()

# Wire keys on ``StatusObservation.current_interaction``. That dict is the Godot
# ``Interaction.to_dict()`` payload verbatim, whose keys are exactly: name,
# description, needs_filled, needs_drained, need_rates,
# act_in_interaction_parameters. There is NO "interaction_name" key, so every
# reader that asked for one silently took its fallback on every decision cycle
# (NPC-1278) — which is why these live as named constants rather than inline
# literals scattered across four call sites.
WIRE_KEY_INTERACTION_NAME = "name"
WIRE_KEY_ACT_PARAMETERS = "act_in_interaction_parameters"

# Sub-keys of one ``act_in_interaction_parameters`` entry. Each entry is a Godot
# ``PropertySpec.to_dict()``: {"type": <string>, "default": <value>,
# "description": <string>}.
WIRE_KEY_PARAM_TYPE = "type"
WIRE_KEY_PARAM_DEFAULT = "default"
WIRE_KEY_PARAM_DESCRIPTION = "description"

# Rendered when the wire carries no usable interaction name. A generic English
# noun, deliberately not any interaction's identifier — nothing in this package
# may name a specific interaction.
UNNAMED_INTERACTION = "interaction"


def _format_parameter_hint(param_name: str, spec: dict) -> str:
    """Render one wire parameter spec as the prose hint the LLM reads.

    ``AvailableAction.parameters`` is a flat ``name -> prose`` mapping, so the
    structured spec has to be flattened. Type and default are carried because
    the simulation rejects an act whose parameters fail ``PropertySpec``
    validation, and the LLM cannot satisfy a contract it cannot see.
    """
    description = str(spec.get(WIRE_KEY_PARAM_DESCRIPTION) or "").strip()
    text = description or f"The {param_name} parameter"

    qualifiers = []
    type_name = str(spec.get(WIRE_KEY_PARAM_TYPE) or "").strip()
    if type_name:
        qualifiers.append(f"type: {type_name}")
    # Membership, not truthiness: ``false`` and ``0`` are legitimate defaults.
    if WIRE_KEY_PARAM_DEFAULT in spec:
        qualifiers.append(f"default: {spec[WIRE_KEY_PARAM_DEFAULT]!r}")

    return f"{text} ({', '.join(qualifiers)})" if qualifiers else text


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
        """Format event as natural language for LLM.

        The ``payload["interaction_name"]`` reads below are CORRECT — do not
        sweep them into ``WIRE_KEY_INTERACTION_NAME``. Event payloads are a
        different wire source from ``StatusObservation.current_interaction``:
        the Godot bid/interaction observation serializers genuinely emit an
        ``interaction_name`` key, while ``Interaction.to_dict()`` (which is what
        ``current_interaction`` carries) spells the same thing ``name``. Rule of
        thumb for any future sweep: ``payload[...]`` readers are right,
        ``current_interaction[...]`` readers were the NPC-1278 bug.
        """
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

    def interaction_display_name(self) -> str:
        """The current interaction's name, as the simulation spells it.

        The single reader of ``WIRE_KEY_INTERACTION_NAME``. Four sites used to
        read a key the wire never carried, so every one of them rendered the
        literal word "interaction" to the LLM regardless of what the NPC was
        actually doing (NPC-1278).

        Falls back to a generic noun rather than raising: this runs on the
        prompt path every decision cycle, and an exception here collapses the
        cycle into the WAIT fallback — an NPC that silently stops acting.
        """
        if not self.current_interaction:
            return UNNAMED_INTERACTION

        name = self.current_interaction.get(WIRE_KEY_INTERACTION_NAME)
        if not isinstance(name, str) or not name.strip():
            logger.warning(
                "current_interaction carries no usable '%s'; falling back to a generic label. "
                "Keys present: %s",
                WIRE_KEY_INTERACTION_NAME,
                sorted(self.current_interaction.keys()),
            )
            return UNNAMED_INTERACTION

        return name.strip()

    def act_parameter_hints(self) -> dict[str, str]:
        """Project the interaction's advertised act parameters onto prose hints.

        The simulation is the sole authority on what ``act_in_interaction``
        accepts: each interaction declares ``act_in_interaction_parameters`` and
        ships it across the wire as ``{name: PropertySpec.to_dict()}``. Deriving
        the hints from that payload is what lets a NEW interaction — or a new
        parameter on an existing one — reach the LLM's action menu with zero
        changes here. Nothing in this module may name an interaction or a
        parameter.

        Never raises. It runs every decision cycle on the prompt path, so a
        malformed payload degrades to fewer hints plus a loud log, never to a
        dead NPC. Degradation:

        * no current interaction -> ``{}``, silent (nothing to advertise)
        * key absent from a non-empty interaction -> ``{}`` + warning (the
          simulation always emits the key, so its absence is a contract break)
        * key present but empty -> ``{}``, silent (a parameterless interaction
          is legitimate — ``sit`` advertises nothing)
        * key present but not a dict -> ``{}`` + warning
        * one malformed entry -> that entry skipped + warning, the good ones kept
        """
        if not self.current_interaction:
            return {}

        if WIRE_KEY_ACT_PARAMETERS not in self.current_interaction:
            logger.warning(
                "current_interaction is missing '%s'; act_in_interaction will be offered with no "
                "parameter hints. Keys present: %s",
                WIRE_KEY_ACT_PARAMETERS,
                sorted(self.current_interaction.keys()),
            )
            return {}

        raw: Any = self.current_interaction[WIRE_KEY_ACT_PARAMETERS]
        if not isinstance(raw, dict):
            logger.warning(
                "'%s' is %s, expected a dict of parameter specs; offering no parameter hints.",
                WIRE_KEY_ACT_PARAMETERS,
                type(raw).__name__,
            )
            return {}

        hints: dict[str, str] = {}
        for param_name, spec in raw.items():
            if not isinstance(spec, dict):
                logger.warning(
                    "Skipping malformed spec for act parameter '%s': expected a dict, got %s.",
                    param_name,
                    type(spec).__name__,
                )
                continue
            hints[str(param_name)] = _format_parameter_hint(str(param_name), spec)

        return hints


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


class VisibleInteraction(BaseModel):
    """The interaction a VISIBLE entity is currently engaged in.

    Named ``VisibleInteraction`` rather than mirroring the simulation's
    ``InteractionSummary`` for the same reason ``RelationshipState`` is not
    ``RelationshipData`` -- and here the confusion it prevents is not
    hypothetical. ``StatusObservation.current_interaction`` is a *different wire
    source with a different shape*: that one is the OBSERVER's own interaction,
    carrying ``Interaction.to_dict()`` (name, description, needs_filled, ...),
    while this one is a bystander's, carrying ``InteractionSummary.to_dict()``.
    They share a wire key and nothing else. Reading one's keys off the other is
    the NPC-1278 shape, one boundary over.

    ``extra`` is deliberately NOT declared. The simulation populates it with
    subclass-specific state and its own docstring says generic consumers should
    not read it, so dropping it here is the intended behaviour rather than an
    oversight -- recorded because an undeclared field and a forgotten field look
    identical from this side, which is exactly how this model came to be missing
    for as long as it was (NPC-1323).
    """

    interaction_id: str = ""
    interaction_name: str = ""
    #: Both identity channels are carried. ``participant_names`` is what renders
    #: into the prompt; ``participant_ids`` stays available for programmatic
    #: consumers that need to act on a participant rather than describe one.
    participant_ids: list[str] = Field(default_factory=list)
    participant_names: list[str] = Field(default_factory=list)
    participant_count: int = 0
    max_participants: int = 1
    min_participants: int = 1
    duration_minutes: float = 0.0
    is_joinable: bool = False
    #: Empty when joinable; a short token like ``at_capacity`` when not.
    joinable_reason: str = ""

    def _companions(self, exclude_entity_id: str = "") -> list[str]:
        """Participant names minus the entity this line is rendered under.

        The simulation's participant lists include EVERY participant, the
        observed entity included, so rendering them verbatim under that
        entity's own bullet produces "Alice ... In Conversation with Alice,
        Bob". ``participant_ids`` and ``participant_names`` are parallel by
        construction (``get_observation_summary`` fills both from the same
        ordered list), which is what makes an id-based exclusion possible --
        and id-based is what we want, because display names are not unique.

        If the two lists ever disagree in length the pairing is unsafe, so this
        returns every name rather than guessing which one to drop: a redundant
        name reads oddly, a wrongly dropped one misinforms.
        """
        names = [n for n in self.participant_names if n]
        if not exclude_entity_id or len(self.participant_ids) != len(self.participant_names):
            return names
        return [
            name
            for pid, name in zip(self.participant_ids, self.participant_names, strict=False)
            if name and pid != exclude_entity_id
        ]

    def render_summary(self, exclude_entity_id: str = "") -> str:
        """One prompt line describing what this entity is doing, and whether
        the observer could join it.

        Joinability is the point of the line. An NPC deciding whether to
        approach a group needs the refusal reason *before* it bids, not after a
        rejected bid comes back. Never raises: this runs on the prompt path
        every decision cycle, where an exception collapses the cycle into the
        WAIT fallback.

        ``exclude_entity_id`` drops the entity whose bullet this renders under;
        see ``_companions``. The participant COUNT is deliberately not adjusted
        to match -- it is capacity information ("4 of 4"), and the observer
        needs the real occupancy to reason about joining.
        """
        name = self.interaction_name.strip() or UNNAMED_INTERACTION
        head = f"In {name}"

        others = self._companions(exclude_entity_id)
        if others:
            head += f" with {', '.join(others)}"

        facts = [f"{self.participant_count} of {self.max_participants}"]
        if self.duration_minutes > 0.0:
            facts.append(f"{self.duration_minutes:.0f} min so far")
        head += f" ({', '.join(facts)})"

        if self.is_joinable:
            return f"{head} -- joinable"
        # The token is data; turning it into words is this layer's job, but it
        # is only reformatted, never translated into a vocabulary the
        # simulation does not have.
        reason = self.joinable_reason.strip().replace("_", " ")
        return f"{head} -- not joinable{f': {reason}' if reason else ''}"


class EntityData(BaseModel):
    """Visible entity with interaction affordances"""

    entity_id: str
    display_name: str
    position: tuple[int, int]
    interactions: dict[str, dict] = Field(default_factory=dict)
    relationship: RelationshipState | None = None
    current_interaction: VisibleInteraction | None = None

    @field_validator("current_interaction", mode="before")
    @classmethod
    def _idle_entity_has_no_interaction(cls, value: Any) -> Any:
        """Map the simulation's idle sentinel onto ``None``.

        ``EntityData.to_dict()`` emits ``current_interaction: {}`` for an idle
        entity rather than omitting the key. Every field on
        ``VisibleInteraction`` has a default, so ``{}`` would parse cleanly into
        an all-default instance -- and the renderer would then announce a
        nameless, zero-participant interaction for every idle entity in view.
        A falsy payload means idle; say so in the type.
        """
        return value or None


class VisionObservation(BaseModel):
    """Visual perception data"""

    visible_entities: list[EntityData]


class ConversationMessage(BaseModel):
    """Single conversation message.

    ``declarations`` carries the speaker's own annotations on the message —
    "I meant this as a goodbye" and whatever kinds follow it. It is a list of
    plain dicts, each with a ``kind`` key, because the vocabulary of kinds is
    owned by the simulation: a kind registered there must reach the LLM with no
    Python change, so nothing here may enumerate or branch on kind values. The
    field is absent from the wire whenever it is empty, hence the default.

    There is deliberately no ``is_farewell`` field. The simulation's pre-wave-0
    payload carried one, but this model never declared it, so the mind has never
    perceived a farewell — that omission IS the bug. Reading only the
    post-migration shape therefore loses nothing: before the simulation change
    lands perception stays exactly as dark as it already was, and after it lands
    it works. A legacy field would be dead the day it merged.
    """

    speaker_id: str
    speaker_name: str
    message: str
    timestamp: int | None = None
    is_system: bool = False
    declarations: list[dict] = Field(default_factory=list)

    def render_markers(self) -> str:
        """Trailing ``[system] [farewell]``-style markers, or "" when there are none.

        Each declaration renders as its ``kind`` key verbatim; there is no kind
        vocabulary in Python (see the class docstring).
        """
        markers = ["[system]"] if self.is_system else []
        for declaration in self.declarations:
            if not isinstance(declaration, dict):
                continue
            kind = declaration.get("kind")
            if isinstance(kind, str) and kind.strip():
                markers.append(f"[{kind.strip()}]")
        return " ".join(markers)


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

                # After the relationship line and before the affordance list:
                # what this entity is doing now bears on whether to approach it
                # at all, which is upstream of which interaction to start.
                if entity.current_interaction:
                    entity_parts.append(
                        f"    {entity.current_interaction.render_summary(entity.entity_id)}"
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
                markers = m.render_markers()
                suffix = f" {markers}" if markers else ""
                if m.speaker_id == self.entity_id:
                    # Mark own messages clearly to prevent self-responding
                    messages.append(f"[YOU] {m.speaker_name}: {m.message}{suffix}")
                else:
                    messages.append(f"{m.speaker_name}: {m.message}{suffix}")

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
                actions.append(
                    AvailableAction(
                        name=ActionType.CONTINUE,
                        description=(
                            f"Wait/pause in the current {self.status.interaction_display_name()} "
                            "for a short moment."
                        ),
                    )
                )

        # Conditional: interaction actions only when the observation confirms an
        # active interaction on BOTH signals (current_interaction + activity_state).
        # Grounding here (not just current_interaction presence) prevents emitting
        # act_in_interaction during interaction teardown — the NPC-688 desync loop.
        if self.is_interacting():
            # Both the label and the parameter hints come from the wire. Naming
            # an interaction here — as the hardcoded `== "conversation"` branch
            # used to — makes every other interaction's parameters unreachable
            # and freezes the menu against the simulation's own schema.
            actions.append(
                AvailableAction(
                    name=ActionType.ACT_IN_INTERACTION,
                    description=(
                        f"Participate in the current {self.status.interaction_display_name()}"
                    ),
                    parameters=self.status.act_parameter_hints(),
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
