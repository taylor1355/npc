"""Observation models for the cognitive architecture"""

import logging
from enum import StrEnum
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

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

# The tail of every log line that sheds a key the simulation sent. Whatever the
# level, the remedy is the same: the simulation is ahead of this checkout.
_UPGRADE_HINT = (
    "This mind is older than the simulation that sent it: pull the mind repository "
    "and restart the server to read it."
)


def _declared_wire_keys(model: type[BaseModel]) -> set[str]:
    """Every key ``model`` accepts on the wire: field names plus their aliases.

    Aliases matter because a key shed by field name alone would silently drop a
    correctly-spelled wire key (``NeedsObservation``'s ``max_need_value`` is the
    live example of a wire spelling differing from the field name).
    """
    keys: set[str] = set()
    for name, field in model.model_fields.items():
        keys.add(name)
        if field.alias:
            keys.add(field.alias)
        alias = field.validation_alias
        if isinstance(alias, str):
            keys.add(alias)
        elif isinstance(alias, AliasChoices):
            keys.update(choice for choice in alias.choices if isinstance(choice, str))
    return keys


def shed_undeclared_keys(
    model: type[BaseModel],
    data: Any,
    *,
    level: int,
    context: str,
    keep: frozenset[str] = frozenset(),
) -> Any:
    """THE degrade path: drop the keys ``model`` does not declare, and say so.

    Every place this module tolerates a simulation that is ahead of it goes
    through here -- an unknown root block on ``Observation``, a key added under an
    unchanged ``contract_version`` where that block's wire contract promises
    additive growth, and every root key of a block whose ``contract_version``
    this mind does not know. One path, so the property "an unread key is never
    dropped silently" is tested once and holds everywhere.

    The drop is always logged at ``level``, naming the model and every key, so
    shedding can degrade what a mind knows but can never hide that it did.
    Nothing is dropped when every key is declared, and nothing is logged.

    ``keep`` names undeclared keys that must stay IN the payload so the model's
    ``extra="forbid"`` still refuses them: retired keys whose presence is a real
    contract breach rather than growth (``PlaceDescriptor``'s ``anchor``).

    Runs as a ``mode="before"`` validator on purpose: an ``after`` validator would
    never fire for exactly the payloads it exists for, because ``extra="forbid"``
    would refuse the keys first.
    """
    if not isinstance(data, dict):
        return data
    declared = _declared_wire_keys(model)
    undeclared = sorted(key for key in data if key not in declared and key not in keep)
    if not undeclared:
        return data
    logger.log(
        level,
        "%s: ignoring undeclared key(s) %s (%s). %s",
        model.__name__,
        ", ".join(repr(key) for key in undeclared),
        context,
        _UPGRADE_HINT,
    )
    return {key: value for key, value in data.items() if key not in undeclared}


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


def _format_cell(value: object) -> str | None:
    """Render a wire ``[x, y]`` destination as ``"(x, y)"``, or None if unusable.

    ``MindEvent.payload`` is an unvalidated ``dict``, so nothing upstream
    guarantees a movement event carries two-element coordinate lists. Returning
    None lets the caller degrade to a plainer sentence; indexing directly is
    what used to raise on the prompt path.
    """
    if isinstance(value, list | tuple) and len(value) >= 2:
        return f"({value[0]}, {value[1]})"
    return None


def _format_bid_details(payload: dict) -> str:
    """Render the parenthetical detail clause shared by the three bid arms.

    Carries the identifiers a bid response needs to target, and the whole
    counter-offer when there is one. The counter fields cross the wire (Godot
    ``InteractionBidObservation.get_data()`` adds them for a counter bid) and
    the simulation renders them in its own prose channel, but ``recent_events``
    is the only path by which they reach the LLM — so dropping them here loses
    them outright rather than merely compacting them.

    ``bid_type`` is deliberately NOT rendered. The wire sends it as a bare
    ``InteractionBid.BidType`` ordinal, and naming an ordinal here would hardcode
    a simulation vocabulary this package may not know. The sibling
    ``MovementObservation`` serializes its enum by name; making the bid
    serializer match is the simulation-side fix.
    """
    parts: list[str] = []

    bidder_id = str(payload.get("bidder_id") or "").strip()
    if bidder_id:
        parts.append(f"from {bidder_id}")

    bid_id = str(payload.get("bid_id") or "").strip()
    if bid_id:
        parts.append(f"bid {bid_id}")

    if str(payload.get("countered_bid_id") or "").strip():
        participants = payload.get("existing_participants")
        if isinstance(participants, list | tuple) and participants:
            who = ", ".join(str(p) for p in participants)
        else:
            who = "others"
        clause = f"counter-offer: join with {who}"
        reason = str(payload.get("counter_reason") or "").strip()
        if reason:
            clause += f", because {reason}"
        parts.append(clause)

    return f" ({'; '.join(parts)})" if parts else ""


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

        This is the rendering of the event buffer that reaches a prompt
        (``nodes.formatting.format_recent_events``, NPC-1335). Conversation
        messages also travel through the typed aggregate rendered by
        ``format_conversation_histories`` (NPC-1298). Two consequences bind the
        remaining arms below:

        1. **It must be total.** The render happens in the argument expression
           that builds the reflection prompt — upstream of ``call_llm``, and so
           upstream of its salvage fallback. An exception here does not degrade
           the cycle, it loses the cycle and its telemetry (the NPC-1195 class).
           Read ``payload`` defensively; it is an unvalidated dict.
        2. **What an arm drops needs another authoritative channel.** Conversation
           bodies are the one such case; the typed aggregate carries the full
           simulation-bounded transcript exactly once.

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
            # target_id is who refused. Kept because "who said no" is what stops
            # the NPC re-bidding at the same entity next cycle; the wire omits
            # the key entirely when it is empty.
            target_id = str(payload.get("target_id") or "").strip()
            text = f"Interaction bid rejected: {interaction_name}"
            if target_id:
                text += f" by {target_id}"
            if reason:
                text += f" (Reason: {reason})"
            return text

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

        # The three bid arms share a detail clause: without the bidder and bid
        # id, two bids in flight for the same interaction are indistinguishable,
        # and a bid response has no id to target.
        elif event_type == MindEventType.INTERACTION_BID_PENDING:
            interaction_name = payload.get("interaction_name", "unknown")
            return f"Interaction bid pending: {interaction_name}{_format_bid_details(payload)}"

        elif event_type == MindEventType.INTERACTION_BID_RECEIVED:
            interaction_name = payload.get("interaction_name", "unknown")
            return f"Interaction bid received: {interaction_name}{_format_bid_details(payload)}"

        elif event_type == MindEventType.INTERACTION_BID_CANCELED:
            interaction_name = payload.get("interaction_name", "unknown")
            return f"Interaction bid canceled: {interaction_name}{_format_bid_details(payload)}"

        elif event_type == MindEventType.INTERACTION_OBSERVATION:
            # Conversation bodies now render from PipelineState's typed, fully
            # aggregated transcript. Keep only arrival metadata here so rolling
            # observation windows do not repeat the same utterances many times.
            history = payload.get("conversation_history")
            if payload.get("interaction_name") == "conversation" and isinstance(history, list):
                interaction_id = payload.get("interaction_id", "unknown")
                total = payload.get("total_message_count", len(history))
                return (
                    f"Conversation update: {interaction_id} "
                    f"({total} total messages; full transcript below)"
                )

            # Other interaction observations have no typed replacement. Preserve
            # their complete payload rather than silently losing domain state.
            return f"Interaction update: {payload}"

        elif event_type == MindEventType.MOVEMENT_COMPLETED:
            status = payload.get("status", "UNKNOWN")
            actual_dest = _format_cell(payload.get("actual_destination"))
            intended_dest = _format_cell(payload.get("intended_destination"))

            # Each arm requires the coordinates it names; a malformed payload
            # falls through to the status-only sentence rather than raising.
            if status == "ARRIVED" and actual_dest:
                return f"Arrived at {actual_dest}"
            elif status == "STOPPED_SHORT" and actual_dest and intended_dest:
                return f"Moved to {actual_dest}, intended destination {intended_dest} was blocked"
            elif status == "BLOCKED" and intended_dest:
                return f"Could not move to {intended_dest}, no valid path"
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
            # Unreachable while every MindEventType member has an arm above.
            # Degrade to a repr rather than to nothing: a member the simulation
            # ships before this package learns about it should reach the model
            # as raw-but-present, not as a content-free line.
            return f"{event_type}: {payload}"


class StatusObservation(BaseModel):
    """Physical and activity state.

    **The place keys are optional-by-absence, and that is a contract, not a
    convenience** (NPC-1476). The simulation
    (``status_observation.gd::get_data``) emits ``current_zone_id`` and
    ``current_zone_name`` TOGETHER or emits NEITHER; it never sends ``""`` and
    never sends null. Absence means "the NPC stands on ground it knows no place
    for", which is a different fact from "a place whose id is empty", and
    ``SpatialTerm`` abstains on it rather than scoring it.

    The value is BELIEF, not ground truth: the sim filters the zones covering
    the NPC's cell to the ones in that NPC's own belief store before picking the
    innermost. An NPC spawned or loaded into a zone without prior knowledge of
    it correctly reports no place; a restored belief can already know the zone.

    ``current_zone_name`` travels rather than being resolved here on purpose.
    The mind has no zone registry, and a lookup would be exactly the omniscient
    path the sim-side filtering exists to close - so the name is only ever the
    one the sim sent, and only ever alongside the id.

    No ``model_config`` here, so pydantic's default ``extra="ignore"`` applies.
    That is deliberate (NPC-1116) and it is what makes the sim safe to ship
    first: a sim that adds nested status keys ahead of this model is silently
    tolerated. Only the root ``Observation`` forbids extras.
    """

    position: tuple[int, int]
    movement_locked: bool = False
    current_interaction: dict = Field(default_factory=dict)
    activity_state: dict = Field(default_factory=dict)

    # Omitted together by the producer; see the class docstring. None means "no
    # place known here" - never confuse it with an empty-string id.
    current_zone_id: str | None = None
    current_zone_name: str | None = None

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
    """Entity needs state.

    The simulation spells the ceiling ``max_need_value``
    (``needs_observation.gd::get_data``); this field is ``max_value``. The key
    was therefore dropped on every cycle and the default silently used instead
    -- correct only by coincidence, since ``needs.gd::MAX_VALUE`` is also 100.0.
    Change that constant and the mind would keep normalizing against a stale
    100.0 with nothing failing anywhere (NPC-1116).

    Both spellings are accepted: the wire name so the real value lands, the
    field name so ``model_dump()`` round-trips and existing keyword
    construction keeps working.
    """

    needs: dict[str, float]
    max_value: float = Field(
        default=100.0,
        validation_alias=AliasChoices("max_need_value", "max_value"),
    )


class GoalDetail(BaseModel):
    """The simulation substrate's active goal.

    ``urgency`` is deliberately unbounded. The simulation's effective-urgency
    domain is wider than [0, 1] (preference alignment scales the template curve
    up; the ceiling crosses the wire as ``GoalObservation.urgency_max``), and
    the percent-of-maximum display conversion is owned by the simulation tier.
    Bounding or percent-formatting the value here would fork that scale, so the
    raw number is carried and rendered plainly.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    urgency: float
    drive_source: str = ""
    template_id: str = ""
    # Cosine similarity of this goal against the NPC's preference vector; may
    # legitimately be negative (a goal the NPC is disinclined toward).
    preference_alignment: float = 0.0
    # How long this goal has been active, in game minutes.
    age_minutes: int = 0
    # Urgency a rival goal must exceed to interrupt this one. Optional on the
    # wire — omitted when unavailable, never sent as a placeholder 0.0.
    interruption_threshold: float | None = None

    def urgency_clause(self) -> str:
        """The parenthesised "(urgency N, arising from your X drive)" tail.

        Shared by every surface that renders this goal, because two independent
        builds of the same clause drift: they already had differed wording before
        this was factored out. The leading phrase is deliberately NOT included --
        the observation's own line frames it as a subconscious pull and the
        reflection prompt's dedicated section as a direction to move toward, and
        that framing difference is intentional where the clause itself must not
        be.

        Urgency stays the raw simulation value; the percent-of-maximum conversion
        belongs to the simulation tier and re-deriving it here would fork the
        display scale.
        """
        drive = f", arising from your {self.drive_source} drive" if self.drive_source else ""
        return f"(urgency {self.urgency:.2f}{drive})"


class GoalSummary(BaseModel):
    """One entry of the substrate's candidate goal field (``goals[]``).

    A deliberately thinner shape than ``GoalDetail``: the list answers *why the
    options exist*, so it carries no age or interruption data, and exactly one
    entry has ``is_active`` true iff an ``active_goal`` is present (matching its
    ``template_id``). There is no per-goal utility on the wire and none may be
    synthesized here — goals carry urgency; option steps carry utility.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    urgency: float
    drive_source: str = ""
    template_id: str = ""
    preference_alignment: float = 0.0
    is_active: bool = False


class GoalStepAction(BaseModel):
    """The simulation-vocabulary action a plan step performs.

    Descriptive only, and documented as lossy on the sim side (a directed
    search-wander and a plain wander serialize identically): the pick is
    answered with ``option_id``, never reconstructed from this. ``name`` is a
    free string rather than an enum because the vocabulary is owned by the
    simulation — a new action name must reach the prompt with no Python change.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict = Field(default_factory=dict)


class GoalStepTarget(BaseModel):
    """The interaction a step is aimed at; ``entity_id`` joins to vision."""

    model_config = ConfigDict(extra="forbid")

    interaction_name: str
    entity_id: str


class GoalStepFactors(BaseModel):
    """The multiplicands of a step's score.

    ``responsiveness`` is ``1.0`` whenever the step has no target —
    multiplicative identity for "habituation does not apply", not a sentinel.
    """

    model_config = ConfigDict(extra="forbid")

    urgency: float
    utility: float
    responsiveness: float
    policy_modifier: float


class GoalOptionStep(BaseModel):
    """One step of one plan segment. ``step_score`` equals the product of the
    factors, and equals the option's ``score`` while options are single-step."""

    model_config = ConfigDict(extra="forbid")

    action: GoalStepAction
    target: GoalStepTarget | None = None
    factors: GoalStepFactors
    step_score: float


class GoalOptionSegment(BaseModel):
    """One goal-scoped stretch of a plan: the goal it serves, then its steps."""

    model_config = ConfigDict(extra="forbid")

    goal_template_id: str
    goal_label: str
    steps: list[GoalOptionStep] = Field(default_factory=list)
    # Reserved by the contract: never emitted at tier 0; a planner writes it
    # additively. Declared so its arrival parses under extra="forbid".
    rationale: str | None = None


class GoalOption(BaseModel):
    """One selectable entry of the substrate's evaluated menu — a serialized
    plan.

    The sim's tier-0 generator emits exactly one segment with one step per
    option, but the contract says to parse the GENERAL shape: when planner
    tiers land, options grow multiple segments and steps under the same
    contract version, and nothing here may assume the degenerate case.

    ``option_id`` is the authoritative handle for a pick and is valid only for
    the cycle that produced it — treat it as opaque, never persist it, never
    compare it across cycles.
    """

    model_config = ConfigDict(extra="forbid")

    option_id: str
    description: str
    score: float
    segments: list[GoalOptionSegment] = Field(default_factory=list)
    # Reserved by the contract: absent until a planner computes it (it is
    # search_confidence x world_confidence — a measurement, never fabricated).
    confidence: float | None = None


# Goal-block wire versions this model set knows how to read. An unknown
# version degrades (warn + best-effort parse), never raises: a future sim must
# not be able to kill the decision cycle by bumping a version number.
KNOWN_GOAL_CONTRACT_VERSIONS = frozenset({1})


class GoalObservation(BaseModel):
    """Substrate goal state: the active pull, the goal field, and the evaluated
    option menu.

    The wire contract lives in the simulation repo at
    ``docs/reference/minds/observations.md`` ("Goal block wire contract").
    ``options`` is the same pool, built by the same code, that the free tier's
    softmax samples from — selecting the argmax is what a low-temperature
    sampler does, selecting elsewhere is a reasoned divergence, and neither is
    an error. ``option_total`` counts the pre-truncation pool, so
    ``option_total > len(options)`` means a longer menu exists server-side.

    ``extra="forbid"`` is this block's parsing posture at every level
    (precedent: ``VectorDBQuery``). Unlike the place descriptors and
    entity-memory rows, the goal contract does not promise open additive growth
    under v1: it names its two reserved keys (``options[].confidence``,
    ``segments[].rationale``), and both are declared below. Any other key is a
    contract drift that must fail loud rather than be silently dropped -- and
    because the refusal is nested, it refuses the whole observation, so a new
    goal key still needs this model deployed first or a version bump.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: int = 1
    # Ceiling of the urgency domain. Optional so a hand-built observation
    # cannot fabricate one; the sim sends it on every cycle.
    urgency_max: float | None = None
    active_goal: GoalDetail | None = None
    goals: list[GoalSummary] = Field(default_factory=list)
    options: list[GoalOption] = Field(default_factory=list)
    option_total: int = 0

    @model_validator(mode="before")
    @classmethod
    def _degrade_on_unknown_contract_version(cls, data):
        """Unknown versions degrade, never raise.

        A raise would collapse the decide_action call into an error response —
        an NPC that silently stops acting because the sim got ahead of the
        mind. So on an unknown version: warn (naming both the received and the
        known versions), shed any root keys this model does not declare (a
        purely additive future version then parses cleanly despite
        ``extra="forbid"``), and parse the rest best-effort. Runs in
        ``mode="before"`` deliberately: an ``after`` validator would never fire
        for exactly the payloads it exists for, because the unknown keys would
        fail ``extra="forbid"`` first. Known-version payloads are untouched —
        for them an undeclared key stays a loud contract drift.
        """
        if not isinstance(data, dict):
            return data
        version = data.get("contract_version", 1)
        if version in KNOWN_GOAL_CONTRACT_VERSIONS:
            return data
        logger.warning(
            "Goal block carries unknown contract_version %s (known: %s); "
            "parsing best-effort under the newest known contract.",
            version,
            sorted(KNOWN_GOAL_CONTRACT_VERSIONS),
        )
        return shed_undeclared_keys(
            cls, data, level=logging.WARNING, context=f"unknown contract_version {version}"
        )


# ---------------------------------------------------------------------------
# Place handles (NPC-1643)
# ---------------------------------------------------------------------------
#
# The prompt names places, never zone ids: a zone id is a UUID the model cannot
# say to another NPC, costs ~20 uncached tokens per mention, and invites the
# model to copy opaque strings around. But MOVE_TO must be able to name a place,
# so each place in a cycle's place block is labelled with a short HANDLE --
# ``p1``, ``p2``, ... -- the way a menu option carries its option_id. The mind
# translates a handle back to the real zone id only at the wire
# (``Action.wire_payload``), so the simulation still receives real ids and the
# UUID never enters a prompt.
#
# A handle is valid for ONE cycle, like an option_id: it is a pure function of
# that cycle's PlaceObservation (``PlaceObservation.place_handles``), so the
# prompt that showed it, the validator that checks it and the translation that
# resolves it all derive the same map from the same object, and nothing has to
# be stored to keep them in agreement.
#
# Format: ``p`` + a 1-based ordinal. Short (one or two tokens), not a word the
# model will use for anything else, and it cannot collide with an option_id
# (``template:ordinal`` always carries a colon) or with a zone id (``zone_...``).
# Rendered in square brackets in the prompt, accepted with or without them.
PLACE_HANDLE_PREFIX = "p"

#: The label for a place with no name. Never the zone id, which is exactly the
#: string the handle scheme exists to keep out of the prompt.
UNNAMED_PLACE_LABEL = "an unnamed place"


def normalize_place_handle(reference: str) -> str:
    """``" [P2] "`` -> ``"p2"``. Leniency on the token only: the simulation never
    sees a handle, so accepting its bracketed or upper-case spelling cannot make
    the two tiers disagree about what was supplied."""
    return reference.strip().strip("[]").strip().lower()


class PlaceKnowledgeSource(StrEnum):
    """How this NPC came to know a place, matching Godot ``PlaceKnowledge.Source``.

    A StrEnum rather than a free ``str``, which is the opposite of the choice
    made for interaction names and declaration kinds. The discriminator is
    open-registry versus closed enum: an interaction is registered in the
    simulation and must reach the LLM with no Python change, whereas ``Source``
    is a four-member GDScript ``enum`` whose serialized names are a save-format
    contract "from birth" with an out-of-enum ``SOURCE_INVALID`` sentinel for
    anything that fails to parse. Nothing can widen it quietly. Same reasoning as
    ``ValenceBand``: structure fails loud.

    ``TOLD`` is the only member that fills ``PlaceDescriptor.told_by``.
    """

    CREATED = "created"
    VISITED = "visited"
    TOLD = "told"
    WITNESSED = "witnessed"


# Descriptor keys the simulation has RETIRED. Unlike a key the mind has not
# learned yet, one of these arriving in a block whose version post-dates its
# retirement means the producer regressed, so it keeps ``extra="forbid"``'s
# refusal instead of the additive-key degrade. ``anchor`` became ``focal_cell``
# at v4; a pre-v4 block's ``anchor`` is translated before descriptors validate.
_RETIRED_DESCRIPTOR_KEYS = frozenset({"anchor"})


class PlaceDescriptor(BaseModel):
    """One place this NPC knows, as the substrate ranked it this cycle.

    Wire producer: the simulation's ``PlaceDescriptor`` (NPC-1299). The list
    that carries these is capped, so every field here is paid for on every
    decision cycle for every MCP NPC -- which is why the renderer below spends
    tokens on some of them and deliberately not on others.

    An undeclared key DEGRADES (logged at WARNING, dropped) rather than refusing
    the descriptor, because the place block's wire contract promises it:
    "descriptors grow keys ... under the *same* ``contract_version``". The one
    exception is the retired ``anchor`` key, which stays refused -- in a v4+
    block it is a contract breach, not growth (see ``_RETIRED_DESCRIPTOR_KEYS``).
    ``extra="forbid"`` stays set so a key that somehow bypasses the logged drop
    still fails loudly.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _degrade_on_additive_keys(cls, data):
        return shed_undeclared_keys(
            cls,
            data,
            level=logging.WARNING,
            context="additive place-descriptor key(s)",
            keep=_RETIRED_DESCRIPTOR_KEYS,
        )

    zone_id: str
    name: str = ""
    #: Serialized ``Zone.Kind``. A free ``str``, not an enum: the kind
    #: vocabulary is the simulation's, ``Zone.kind_from_string`` routes an
    #: unrecognised name to ``KIND_INVALID`` there, and a new kind must not need
    #: a second edit in this repository.
    kind: str = ""
    #: The zone's focal cell: the one cell that stands for the place, and where a
    #: journey to it walks. ``[x, y]`` on the wire -- Godot has no JSON vector
    #: type, so every observation converts (``status_observation.gd``,
    #: ``entity_data.gd``), and this matches ``StatusObservation.position``.
    #:
    #: ``None`` when the wire carries no cell. Not ``(0, 0)``: the origin is a
    #: real cell on every board the simulation builds, so it cannot also mean
    #: "none supplied".
    focal_cell: tuple[int, int] | None = None
    #: Chebyshev distance from the observer, in cells.
    distance: int = 0
    #: Whether this NPC has ever LOOKED INSIDE this place (NPC-1473).
    #:
    #: The discriminator for the two fields below, and it exists because no
    #: value of theirs can carry it: zero providers, no affordances and an age
    #: of zero are all legitimate readings of a place seen to be bare. The wire
    #: OMITS ``affords``, ``provider_count`` and ``witnessed_age_minutes``
    #: entirely when this is false, so their defaults below are never a
    #: witnessed reading -- read this flag, never their emptiness.
    #:
    #: A place merely heard of is unwitnessed. That is not a gap to be filled:
    #: "I have never looked" and "I looked and it was bare" must produce
    #: different behaviour, and collapsing them is what the flag prevents.
    witnessed: bool = False
    #: The dominant interaction names this place affords, capped simulation-side.
    #: AS LAST WITNESSED and possibly wrong -- see ``witnessed``. At contract
    #: version 1 this was live ground truth; the v2 bump exists for exactly this
    #: change of meaning, since the shape did not change.
    affords: list[str] = Field(default_factory=list)
    #: Provider count AS LAST WITNESSED. See ``affords``.
    provider_count: int = 0
    #: Game minutes since this NPC last looked inside. Omitted when unwitnessed.
    #:
    #: Distinct from ``age_minutes``, which is how long ago the place was
    #: LEARNED. A place told about last week and never visited has a large
    #: age and no witnessed age at all; one learned long ago and checked this
    #: morning has a large age and a small witnessed age. Staleness of BELIEF is
    #: this field, not that one.
    witnessed_age_minutes: int = 0
    source: PlaceKnowledgeSource = PlaceKnowledgeSource.VISITED
    #: Who told this NPC, for ``TOLD`` only; the wire omits the key otherwise.
    #: ``source`` is the discriminator, never the emptiness of this string -- but
    #: an entity id is never blank, so "" is out of domain rather than a value
    #: masquerading as one.
    told_by: str = ""
    #: How long ago this NPC learned the place, in game minutes.
    age_minutes: int = 0
    #: True when the place is known but not currently visible. Load-bearing for
    #: marking, which the simulation structurally refuses for ground the marker
    #: cannot see.
    beyond_vision: bool = False
    #: The ``world_confidence`` analogue. Deliberately UNBOUNDED, following
    #: ``GoalOption.confidence`` on this same wire family: bounding it here would
    #: convert a cosmetic numeric excursion into a ValidationError, and in this
    #: pipeline a ValidationError collapses the cycle into the WAIT fallback.
    #:
    #: At contract version 3 (NPC-1479) it is a PRECISION reading -- the share of
    #: the belief that came from observation rather than from the prior -- where
    #: before it was a discount factor. Same name, same shape, different meaning:
    #: never multiply a score by it.
    confidence: float = 0.0
    #: The posterior mean number of item providers the NPC expects to find in
    #: this place NOW (NPC-1479, contract version 3). Emitted for every
    #: descriptor, witnessed or not: for a place never looked inside it is the
    #: prior mean, which is still a belief.
    #:
    #: ``None`` means the producer predates version 3 and computed no posterior
    #: at all. Not ``0.0``: zero expected providers is a real belief ("nothing
    #: there"), and a default must not put a claim in the model's mouth.
    expected_providers: float | None = None
    #: Standard deviation of ``expected_providers`` -- how unsure the NPC is.
    #: ``None`` for the same reason. Not an information-gain reading: a rich
    #: observation can raise the mean and the deviation together.
    belief_deviation: float | None = None

    def label(self) -> str:
        """The name the prompt uses for this place -- never its zone id."""
        return self.name.strip() or UNNAMED_PLACE_LABEL

    def render_summary(self, here_zone_id: str = "", handle: str = "") -> str:
        """One prompt clause naming this place and why it matters.

        Name-first, because a name is what an NPC can say to another NPC and a
        zone id is not. ``handle`` prefixes the clause as ``[p1]`` when given: it
        is how a MOVE_TO names this place, and it stands in for the zone id so a
        nameless place is still addressable without the id reaching the prompt.

        ``confidence`` and ``age_minutes`` are deliberately NOT rendered: both are ranking inputs the simulation already applied when it
        ordered and capped this list, so spending per-cycle tokens restating
        them buys the model nothing it cannot read from the ordering.

        Never raises -- this runs on the prompt path, where an exception
        collapses the decision cycle.
        """
        facts: list[str] = []
        if here_zone_id and self.zone_id == here_zone_id:
            facts.append("here")
        else:
            facts.append(f"{self.distance} away")
            if self.beyond_vision:
                facts.append("out of sight")

        if self.affords:
            clause = ", ".join(self.affords)
            if self.provider_count:
                clause += f" x{self.provider_count}"
            facts.append(clause)
            # The belief beside the memory, mirroring the simulation's
            # ``place_descriptor.gd::format_for_npc``: rendered only for a place
            # whose contents were seen, because "about 1.0, give or take 1.0" for
            # a place never looked inside says nothing the absent affordances
            # have not already said.
            if (
                self.witnessed
                and self.expected_providers is not None
                and self.belief_deviation is not None
            ):
                facts.append(
                    f"expect about {self.expected_providers:.1f}, +/- {self.belief_deviation:.1f}"
                )

        if self.source == PlaceKnowledgeSource.TOLD and self.told_by:
            facts.append(f"told by {self.told_by}")
        elif self.source == PlaceKnowledgeSource.CREATED:
            facts.append("you named it")

        prefix = f"[{handle}] " if handle else ""
        return f"{prefix}{self.label()} ({', '.join(facts)})"


class HabitSpotDescriptor(BaseModel):
    """One unnamed place derived from this NPC's own movement history.

    The simulation re-derives these rows from ``SpatialHabitMap`` every place
    observation pass. There is deliberately no zone id, name, contents belief
    or source field: promotion is what turns the ground into a stable, tellable
    place. ``focal_cell`` remains actionable through MOVE_TO's ordinary
    ``destination`` parameter without inventing a persistent reference.
    """

    model_config = ConfigDict(extra="forbid")

    focal_cell: tuple[int, int]
    distance: int = 0
    beyond_vision: bool = False
    rank: int = 0
    direction: str = "here"

    def label(self) -> str:
        """Possessive place vocabulary, never a generated stable name."""
        if self.rank == 0:
            return "Your usual spot"
        if self.rank == 1:
            return "Your other usual spot"
        return f"Your usual spot #{self.rank + 1}"

    def render_summary(self) -> str:
        """Prompt clause naming the spot and its ordinary cell destination."""
        x, y = self.focal_cell
        if self.direction == "here":
            return f"{self.label()} is here (destination [{x}, {y}])"
        where = f"to the {self.direction}"
        sight = ", out of sight" if self.beyond_vision else ""
        return f"{self.label()} is {where} ({self.distance} away{sight}; destination [{x}, {y}])"


class MarkBudgetState(BaseModel):
    """How many places this NPC is currently holding, and the wait for the next.

    ``next_slot_in_minutes`` carries the simulation's OWN sentinel:
    ``MarkBudget.minutes_until_next_slot`` returns ``-1.0`` when a slot is
    already free, deliberately not ``0.0``, because a genuinely-zero wait is a
    real answer (a mark whose window expires this very minute). So this field is
    not a duration until it is known to be non-negative, and ``render_summary``
    guards it rather than formatting it.
    """

    model_config = ConfigDict(extra="forbid")

    active: int = 0
    cap: int = 0
    next_slot_in_minutes: float = -1.0

    def render_summary(self) -> str:
        """The "you are holding N of M" line, with the wait only when there is one.

        Occupancy is read from ``active``/``cap`` rather than from the sentinel:
        those two cannot be anything but what they say, whereas a negative
        ``next_slot_in_minutes`` is a flag wearing a number's clothes. Both
        guards are applied, so even a self-contradictory pair renders a true
        sentence rather than "the next frees in -1 minutes".
        """
        held = f"You are holding {self.active} of {self.cap} marks."
        if self.active < self.cap or self.next_slot_in_minutes < 0.0:
            return held
        return f"{held[:-1]}; the next frees in {self.next_slot_in_minutes:.0f} minutes."


# Place-block wire versions this model set knows how to read. Unknown versions
# degrade exactly as the goal block's do -- see the validator below.
# v2 (NPC-1473) is a SEMANTIC bump, not a structural one: at v1 provider_count
# and affords were ground truth read live, at v2 they are what the NPC last
# WITNESSED and may be wrong. Nothing about their shape changed, which is
# exactly why it needed a version -- a v1 reader keeps parsing while quietly
# meaning something else.
# v3 (NPC-1479) is the same case again, plus two additive descriptor keys
# (``expected_providers``, ``belief_deviation``): ``confidence`` kept its name and
# shape while becoming a precision reading instead of a discount factor. The
# additive keys are what made its absence here FATAL rather than degraded -- they
# are nested inside each descriptor, the fallback below sheds only root keys, and
# the descriptor forbids extras.
# v4 (NPC-1666) is a pure structural rename: each descriptor's ``anchor`` key is
# now ``focal_cell``, because "anchor" had come to name three different things
# in the simulation. See ``_FOCAL_CELL_CONTRACT_VERSION`` for how older blocks
# still spelling it ``anchor`` are read.
# v5 (NPC-1643) is the v2/v3 case a third time: ``target_place`` keeps its key,
# its shape and its omission rule while ceasing to mean "the place my active goal
# names" and starting to mean "the place I am travelling to". Goals stopped
# carrying places simulation-side -- the destination is chosen at action time --
# so the old meaning has no referent left, and a v4 reader keeps parsing while
# quietly believing something about the NPC's GOAL. Nothing structural moved,
# which is exactly why it needed a version.
# v6 (NPC-1481) adds ``query_result``: descriptors selected from this NPC's own
# uncapped place-knowledge pool in response to the previous cycle's query. It is
# a versioned addition because this model is extra="forbid" and because dropping
# the result would make the deferred request silently inert.
# v7 (NPC-1474) adds ``habit_spots``, a sibling list whose deliberately lighter
# descriptor carries no zone reference. It is versioned because dropping this
# strict root key would make the new perception channel silently inert.
KNOWN_PLACE_CONTRACT_VERSIONS = frozenset({1, 2, 3, 4, 5, 6, 7})

# The first place-block version whose descriptors spell the representative cell
# ``focal_cell``. Every KNOWN version below it spells it ``anchor``.
_FOCAL_CELL_CONTRACT_VERSION = 4

# The descriptor slots of one place block, for the pre-v4 key translation.
_SINGLE_DESCRIPTOR_KEYS = ("current_place", "target_place")
_DESCRIPTOR_LIST_KEY = "known_places"


def _descriptor_with_focal_cell_key(descriptor: Any) -> Any:
    """One pre-v4 descriptor with ``anchor`` respelled ``focal_cell``.

    Untouched when it is not a dict, carries no ``anchor``, or ALREADY carries
    ``focal_cell`` -- a descriptor spelling both is malformed, and leaving the
    stray ``anchor`` in place is what lets ``extra="forbid"`` say so.
    """
    if not isinstance(descriptor, dict) or "anchor" not in descriptor:
        return descriptor
    if "focal_cell" in descriptor:
        return descriptor
    translated = {key: value for key, value in descriptor.items() if key != "anchor"}
    translated["focal_cell"] = descriptor["anchor"]
    return translated


class PlaceObservation(BaseModel):
    """The substrate's place knowledge: where you are, what you know, what you hold.

    Wire producer: the simulation's ``place_observation.gd``. This model is
    matched against SHIPPED producer code -- see the provenance note on
    ``PLACE_BLOCK_CONTRACT_SAMPLE`` in
    ``tests/unit/observations/test_place_observation.py``, which records exactly
    what it was re-derived from.

    It was previously written AHEAD of that producer, against an approved
    specification, and the two disagreed once the producer landed: this model
    declared ``here`` / ``known`` / ``target`` and narrowed the first to a
    three-field object, where the producer emits ``current_place`` /
    ``known_places`` / ``target_place`` and sends a FULL descriptor for the
    first. Because this model and ``Observations`` are both ``extra="forbid"``,
    that did not degrade -- it raised, and the nested failure refused the WHOLE
    observation. Recorded because the shape recurs: a model written against a
    specification is a hypothesis about a producer, and only running the two
    together settles it.

    ``known_places`` is CAPPED simulation-side and ``known_total`` counts the
    whole set, so ``known_total > len(known_places)`` means "you know more places
    than are listed" -- the distinction between knowing three places and being
    shown three of thirty. ``current_place`` and ``target_place`` are
    force-included in the ranking when they exist, so a place named in either is
    also findable in ``known_places`` -- and both carry the same
    ``PlaceDescriptor`` shape that list does, never a narrowed one.

    The block ROOT is ``extra="forbid"`` under a known version, matching every
    ``Goal*`` model and ``InventoryObservation``: the contract adds root keys
    only by bumping ``contract_version`` (v6 ``query_result``, v7
    ``habit_spots``), so an undeclared root key under a known version is a
    contract breach. The DESCRIPTORS inside it are different -- the contract
    promises they grow keys under the same version, so ``PlaceDescriptor``
    degrades on an undeclared key instead. A nested refusal still refuses the
    whole observation: ``Observation``'s root drop sheds unknown BLOCKS only.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: int = 1
    # Root key names are the SIMULATION's, not this module's preference. They
    # were `here` / `known` / `target` while this model was written ahead of its
    # producer; the producer emits `current_place` / `known_places` /
    # `target_place`, and with extra="forbid" a mismatch does not degrade -- it
    # raises, and because Observations is also extra="forbid" the nested error
    # refuses the WHOLE observation. Every MCP NPC would wait, every cycle.
    #: A FULL descriptor, not a narrowed one. The producer sends
    #: ``current_place.to_dict()`` off the same PlaceDescriptor it puts in
    #: ``known_places`` -- and says so: the current and target places take
    #: guaranteed slots IN that menu rather than in addition to it. A narrower
    #: model here (this field was once a three-field ``CurrentPlace``) does not
    #: merely lose the extra keys: extra="forbid" makes each one an error, and
    #: the nested failure refuses the whole observation.
    current_place: PlaceDescriptor | None = None
    known_places: list[PlaceDescriptor] = Field(default_factory=list)
    known_total: int = 0
    #: The place this NPC is TRAVELLING to, absent when it is not travelling
    #: (NPC-1643, block version 5). Below v5 the same key meant "the place my
    #: active goal names" -- a value frozen at goal activation, which could name a
    #: different place than the travel option offered on the same menu. Read it as
    #: a journey, never as a property of the goal; ``render_summary`` says so in
    #: the prose the model actually sees.
    target_place: PlaceDescriptor | None = None
    #: The answer to the place query issued with the PREVIOUS decision, delivered
    #: one cycle later (NPC-1481, block version 6). These are full descriptors
    #: from this NPC's own knowledge, never registry discoveries. They sit beside
    #: ``known_places`` rather than replacing it: the ordinary cap remains the
    #: cycle's passive context and this is the explicit attention pull.
    query_result: list[PlaceDescriptor] | None = None
    #: Unnamed places derived from this NPC's own occupancy history (NPC-1474,
    #: block version 7). Always a sibling of the zone-typed lists: no id means it
    #: cannot enter ``place_handles`` or be mistaken for a tellable place.
    habit_spots: list[HabitSpotDescriptor] = Field(default_factory=list)
    mark_budget: MarkBudgetState | None = None

    @model_validator(mode="before")
    @classmethod
    def _degrade_on_unknown_contract_version(cls, data):
        """Unknown versions degrade, never raise.

        Identical in shape and reasoning to ``GoalObservation``'s: a raise here
        collapses ``decide_action`` into an error response, which is an NPC that
        silently stops acting because the simulation got ahead of the mind. Warn,
        shed undeclared root keys so a purely additive future version parses
        despite ``extra="forbid"``, and parse the rest best-effort. Known-version
        payloads are untouched, so for them an undeclared key stays loud.
        """
        if not isinstance(data, dict):
            return data
        version = data.get("contract_version", 1)
        if version in KNOWN_PLACE_CONTRACT_VERSIONS:
            if version < _FOCAL_CELL_CONTRACT_VERSION:
                return cls._with_focal_cell_keys(data)
            return data
        logger.warning(
            "Place block carries unknown contract_version %s (known: %s); "
            "parsing best-effort under the newest known contract.",
            version,
            sorted(KNOWN_PLACE_CONTRACT_VERSIONS),
        )
        return shed_undeclared_keys(
            cls, data, level=logging.WARNING, context=f"unknown contract_version {version}"
        )

    @staticmethod
    def _with_focal_cell_keys(data: dict) -> dict:
        """A pre-v4 block with every descriptor's ``anchor`` read as ``focal_cell``.

        A cross-repository deployment window, not an in-repo alias: this model can
        merge before the simulation's v4 producer, and until that deploys it
        receives v3 blocks spelling the key the old way. Refusing them would send
        every MCP NPC that knows a place to the wait fallback. Keyed on the
        BLOCK's version -- a v4 descriptor carrying ``anchor`` still fails loud.
        Once no pre-v4 producer remains, this and ``_FOCAL_CELL_CONTRACT_VERSION``
        can go together.
        """
        translated = dict(data)
        for key in _SINGLE_DESCRIPTOR_KEYS:
            if key in translated:
                translated[key] = _descriptor_with_focal_cell_key(translated[key])
        places = translated.get(_DESCRIPTOR_LIST_KEY)
        if isinstance(places, list):
            translated[_DESCRIPTOR_LIST_KEY] = [
                _descriptor_with_focal_cell_key(place) for place in places
            ]
        return translated

    def place_handles(self) -> dict[str, PlaceDescriptor]:
        """This cycle's place handles, ``{"p1": descriptor, ...}`` (NPC-1643).

        A PURE FUNCTION of this block, so the prompt that shows a handle, the
        validator that checks it and the wire translation that resolves it all
        derive one map from one observation -- the map is local to the cycle by
        construction, with nothing stored that could outlive it.

        Numbered in the order the prompt reads: ``known_places`` in the
        simulation's rank order first, then ``current_place`` and
        ``target_place`` only if the ranking did not already carry them (the
        simulation force-includes both, so normally it does). One handle per zone
        id, so a place that appears in two sentences is one place to the model.

        Extension point: any later block that lists places this cycle (PR-3's
        ``query_result``) must be numbered by THIS function, continuing after the
        ones above -- never by a second scheme, or ``p1`` would mean two places
        in one prompt.
        """
        handles: dict[str, PlaceDescriptor] = {}
        seen: set[str] = set()
        ordered = [
            *self.known_places,
            self.current_place,
            self.target_place,
            *(self.query_result or []),
        ]
        for descriptor in ordered:
            if descriptor is None or descriptor.zone_id in seen:
                continue
            seen.add(descriptor.zone_id)
            handles[f"{PLACE_HANDLE_PREFIX}{len(handles) + 1}"] = descriptor
        return handles

    def handle_for(self, zone_id: str) -> str:
        """The handle this cycle gave ``zone_id``, or "" when it has none."""
        for handle, descriptor in self.place_handles().items():
            if descriptor.zone_id == zone_id:
                return handle
        return ""

    def resolve_place_handle(self, reference: str) -> PlaceDescriptor | None:
        """The place a MOVE_TO's ``zone_id`` names, or None when it names none.

        Accepts a handle only -- ``p2``, ``[p2]``, ``P2``. A raw zone id is
        REFUSED even when it belongs to a place in this block: the prompt never
        shows one, so a mind holding one got it through some channel that should
        not exist, and refusing surfaces that leak where accepting would hide it.
        The cost is one retry in a case that should never arise.
        """
        return self.place_handles().get(normalize_place_handle(reference))

    def render_summary(self) -> str:
        """The place block as prompt prose, or "" when there is nothing to say.

        Returning "" for an empty block is what keeps every existing fixture
        rendering byte-identically: an NPC that knows no places and holds no
        marks reads exactly as it did before this block existed.

        Every place is labelled with its ``[pN]`` handle and named by its name
        (``PlaceDescriptor.label``); no zone id is ever rendered.

        Never raises; this runs on the prompt path.
        """
        lines: list[str] = []
        here_zone_id = self.current_place.zone_id if self.current_place else ""
        handles = {
            descriptor.zone_id: handle for handle, descriptor in self.place_handles().items()
        }

        if self.current_place:
            handle = handles.get(self.current_place.zone_id, "")
            lines.append(f"You are at [{handle}] {self.current_place.label()}.")

        if self.known_places:
            listed = "; ".join(
                place.render_summary(here_zone_id, handles.get(place.zone_id, ""))
                for place in self.known_places
            )
            # The count is rendered only when it tells the model something it
            # cannot see: that the list it is reading is a truncation.
            scope = (
                f" ({len(self.known_places)} of {self.known_total})"
                if (self.known_total > len(self.known_places))
                else ""
            )
            lines.append(f"Places you know{scope}: {listed}")

        if self.target_place and self.target_place.zone_id != here_zone_id:
            label = f"[{handles.get(self.target_place.zone_id, '')}] {self.target_place.label()}"
            # "Headed for", not "my goal is aimed at": since block version 5 this
            # is the NPC's journey and says nothing about its goal (NPC-1643). The
            # wording matches the simulation's own ``place_observation.gd::
            # format_for_npc``, so the two tiers cannot describe one fact two ways.
            lines.append(f"You are headed for {label}.")

        if self.query_result is not None:
            if self.query_result:
                listed = "; ".join(
                    place.render_summary(here_zone_id, handles.get(place.zone_id, ""))
                    for place in self.query_result
                )
                lines.append(f"Place query result: {listed}")
            else:
                lines.append("Place query result: no matching places.")

        if self.habit_spots:
            listed = "; ".join(spot.render_summary() for spot in self.habit_spots)
            lines.append(f"Personal habit spots: {listed}.")

        if self.mark_budget:
            lines.append(self.mark_budget.render_summary())

        return "\n".join(lines)


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


class InventoryObservation(BaseModel):
    """What this entity is carrying.

    Wire producer: ``inventory_observation.gd::get_data`` -- exactly
    ``owner_id`` / ``capacity`` / ``used_slots`` / ``items``, emitted every
    decision cycle for any entity with an InventoryComponent (all production
    NPC configs). This model's absence is what blocked ``Observation``'s
    ``extra="forbid"`` (NPC-1116 / NPC-1321): the whole block was discarded
    before reaching the LLM, silently, on every cycle.

    ``items`` are ``EntityData`` -- the SAME shape vision carries, because the
    simulation builds both with ``EntityData.to_dict()``. Carried items are
    co-located with the carrier by construction, so the simulation stomps
    their ``distance_to_observer`` to 0; that field is not on the wire and is
    deliberately not declared here.

    ``extra="forbid"`` (precedent: every ``Goal*`` model): this block is new,
    so there is no legacy payload to break, and a key added simulation-side
    must be a lockstep signal rather than a silent drop.
    """

    model_config = ConfigDict(extra="forbid")

    owner_id: str = ""
    capacity: int = 0
    #: The simulation's own count, carried rather than re-derived from
    #: ``len(items)``: they are the same today, and a future partial or paged
    #: items list must not be able to silently misreport occupancy.
    used_slots: int = 0
    items: list[EntityData] = Field(default_factory=list)


# Entity-memory-block wire versions this model set knows how to read. Unknown
# versions degrade exactly as the goal and place blocks' do -- see the validator
# on ``EntityMemoryObservation``. Like the place block's fallback, it sheds only
# the block's ROOT keys; a key added to a row degrades through the row's own
# additive-key validator, at any version.
KNOWN_ENTITY_MEMORY_CONTRACT_VERSIONS = frozenset({1})


class RememberedEntity(BaseModel):
    """One thing this NPC remembers but may no longer see.

    Wire producer: the simulation's ``EntityMemoryObservation.Remembered.to_dict``
    (``src/minds/observations/entity_memory_observation.gd``, NPC-1504) --
    exactly ``entity_id`` / ``name`` / ``last_cell`` / ``present`` /
    ``age_minutes``, all five emitted on every row. Deliberately NOT the scoring
    surface (confidence, expected yields, interaction names): those stay
    simulation-side on ``EntityDescriptor``.

    An undeclared key DEGRADES (logged at WARNING, dropped) rather than refusing
    the row, because the entity-memory wire contract promises it: "rows grow keys
    under the *same* ``contract_version``". ``extra="forbid"`` stays set as the
    backstop for anything that bypasses the logged drop.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _degrade_on_additive_keys(cls, data):
        return shed_undeclared_keys(
            cls, data, level=logging.WARNING, context="additive entity-memory row key(s)"
        )

    entity_id: str
    #: A snapshot, prose only -- it may name something that no longer exists,
    #: and is never a lookup key.
    name: str = ""
    #: The cell it was last seen at, ``[x, y]`` on the wire (Godot has no JSON
    #: vector type; matches ``PlaceDescriptor.focal_cell``). Required: the
    #: producer always emits it, and no default cell could stand for "unknown"
    #: -- the origin is a real cell on every board.
    last_cell: tuple[int, int]
    #: ``False`` is a DISPROVED belief, not an unknown one: written only by an
    #: arrival that found nothing there. Required because either default would
    #: fabricate evidence -- ``True`` a belief, ``False`` a disproof.
    present: bool
    #: Game minutes since this NPC last saw it.
    age_minutes: int


class EntityMemoryObservation(BaseModel):
    """What this NPC remembers about things it can no longer see (NPC-1504).

    Wire producer: the simulation's ``entity_memory_observation.gd::get_data``
    -- exactly ``contract_version`` / ``remembered`` / ``known_total``. The
    wire contract lives in the simulation repo at
    ``docs/reference/minds/observations.md`` ("Entity-memory block wire
    contract"). Emitted every cycle for any NPC with a ``SubstrateComponent``,
    even when it remembers nothing, so an empty block ("no memories") is
    distinguishable from an absent one ("no block").

    ``remembered`` is nearest-first and CAPPED simulation-side
    (``MAX_SERIALIZED``); ``known_total`` counts the whole set, so
    ``known_total > len(remembered)`` means a longer list exists -- the same
    convention as ``PlaceObservation.known_places``.

    The block's ROOT stays strict (``extra="forbid"``) under a known version:
    the contract promises additive growth for rows, not for the block, so a new
    root key at v1 is a real contract breach and refuses the block. The
    ``Observation`` root drop does not rescue it -- that sheds undeclared ROOT
    BLOCKS, not keys inside a declared one.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: int = 1
    remembered: list[RememberedEntity] = Field(default_factory=list)
    known_total: int = 0

    @model_validator(mode="before")
    @classmethod
    def _degrade_on_unknown_contract_version(cls, data):
        """Unknown versions degrade, never raise.

        Identical in shape and reasoning to ``GoalObservation``'s: a raise here
        collapses ``decide_action`` into an error response, which is an NPC that
        silently stops acting because the simulation got ahead of the mind. Warn,
        shed undeclared root keys so a purely additive future version parses
        despite ``extra="forbid"``, and parse the rest best-effort. Known-version
        payloads are untouched, so for them an undeclared key stays loud.
        """
        if not isinstance(data, dict):
            return data
        version = data.get("contract_version", 1)
        if version in KNOWN_ENTITY_MEMORY_CONTRACT_VERSIONS:
            return data
        logger.warning(
            "Entity-memory block carries unknown contract_version %s (known: %s); "
            "parsing best-effort under the newest known contract.",
            version,
            sorted(KNOWN_ENTITY_MEMORY_CONTRACT_VERSIONS),
        )
        return shed_undeclared_keys(
            cls, data, level=logging.WARNING, context=f"unknown contract_version {version}"
        )


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

    ``id`` is the simulation's per-message identity and is **required**. The
    simulation mints it in ``Message._init`` as a class invariant and mints one
    for any pre-id save on load, so every message a current simulation can emit
    carries one. Accepting an id-less message would mean keeping a parallel
    legacy path alive for a producer that does not exist; an id-less payload is
    instead refused loudly at the parse boundary (see
    ``_extract_conversation_observations``).
    """

    speaker_id: str
    speaker_name: str
    message: str
    id: str
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
    """Complete structured observation.

    Every root key the simulation emits is declared below -- verified against
    simulation ``origin/main`` @ 3d07f5192 by resolving ``get_type()`` for every
    observation added in ``entity_controller.gd`` and
    ``npc_controller.gd::get_current_state_observation``. The wire root key set
    is mechanically ``{entity_id, current_simulation_time}`` plus one key per
    ``get_type()``, because ``composite_observation.gd::get_data`` builds it
    that way and nothing filters it afterwards. ``TestObservationRootDegrade``
    pins this field set against ``wire_full_root_payload``, so drift shows up
    in this repository's CI rather than on a live NPC.

    AN UNDECLARED ROOT KEY DEGRADES, IT DOES NOT REFUSE. It is dropped before
    validation and logged at ERROR, naming every key and saying this mind is
    older than the simulation and needs a pull and a restart; the mind then
    decides on the rest of the observation. This reverses NPC-1116's root
    forbid, which made the failure loud but TOTAL: when the simulation began
    emitting ``entity_memory`` (sim dadd2a5aa) before this model declared it,
    every ``decide_action`` was refused and every MCP NPC waited, every cycle,
    for nine days. A missing block costs one input; a refused observation costs
    the NPC.

    The drop goes through ``shed_undeclared_keys``, the same path the per-block
    unknown-``contract_version`` degrades use, so there is one degrade
    mechanism and it always logs. ``extra="forbid"`` stays set as the backstop:
    anything that reaches validation undeclared still raises, so an unknown key
    is either logged-and-dropped or refused -- never silently ignored.

    RELEASE ORDERING, what still binds: a new root block no longer needs to land
    here first to keep MCP NPCs acting, but it is INVISIBLE to the mind until it
    is declared here and the server is restarted. And the drop is root-only: a
    new key INSIDE a declared block is governed by that block's own policy --
    degraded where its wire contract promises additive growth
    (``PlaceDescriptor``, ``RememberedEntity``), refused where it does not
    (the ``Goal*`` family, ``InventoryObservation``, the ``place`` and
    ``entity_memory`` block roots under a known version) -- so those still need
    the mind change deployed first, or a ``contract_version`` bump.

    ``conversations`` is declared but never on the wire: it is lifted out of
    ``INTERACTION_OBSERVATION`` events by
    ``server.py::_extract_conversation_observations``. It stays declared so
    ``model_dump()`` round-trips through ``model_validate``.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str  # Mind's entity ID in simulation
    current_simulation_time: int

    status: StatusObservation | None = None
    needs: NeedsObservation | None = None
    goal: GoalObservation | None = None
    # OPTIONAL, and that is load-bearing rather than incidental: this model may
    # merge and deploy before the simulation emits the block (the field simply
    # reads None). NPC-1299 owns the producer.
    place: PlaceObservation | None = None
    # Optional for the same reason ``place`` is. The simulation began emitting
    # this block (NPC-1504, sim dadd2a5aa) before this field existed, which is the
    # outage that retired root forbid. Parsed only: nothing mind-side renders or
    # stores it yet.
    entity_memory: EntityMemoryObservation | None = None
    mood: MoodObservation | None = None
    inventory: InventoryObservation | None = None
    vision: VisionObservation | None = None
    conversations: list[ConversationObservation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _degrade_on_undeclared_root_keys(cls, data):
        """Drop an unknown root block and keep deciding; see the class doc."""
        return shed_undeclared_keys(
            cls,
            data,
            level=logging.ERROR,
            context="unknown root observation block(s); deciding without them",
        )

    def __str__(self) -> str:
        """Format observation as natural language for LLM"""
        parts = []

        if self.status:
            # Place first when the NPC knows one - a mind reasons about "the
            # berry grounds", not about a pair of coordinates. The cell is not
            # dropped: travel and distance reasoning still read it.
            if self.status.current_zone_name:
                parts.append(f"Position: {self.status.current_zone_name} {self.status.position}")
            else:
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

        # Rendered only when the block carries something. An observation with no
        # place knowledge is byte-identical to the pre-place rendering, which is
        # what lets every existing fixture stand as a control arm. Placed after
        # the goal pull and before mood because where you are and what you know
        # of it read together with what you are drawn toward.
        if self.place:
            place_text = self.place.render_summary()
            if place_text:
                parts.append(place_text)

        if self.mood:
            parts.append(
                f"Mood: {self.mood.label} "
                f"(valence {self.mood.valence:+.2f} against a resting "
                f"{self.mood.valence_baseline:+.2f}; arousal {self.mood.arousal:.2f} "
                f"against a resting {self.mood.arousal_baseline:.2f})"
            )

        # Rendered only when actually carrying something: an absent line means
        # an empty bag, and every existing fixture without an inventory must
        # render byte-identically (the control-arm pattern the enrichment
        # fixtures rely on). Placed between mood and vision because inventory
        # and vision are the two affordance sources -- they read together.
        if self.inventory and self.inventory.items:
            inv = self.inventory
            carried = []
            for item in inv.items:
                affordances = ", ".join(item.interactions.keys())
                carried.append(
                    f"  - {item.display_name} (ID: {item.entity_id})"
                    + (f" [{affordances}]" if affordances else "")
                )
            parts.append(f"Carrying ({inv.used_slots} of {inv.capacity}):\n" + "\n".join(carried))

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

    def actionable_entities(self) -> list[EntityData]:
        """Entities the actor can target now, from vision or carried inventory.

        Inventory and vision deliberately share ``EntityData`` on the wire. A
        dictionary keeps one entry per id when an item briefly appears in both
        sources during pickup/drop boundary frames.
        """
        by_id: dict[str, EntityData] = {}
        if self.vision:
            for entity in self.vision.visible_entities:
                by_id[entity.entity_id] = entity
        if self.inventory:
            for item in self.inventory.items:
                by_id[item.entity_id] = item
        return list(by_id.values())

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

        # Starting a new activity is unavailable while movement is locked or an
        # interaction is active. Advertising these actions caused the model to
        # repeatedly select options the simulation could only reject.
        can_start_activity = not self.is_interacting() and not (
            self.status and self.status.movement_locked
        )
        if can_start_activity:
            # zone_id is advertised only when this cycle HAS a place handle to put
            # in it (NPC-1643). With no places known it could only be refused, and
            # these tokens sit below the cache breakpoint on every cycle.
            if self.place and self.place.place_handles():
                actions.append(
                    AvailableAction(
                        name=ActionType.MOVE_TO,
                        description="Go to a grid position, or to a place you know",
                        parameters={
                            "destination": "Grid coordinates as tuple (x, y). Give this OR zone_id",
                            "zone_id": (
                                "A place's handle from the brackets beside it, e.g. p1. "
                                "Give this OR destination"
                            ),
                        },
                    )
                )
            else:
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

        # Marking is offered whenever the NPC can SEE, and gated on nothing else.
        # Each half of that is a decision:
        #
        # * Gated on vision because a mark whose extent is not visible to the
        #   marker is structurally refused simulation-side (ZonePresence), so
        #   offering it blind advertises a guaranteed refusal.
        # * ``is not None``, not truthiness: a VisionObservation carrying no
        #   entities is a legitimate "I can see, and there is nothing there" --
        #   which is a perfectly good moment to name empty ground.
        # * NOT gated on the mark budget or on places already known. Neither is
        #   on the wire (NPC-1299 owns them), and a field declared here for them
        #   would parse and read None forever. The interim backstop is the
        #   simulation's own refusal, which names when the next slot frees.
        # * NOT gated on is_interacting(). MarkZoneAction is dispatched by the
        #   component-handler registry BEFORE the state machine and its
        #   get_target_state() is null, so a mark cannot disturb an active
        #   interaction.
        #
        # Keep this prose tight: the menu renders below the prompt's cache
        # breakpoint, so every token here is uncached input on every decision
        # cycle for every MCP NPC.
        if self.vision is not None:
            actions.append(
                AvailableAction(
                    name=ActionType.MARK_ZONE,
                    description=(
                        "Declare a stretch of ground you can see to be a named place. "
                        "Name it yourself, in your own words."
                    ),
                    parameters={
                        "cells": "Cells to mark, as [[x, y], ...]. Give this OR radius, never both",
                        "radius": "Disc radius in cells around you. Give this OR cells, never both",
                        "name": "Required - the name you are giving this place",
                        "kind": "Optional kind of place; 'gathering_ground' is the known one",
                    },
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

        # Interaction-based actions from visible entities and carried items.
        if can_start_activity:
            for entity in self.actionable_entities():
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
