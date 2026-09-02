"""Action models for the cognitive architecture"""

from enum import Enum

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from mind.cognitive_architecture.actions.exceptions import (
    ActivityLockedError,
    InvalidEntityError,
    InvalidInteractionError,
    InvalidSelectedOptionError,
    MissingRequiredParameterError,
    MovementLockedError,
    MutuallyExclusiveParametersError,
    NoAdvertisedParameterError,
)


class ActionType(str, Enum):
    """Available action types"""

    MOVE_TO = "move_to"
    MOVE_DIRECTION = "move_direction"
    INTERACT_WITH = "interact_with"
    WANDER = "wander"
    WAIT = "wait"
    CONTINUE = "continue"
    CANCEL_INTERACTION = "cancel_interaction"
    ACT_IN_INTERACTION = "act_in_interaction"
    RESPOND_TO_INTERACTION_BID = "respond_to_interaction_bid"
    BATCH_REJECT_INTERACTION_BIDS = "batch_reject_interaction_bids"
    # The deliberate act of declaring visible ground a named place (NPC-1224 /
    # NPC-1309). The simulation matches this name upper-cased in
    # McpMindClient._create_action_from_mcp_response; a name absent from that
    # match falls through to WaitAction with only a warning, which is why
    # test_mark_zone_contract.py pins the two sides against each other.
    MARK_ZONE = "mark_zone"


# There is deliberately NO MARK_ZONE_RADIUS_ABSENT constant, and it must not be
# re-added as one.
#
# MarkZoneAction.radius declares -1 as its "not supplied" default, and -1 rather
# than 0 because a radius of zero is a legitimate mark ("this cell, where I
# stand"). But the simulation's PREDICATE is `mark.radius >= 0`, not
# `mark.radius != -1`: every negative value reads as not-supplied there. A
# constant anchoring the check would therefore have to be compared as
# `radius != ABSENT`, which accepts -2 as a supplied radius and diverges from the
# simulation in exactly the direction this module exists to prevent.
#
# So the sentinel VALUE is documentation (it explains why the default is -1) and
# the sentinel TEST is `>= 0`. Binding them to one name would make the two look
# interchangeable when they are not.

# The two ways one act may name its extent. Exactly one, never both, never
# neither -- a precedence rule between them would silently discard half of what
# the caller asked for.
MARK_ZONE_EXTENT_PARAMS = ["cells", "radius"]


class Action(BaseModel):
    """Action to be executed"""

    model_config = {"use_enum_values": True}

    action: ActionType
    parameters: dict = Field(
        default_factory=dict,
        description="Action parameters as key-value pairs. Use exact parameter names from the action description.",
    )
    # Selection-output fields for the substrate's Goal Options menu. Both are
    # optional: an absent selected_option_id is a fully legal off-menu answer
    # (bid responses, act_in_interaction, or simply an action the menu did not
    # offer). When present, the simulation resolves the id against the plan it
    # retained for this cycle, so it must be echoed verbatim — the sim treats
    # an unresolvable id loudly, never silently.
    selected_option_id: str | None = Field(
        default=None,
        description=(
            "When the chosen action takes one of the entries under 'Goal Options', "
            "the option_id of that entry, echoed verbatim. Omit entirely when "
            "acting off-menu."
        ),
    )
    selection_rationale: str | None = Field(
        default=None,
        description=(
            "One short sentence on why this option was chosen over the others "
            "(or why an off-menu action beat the menu). Omit when there is "
            "nothing to say."
        ),
    )

    def __str__(self) -> str:
        """Format action for LLM consumption"""
        params_str = (
            ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
            if self.parameters
            else "no parameters"
        )
        return f"{self.action}({params_str})"

    @model_validator(mode="after")
    def validate_executable(self, info: ValidationInfo):
        """Validate action can be executed given pipeline state.

        Requires 'state' in validation context.
        Raises ValidationError (wrapping ActionValidationError) if invalid.
        """
        if not info.context:
            raise ValueError("Action validation requires context with 'state'")

        state = info.context.get("state")
        if not state:
            raise ValueError("Action validation requires 'state' in context")

        observation = state.observation

        # Run validation checks
        self._validate_movement_lock(observation)

        if self.action == ActionType.INTERACT_WITH:
            self._validate_interact_with(observation)
        elif self.action == ActionType.MOVE_TO:
            self._validate_move_to()
        elif self.action == ActionType.RESPOND_TO_INTERACTION_BID:
            self._validate_respond_to_bid(state)
        elif self.action == ActionType.BATCH_REJECT_INTERACTION_BIDS:
            self._validate_batch_reject_bids(state)
        elif self.action == ActionType.ACT_IN_INTERACTION:
            self._validate_act_in_interaction(observation)
        elif self.action == ActionType.MARK_ZONE:
            self._validate_mark_zone()

        # Bid responses are grounded in the live pending-bid set above, not in
        # the goal planner's menu. Models occasionally echo the option they were
        # considering before an invitation arrived. That handle is authoritative
        # in the simulation, so shipping it would execute the retained plan
        # instead of the valid response. Normalize this reactive action back to
        # the off-menu wire shape while retaining its explanatory rationale.
        if self.action in (
            ActionType.RESPOND_TO_INTERACTION_BID,
            ActionType.BATCH_REJECT_INTERACTION_BIDS,
        ):
            self.selected_option_id = None

        self._validate_selected_option(observation)

        return self

    def _validate_movement_lock(self, observation):
        """Check if movement-based actions are blocked"""
        if observation.status and observation.status.movement_locked:
            if self.action in (ActionType.MOVE_TO, ActionType.MOVE_DIRECTION, ActionType.WANDER):
                raise MovementLockedError()

    def _validate_interact_with(self, observation):
        """Validate INTERACT_WITH action against observation"""
        entity_id = self.parameters.get("entity_id")
        interaction_name = self.parameters.get("interaction_name")

        if not entity_id:
            raise MissingRequiredParameterError("entity_id", self.action)
        if not interaction_name:
            raise MissingRequiredParameterError("interaction_name", self.action)

        if observation.is_interacting():
            raise ActivityLockedError(
                self.action,
                "another interaction is active; use 'cancel_interaction', 'continue', "
                "or 'act_in_interaction' instead",
            )
        if observation.status and observation.status.movement_locked:
            raise ActivityLockedError(self.action, "movement is locked by the current activity")

        # Both visible entities and items in the actor's own inventory are
        # actionable. The simulation uses the same EntityData wire shape for
        # both; excluding inventory here made an item visible to the LLM but
        # structurally impossible to consume.
        actionable_entities = observation.actionable_entities()
        if observation.vision is not None or observation.inventory is not None:
            actionable_ids = [entity.entity_id for entity in actionable_entities]

            if entity_id not in actionable_ids:
                raise InvalidEntityError(entity_id, actionable_ids)

            # Check interaction availability
            entity = next(entity for entity in actionable_entities if entity.entity_id == entity_id)
            if interaction_name not in entity.interactions:
                raise InvalidInteractionError(
                    interaction_name, entity_id, list(entity.interactions.keys())
                )

    def _validate_selected_option(self, observation):
        """Require an option echo to resolve and describe the same first step.

        ``selected_option_id`` is an authoritative handle in the simulation. A
        fabricated id or a contradictory action therefore changes what the NPC
        does after validation. Rejecting the pair here gives reflection one
        chance to repair its structured output instead of shipping two answers.
        """
        if self.selected_option_id is None:
            return

        options = observation.goal.options if observation.goal else []
        option = next(
            (candidate for candidate in options if candidate.option_id == self.selected_option_id),
            None,
        )
        if option is None:
            available = [candidate.option_id for candidate in options]
            raise InvalidSelectedOptionError(
                self.selected_option_id,
                f"not in this cycle's Goal Options. Available option_ids: {available}",
            )

        step = next(
            (segment.steps[0] for segment in option.segments if segment.steps),
            None,
        )
        if step is None:
            raise InvalidSelectedOptionError(
                self.selected_option_id,
                "the selected Goal Option has no executable first step",
            )

        expected_action = step.action.name.lower()
        actual_action = str(self.action).lower()
        if expected_action != actual_action or not self._parameters_match(
            self.parameters, step.action.parameters
        ):
            raise InvalidSelectedOptionError(
                self.selected_option_id,
                f"action echo {actual_action}({self.parameters}) does not match the selected "
                f"Goal Option first step {expected_action}({step.action.parameters})",
            )

    @staticmethod
    def _parameters_match(actual: dict, expected: dict) -> bool:
        """Compare wire parameters while treating tuples and lists alike."""

        def normalize(value):
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            return value

        return normalize(actual) == normalize(expected)

    def _validate_move_to(self):
        """Validate MOVE_TO action parameters"""
        if "destination" not in self.parameters:
            raise MissingRequiredParameterError("destination", self.action)

    def _validate_respond_to_bid(self, state):
        """Validate RESPOND_TO_INTERACTION_BID action against pending bids"""
        bid_id = self.parameters.get("bid_id")
        accept = self.parameters.get("accept")

        if not bid_id:
            raise MissingRequiredParameterError("bid_id", self.action)
        if accept is None:
            raise MissingRequiredParameterError("accept", self.action)

        # Check bid exists in pending bids
        if bid_id not in state.pending_incoming_bids:
            raise ValueError(
                f"Invalid bid_id '{bid_id}'. Available bids: {list(state.pending_incoming_bids.keys())}"
            )

        # If rejecting, reason is required
        if not accept and not self.parameters.get("reason"):
            raise MissingRequiredParameterError("reason", self.action)

    def _validate_batch_reject_bids(self, state):
        """Validate BATCH_REJECT_INTERACTION_BIDS action against pending bids"""
        ids = self.parameters.get("ids")
        reason = self.parameters.get("reason")

        if not ids:
            raise MissingRequiredParameterError("ids", self.action)
        if not reason:
            raise MissingRequiredParameterError("reason", self.action)

        # If not wildcard, validate the specified IDs exist
        if ids != "*":
            if not isinstance(ids, list):
                raise ValueError(f"Parameter 'ids' must be '*' or a list, got: {type(ids)}")

            # Check if these are bid IDs or entity IDs
            pending_bid_ids = set(state.pending_incoming_bids.keys())
            pending_entity_ids = {
                event.payload.get("bidder_id") for event in state.pending_incoming_bids.values()
            }

            # Validate each item is either a valid bid_id or entity_id
            for item in ids:
                if item not in pending_bid_ids and item not in pending_entity_ids:
                    raise ValueError(
                        f"'{item}' is not a valid bid_id or entity_id. "
                        f"Available bid_ids: {list(pending_bid_ids)}, "
                        f"Available entity_ids: {list(pending_entity_ids)}"
                    )

    def _validate_act_in_interaction(self, observation):
        """Validate ACT_IN_INTERACTION action requires appropriate parameters.

        Grounds validity in the observation's authoritative interaction signals
        (current_interaction AND activity_state == interacting). A stale
        working-memory belief that "I'm in a conversation" can no longer pass
        validation once the observation says the interaction has ended
        (NPC-688). Missing status defaults to rejected.

        The parameter check is schema-derived: whatever the interaction
        advertises is what an act may carry, and an act must carry at least one
        of them. This names no interaction and no parameter, so a new
        interaction registered in the simulation is validated here with no
        Python change (NPC-1278). The predecessor keyed on a hardcoded
        interaction name read from a wire key that does not exist, so it never
        fired for anything.

        "At least one" rather than "all": the simulation's PropertySpec layer
        supplies defaults for anything omitted, so a partial act is legitimate,
        while a bare ``{}`` is the wasted turn worth converting into a visible,
        retryable error.
        """
        if not observation.is_interacting():
            raise ValueError("ACT_IN_INTERACTION requires an active interaction")

        advertised = observation.status.act_parameter_hints()
        if not advertised:
            # A parameterless interaction accepts a bare act; nothing to check.
            return

        if not any(name in self.parameters for name in advertised):
            raise NoAdvertisedParameterError(list(advertised), self.action)

    def _validate_mark_zone(self):
        """Validate MARK_ZONE names exactly one extent, and names the place.

        The two "was this supplied?" predicates are the simulation's own,
        mirrored EXACTLY from ``substrate_component.gd::_extent_for``::

            has_cells  := not mark.cells.is_empty()
            has_radius := mark.radius >= 0

        Neither is "is the key present", and keying on presence would drift in
        BOTH directions: it would reject ``{"cells": [], "radius": 3}`` and
        ``{"cells": [[1, 2]], "radius": -1}``, which the simulation accepts, and
        accept ``{"cells": [], "radius": -1}``, which it refuses.

        Deliberately absent, each for its own reason:

        - The radius BOUND. The simulation refuses (never clamps) any radius
          past the marker's sight and names both numbers. Sight radius does not
          cross the wire, so a bound here would be a guess that could only be
          wrong.
        - The ``kind`` VOCABULARY. ``Zone.kind_from_string`` routes anything
          unrecognised to ``KIND_INVALID`` and the refusal enumerates
          ``Zone.kind_names()``. Hardcoding it here would make a new
          ``Zone.Kind`` member need a second edit in this repository -- a new
          drift surface, traded for nothing.
        - ``purpose_tags``. Nothing in the simulation's ``src/`` reads
          ``ZoneAttributes.purpose_tags``, so declaring it would advertise
          configuration nothing honours (NPC-1229 cut ``purpose=`` on the
          identical falsifier).
        """
        cells = self.parameters.get("cells")
        has_cells = isinstance(cells, list) and len(cells) > 0

        radius = self.parameters.get("radius")
        # bool is a subclass of int; `radius: true` is not a radius.
        if isinstance(radius, bool):
            radius = None
        elif isinstance(radius, float) and radius.is_integer():
            # The simulation's TypeConverters._convert_to_int accepts a float and
            # truncates, so an integral float is a payload it would honour. A
            # NON-integral one is refused here rather than silently truncated:
            # `radius: 2.7` means something the caller cannot have meant.
            radius = int(radius)
        has_radius = isinstance(radius, int) and radius >= 0

        if has_cells == has_radius:
            supplied = [
                name
                for name, present in zip(
                    MARK_ZONE_EXTENT_PARAMS, (has_cells, has_radius), strict=True
                )
                if present
            ]
            raise MutuallyExclusiveParametersError(MARK_ZONE_EXTENT_PARAMS, self.action, supplied)

        if has_cells:
            for pair in cells:
                if not isinstance(pair, list | tuple) or len(pair) < 2:
                    raise ValueError(
                        f"Malformed cell entry {pair!r} in 'cells' - cells are [x, y] pairs."
                    )

        # Required here, though the simulation would accept a blank name and let
        # ZoneNamer derive one. Naming what you mark in the same act is the
        # design intent (zone-layer-design.md, section "Names": minds that can
        # coin, coin), and a mark without a name yields a place this mind did
        # not choose plus a second round trip to rename it. SimpleMind keeps the
        # blank path in the simulation; it has no language to coin with.
        if not str(self.parameters.get("name") or "").strip():
            raise MissingRequiredParameterError("name", self.action)


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
