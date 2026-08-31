"""The ``mark_zone`` cross-repository contract (NPC-1224 / NPC-1309).

The simulation has carried a live ``MARK_ZONE`` arm since NPC-1224, and no mind
could reach it, because ``mark_zone`` was absent from ``ActionType`` and from the
per-cycle action menu. Nothing went red for that: an action name the simulation
does not match falls through to ``WaitAction`` with a single warning, so the arm
sat there looking shipped. This module is the tripwire for both directions of
that drift.
"""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action, ActionType
from mind.cognitive_architecture.actions.exceptions import (
    MissingRequiredParameterError,
    MutuallyExclusiveParametersError,
)
from mind.cognitive_architecture.observations import (
    Observation,
    StatusObservation,
    VisionObservation,
)

# The action names ``McpMindClient._create_action_from_mcp_response`` matches,
# read from the simulation repository at origin/main @ 0d5fc725
# (``src/clients/mcp/mcp_mind_client.gd``, the ``match action_name.to_upper()``).
#
# THIS SET AND ``ActionType`` MUST BE EQUAL, and the equality is the point.
#   - A name in ``ActionType`` and NOT here: this mind can emit something the
#     simulation does not match. It falls through to ``WaitAction`` with one
#     warning -- no exception, no failed decision cycle, no red anywhere.
#   - A name here and NOT in ``ActionType``: the simulation carries a live arm no
#     mind can reach. That is exactly NPC-1309 -- ``mark_zone`` sat in this set
#     and not in the enum, and nothing anywhere went red for it.
#
# Written as a literal because the simulation is a different repository and this
# suite cannot read it. Re-derive it, and update the commit above, whenever an
# arm is added to or removed from that match.
SIMULATION_ACTION_ARMS = frozenset(
    {
        "continue",
        "wait",
        "wander",
        "move_to",
        "move_direction",
        "mark_zone",
        "interact_with",
        "act_in_interaction",
        "cancel_interaction",
        "respond_to_interaction_bid",
        "batch_reject_interaction_bids",
    }
)

# The parameter names ``MarkZoneAction._get_property_specs`` declares, read from
# the same simulation commit (``src/contracts/actions/mark_zone_action.gd``).
#
# ``purpose_tags`` is ABSENT and must stay absent. Nothing in the simulation's
# ``src/`` reads ``ZoneAttributes.purpose_tags``, and ``ZoneCommands.
# CREATE_OPTION_KEYS`` cut ``purpose=`` on that identical falsifier (NPC-1229),
# so declaring it here would advertise configuration nothing honours. The
# ``ZoneAttributes`` bag is the seam when a consumer arrives; adding the
# parameter then is purely additive.
SIMULATION_MARK_ZONE_PARAMS = frozenset({"cells", "radius", "name", "kind"})

# The payload the simulation's own documentation uses as its worked example, at
# the same commit. Pinned so the shape a reader is taught is the shape this
# package actually accepts.
MARK_ZONE_CONTRACT_SAMPLE = {
    "action": "mark_zone",
    "parameters": {"radius": 2, "name": "the berry grounds"},
}


def _observation(*, with_vision: bool = True, interacting: bool = False) -> Observation:
    """An observation carrying only what the mark_zone paths actually read."""
    interaction = (
        {
            "name": "conversation",
            "description": "Talk",
            "act_in_interaction_parameters": {},
        }
        if interacting
        else {}
    )
    return Observation(
        entity_id="npc_alice",
        current_simulation_time=100,
        status=StatusObservation(
            position=(5, 5),
            movement_locked=False,
            current_interaction=interaction,
            activity_state={"state_name": "interacting" if interacting else "idle"},
        ),
        vision=VisionObservation(visible_entities=[]) if with_vision else None,
    )


def _state(observation: Observation) -> Mock:
    state = Mock()
    state.observation = observation
    return state


def _validate(parameters: dict, observation: Observation | None = None) -> Action:
    return Action.model_validate(
        {"action": "mark_zone", "parameters": parameters},
        context={"state": _state(observation or _observation())},
    )


def _domain_error(exc_info):
    """Reach the domain exception pydantic wrapped, to assert on its type."""
    return exc_info.value.errors()[0]["ctx"]["error"]


def test_the_emittable_action_names_are_exactly_the_simulation_s_arms():
    """Every name this mind can emit is matched there, and vice versa.

    ``==`` rather than ``<=`` deliberately: the subset direction alone would
    miss the failure NPC-1309 actually was, where the simulation held an arm no
    mind could reach.
    """
    assert {action.value for action in ActionType} == SIMULATION_ACTION_ARMS


def test_the_advertised_parameters_are_exactly_the_simulation_s_property_specs():
    """The menu advertises the simulation's parameter surface, no more, no less.

    An advertised parameter the simulation does not declare is silently dropped
    by ``BaseAction._init`` (it only reads declared ``PropertySpec``s), so the
    LLM would be taught a knob that does nothing. An undeclared one the
    simulation does honour is capability the mind can never reach.
    """
    entry = next(
        action
        for action in _observation().get_available_actions()
        if action.name == ActionType.MARK_ZONE
    )
    assert set(entry.parameters) == SIMULATION_MARK_ZONE_PARAMS


def test_the_documented_contract_sample_validates_and_round_trips():
    action = Action.model_validate(
        MARK_ZONE_CONTRACT_SAMPLE, context={"state": _state(_observation())}
    )
    dumped = action.model_dump()

    # use_enum_values, so the enum dumps as the bare string the simulation
    # upper-cases and matches.
    assert dumped["action"] == "mark_zone"
    assert dumped["parameters"] == MARK_ZONE_CONTRACT_SAMPLE["parameters"]
    # Verbatim: the coined name reaches ZoneSpec.name_override untouched, never
    # normalised into a key. Coined names are unconstrained text by design.
    assert dumped["parameters"]["name"] == "the berry grounds"


class TestMarkZoneExtent:
    """Exactly one of cells/radius, mirroring ``substrate_component.gd::_extent_for``."""

    def test_cells_alone_is_accepted(self):
        action = _validate({"cells": [[1, 2], [1, 3]], "name": "x"})
        assert action.parameters["cells"] == [[1, 2], [1, 3]]

    def test_radius_alone_is_accepted(self):
        action = _validate({"radius": 3, "name": "x"})
        assert action.parameters["radius"] == 3

    def test_a_radius_of_zero_is_a_legitimate_one_cell_mark(self):
        """The sentinel-collision test.

        A zero radius marks the single cell you stand on. If the "not supplied"
        sentinel were ``0`` instead of ``-1`` this payload would be unspeakable;
        the simulation pins the same case in
        ``test_a_radius_of_zero_marks_the_single_cell_you_stand_on``.
        """
        action = _validate({"radius": 0, "name": "x"})
        assert action.parameters["radius"] == 0

    def test_an_integral_float_radius_is_accepted(self):
        """The simulation's TypeConverters would coerce it, so this side must too."""
        action = _validate({"radius": 2.0, "name": "x"})
        assert action.parameters["radius"] == 2.0

    def test_a_fractional_radius_is_refused_not_truncated(self):
        """``radius: 2.7`` means something the caller cannot have meant."""
        with pytest.raises(ValidationError):
            _validate({"radius": 2.7, "name": "x"})

    def test_a_boolean_is_not_a_radius(self):
        """``bool`` is a subclass of ``int``; ``radius: true`` must not read as 1."""
        with pytest.raises(ValidationError) as exc_info:
            _validate({"radius": True, "name": "x"})
        assert isinstance(_domain_error(exc_info), MutuallyExclusiveParametersError)

    def test_naming_both_is_refused(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate({"cells": [[1, 2]], "radius": 3, "name": "x"})
        error = _domain_error(exc_info)
        assert isinstance(error, MutuallyExclusiveParametersError)
        assert error.supplied == ["cells", "radius"]
        assert "both cells and radius" in str(error)

    def test_naming_neither_is_refused(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate({"name": "x"})
        error = _domain_error(exc_info)
        assert isinstance(error, MutuallyExclusiveParametersError)
        assert error.supplied == []
        assert "neither cells nor radius" in str(error)

    def test_an_empty_cells_list_is_not_a_supplied_extent(self):
        """Simulation fidelity: ``not mark.cells.is_empty()`` is the predicate.

        Keying on KEY PRESENCE instead would reject this payload, which the
        simulation accepts (radius wins because cells supplies no extent).
        """
        action = _validate({"cells": [], "radius": 3, "name": "x"})
        assert action.parameters["radius"] == 3

    def test_the_radius_sentinel_is_not_a_supplied_extent(self):
        """Simulation fidelity: ``mark.radius >= 0`` is the predicate.

        Keying on key presence would reject this too, and the simulation accepts
        it -- ``-1`` means "not supplied", so the cells win.
        """
        action = _validate({"cells": [[1, 2]], "radius": -1, "name": "x"})
        assert action.parameters["cells"] == [[1, 2]]

    def test_empty_cells_and_the_sentinel_together_are_neither(self):
        """The third payload key-presence would get wrong -- in the other direction.

        Both keys are PRESENT and neither supplies an extent, so the simulation
        refuses it. A presence check would have accepted it.
        """
        with pytest.raises(ValidationError) as exc_info:
            _validate({"cells": [], "radius": -1, "name": "x"})
        assert "neither cells nor radius" in str(_domain_error(exc_info))

    def test_a_malformed_cell_pair_is_refused(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate({"cells": [[1]], "name": "x"})
        assert "cells are [x, y] pairs" in str(exc_info.value)


class TestMarkZoneName:
    """``name`` is required here, and deliberately stricter than the simulation."""

    def test_a_mark_with_no_name_is_refused(self):
        """Stricter than the simulation, on purpose.

        The simulation accepts a blank name and lets ``ZoneNamer`` derive one.
        A mind that can coin, coins (zone-layer-design.md, section "Names"):
        marking without naming yields a place this mind did not choose, plus a
        second round trip to rename it.
        """
        with pytest.raises(ValidationError) as exc_info:
            _validate({"radius": 2})
        error = _domain_error(exc_info)
        assert isinstance(error, MissingRequiredParameterError)
        assert error.param_name == "name"

    def test_a_whitespace_name_is_not_a_name(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate({"radius": 2, "name": "   "})
        assert isinstance(_domain_error(exc_info), MissingRequiredParameterError)


def test_an_unknown_kind_is_left_to_the_simulation():
    """An assertion that this package does NOT validate something.

    ``Zone.kind_from_string`` routes anything unrecognised to ``KIND_INVALID``
    and the refusal enumerates ``Zone.kind_names()``, so the vocabulary is
    authoritative and self-describing there. A mind-side allowlist would make a
    new ``Zone.Kind`` member need a second edit in this repository -- a new drift
    surface traded for nothing. If a future contributor adds one "helpfully",
    this test goes red and this docstring says why.
    """
    action = _validate({"radius": 2, "name": "x", "kind": "orchard"})
    assert action.parameters["kind"] == "orchard"


def test_an_enormous_radius_is_left_to_the_simulation():
    """The other deliberate non-validation: the radius BOUND.

    The simulation refuses (never clamps) any radius past the marker's sight and
    names BOTH numbers in the refusal, which teaches the model more than a
    mind-side guess could. Sight radius does not cross the wire, so a bound here
    could only ever be a guess. The simulation pins its half in
    ``test_an_enormous_radius_is_refused_without_ever_allocating_it``.
    """
    action = _validate({"radius": 9999, "name": "x"})
    assert action.parameters["radius"] == 9999


class TestMarkZoneMenu:
    """When the per-cycle menu offers marking."""

    def _names(self, observation: Observation) -> list[str]:
        return [action.name for action in observation.get_available_actions()]

    def test_marking_is_offered_when_the_npc_can_see(self):
        assert ActionType.MARK_ZONE in self._names(_observation())

    def test_marking_is_not_offered_without_vision(self):
        """Marking ground you cannot see is structurally refused by ZonePresence,
        so offering it blind would advertise a guaranteed refusal."""
        assert ActionType.MARK_ZONE not in self._names(_observation(with_vision=False))

    def test_an_empty_field_of_view_still_offers_marking(self):
        """``is not None``, not truthiness.

        A VisionObservation with no entities is a legitimate "I can see, and
        there is nothing there" -- which is a perfectly good moment to name empty
        ground. A truthiness gate would silently withdraw the verb exactly then.
        """
        observation = _observation()
        assert observation.vision.visible_entities == []
        assert ActionType.MARK_ZONE in self._names(observation)

    def test_marking_is_still_offered_while_interacting(self):
        """Pins the deliberate NON-gating on is_interacting().

        ``MarkZoneAction`` is dispatched by the component-handler registry
        BEFORE the state machine and its ``get_target_state()`` is null, so a
        mark cannot disturb an active interaction.
        """
        observation = _observation(interacting=True)
        assert observation.is_interacting()
        assert ActionType.MARK_ZONE in self._names(observation)

    def test_marking_while_interacting_also_validates(self):
        """The menu and the validator must agree; an offer the validator refuses
        would burn two retries and land on WAIT."""
        action = _validate({"radius": 2, "name": "x"}, _observation(interacting=True))
        assert action.action == "mark_zone"
