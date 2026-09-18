"""The ``move_to`` cross-repository contract: a cell OR a known place (NPC-1643).

A mind may name where it is going two ways -- ``destination``, a grid cell, or
``zone_id``, a place it knows -- and must name exactly one. The simulation
resolves a named place in ``plan_execution.gd::apply_place_travel``; this module
pins that the mind's validator agrees with the simulation's two "was this
supplied?" predicates EXACTLY, because any disagreement ships an action one side
accepts and the other refuses.

ORDERING: this change must merge AFTER the simulation's (finding F4). A
simulation that does not understand ``zone_id`` reads ``{"zone_id": ...}`` as a
``MoveToAction`` whose destination defaults to ``Vector2i.ZERO`` -- a legal cell
-- and walks the NPC to the origin, silently.
"""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action
from mind.cognitive_architecture.actions.exceptions import MutuallyExclusiveParametersError
from mind.cognitive_architecture.actions.models import MOVE_TO_TARGET_PARAMS
from mind.cognitive_architecture.observations import Observation, StatusObservation

# The parameter names ``MoveToAction._get_property_specs`` declares, read from the
# simulation at branch feature/NPC-1643-pr2-named-place-move @ 2a1c2c67
# (``src/contracts/actions/move_to_action.gd``). Re-derive, and update the commit,
# whenever that spec changes.
SIMULATION_MOVE_TO_PARAMS = frozenset({"destination", "zone_id"})

# ``PlanExecution.PLACE_TRAVEL_AMBIGUOUS`` at the same commit, with its ``%s``
# filled by each of the two cases the simulation formats into it. The mind's
# error must carry the same sentence, so an operator reading both logs reads one
# rule rather than two dialects of it.
SIMULATION_AMBIGUOUS_TEMPLATE = "names {} — name exactly one of destination, zone_id"
SIMULATION_BOTH = "both destination and zone_id"
SIMULATION_NEITHER = "neither destination nor zone_id"


def _state() -> Mock:
    state = Mock()
    state.observation = Observation(
        entity_id="npc_alice",
        current_simulation_time=100,
        status=StatusObservation(position=(5, 5), movement_locked=False),
    )
    return state


def _validate(parameters: dict) -> Action:
    return Action.model_validate(
        {"action": "move_to", "parameters": parameters}, context={"state": _state()}
    )


def _domain_error(exc_info):
    """Reach the domain exception pydantic wrapped, to assert on its type."""
    return exc_info.value.errors()[0]["ctx"]["error"]


def _mind_sentence(error: MutuallyExclusiveParametersError) -> str:
    """The rule half of the mind's message, in the simulation's punctuation.

    The mind's exception joins with an ASCII hyphen where the simulation uses an
    em dash; the WORDS are the contract, so the separator is normalized here
    rather than the product changed.
    """
    return str(error).replace(" - name exactly one", " — name exactly one")


def test_the_target_params_are_exactly_the_simulation_s():
    """Both ends name the same two alternatives, in the refusal's order."""
    assert frozenset(MOVE_TO_TARGET_PARAMS) == SIMULATION_MOVE_TO_PARAMS
    assert MOVE_TO_TARGET_PARAMS == ["destination", "zone_id"]


class TestMoveToTarget:
    """Exactly one of destination / zone_id, in all four combinations."""

    def test_a_destination_alone_is_accepted(self):
        action = _validate({"destination": [10, 20]})
        assert action.parameters["destination"] == [10, 20]

    def test_a_zone_id_alone_is_accepted(self):
        """The new capability: a place this mind knows, by id."""
        action = _validate({"zone_id": "zone_berry"})
        assert action.parameters["zone_id"] == "zone_berry"

    def test_naming_both_is_refused(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate({"destination": [10, 20], "zone_id": "zone_berry"})
        error = _domain_error(exc_info)
        assert isinstance(error, MutuallyExclusiveParametersError)
        assert error.supplied == ["destination", "zone_id"]
        assert SIMULATION_AMBIGUOUS_TEMPLATE.format(SIMULATION_BOTH) in _mind_sentence(error)

    def test_naming_neither_is_refused(self):
        """The residual half of F4: an empty move used to walk to the origin."""
        with pytest.raises(ValidationError) as exc_info:
            _validate({})
        error = _domain_error(exc_info)
        assert isinstance(error, MutuallyExclusiveParametersError)
        assert error.supplied == []
        assert SIMULATION_AMBIGUOUS_TEMPLATE.format(SIMULATION_NEITHER) in _mind_sentence(error)


class TestMirrorsTheSimulationPredicates:
    """Where the two predicates are easy to get subtly wrong."""

    def test_the_origin_is_a_legal_destination(self):
        """``has_dest`` is key presence, never a value test.

        ``Vector2i.ZERO`` is a legal cell (NPC-1327), so the simulation cannot use
        a value as a sentinel, and a mind that did would refuse a move the
        simulation accepts.
        """
        action = _validate({"destination": [0, 0]})
        assert action.parameters["destination"] == [0, 0]

    def test_a_null_destination_reads_as_omitted(self):
        """``McpMindClient`` erases a JSON-null destination before constructing."""
        action = _validate({"destination": None, "zone_id": "zone_berry"})
        assert action.parameters["zone_id"] == "zone_berry"

    def test_a_null_zone_id_reads_as_omitted(self):
        """``_optional_string_parameter`` maps JSON null to ``""``."""
        action = _validate({"destination": [1, 2], "zone_id": None})
        assert action.parameters["destination"] == [1, 2]

    def test_an_empty_zone_id_reads_as_omitted(self):
        """The simulation's predicate is ``not zone_id.is_empty()``."""
        action = _validate({"destination": [1, 2], "zone_id": ""})
        assert action.parameters["destination"] == [1, 2]

    def test_both_null_is_neither(self):
        with pytest.raises(ValidationError) as exc_info:
            _validate({"destination": None, "zone_id": None})
        assert _domain_error(exc_info).supplied == []

    def test_a_whitespace_zone_id_is_NOT_stripped(self):
        """The simulation does not strip, so neither may the mind.

        ``"  "`` is a supplied id there (refused later as a place the NPC does not
        know). Stripping here would pass ``{destination, "  "}`` as a plain move
        that the simulation refuses as naming both.
        """
        with pytest.raises(ValidationError) as exc_info:
            _validate({"destination": [1, 2], "zone_id": "  "})
        assert _domain_error(exc_info).supplied == ["destination", "zone_id"]
