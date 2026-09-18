"""Place handles: how a mind names a place without ever seeing its zone id (NPC-1643).

The prompt labels each place with a per-cycle handle (``[p1]``, ``[p2]``, ...) the
way a menu option carries its option_id. A MOVE_TO sends the handle; the mind
refuses any handle this cycle did not issue, and translates a valid one to the real
zone id only as the action leaves the server (``Action.wire_payload``). So the
simulation receives real ids, and no UUID enters a prompt -- not this cycle's, and
not a later one's through the memory of what was chosen.

Four properties, each with a test that fails without it:

1. A handle maps to the right place, and translation happens at the wire.
2. A handle this cycle did not issue is refused loudly -- never passed through.
3. Handles are stable within a cycle: the prompt, the validator and the
   translation agree, because all three derive one map from one observation.
4. No UUID appears anywhere in the rendered prompt, including the next cycle's.
"""

import re
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action, ActionType
from mind.cognitive_architecture.actions.exceptions import UnknownPlaceHandleError
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.observations.models import PlaceObservation
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import WorkingMemory

# Real-shaped ids: the simulation mints ``zone_`` + a UUID4.
ZONE_BERRY = "zone_141767cd-68f4-4be2-abff-539a917106a6"
ZONE_POND = "zone_d7ecbf9e-2be7-406a-9d4f-d41a7fedb45a"
ZONE_NAMELESS = "zone_71140def-17a9-4a01-a148-7cd7b3703c19"
ALL_ZONE_IDS = (ZONE_BERRY, ZONE_POND, ZONE_NAMELESS)

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# The simulation's MoveToAction property specs; see test_move_to_contract.py.
SIMULATION_MOVE_TO_PARAMS = frozenset({"destination", "zone_id"})


def _place_block(order: tuple[str, ...] = (ZONE_BERRY, ZONE_POND, ZONE_NAMELESS)) -> dict:
    names = {ZONE_BERRY: "the berry grounds", ZONE_POND: "the pond bend", ZONE_NAMELESS: ""}
    places = [
        {"zone_id": zone_id, "name": names[zone_id], "distance": 10 + i, "beyond_vision": True}
        for i, zone_id in enumerate(order)
    ]
    return {
        "contract_version": 5,
        "current_place": places[0],
        "known_places": places,
        "known_total": len(places),
        "target_place": places[1],
    }


def _observation(order: tuple[str, ...] = (ZONE_BERRY, ZONE_POND, ZONE_NAMELESS)) -> Observation:
    return Observation(
        entity_id="npc_alice",
        current_simulation_time=100,
        # current_zone_id is on the status wire too; it must not render either.
        status=StatusObservation(position=(5, 5), movement_locked=False, current_zone_id=order[0]),
        place=PlaceObservation.model_validate(_place_block(order)),
    )


def _validate(parameters: dict, observation: Observation | None = None) -> Action:
    state = Mock()
    state.observation = observation or _observation()
    return Action.model_validate(
        {"action": "move_to", "parameters": parameters}, context={"state": state}
    )


def _domain_error(exc_info):
    return exc_info.value.errors()[0]["ctx"]["error"]


# ---------------------------------------------------------------------------
# 1. A handle maps to the right place
# ---------------------------------------------------------------------------


class TestHandleResolution:
    def test_handles_number_the_ranked_list_in_prompt_order(self):
        handles = _observation().place.place_handles()
        assert {h: d.zone_id for h, d in handles.items()} == {
            "p1": ZONE_BERRY,
            "p2": ZONE_POND,
            "p3": ZONE_NAMELESS,
        }

    def test_a_handle_translates_to_its_real_zone_id_at_the_wire(self):
        observation = _observation()
        action = _validate({"zone_id": "p2"}, observation)
        assert action.parameters["zone_id"] == "p2", (
            "the action stays in handle-space; only the wire payload carries the id"
        )
        assert action.wire_payload(observation)["parameters"]["zone_id"] == ZONE_POND

    @pytest.mark.parametrize("spelling", ["p2", "[p2]", "P2", " [P2] "])
    def test_the_bracketed_and_upper_case_spellings_resolve(self, spelling):
        """The simulation never sees a handle, so leniency here cannot make the
        two tiers disagree about what was supplied."""
        observation = _observation()
        payload = _validate({"zone_id": spelling}, observation).wire_payload(observation)
        assert payload["parameters"]["zone_id"] == ZONE_POND

    def test_a_coordinate_move_passes_through_the_wire_untouched(self):
        observation = _observation()
        action = _validate({"destination": [3, 4]}, observation)
        assert action.wire_payload(observation) == action.model_dump()


# ---------------------------------------------------------------------------
# 2. A handle this cycle did not issue is refused
# ---------------------------------------------------------------------------


class TestUnknownReferencesAreRefused:
    @pytest.mark.parametrize(
        "reference",
        [
            "p9",  # never issued
            "the berry grounds",  # a name, not a handle
            ZONE_BERRY,  # a real id of a place IN this block: refused by decision
        ],
    )
    def test_anything_but_this_cycle_s_handle_is_refused(self, reference):
        with pytest.raises(ValidationError) as exc_info:
            _validate({"zone_id": reference})
        error = _domain_error(exc_info)
        assert isinstance(error, UnknownPlaceHandleError)
        assert error.available == ["p1", "p2", "p3"], "the refusal lists the valid handles"
        assert "p1, p2, p3" in str(error)

    def test_a_mind_that_knows_no_places_is_told_to_give_a_destination(self):
        observation = Observation(
            entity_id="npc_alice",
            current_simulation_time=100,
            status=StatusObservation(position=(5, 5), movement_locked=False),
        )
        with pytest.raises(ValidationError) as exc_info:
            _validate({"zone_id": "p1"}, observation)
        error = _domain_error(exc_info)
        assert isinstance(error, UnknownPlaceHandleError)
        assert "give a destination" in str(error)

    def test_a_handle_is_not_valid_in_another_cycle(self):
        """The map is local to the cycle that built it. An action validated in
        one cycle and translated against another cycle's block whose list no
        longer reaches that handle raises rather than sending a guess."""
        this_cycle = _observation()
        action = _validate({"zone_id": "p3"}, this_cycle)
        next_cycle = Observation(
            entity_id="npc_alice",
            current_simulation_time=160,
            place=PlaceObservation.model_validate(
                {"known_places": [{"zone_id": ZONE_POND, "name": "the pond bend"}]}
            ),
        )
        with pytest.raises(UnknownPlaceHandleError):
            action.wire_payload(next_cycle)


# ---------------------------------------------------------------------------
# 3. Stable within a cycle: prompt, validator and wire agree
# ---------------------------------------------------------------------------


class TestHandlesAreStableWithinACycle:
    def test_repeated_derivations_agree(self):
        place = _observation().place
        first = {h: d.zone_id for h, d in place.place_handles().items()}
        second = {h: d.zone_id for h, d in place.place_handles().items()}
        assert first == second

    def test_every_rendered_handle_resolves_to_the_place_it_labels(self):
        """What the model READS beside a name is what the validator resolves."""
        place = _observation().place
        rendered = place.render_summary()
        for handle, descriptor in place.place_handles().items():
            assert f"[{handle}] {descriptor.label()}" in rendered
            assert place.resolve_place_handle(handle).zone_id == descriptor.zone_id

    def test_a_place_named_in_two_sentences_carries_one_handle(self):
        """The current and target places are also in the ranked list; each is one
        place to the model, so each gets the handle the list gave it."""
        rendered = _observation().place.render_summary()
        assert "You are at [p1] the berry grounds." in rendered
        assert "You are headed for [p2] the pond bend." in rendered
        assert "[p4]" not in rendered


# ---------------------------------------------------------------------------
# The menu advertises what the simulation accepts, and only when it can be used
# ---------------------------------------------------------------------------


class TestMoveToMenuEntry:
    def _entry(self, observation: Observation):
        return next(
            action
            for action in observation.get_available_actions()
            if action.name == ActionType.MOVE_TO
        )

    def test_with_places_the_menu_advertises_the_simulation_s_parameters(self):
        assert set(self._entry(_observation()).parameters) == SIMULATION_MOVE_TO_PARAMS

    def test_with_no_places_zone_id_is_not_advertised(self):
        """It could only be refused, and these tokens are uncached every cycle."""
        bare = Observation(
            entity_id="npc_alice",
            current_simulation_time=100,
            status=StatusObservation(position=(5, 5), movement_locked=False),
        )
        assert set(self._entry(bare).parameters) == {"destination"}


# ---------------------------------------------------------------------------
# 4. No UUID anywhere in the rendered prompt -- this cycle's or the next
# ---------------------------------------------------------------------------


def _llm_choosing(parameters_json: str) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(
        content=(
            '{"updated_working_memory": {"situation_assessment": "hungry"}, '
            '"new_memories": [], '
            f'"chosen_action": {{"action": "move_to", "parameters": {parameters_json}}}}}'
        ),
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )
    return llm


def _state(observation: Observation, recent_events=None) -> PipelineState:
    """Built the way ``Mind.build_pipeline_state`` builds it."""
    return PipelineState(
        observation=observation,
        available_actions=observation.get_available_actions(),
        working_memory=WorkingMemory(situation_assessment="hungry"),
        recent_events=list(recent_events or []),
    )


def _sent_prompt(llm: AsyncMock) -> str:
    content = llm.ainvoke.call_args[0][0][0].content
    return content if isinstance(content, str) else "".join(block["text"] for block in content)


class TestNoUuidReachesThePrompt:
    def test_the_detector_can_see_a_zone_id(self):
        """Known-positive control: an empty search below means something only if
        the pattern matches the ids it is looking for."""
        for zone_id in ALL_ZONE_IDS:
            assert UUID_RE.search(zone_id)

    async def test_this_cycle_s_prompt_carries_handles_and_no_zone_id(self):
        observation = _observation()
        llm = _llm_choosing('{"zone_id": "p1"}')
        state = await ReflectionNode(llm).process(_state(observation))

        prompt = _sent_prompt(llm)
        assert "[p1] the berry grounds" in prompt, "control: the place block was rendered"
        assert "e.g. p1" in prompt, "control: the menu entry was rendered"
        assert UUID_RE.search(prompt) is None
        for zone_id in ALL_ZONE_IDS:
            assert zone_id not in prompt

        assert state.chosen_action.wire_payload(observation)["parameters"]["zone_id"] == (
            ZONE_BERRY
        ), "and the simulation still receives the real id"

    async def test_the_next_cycle_remembers_the_place_by_name_not_by_handle_or_id(self):
        """``ACTION_CHOSEN`` renders into LATER prompts. The id would be a UUID in
        the prompt; the handle would outlive its cycle -- below, ``p1`` names a
        different place next cycle, so "I chose p1" would be a false memory."""
        first = _observation()
        chose = await ReflectionNode(_llm_choosing('{"zone_id": "p1"}')).process(_state(first))

        reordered = _observation(order=(ZONE_POND, ZONE_BERRY, ZONE_NAMELESS))
        assert reordered.place.resolve_place_handle("p1").zone_id == ZONE_POND, (
            "precondition: p1 means a different place this cycle"
        )
        llm = _llm_choosing('{"destination": [1, 2]}')
        await ReflectionNode(llm).process(_state(reordered, chose.recent_events))

        prompt = _sent_prompt(llm)
        assert "Chose action: move_to(place=the berry grounds)" in prompt
        assert "zone_id=p1" not in prompt
        assert UUID_RE.search(prompt) is None
