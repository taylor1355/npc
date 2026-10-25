"""Reference parameters: an act names a message or a place by handle, never by id (NPC-2021).

The simulation marks which act parameters refer to something with an optional
``refers_to`` key on the parameter's spec. The mind reads that generically
(``StatusObservation.act_parameter_references``), shows the referent with a
per-cycle handle, refuses a handle that names nothing -- through the reflection
node's existing retry loop -- and swaps the handle for the real id once, in
``Action.wire_payload``. The parameter names below are the simulation's, as the
fixture declares them; they are test data here, exactly as they would arrive on
the wire.
"""

import json
import re
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action
from mind.cognitive_architecture.actions.exceptions import UnknownReferenceHandleError
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from mind.cognitive_architecture.observations import ConversationMessage
from mind.cognitive_architecture.observations.models import PlaceObservation
from mind.cognitive_architecture.references import ParameterReference
from mind.cognitive_architecture.state import PipelineState
from tests.fixtures.observations import wire_current_interaction, wire_property_spec
from tests.fixtures.remark_catalogue import remark_conversation_state

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

ZONE_ORCHARD = "zone_141767cd-68f4-4be2-abff-539a917106a6"
ZONE_POND = "zone_d7ecbf9e-2be7-406a-9d4f-d41a7fedb45a"
MSG_GREETING = "msg_7a1c0d9e-1111-4222-8333-444455556661"
MSG_QUESTION = "msg_7a1c0d9e-1111-4222-8333-444455556662"

REPLY = "in_reply_to"
SUBJECT = "share_belief_subject_id"
KIND = "share_belief_kind"
ASK_SUBJECT = "ask_who_told_subject_id"


def _state() -> PipelineState:
    """A remark-speaking conversation with two messages and two known places."""
    state = remark_conversation_state(with_catalogue=False)
    state.conversation_histories = {
        "conv_1": [
            ConversationMessage(
                id=MSG_GREETING, speaker_id="npc_wren", speaker_name="Wren", message="Hello!"
            ),
            ConversationMessage(
                id=MSG_QUESTION,
                speaker_id="npc_wren",
                speaker_name="Wren",
                message="Who told you about the Old Orchard?",
            ),
        ]
    }
    state.observation.place = PlaceObservation.model_validate(
        {
            "contract_version": 5,
            "known_places": [
                {"zone_id": ZONE_ORCHARD, "name": "the Old Orchard", "distance": 9},
                {"zone_id": ZONE_POND, "name": "Mill Pond", "distance": 12},
            ],
            "known_total": 2,
        }
    )
    return state


def _act(parameters: dict, state: PipelineState) -> Action:
    return Action.model_validate(
        {"action": "act_in_interaction", "parameters": {"message": "Words.", **parameters}},
        context={"state": state},
    )


def _error(exc_info) -> Exception:
    return exc_info.value.errors()[0]["ctx"]["error"]


class TestParameterReferencesAreReadOffTheWire:
    def test_the_advertised_references_are_parsed_from_the_specs(self):
        references = _state().observation.status.act_parameter_references()

        assert references[REPLY].domain == "message"
        assert references[ASK_SUBJECT].domain == "place"
        assert references[SUBJECT].by == KIND
        assert set(references) == {REPLY, ASK_SUBJECT, SUBJECT}, "plain parameters are absent"

    def test_a_conditional_reference_picks_its_domain_from_the_sibling(self):
        reference = _state().observation.status.act_parameter_references()[SUBJECT]

        assert reference.domain_for({KIND: "place_knowledge"}) == "place"
        assert reference.domain_for({}) == "", "no selector: undetermined, so plain"
        assert reference.domain_for({KIND: "something_new"}) == ""

    def test_an_older_simulation_without_the_key_has_no_references(self):
        state = _state()
        specs = state.observation.status.current_interaction["act_in_interaction_parameters"]
        for spec in specs.values():
            spec.pop("refers_to", None)

        assert state.observation.status.act_parameter_references() == {}
        _act({REPLY: MSG_QUESTION}, state)  # unchanged behaviour: an id passes through

    @pytest.mark.parametrize(
        "bad", [7, ["message"], {"by": 3}, {"by": "x"}, {"by": "x", "domains": {}}]
    )
    def test_a_malformed_value_degrades_that_parameter_to_plain(self, bad):
        assert ParameterReference.from_wire(bad, parameter="p") is None

    def test_an_unhandled_domain_is_passed_through(self):
        """A newer simulation can name a domain this mind has no handle for; the
        parameter stays plain rather than being refused."""
        state = _state()
        state.observation.status.current_interaction = wire_current_interaction(
            name="conversation",
            act_parameters={"target": wire_property_spec("string", "", "x", refers_to="gizmo")},
        )

        Action.model_validate(
            {"action": "act_in_interaction", "parameters": {"target": "anything"}},
            context={"state": state},
        )

    def test_the_prompt_hint_tells_the_model_to_give_a_handle(self):
        hints = _state().observation.status.act_parameter_hints()

        assert "[mN]" in hints[REPLY] and "never an id" in hints[REPLY]
        assert "[pN]" in hints[ASK_SUBJECT]
        assert "[pN]" in hints[SUBJECT] and "[mN]" not in hints[SUBJECT]
        assert "refers to" not in hints["message"]


class TestValidation:
    def test_a_valid_message_handle_passes_and_stays_in_handle_space(self):
        action = _act({REPLY: "m2"}, _state())

        assert action.parameters[REPLY] == "m2"

    def test_a_valid_place_handle_passes_on_a_place_reference(self):
        _act({ASK_SUBJECT: "[p2]"}, _state())

    def test_the_conditional_reference_validates_as_a_place_for_a_place_share(self):
        _act({KIND: "place_knowledge", SUBJECT: "p1"}, _state())
        with pytest.raises(ValidationError) as exc_info:
            _act({KIND: "place_knowledge", SUBJECT: "m1"}, _state())
        assert _error(exc_info).domain == "place"

    def test_an_unhandled_conditional_branch_is_left_to_the_simulation(self):
        _act({KIND: "entity_knowledge", SUBJECT: "npc_whoever"}, _state())

    def test_an_unset_reference_is_not_validated(self):
        _act({REPLY: "", ASK_SUBJECT: None}, _state())

    @pytest.mark.parametrize("bogus", ["m9", "m0", "nonsense", "p1"])
    def test_an_unknown_message_handle_is_refused_with_the_valid_ones(self, bogus):
        with pytest.raises(ValidationError) as exc_info:
            _act({REPLY: bogus}, _state())

        error = _error(exc_info)
        assert isinstance(error, UnknownReferenceHandleError)
        assert error.available == ["m1", "m2"]
        assert "m1, m2" in str(error)

    def test_a_raw_message_id_is_refused(self):
        with pytest.raises(ValidationError) as exc_info:
            _act({REPLY: MSG_QUESTION}, _state())
        assert isinstance(_error(exc_info), UnknownReferenceHandleError)

    def test_an_invented_place_handle_is_refused(self):
        with pytest.raises(ValidationError) as exc_info:
            _act({ASK_SUBJECT: "zone_nearby"}, _state())
        error = _error(exc_info)
        assert error.domain == "place" and error.available == ["p1", "p2"]

    def test_a_raw_zone_id_is_refused(self):
        with pytest.raises(ValidationError) as exc_info:
            _act({ASK_SUBJECT: ZONE_POND}, _state())
        assert isinstance(_error(exc_info), UnknownReferenceHandleError)

    def test_with_nothing_to_refer_to_the_error_says_to_leave_it_unset(self):
        state = _state()
        state.conversation_histories = {}
        with pytest.raises(ValidationError, match="leave 'in_reply_to' unset"):
            _act({REPLY: "m1"}, state)


class TestTranslation:
    def test_every_handle_becomes_its_real_id_at_the_wire(self):
        state = _state()
        action = _act({REPLY: "m2", KIND: "place_knowledge", SUBJECT: "[P1]"}, state)

        wire = action.wire_payload(state)["parameters"]

        assert wire[REPLY] == MSG_QUESTION
        assert wire[SUBJECT] == ZONE_ORCHARD
        assert wire[KIND] == "place_knowledge" and wire["message"] == "Words."

    def test_translation_does_not_mutate_the_action(self):
        state = _state()
        action = _act({REPLY: "m1"}, state)

        action.wire_payload(state)

        assert action.parameters[REPLY] == "m1"

    def test_an_unhandled_branch_passes_through_the_wire_unchanged(self):
        state = _state()
        action = _act({KIND: "entity_knowledge", SUBJECT: "npc_whoever"}, state)

        assert action.wire_payload(state)["parameters"][SUBJECT] == "npc_whoever"

    def test_a_handle_the_state_no_longer_reaches_raises_rather_than_sending_a_guess(self):
        state = _state()
        action = _act({REPLY: "m2"}, state)
        later = _state()
        later.conversation_histories = {"conv_1": later.conversation_histories["conv_1"][:1]}

        with pytest.raises(UnknownReferenceHandleError):
            action.wire_payload(later)

    def test_a_retry_derives_the_same_map_as_the_first_attempt(self):
        state = _state()
        first = _act({REPLY: "m2"}, state).wire_payload(state)
        second = _act({REPLY: "m2"}, state).wire_payload(state)

        assert first == second


class TestMemory:
    def test_the_remembered_action_carries_prose_not_handles_or_ids(self):
        state = _state()
        action = _act({REPLY: "m2", ASK_SUBJECT: "p2"}, state)

        remembered = action.memory_parameters(state)

        assert "Old Orchard" in remembered[REPLY] and "Wren" in remembered[REPLY]
        assert remembered[ASK_SUBJECT] == "Mill Pond"
        assert not UUID_RE.search(json.dumps(remembered))
        assert "m2" not in remembered[REPLY] and "p2" not in remembered[ASK_SUBJECT]

    def test_a_handle_the_state_does_not_reach_is_remembered_as_unknown(self):
        state = _state()
        action = _act({REPLY: "m2"}, state)
        state.conversation_histories = {}

        assert action.memory_parameters(state)[REPLY] == "an unknown message"


# --- through the reflection node: the existing retry loop re-asks -------------


def _reflection_json(parameters: dict) -> str:
    return json.dumps(
        {
            "updated_working_memory": {"situation_assessment": "talking"},
            "new_memories": [],
            "chosen_action": {"action": "act_in_interaction", "parameters": parameters},
        }
    )


def _llm_returning(*contents: str) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke.side_effect = [
        AIMessage(
            content=c, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )
        for c in contents
    ]
    return llm


def _sent(llm: AsyncMock, call: int = 0) -> str:
    content = llm.ainvoke.await_args_list[call].args[0][-1].content
    return content if isinstance(content, str) else "".join(b["text"] for b in content)


class TestThroughTheReflectionNode:
    @pytest.mark.asyncio
    async def test_a_fabricated_handle_is_retried_with_the_valid_ones_and_recorded(self):
        llm = _llm_returning(
            _reflection_json({"message": "I heard it from Tobin.", REPLY: MSG_QUESTION}),
            _reflection_json({"message": "I heard it from Tobin.", REPLY: "m2"}),
        )

        result = await ReflectionNode(llm).process(_state())

        assert result.chosen_action.parameters[REPLY] == "m2"
        assert llm.ainvoke.await_count == 2
        failures = result.validation_failures["reflection"]
        assert len(failures) == 1 and "not a message handle" in failures[0]
        assert "m1, m2" in _sent(llm, 1), "the retry must carry the valid handles"

    @pytest.mark.asyncio
    async def test_the_prompt_shows_handles_and_no_real_id(self):
        llm = _llm_returning(_reflection_json({"message": "Hi"}))

        await ReflectionNode(llm).process(_state())

        prompt = _sent(llm)
        assert "[m1] " in prompt and "[m2] " in prompt, "control: the rows were annotated"
        assert "[p1] the Old Orchard" in prompt, "control: the place block was rendered"
        for real_id in (MSG_GREETING, MSG_QUESTION, ZONE_ORCHARD, ZONE_POND):
            assert real_id not in prompt
        assert UUID_RE.search(prompt) is None

    @pytest.mark.asyncio
    async def test_the_action_event_a_later_prompt_renders_holds_no_handle_or_id(self):
        llm = _llm_returning(_reflection_json({"message": "Tobin.", REPLY: "m2"}))

        result = await ReflectionNode(llm).process(_state())

        text = json.dumps(result.recent_events[-1].payload)
        assert UUID_RE.search(text) is None
        assert "Old Orchard" in text
        assert '"m2"' not in text
