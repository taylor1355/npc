"""Required act parameters: a spec the simulation marks ``required`` must be set.

The simulation's PropertySpec layer fills a default in for any omitted act
parameter, which is why ``_validate_act_in_interaction`` only asks for "at least
one" advertised parameter. A spec carrying ``"required": true`` opts out of that:
the simulation fills nothing in and refuses an act that omits it. The mind reads
the key generically (``StatusObservation.act_parameter_required``), says so in the
prompt hint, and refuses the omission through the reflection node's existing
retry loop, so a model that forgets is re-asked rather than losing the turn.

The parameter names below are invented test data; nothing in ``mind/src`` may
name an act parameter, and nothing here needs to.
"""

import json
import logging
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action, ActionType
from mind.cognitive_architecture.actions.exceptions import MissingRequiredParameterError
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.state import PipelineState
from tests.fixtures.observations import wire_current_interaction, wire_property_spec

WORDS = "words"
SHADE = "shade"
VOLUME = "volume"


def _interaction(required: bool = True, flag=None) -> dict:
    shade = wire_property_spec("float", 0.0, "How the words land, -1..1", required=required)
    if flag is not None:
        shade["required"] = flag
    return wire_current_interaction(
        "chatter",
        act_parameters={
            WORDS: wire_property_spec("string", "", "What to say"),
            SHADE: shade,
            VOLUME: wire_property_spec("int", 1, "How loud"),
        },
    )


def _observation(current_interaction: dict) -> Observation:
    return Observation(
        entity_id="npc",
        current_simulation_time=1,
        status=StatusObservation(
            position=(10, 10),
            movement_locked=True,
            current_interaction=current_interaction,
            activity_state={"state_name": "interacting"},
        ),
    )


def _state(current_interaction: dict | None = None) -> PipelineState:
    observation = _observation(current_interaction or _interaction())
    # The menu the perception step would have offered, so the prompt renders the hints.
    return PipelineState(
        observation=observation, available_actions=observation.get_available_actions()
    )


def _act(parameters: dict, state: PipelineState) -> Action:
    return Action.model_validate(
        {"action": "act_in_interaction", "parameters": parameters}, context={"state": state}
    )


def _error(exc_info) -> Exception:
    return exc_info.value.errors()[0]["ctx"]["error"]


def _hints(current_interaction: dict) -> dict[str, str]:
    for action in _observation(current_interaction).get_available_actions():
        if action.name == ActionType.ACT_IN_INTERACTION:
            return action.parameters
    raise AssertionError("act_in_interaction was not offered")


class TestRequiredIsReadOffTheWire:
    def test_the_required_parameters_are_listed(self):
        assert _observation(_interaction()).status.act_parameter_required() == [SHADE]

    def test_an_older_simulation_without_the_key_requires_nothing(self):
        assert _observation(_interaction(required=False)).status.act_parameter_required() == []

    def test_a_false_flag_requires_nothing(self):
        observation = _observation(_interaction(required=False, flag=False))

        assert observation.status.act_parameter_required() == []

    @pytest.mark.parametrize("bad", ["true", 1, "yes", [True]])
    def test_a_non_boolean_flag_is_optional_and_logged(self, bad, caplog):
        observation = _observation(_interaction(required=False, flag=bad))

        with caplog.at_level(logging.WARNING):
            assert observation.status.act_parameter_required() == []
        assert any("non-boolean" in r.getMessage() for r in caplog.records)

    def test_no_current_interaction_requires_nothing(self):
        assert _observation({}).status.act_parameter_required() == []


class TestThePromptHintSaysSo:
    def test_a_required_hint_says_required_and_shows_no_default(self):
        hint = _hints(_interaction())[SHADE]

        assert "required" in hint
        assert "default" not in hint

    def test_an_optional_hint_is_unchanged(self):
        hint = _hints(_interaction())[VOLUME]

        assert "default: 1" in hint
        assert "required" not in hint

    def test_a_required_flag_beats_a_stray_default_on_the_wire(self):
        interaction = _interaction()
        interaction["act_in_interaction_parameters"][SHADE]["default"] = 0.0

        assert "default" not in _hints(interaction)[SHADE]


class TestValidation:
    def test_an_act_omitting_a_required_parameter_is_refused_by_name(self):
        with pytest.raises(ValidationError) as exc_info:
            _act({WORDS: "Fine day."}, _state())

        error = _error(exc_info)
        assert isinstance(error, MissingRequiredParameterError)
        assert error.param_name == SHADE

    def test_an_act_setting_it_passes(self):
        action = _act({WORDS: "Fine day.", SHADE: 0.4}, _state())

        assert action.parameters[SHADE] == 0.4

    def test_a_zero_value_counts_as_set(self):
        # Presence, not truthiness: 0.0 is a legitimate neutral value.
        assert _act({WORDS: "Fine day.", SHADE: 0.0}, _state()).parameters[SHADE] == 0.0

    def test_the_required_parameter_alone_satisfies_at_least_one(self):
        assert _act({SHADE: -0.2}, _state()).action == ActionType.ACT_IN_INTERACTION

    def test_without_the_key_the_omission_is_still_legitimate(self):
        state = _state(_interaction(required=False))

        assert _act({WORDS: "Fine day."}, state).action == ActionType.ACT_IN_INTERACTION

    def test_an_unadvertised_extra_is_still_refused_first(self):
        # The extras check keeps its precedence: an act naming a parameter the
        # interaction does not advertise is told so, whatever else it omits.
        with pytest.raises(ValidationError) as exc_info:
            _act({WORDS: "Fine day.", "pitch": 3}, _state())

        assert "pitch" in str(exc_info.value)


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
    async def test_an_omitted_required_parameter_is_retried_and_recorded(self):
        llm = _llm_returning(
            _reflection_json({WORDS: "Fine day."}),
            _reflection_json({WORDS: "Fine day.", SHADE: 0.3}),
        )

        result = await ReflectionNode(llm).process(_state())

        assert result.chosen_action.parameters[SHADE] == 0.3
        assert llm.ainvoke.await_count == 2
        failures = result.validation_failures["reflection"]
        assert len(failures) == 1 and SHADE in failures[0]
        assert SHADE in _sent(llm, 1), "the retry must name the missing parameter"

    @pytest.mark.asyncio
    async def test_the_first_prompt_already_marks_it_required(self):
        llm = _llm_returning(_reflection_json({WORDS: "Fine day.", SHADE: 0.1}))

        await ReflectionNode(llm).process(_state())

        assert "required: every act must set it" in _sent(llm, 0)
