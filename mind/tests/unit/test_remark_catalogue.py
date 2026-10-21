"""The remark catalogue (NPC-1916): its contract, its config boundary, and the
required-key rule on conversational turns (Decision 2).

Keys used in the validator cases come from the committed catalogue fixture by
role (a phatic one, one per kind), never spelled here, so the tests describe
the rule rather than today's vocabulary.
"""

import json
import logging
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from mind.cognitive_architecture.references import is_parameter_set
from mind.cognitive_architecture.remark_catalogue import (
    REMARK_CATALOGUE_CONTRACT_VERSION,
    RemarkCatalogue,
)
from mind.cognitive_architecture.working_memory import WorkingMemory
from mind.interfaces.mcp.mind import Mind
from mind.interfaces.mcp.models import MindConfig
from mind.interfaces.mcp.server import MCPServer
from tests.fixtures.remark_catalogue import (
    GAME_SHA,
    catalogue,
    declaring_values,
    remark_conversation_observation,
    remark_conversation_state,
    wire_catalogue,
)

# --- Contract literal ------------------------------------------------------


class TestCatalogueContractLiteral:
    """The wire shape the game's builder emits at GAME_SHA (plan section 2.1)."""

    def test_fixture_records_its_game_commit(self):
        assert GAME_SHA == "4e38abc0f"

    def test_top_level_shape(self):
        wire = wire_catalogue()

        assert set(wire) == {"contract_version", "remarks"}
        assert wire["contract_version"] == REMARK_CATALOGUE_CONTRACT_VERSION == 1

    def test_every_row_has_exactly_the_contract_columns_all_strings(self):
        for row in wire_catalogue()["remarks"]:
            assert set(row) == {"key", "gloss", "declares", "declare_with", "slots"}, row
            assert isinstance(row["key"], str) and row["key"]
            assert isinstance(row["gloss"], str) and row["gloss"]
            assert isinstance(row["declares"], str)
            assert all(isinstance(v, str) for v in row["declare_with"] + row["slots"])

    def test_twenty_wire_remarks_sorted_by_key(self):
        keys = [row["key"] for row in wire_catalogue()["remarks"]]

        assert len(keys) == 20
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)

    def test_glosses_name_their_slots_as_placeholders(self):
        for row in wire_catalogue()["remarks"]:
            for slot in row["slots"]:
                assert "{" + slot + "}" in row["gloss"], row["key"]

    def test_fixture_parses_into_the_model(self):
        parsed = catalogue()

        assert parsed.keys() == [row["key"] for row in wire_catalogue()["remarks"]]
        assert parsed.keys_for(""), "the catalogue must carry phatic remarks"
        assert len(parsed.declaring_parameters()) >= 2


# --- Model and config ------------------------------------------------------


def _config(**extra) -> dict:
    return {"traits": ["curious"], **extra}


class TestRemarkCatalogueModel:
    def test_extra_columns_on_a_row_are_ignored(self):
        """Rows grow columns additively (stance ``fields`` is the first expected)."""
        wire = wire_catalogue()
        wire["remarks"][0]["fields"] = {"anything": "goes"}

        parsed = RemarkCatalogue.model_validate(wire)

        assert parsed.remarks[0].key == wire["remarks"][0]["key"]

    def test_duplicate_keys_are_refused(self):
        wire = wire_catalogue()
        wire["remarks"].append(dict(wire["remarks"][0]))

        with pytest.raises(ValidationError, match="repeats keys"):
            RemarkCatalogue.model_validate(wire)

    def test_a_typed_row_without_declare_with_is_refused(self):
        wire = wire_catalogue()
        typed = next(r for r in wire["remarks"] if r["declares"])
        typed["declare_with"] = []

        with pytest.raises(ValidationError, match="no declare_with"):
            RemarkCatalogue.model_validate(wire)

    def test_a_phatic_row_with_declare_with_is_refused(self):
        wire = wire_catalogue()
        phatic = next(r for r in wire["remarks"] if not r["declares"])
        phatic["declare_with"] = ["something"]

        with pytest.raises(ValidationError, match="phatic"):
            RemarkCatalogue.model_validate(wire)

    def test_prompt_line_format(self):
        parsed = catalogue()
        phatic = parsed.get(parsed.keys_for("")[0])
        typed = next(r for r in parsed.remarks if r.slots and r.declares)

        assert phatic.prompt_line() == f'{phatic.key}: "{phatic.gloss}"'
        assert typed.prompt_line() == (
            f'{typed.key}: "{typed.gloss}" [declares {typed.declares} via '
            f"{', '.join(typed.declare_with)}] [slots {', '.join(typed.slots)}]"
        )
        assert parsed.prompt_block().count("\n") == len(parsed.remarks) - 1


class TestMindConfigCatalogue:
    def test_absent_is_none(self):
        assert MindConfig.model_validate(_config()).remark_catalogue is None

    def test_present_parses(self):
        config = MindConfig.model_validate(_config(remark_catalogue=wire_catalogue()))

        assert config.remark_catalogue == catalogue()

    def test_unknown_contract_version_degrades_with_a_warning(self, caplog):
        wire = wire_catalogue()
        wire["contract_version"] = 99

        with caplog.at_level(logging.WARNING):
            config = MindConfig.model_validate(_config(remark_catalogue=wire))

        assert config.remark_catalogue is None
        assert "contract_version 99" in caplog.text

    def test_empty_remark_list_degrades_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = MindConfig.model_validate(
                _config(remark_catalogue={"contract_version": 1, "remarks": []})
            )

        assert config.remark_catalogue is None
        assert "lists no remarks" in caplog.text

    def test_a_malformed_known_version_is_refused_loudly(self):
        wire = wire_catalogue()
        del wire["remarks"][0]["key"]

        with pytest.raises(ValidationError):
            MindConfig.model_validate(_config(remark_catalogue=wire))

    def test_recorded_config_keeps_the_catalogue(self):
        """A relinked mind must validate against the grammar it was created with."""
        config = MindConfig.model_validate(
            _config(remark_catalogue=wire_catalogue(), initial_long_term_memories=["seed"])
        )

        recorded = MCPServer._config_to_record(config)

        assert recorded.remark_catalogue == catalogue()
        assert recorded.initial_long_term_memories == []

    def test_the_mind_threads_the_catalogue_into_every_pipeline_state(self):
        mind = Mind(
            mind_id="m",
            entity_id="npc_a",
            traits=[],
            pipeline=None,
            memory_store=None,
            working_memory=WorkingMemory(),
            llm_model="any",
            remark_catalogue=catalogue(),
        )

        state = mind.build_pipeline_state(remark_conversation_observation())

        assert state.remark_catalogue == catalogue()


# --- Required-key validation (Decision 2, plan section 4) -------------------


def _phatic_key() -> str:
    return catalogue().keys_for("")[0]


def _typed(kind_index: int = 0):
    parsed = catalogue()
    kind = sorted(parsed.declaring_parameters())[kind_index]
    return parsed.get(parsed.keys_for(kind)[0])


def _declaring(remark) -> dict:
    """Parameters that declare ``remark``'s kind, every declare_with set."""
    return declaring_values(remark.declare_with, remark_conversation_state())


def _act(parameters: dict, state) -> Action:
    return Action.model_validate(
        {"action": "act_in_interaction", "parameters": {"message": "Words.", **parameters}},
        context={"state": state},
    )


class TestRequiredRemarkKey:
    def test_agreeing_phatic_key_passes(self):
        action = _act({"utterance_element": _phatic_key()}, remark_conversation_state())

        assert action.parameters["utterance_element"] == _phatic_key()

    def test_agreeing_typed_key_passes(self):
        remark = _typed()
        params = {"utterance_element": remark.key, **_declaring(remark)}

        _act(params, remark_conversation_state())

    def test_missing_key_fails_when_a_catalogue_is_present(self):
        with pytest.raises(ValidationError, match="required on every conversational turn"):
            _act({}, remark_conversation_state())

    def test_missing_key_passes_when_no_catalogue_is_present(self):
        _act({}, remark_conversation_state(with_catalogue=False))

    def test_unknown_key_fails_and_lists_the_valid_keys(self):
        with pytest.raises(ValidationError) as excinfo:
            _act({"utterance_element": "no_such_remark"}, remark_conversation_state())

        text = str(excinfo.value)
        assert "'no_such_remark' is not a remark" in text
        assert _phatic_key() in text, "nothing declared, so the phatic keys are the valid ones"

    def test_phatic_key_with_a_declaration_fails(self):
        remark = _typed()
        params = {"utterance_element": _phatic_key(), **_declaring(remark)}

        with pytest.raises(ValidationError) as excinfo:
            _act(params, remark_conversation_state())

        text = str(excinfo.value)
        assert "declares nothing" in text
        assert remark.key in text, "the valid keys listed are the declared kind's"

    def test_typed_key_without_its_parameter_fails(self):
        remark = _typed()

        with pytest.raises(ValidationError, match="but the turn declares nothing"):
            _act({"utterance_element": remark.key}, remark_conversation_state())

    def test_typed_key_with_only_part_of_its_parameters_fails(self):
        remark = next(r for r in catalogue().remarks if len(r.declare_with) > 1)
        params = {"utterance_element": remark.key, remark.declare_with[0]: "x"}

        with pytest.raises(ValidationError, match="unset: "):
            _act(params, remark_conversation_state())

    def test_typed_key_with_another_kinds_declaration_fails(self):
        remark, other = _typed(0), _typed(1)
        params = {"utterance_element": remark.key, **_declaring(other)}

        with pytest.raises(ValidationError, match=f"the turn declares '{other.declares}'"):
            _act(params, remark_conversation_state())

    def test_two_declared_kinds_fail_even_with_a_matching_key(self):
        remark, other = _typed(0), _typed(1)
        params = {"utterance_element": remark.key, **_declaring(remark), **_declaring(other)}

        with pytest.raises(ValidationError, match="declares 2 kinds at once"):
            _act(params, remark_conversation_state())

    def test_an_interaction_that_does_not_advertise_the_key_is_not_checked(self):
        """Only interactions advertising utterance_element speak in remarks."""
        state = remark_conversation_state()
        advertised = state.observation.status.current_interaction["act_in_interaction_parameters"]
        del advertised["utterance_element"]

        _act({}, state)


class TestIsParameterSet:
    @pytest.mark.parametrize("value", [None, False, "", "   ", "false", " False ", [], {}])
    def test_sentinels_read_as_not_set(self, value):
        assert not is_parameter_set(value)

    @pytest.mark.parametrize("value", [True, "x", "true", ["a"], {"a": 1}, 0, 3])
    def test_values_read_as_set(self, value):
        assert is_parameter_set(value)


# --- The existing retry loop re-asks, and the retries are recorded ----------


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
            content=c,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        for c in contents
    ]
    return llm


class TestRetryOnRemarkMismatch:
    @pytest.mark.asyncio
    async def test_a_missing_key_is_retried_with_the_valid_keys_and_recorded(self):
        llm = _llm_returning(
            _reflection_json({"message": "Hello!"}),
            _reflection_json({"message": "Hello!", "utterance_element": _phatic_key()}),
        )
        state = remark_conversation_state()

        result = await ReflectionNode(llm).process(state)

        assert result.chosen_action.parameters["utterance_element"] == _phatic_key()
        assert llm.ainvoke.await_count == 2
        failures = result.validation_failures["reflection"]
        assert len(failures) == 1 and "Remark mismatch" in failures[0]
        retry_prompt = llm.ainvoke.await_args_list[1].args[0][-1].content
        assert _phatic_key() in retry_prompt, "the retry must carry the valid keys"

    @pytest.mark.asyncio
    async def test_a_first_attempt_success_records_an_empty_failure_list(self):
        llm = _llm_returning(
            _reflection_json({"message": "Hello!", "utterance_element": _phatic_key()})
        )

        result = await ReflectionNode(llm).process(remark_conversation_state())

        assert result.validation_failures["reflection"] == []
        assert llm.ainvoke.await_count == 1
