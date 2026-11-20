"""The dimension catalogue (NPC-2312): contract, config boundary, prompt rendering,
band names on heard lines, and the required graded value on every turn.

Dimension, band and parameter names used in cases come from the committed contract
fixture by role (its first dimension, its first and last band), never spelled
here, so the tests describe the rule rather than today's vocabulary.
"""

import ast
import copy
import json
import logging
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action
from mind.cognitive_architecture.dimension_catalogue import (
    DIMENSION_CATALOGUE_CONTRACT_VERSION,
    DimensionCatalogue,
)
from mind.cognitive_architecture.nodes.formatting import format_conversation_histories
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from mind.interfaces.mcp.models import MindConfig
from mind.interfaces.mcp.server import MCPServer
from tests.fixtures.dimension_catalogue import (
    dimensions,
    graded_conversation_state,
    wire_dimensions,
)

SRC = Path(__file__).parents[2] / "src" / "mind"


def _first():
    return dimensions().dimensions[0]


def _param() -> str:
    return _first().parameter


def _mid(band) -> float:
    return (band.low + band.high) / 2


def _wire() -> dict:
    return copy.deepcopy(wire_dimensions())


def _config(**extra) -> dict:
    return {"traits": ["curious"], **extra}


# --- Contract literal ------------------------------------------------------


class TestContractLiteral:
    """The wire shape the simulation's builder is pinned against."""

    def test_top_level_shape(self):
        wire = wire_dimensions()

        assert set(wire) == {"contract_version", "dimensions"}
        assert wire["contract_version"] == DIMENSION_CATALOGUE_CONTRACT_VERSION == 1

    def test_every_dimension_has_exactly_the_contract_columns(self):
        for row in wire_dimensions()["dimensions"]:
            assert set(row) == {
                "key",
                "parameter",
                "declaration_kind",
                "declaration_field",
                "about",
                "contrast",
                "low",
                "high",
                "bands",
            }, row

    def test_every_band_has_exactly_the_contract_columns(self):
        for row in wire_dimensions()["dimensions"]:
            for band in row["bands"]:
                assert set(band) == {"name", "low", "high", "meaning", "effect", "landmarks"}
                assert all(isinstance(line, str) and line for line in band["landmarks"])

    def test_fixture_parses_and_its_bands_tile_the_range(self):
        dimension = _first()

        assert dimension.bands[0].low == dimension.low
        assert dimension.bands[-1].high == dimension.high
        assert [b.high for b in dimension.bands[:-1]] == [b.low for b in dimension.bands[1:]]


# --- Model ------------------------------------------------------------------


class TestModel:
    def test_extra_columns_are_ignored(self):
        wire = _wire()
        wire["dimensions"][0]["future_column"] = {"anything": "goes"}
        wire["dimensions"][0]["bands"][0]["future_column"] = 1

        assert DimensionCatalogue.model_validate(wire).dimensions[0].key == _first().key

    def test_a_gap_between_bands_is_refused(self):
        wire = _wire()
        wire["dimensions"][0]["bands"][1]["low"] += 0.05

        with pytest.raises(ValidationError, match="gap or overlap"):
            DimensionCatalogue.model_validate(wire)

    def test_an_overlap_between_bands_is_refused(self):
        wire = _wire()
        wire["dimensions"][0]["bands"][1]["low"] -= 0.05

        with pytest.raises(ValidationError, match="gap or overlap"):
            DimensionCatalogue.model_validate(wire)

    def test_bands_that_stop_short_of_the_range_are_refused(self):
        wire = _wire()
        wire["dimensions"][0]["bands"][-1]["high"] -= 0.1

        with pytest.raises(ValidationError, match="last band ends"):
            DimensionCatalogue.model_validate(wire)

    def test_bands_that_start_after_the_range_are_refused(self):
        wire = _wire()
        wire["dimensions"][0]["bands"][0]["low"] += 0.1

        with pytest.raises(ValidationError, match="first band starts"):
            DimensionCatalogue.model_validate(wire)

    def test_a_backwards_band_is_refused(self):
        wire = _wire()
        band = wire["dimensions"][0]["bands"][0]
        band["low"], band["high"] = band["high"], band["low"]

        with pytest.raises(ValidationError):
            DimensionCatalogue.model_validate(wire)

    def test_repeated_band_names_are_refused(self):
        wire = _wire()
        wire["dimensions"][0]["bands"][1]["name"] = wire["dimensions"][0]["bands"][0]["name"]

        with pytest.raises(ValidationError, match="repeats band names"):
            DimensionCatalogue.model_validate(wire)

    def test_a_dimension_with_no_bands_is_refused(self):
        wire = _wire()
        wire["dimensions"][0]["bands"] = []

        with pytest.raises(ValidationError):
            DimensionCatalogue.model_validate(wire)

    def test_repeated_parameters_are_refused(self):
        wire = _wire()
        twin = copy.deepcopy(wire["dimensions"][0])
        twin["key"] = "other"
        twin["declaration_kind"] = "other"
        wire["dimensions"].append(twin)

        with pytest.raises(ValidationError, match="repeats parameter"):
            DimensionCatalogue.model_validate(wire)

    def test_a_blank_parameter_is_refused(self):
        wire = _wire()
        wire["dimensions"][0]["parameter"] = "  "

        with pytest.raises(ValidationError):
            DimensionCatalogue.model_validate(wire)


class TestBandLookup:
    def test_every_band_owns_its_lower_edge_and_the_last_owns_the_top(self):
        dimension = _first()

        for band in dimension.bands:
            assert dimension.band_for(band.low) is band
        assert dimension.band_for(dimension.high) is dimension.bands[-1]

    def test_a_value_just_under_an_inner_edge_is_in_the_band_below(self):
        dimension = _first()

        for below, above in zip(dimension.bands, dimension.bands[1:], strict=False):
            assert dimension.band_for(above.low - 1e-6) is below

    def test_outside_the_range_raises(self):
        dimension = _first()

        with pytest.raises(ValueError):
            dimension.band_for(dimension.high + 0.01)
        with pytest.raises(ValueError):
            dimension.band_for(dimension.low - 0.01)


# --- Config boundary --------------------------------------------------------


class TestMindConfigDimensions:
    def test_absent_is_none_so_an_older_simulation_still_works(self):
        assert MindConfig.model_validate(_config()).dimensions is None

    def test_present_parses(self):
        config = MindConfig.model_validate(_config(dimensions=wire_dimensions()))

        assert config.dimensions == dimensions()

    def test_unknown_contract_version_degrades_with_a_warning(self, caplog):
        wire = _wire()
        wire["contract_version"] = 99

        with caplog.at_level(logging.WARNING):
            config = MindConfig.model_validate(_config(dimensions=wire))

        assert config.dimensions is None
        assert "contract_version 99" in caplog.text

    def test_empty_dimension_list_degrades_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = MindConfig.model_validate(
                _config(dimensions={"contract_version": 1, "dimensions": []})
            )

        assert config.dimensions is None
        assert "lists no dimension" in caplog.text

    def test_a_malformed_known_version_is_refused_loudly(self):
        wire = _wire()
        del wire["dimensions"][0]["parameter"]

        with pytest.raises(ValidationError):
            MindConfig.model_validate(_config(dimensions=wire))

    def test_a_table_with_a_gap_is_refused_at_create_mind(self):
        wire = _wire()
        wire["dimensions"][0]["bands"][2]["low"] += 0.05

        with pytest.raises(ValidationError, match="gap or overlap"):
            MindConfig.model_validate(_config(dimensions=wire))

    def test_the_recorded_config_keeps_the_catalogue(self):
        """A relinked mind must keep refusing a turn that omits its graded value."""
        config = MindConfig.model_validate(
            _config(dimensions=wire_dimensions(), initial_long_term_memories=["seed"])
        )

        recorded = MCPServer._config_to_record(config)

        assert recorded.dimensions == dimensions()


# --- Calibration rendering --------------------------------------------------


class TestCalibrationRendering:
    def test_the_block_carries_the_range_each_band_its_effect_and_its_landmarks(self):
        dimension = _first()
        block = dimensions().prompt_block({dimension.parameter: "hint"})

        assert dimension.range_text() in block
        assert dimension.about in block
        for band in dimension.bands:
            assert band.name in block
            assert band.meaning in block
            for landmark in band.landmarks:
                assert f'"{landmark}"' in block
        assert any(b.effect and b.effect in block for b in dimension.bands)

    def test_the_block_says_how_the_dimension_differs_from_its_neighbour(self):
        dimension = _first()

        block = dimensions().prompt_block({dimension.parameter: "hint"})

        assert dimension.contrast and dimension.contrast in block

    def test_the_block_says_the_value_is_required_and_names_the_parameter(self):
        block = dimensions().prompt_block({_param(): "hint"})

        assert f"`{_param()}`" in block
        assert "on every line you say" in block
        assert "is refused" in block

    def test_the_block_says_how_to_read_an_unmarked_line(self):
        dimension = _first()

        block = dimensions().prompt_block({dimension.parameter: "hint"})

        assert f"{dimension.band_for(0.0).name}" in block.splitlines()[-1]
        assert f"[{dimension.declaration_kind}: Band value]" in block.splitlines()[-1]

    def test_nothing_renders_for_an_interaction_that_does_not_advertise_the_parameter(self):
        assert dimensions().prompt_block({"message": "hint"}) == ""

    @pytest.mark.asyncio
    async def test_the_block_sits_below_the_cache_breakpoint_in_a_graded_conversation(self):
        state = graded_conversation_state()
        llm = _llm_returning(_reflection_json({"message": "Hi", _param(): 0.0}))
        node = ReflectionNode(llm)

        await node.process(state)

        sent = llm.ainvoke.call_args[0][0][0].content
        block = state.dimensions.prompt_block(state.observation.status.act_parameter_hints())
        assert "### How Your Words Are Weighed" in sent and block in sent
        assert sent.startswith(node.static_prefix)
        assert block not in node.static_prefix

    @pytest.mark.asyncio
    async def test_no_catalogue_means_no_block(self):
        llm = _llm_returning(_reflection_json({"message": "Hi"}))

        await ReflectionNode(llm).process(_unflagged_state(with_dimensions=False))

        assert "How Your Words Are Weighed" not in llm.ainvoke.call_args[0][0][0].content


# --- Heard lines ------------------------------------------------------------


def _declaration(value, dimension=None) -> dict:
    dimension = dimension or _first()
    return {"kind": dimension.declaration_kind, dimension.declaration_field: value}


class TestHeardLinesShowTheirBand:
    def test_a_heard_declaration_shows_band_name_and_value(self):
        dimension = _first()
        band = dimension.bands[1]
        value = round(_mid(band), 2)
        state = graded_conversation_state(declarations=[_declaration(value)])

        text = format_conversation_histories(
            state.conversation_histories, "npc_a", dimensions=state.dimensions
        )

        assert f"[{dimension.declaration_kind}: {band.name} {value:g}]" in text

    def test_each_band_is_named_for_a_value_inside_it(self):
        dimension = _first()

        for band in dimension.bands:
            marker = dimensions().marker_for(_declaration(_mid(band)))
            assert marker is not None and f": {band.name} " in marker

    def test_an_integer_value_is_accepted(self):
        dimension = _first()
        marker = dimensions().marker_for(_declaration(int(dimension.high)))

        assert marker is not None and dimension.bands[-1].name in marker

    def test_without_a_catalogue_the_declaration_renders_as_its_bare_kind(self):
        dimension = _first()
        state = graded_conversation_state(declarations=[_declaration(0.5)])

        text = format_conversation_histories(state.conversation_histories, "npc_a")

        assert f"[{dimension.declaration_kind}]" in text
        assert dimension.bands[-1].name not in text

    def test_an_unplaceable_stored_value_falls_back_to_the_bare_kind_and_warns(self, caplog):
        dimension = _first()
        state = graded_conversation_state(
            declarations=[_declaration(dimension.high + 5), _declaration("warm")]
        )

        with caplog.at_level(logging.WARNING):
            text = format_conversation_histories(
                state.conversation_histories, "npc_a", dimensions=state.dimensions
            )

        assert text.count(f"[{dimension.declaration_kind}]") == 2
        assert "showing it without a band" in caplog.text

    def test_a_declaration_of_another_kind_renders_as_before(self):
        state = graded_conversation_state(declarations=[{"kind": "some_other_kind"}])

        text = format_conversation_histories(
            state.conversation_histories, "npc_a", dimensions=state.dimensions
        )

        assert "[some_other_kind]" in text


# --- The required structured-output field ------------------------------------


def _unflagged_state(with_dimensions: bool = True):
    """The conversation as an interaction that does not flag the parameter required.

    The simulation's own ``required`` flag refuses a missing value before this
    server's rule runs; removing it isolates the rule under test.
    """
    state = graded_conversation_state(with_dimensions=with_dimensions)
    spec = state.observation.status.current_interaction["act_in_interaction_parameters"][_param()]
    del spec["required"]
    spec["default"] = None
    return state


def _act(parameters: dict, state=None) -> Action:
    state = state or graded_conversation_state()
    return Action.model_validate(
        {"action": "act_in_interaction", "parameters": parameters}, context={"state": state}
    )


class TestRequiredValue:
    def test_a_value_inside_the_range_passes_and_is_kept_verbatim(self):
        action = _act({"message": "Hi", _param(): -0.25})

        assert action.parameters[_param()] == -0.25

    def test_zero_is_a_real_value_not_a_missing_one(self):
        assert _act({"message": "Hi", _param(): 0}).parameters[_param()] == 0

    def test_both_range_edges_pass(self):
        dimension = _first()

        _act({"message": "Hi", _param(): dimension.low})
        _act({"message": "Hi", _param(): dimension.high})

    def test_a_missing_value_is_refused_and_names_the_parameter_and_range(self):
        with pytest.raises(ValidationError) as excinfo:
            _act({"message": "Hi"}, _unflagged_state())

        text = str(excinfo.value)
        assert f"'{_param()}' is missing from the turn" in text
        assert _first().range_text() in text

    def test_a_missing_value_is_refused_by_the_simulations_own_required_flag_too(self):
        with pytest.raises(ValidationError, match="Required parameter"):
            _act({"message": "Hi"})

    def test_a_null_value_is_refused_not_read_as_zero(self):
        with pytest.raises(ValidationError, match="is missing from the turn"):
            _act({"message": "Hi", _param(): None}, _unflagged_state())

    @pytest.mark.parametrize("beyond", [0.01, 1.0, 100.0])
    def test_a_value_above_the_range_is_refused_not_clamped(self, beyond):
        with pytest.raises(ValidationError, match="outside its range"):
            _act({"message": "Hi", _param(): _first().high + beyond})

    @pytest.mark.parametrize("beyond", [0.01, 1.0, 100.0])
    def test_a_value_below_the_range_is_refused_not_clamped(self, beyond):
        with pytest.raises(ValidationError, match="outside its range"):
            _act({"message": "Hi", _param(): _first().low - beyond})

    @pytest.mark.parametrize("bad", ["0.5", "warm", True, False, [0.5], {"v": 1}])
    def test_a_non_number_is_refused_not_coerced(self, bad):
        with pytest.raises(ValidationError, match="must be a number"):
            _act({"message": "Hi", _param(): bad})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_number_is_refused(self, bad):
        with pytest.raises(ValidationError, match="must be a number"):
            _act({"message": "Hi", _param(): bad})

    def test_no_catalogue_means_no_added_requirement(self):
        """An older simulation sends no dimensions; the mind adds nothing to its spec."""
        _act({"message": "Hi"}, _unflagged_state(with_dimensions=False))

    def test_an_interaction_that_does_not_advertise_the_parameter_needs_no_value(self):
        state = graded_conversation_state()
        del state.observation.status.current_interaction["act_in_interaction_parameters"][_param()]

        _act({"message": "Hi"}, state)


class TestRetryOnDimensionMismatch:
    @pytest.mark.asyncio
    async def test_a_missing_value_is_retried_and_the_retry_carries_the_range(self):
        llm = _llm_returning(
            _reflection_json({"message": "Hi"}),
            _reflection_json({"message": "Hi", _param(): -0.5}),
        )

        result = await ReflectionNode(llm).process(_unflagged_state())

        assert result.chosen_action.parameters[_param()] == -0.5
        assert llm.ainvoke.await_count == 2
        failures = result.validation_failures["reflection"]
        assert len(failures) == 1 and "Dimension value" in failures[0]
        retry_prompt = llm.ainvoke.await_args_list[1].args[0][-1].content
        assert _first().range_text() in retry_prompt

    @pytest.mark.asyncio
    async def test_an_out_of_range_value_is_retried_never_sent_clamped(self):
        llm = _llm_returning(
            _reflection_json({"message": "Hi", _param(): 7}),
            _reflection_json({"message": "Hi", _param(): 0.4}),
        )

        result = await ReflectionNode(llm).process(graded_conversation_state())

        assert result.chosen_action.parameters[_param()] == 0.4
        assert llm.ainvoke.await_count == 2
        assert "outside its range" in result.validation_failures["reflection"][0]

    @pytest.mark.asyncio
    async def test_a_valid_first_attempt_records_no_failure(self):
        llm = _llm_returning(_reflection_json({"message": "Hi", _param(): 0.0}))

        result = await ReflectionNode(llm).process(graded_conversation_state())

        assert result.validation_failures["reflection"] == []
        assert llm.ainvoke.await_count == 1


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


# --- No vocabulary in src ----------------------------------------------------


def _vocabulary() -> set[str]:
    words: set[str] = set()
    for row in wire_dimensions()["dimensions"]:
        words.update(
            {row["key"], row["parameter"], row["declaration_kind"], row["declaration_field"]}
        )
        words.update(band["name"] for band in row["bands"])
    return words


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _offenders(source: str, vocabulary: set[str]) -> list[tuple[int, str, str]]:
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in docstrings:
            continue
        for word in vocabulary:
            if _names(node.value, word):
                hits.append((node.lineno, node.value, word))
    return hits


def _names(literal: str, word: str) -> bool:
    """Whether a string literal spells ``word`` as vocabulary.

    A plain English word that is also a band name ("plain", "warm") counts only
    when it IS the literal, so a log line saying "plain text" is prose rather than
    a hardcoded band. A compound word (``some_param``) counts wherever it appears
    as a whole token.
    """
    if literal.strip().lower() == word.lower():
        return True
    pattern = rf"(?<![\w.]){re.escape(word)}(?![\w.])"
    return "_" in word and bool(re.search(pattern, literal))


def test_the_vocabulary_is_not_empty():
    """An empty vocabulary would make the scan below pass over nothing."""
    assert len(_vocabulary()) >= 6


def test_src_names_no_dimension_band_or_declaring_parameter():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        offenders += [(path.name, *hit) for hit in _offenders(path.read_text(), _vocabulary())]
    assert not offenders, offenders


def test_the_scan_catches_a_hardcoded_band_name():
    """Known positive: the scan above must be able to go red."""
    band = _first().bands[0].name

    assert _offenders(f'x = "{band}"\n', _vocabulary())
    assert _offenders(f'"""{band}"""\nx = 1\n', _vocabulary()) == []
