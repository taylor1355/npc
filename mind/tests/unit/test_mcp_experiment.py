"""The MCP experiment tier (NPC-1916, Decision 4): its matrix, its record, and its
four analyses -- everything that can be checked without a model.

The analyses run over ``tests/fixtures/mcp_experiment_artifact.jsonl``, a
hand-built seven-record artifact whose every number below is derivable by hand
from the records (the derivations are in the comments), so a regression in an
analysis shows as a wrong number rather than as a crash.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from mind.cognitive_architecture.actions import Action
from mind.cognitive_architecture.actions.models import normalize_remark_reply
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from mind.cognitive_architecture.references import HandleTable
from tests.fixtures.mcp_experiment import (
    ARM_CATALOGUE,
    ARM_RED,
    CASES,
    PERSONAS,
    RecordingLLM,
    run_turn,
)
from tests.fixtures.mcp_experiment_analysis import (
    check_floors,
    cost_report,
    derive_floors,
    export_dataset,
    judge_pass,
    load_artifact,
    parse_score,
    validity_report,
)
from tests.fixtures.remark_catalogue import catalogue, declaring_values

ARTIFACT = Path(__file__).parents[1] / "fixtures" / "mcp_experiment_artifact.jsonl"

# --- The matrix ---------------------------------------------------------------


class TestMatrix:
    def test_about_a_dozen_cases_with_unique_ids_and_two_personas(self):
        ids = [case.id for case in CASES]

        assert len(ids) >= 12
        assert len(set(ids)) == len(ids)
        assert len(PERSONAS) >= 2

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
    def test_every_apt_key_is_a_catalogue_remark_of_the_apt_kind(self, case):
        parsed = catalogue()
        for key in case.apt_keys:
            entry = parsed.get(key)
            assert entry is not None, f"{case.id}: '{key}' is not in the catalogue"
            assert entry.declares == case.apt_kind, f"{case.id}: '{key}' declares {entry.declares}"

    def test_the_matrix_covers_every_kind_and_the_phatic_roots(self):
        kinds = {case.apt_kind for case in CASES}

        assert kinds >= {""} | set(catalogue().declaring_parameters())

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
    def test_an_apt_act_validates_against_each_case_state(self, case):
        """The states are well-formed remark turns: an apt answer passes validation."""
        entry = catalogue().get(case.apt_keys[0])
        for persona in PERSONAS:
            state = case.state(persona, with_catalogue=True)
            params = {"message": "Words.", "utterance_element": entry.key}
            params.update(declaring_values(entry.declare_with, state))
            params.update(entry.fields)
            reply_parameter = catalogue().reply_parameter(entry)
            if reply_parameter is not None:
                # A reply names the partner's last line, by its handle.
                table = HandleTable(state.observation, state.conversation_histories)
                params[reply_parameter] = table.handles("message")[-1]
            # The node normalizes a reply's omitted addressee before validating.
            action = normalize_remark_reply(
                {"action": "act_in_interaction", "parameters": params}, state
            )
            Action.model_validate(action, context={"state": state})

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
    def test_the_red_arm_state_carries_no_catalogue_and_requires_no_key(self, case):
        state = case.state(PERSONAS[0], with_catalogue=False)

        assert state.remark_catalogue is None
        Action.model_validate(
            {"action": "act_in_interaction", "parameters": {"message": "Hi."}},
            context={"state": state},
        )


# --- Running a turn (no model: a scripted one) --------------------------------


class _Scripted:
    """A chat model double that returns scripted outputs in order."""

    model_name = None

    def __init__(self, *outputs: str):
        self._outputs = list(outputs)

    async def ainvoke(self, messages, *args, **kwargs):
        return AIMessage(
            content=self._outputs.pop(0),
            usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
        )


def _reflection(params: dict) -> str:
    return json.dumps(
        {
            "updated_working_memory": {"situation_assessment": "x"},
            "chosen_action": {"action": "act_in_interaction", "parameters": params},
        }
    )


@pytest.mark.asyncio
async def test_run_turn_records_attempts_retries_and_the_named_remark():
    case = next(c for c in CASES if c.apt_kind == "")
    key = case.apt_keys[0]
    recorder = RecordingLLM(
        _Scripted(
            _reflection({"message": "Hi"}), _reflection({"message": "Hi", "utterance_element": key})
        )
    )

    record = await run_turn(
        ReflectionNode(recorder),
        recorder,
        case,
        PERSONAS[0],
        ARM_CATALOGUE,
        rep=0,
        run_id="r",
        backend="fake",
        model="fake",
    )

    assert record["remark_named"] == key
    assert record["words"] == "Hi"
    assert record["retries"] == 1 and not record["salvaged"]
    assert len(record["attempts"]) == 2
    assert "### Remarks You Can Name" in record["attempts"][0]["sent"]
    assert "Remark mismatch" in record["attempts"][1]["sent"]
    assert record["usage"]["model_calls"] == 2
    assert record["game_sha"] == "4e38abc0f"
    json.dumps(record)  # an artifact line must serialize


@pytest.mark.asyncio
async def test_run_turn_red_arm_shows_no_catalogue_and_never_retries_for_a_key():
    case = CASES[0]
    recorder = RecordingLLM(_Scripted(_reflection({"message": "Hi"})))

    record = await run_turn(
        ReflectionNode(recorder),
        recorder,
        case,
        PERSONAS[0],
        ARM_RED,
        rep=0,
        run_id="r",
        backend="fake",
        model="fake",
    )

    assert record["retries"] == 0 and record["remark_named"] == ""
    assert "### Remarks You Can Name" not in record["attempts"][0]["sent"]


# --- (a) validity --------------------------------------------------------------


@pytest.fixture
def records():
    return load_artifact(ARTIFACT)


class TestValidity:
    def test_catalogue_arm_rates(self, records):
        row = validity_report(records, catalogue())[ARM_CATALOGUE]

        # 5 turns; 4 acts (one salvaged to wait); every act names a catalogue key
        # that agrees; 3 of 4 are apt (the gruff leaver said `response`).
        assert row["turns"] == 5
        assert row["act_rate"] == 0.8
        assert row["valid_key_rate"] == 1.0
        assert row["agreement_rate"] == 1.0
        assert row["apt_rate"] == 0.75
        assert row["kind_apt_rate"] == 0.75
        # Retries per turn [0, 1, 0, 0, 2]; three remark failures, one other.
        assert row["first_try_rate"] == 0.6
        assert row["mean_retries"] == 0.6
        assert row["remark_failures"] == 3
        assert row["other_failures"] == 1
        assert row["salvage_rate"] == 0.2

    def test_red_arm_collapses(self, records):
        row = validity_report(records, catalogue())[ARM_RED]

        # One turn named nothing, one named a key the catalogue lacks.
        assert row["valid_key_rate"] == 0.0
        assert row["agreement_rate"] == 0.0

    def test_floors_derive_from_the_measurement_and_pass_against_it(self, records):
        report = validity_report(records, catalogue())
        floors = derive_floors(report, source="fixture")

        assert floors["floors"]["agreement_rate"] == 1.0
        assert check_floors(report, floors) == []

    def test_floors_fail_when_a_rate_drops(self, records):
        report = validity_report(records, catalogue())
        floors = derive_floors(report, source="fixture")
        floors["floors"]["apt_rate"] = 0.9

        assert check_floors(report, floors) == ["apt_rate: 0.75 below floor 0.9"]

    def test_floors_fail_when_the_red_arm_fails_to_fail(self, records):
        red_agreeing = [
            {
                **r,
                "remark_named": "greeting_response",
                "parameters": {"utterance_element": "greeting_response"},
            }
            if r["arm"] == ARM_RED
            else r
            for r in records
        ]
        report = validity_report(red_agreeing, catalogue())

        violations = check_floors(report, derive_floors(report, source="fixture"))

        assert any("red arm failed to fail" in v for v in violations)

    def test_an_empty_artifact_is_refused(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no records"):
            load_artifact(empty)


# --- (b) dataset -----------------------------------------------------------------


class TestDataset:
    def test_rows_are_agreeing_catalogue_arm_turns_with_provenance(self, records):
        rows = export_dataset(records, catalogue())

        assert [(r["text"], r["label"]) for r in rows] == [
            ("Hi! How's it going?", "greeting_response"),
            ("Hm. Hello.", "greeting_response"),
            ("I must go eat. Bye!", "farewell"),
            ("Food. Now.", "response"),
        ]
        assert {r["game_sha"] for r in rows} == {"4e38abc0f"}
        assert rows[2]["declares"] == "farewell" and rows[0]["declares"] == ""
        assert [r["apt"] for r in rows] == [True, True, True, False]

    def test_the_red_arm_never_contributes_labels(self, records):
        assert all(r["case_id"] and r["run_id"] for r in export_dataset(records, catalogue()))
        red_texts = {r["words"] for r in records if r["arm"] == ARM_RED}
        assert not red_texts & {r["text"] for r in export_dataset(records, catalogue())}


# --- (c) judge -------------------------------------------------------------------


class TestJudge:
    def test_parse_score_never_defaults(self):
        assert parse_score('{"score": 4, "reason": "ok"}') == 4
        assert parse_score("I would say four") is None
        assert parse_score('{"score": 9}') is None

    def test_scores_replies_and_persona_pairs_with_the_given_judge(self, records):
        prompts = []

        def judge(prompt: str) -> str:
            prompts.append(prompt)
            if "DIFFERENT people" in prompt:
                return '{"score": 2, "reason": "similar"}'
            if "Food. Now." in prompt:
                return "no score here"
            return '{"score": 4, "reason": "fine"}'

        report = judge_pass(records, judge)

        # 4 catalogue-arm replies with words; one unscoreable.
        assert report["on_topic"]["judged"] == 4
        assert report["on_topic"]["unscored"] == 1
        assert report["on_topic"]["mean"] == 4.0
        # Two cases answered by both personas in rep 0: two pairs.
        assert report["persona_distinctness"]["judged"] == 2
        assert report["persona_distinctness"]["mean"] == 2.0
        assert any("Wren: Hello there! <greeting>" in p for p in prompts), (
            "the judge must see the transcript taken from the prompt that was sent"
        )


# --- (d) cost ----------------------------------------------------------------------


class TestCost:
    def test_catalogue_arm_costs(self, records):
        row = cost_report(records, catalogue())[ARM_CATALOGUE]

        assert row["model_calls"] == 6
        assert row["calls_per_turn"] == 1.2
        assert row["retry_calls"] == 1
        assert row["retry_tokens"] == 1300
        assert row["prompt_tokens_per_turn"] == 1240.0
        assert row["latency_ms_mean"] == 1600.0
        assert row["latency_ms_p50"] == 1000
        assert row["latency_ms_p95"] == 3000
        # 600 completion tokens over 8 s of call time.
        assert row["effective_completion_tokens_per_second"] == 75.0

    def test_catalogue_block_size_is_reported(self, records):
        block = cost_report(records, catalogue())["catalogue_block"]

        assert block["remarks"] == len(catalogue().remarks)
        assert block["block_chars"] == len(catalogue().prompt_block())


# --- The CLIs ------------------------------------------------------------------------


def test_the_experiment_and_analysis_clis_import():
    """``tools/`` is outside ruff and pytest; importing is what keeps them from rotting."""
    import tools.mcp_experiment_cost as cost
    import tools.mcp_experiment_dataset as dataset
    import tools.mcp_experiment_judge as judge
    import tools.mcp_experiment_validity as validity
    import tools.run_mcp_experiment as runner

    for module in (cost, dataset, judge, validity, runner):
        assert callable(module.main)


def test_the_offline_clis_run_over_the_fixture_artifact(tmp_path, capsys):
    import tools.mcp_experiment_cost as cost
    import tools.mcp_experiment_dataset as dataset
    import tools.mcp_experiment_validity as validity

    floors = tmp_path / "floors.json"
    assert validity.main([str(ARTIFACT), "--write-floors", str(floors)]) == 0
    assert validity.main([str(ARTIFACT), "--check-floors", str(floors)]) == 0
    assert cost.main([str(ARTIFACT)]) == 0
    out = tmp_path / "data.jsonl"
    assert dataset.main([str(ARTIFACT), "--out", str(out)]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4
    assert "floors: PASS" in capsys.readouterr().out


def test_the_runner_dry_run_spends_nothing(capsys):
    import tools.run_mcp_experiment as runner

    assert runner.main(["--dry-run", "--reps", "1"]) == 0
    assert f"{len(CASES) * len(PERSONAS) * 2} turns would run" in capsys.readouterr().out


def test_the_runner_refuses_reps_that_would_replay_identical_samples(capsys):
    import tools.run_mcp_experiment as runner

    with pytest.raises(SystemExit) as refused:
        runner.main(["--dry-run", "--reps", "5"])

    assert refused.value.code == 2
    assert "identical samples" in capsys.readouterr().err
    # The control: the same reps at a sampling temperature, and the default, run.
    assert runner.main(["--dry-run", "--reps", "5", "--temperature", "0.7"]) == 0
    assert runner.main(["--dry-run"]) == 0


def test_the_committed_floors_came_from_a_measured_run():
    """Floors are derived, never guessed: the sidecar names the run it came from."""
    from tests.fixtures.mcp_experiment_analysis import FLOOR_METRICS

    floors = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "mcp_experiment_floors.json").read_text(
            encoding="utf-8"
        )
    )

    assert "first measured run" in floors["source"] and floors["model"] in floors["source"]
    assert set(floors["floors"]) <= set(FLOOR_METRICS)
    assert all(0.0 <= value <= 1.0 for value in floors["floors"].values())


# --- Reference handles (NPC-2021) ------------------------------------------------


class TestReferenceMatrix:
    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
    def test_every_expected_id_is_a_real_id_of_the_case(self, case):
        assert set(case.expect_message_ids()) <= set(case.known_message_ids())
        assert set(case.expect_zone_ids) <= set(case.known_zone_ids())

    def test_the_cases_that_declare_a_reference_expect_one(self):
        expecting = {c.id for c in CASES if c.expect_message_ids() or c.expect_zone_ids}
        assert {
            "asked_who_told_told",
            "partner_hungry_share_place",
            "ask_who_told_them",
        } <= expecting


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
async def test_no_prompt_of_the_matrix_carries_a_real_id(case):
    """The experiment only measures handles if the prompt cannot leak an id."""
    recorder = RecordingLLM(_Scripted(_reflection({"message": "Hi"})))

    record = await run_turn(
        ReflectionNode(recorder),
        recorder,
        case,
        PERSONAS[0],
        ARM_CATALOGUE,
        rep=0,
        run_id="r",
        backend="fake",
        model="fake",
    )

    sent = record["attempts"][0]["sent"]
    assert "[m1]" in sent, "control: the transcript was annotated"
    for real_id in case.known_message_ids() + case.known_zone_ids():
        assert real_id not in sent


@pytest.mark.asyncio
async def test_run_turn_records_the_reference_translation_and_the_refused_attempt():
    case = next(c for c in CASES if c.id == "asked_who_told_told")
    key = "answer_who_told.told"
    fabricated = {
        "message": "Tobin.",
        "utterance_element": key,
        "is_answer": True,
        "in_reply_to": "msg-123",
    }
    valid = {**fabricated, "in_reply_to": "m3"}
    recorder = RecordingLLM(_Scripted(_reflection(fabricated), _reflection(valid)))

    record = await run_turn(
        ReflectionNode(recorder),
        recorder,
        case,
        PERSONAS[0],
        ARM_CATALOGUE,
        rep=0,
        run_id="r",
        backend="fake",
        model="fake",
    )

    assert record["retries"] == 1
    assert "handle this cycle" in record["validation_failures"][0]
    assert record["parameters"]["in_reply_to"] == "m3", "what the model wrote"
    assert record["wire_parameters"]["in_reply_to"] == case.message_id(2)
    assert record["references"] == [
        {
            "parameter": "in_reply_to",
            "domain": "message",
            "written": "m3",
            "wire_value": case.message_id(2),
        }
    ]
    for attempt in record["attempts"]:
        assert case.message_id(2) not in attempt["sent"], "the prompt never carries an id"
    assert "[m3]" in record["attempts"][0]["sent"]


class TestReferenceMetrics:
    def _record(self, **overrides):
        base = {
            "arm": ARM_CATALOGUE,
            "case_id": "c",
            "action": "act_in_interaction",
            "parameters": {},
            "remark_named": "",
            "apt_keys": [],
            "apt_kind": "",
            "known_message_ids": ["id1", "id2"],
            "known_zone_ids": ["zone1"],
            "expect_message_ids": ["id2"],
            "expect_zone_ids": [],
            "validation_failures": [],
            "retries": 0,
            "references": [],
        }
        return {**base, **overrides}

    def _report(self, records):
        return validity_report(records, catalogue())[ARM_CATALOGUE]

    def test_a_real_expected_id_resolves_and_is_apt(self):
        row = {"parameter": "p", "domain": "message", "written": "m2", "wire_value": "id2"}

        report = self._report([self._record(references=[row])])

        assert report["reference_resolved_rate"] == 1.0
        assert report["reference_apt_rate"] == 1.0
        assert report["fabricated_reference_count"] == 0

    def test_a_real_but_unexpected_id_resolves_without_being_apt(self):
        row = {"parameter": "p", "domain": "message", "written": "m1", "wire_value": "id1"}

        report = self._report([self._record(references=[row])])

        assert report["reference_resolved_rate"] == 1.0 and report["reference_apt_rate"] == 0.0

    def test_an_invented_id_is_a_fabrication(self):
        rows = [
            {"parameter": "p", "domain": "message", "written": "x", "wire_value": "invented"},
            {"parameter": "q", "domain": "place", "written": "p1", "wire_value": "zone1"},
        ]

        report = self._report([self._record(references=rows)])

        assert report["reference_resolved_rate"] == 0.5
        assert report["fabricated_reference_count"] == 1
        assert report["reference_apt_rate"] == 0.0, "only the message row defines an expectation"

    def test_a_case_with_no_expectation_scores_no_aptness(self):
        row = {"parameter": "q", "domain": "place", "written": "p1", "wire_value": "zone1"}

        report = self._report([self._record(references=[row])])

        assert report["reference_apt_rate"] is None, "nothing measured is None, never 0"

    def test_no_references_measures_nothing(self):
        report = self._report([self._record()])

        assert report["reference_resolved_rate"] is None
        assert report["fabricated_reference_count"] == 0 and report["reference_values"] == 0

    def test_reference_failures_are_counted_apart_from_the_others(self):
        refused = "Action 'a' sets 'p' to 'x', which is not a message handle this cycle - use m1."
        failures = [refused, "Remark mismatch: x. keys", "something else"]

        report = self._report([self._record(validation_failures=failures, retries=2)])

        assert (report["reference_failures"], report["remark_failures"]) == (1, 1)
        assert report["other_failures"] == 1

    def test_the_failure_phrase_matches_what_both_exceptions_say(self):
        from mind.cognitive_architecture.actions.exceptions import (
            UnknownPlaceHandleError,
            UnknownReferenceHandleError,
        )
        from tests.fixtures.mcp_experiment_analysis import REFERENCE_FAILURE

        assert REFERENCE_FAILURE in str(UnknownReferenceHandleError("p", "x", "message", "a", []))
        assert REFERENCE_FAILURE in str(UnknownPlaceHandleError("x", "a", ["p1"]))
