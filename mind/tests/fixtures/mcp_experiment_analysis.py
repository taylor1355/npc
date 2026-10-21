"""The four analyses over an MCP experiment artifact (NPC-1916, Decision 4).

The experiment run is the expensive part, so it is done once and stored as
JSONL (``tools/run_mcp_experiment.py``); everything here reads that artifact
and spends nothing, except the judge pass, which takes its judge as a callable
so the live model is the CLI's business and the logic is testable offline.

- (a) ``validity_report``: valid-key, agreement and apt rates, retries per turn.
- (b) ``export_dataset``: words -> remark key rows for NPC-1929's classifier.
- (c) ``judge_pass``: on-topic and persona-distinctness scores from a judge.
- (d) ``cost_report``: tokens, retry overhead, latency and catalogue size.

Lives under ``tests/fixtures`` (linted, pinned by
``tests/unit/test_mcp_experiment.py``) for the reason
``tests/fixtures/measurement.py`` does; the ``tools/mcp_experiment_*.py``
scripts are argument parsing and printing.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import combinations
from pathlib import Path

from mind.cognitive_architecture.actions.exceptions import RemarkElementError
from mind.cognitive_architecture.remark_catalogue import RemarkCatalogue

from .mcp_experiment import ARM_CATALOGUE, ARM_RED

ACT = "act_in_interaction"

# The prefix RemarkElementError puts on every message it raises.
REMARK_FAILURE = "Remark mismatch"
# The phrase every reference-handle refusal carries ("... is not a place handle this
# cycle - use one of: ..."): UnknownReferenceHandleError and UnknownPlaceHandleError.
# Pinned against both exceptions by tests/unit/test_mcp_experiment.py.
REFERENCE_FAILURE = "handle this cycle"

# Per reference domain, the record's keys holding every real id of the case and the
# ids an apt turn would name (NPC-2021).
_REFERENCE_ID_SETS = {
    "message": ("known_message_ids", "expect_message_ids"),
    "place": ("known_zone_ids", "expect_zone_ids"),
}


def load_artifact(path: str | Path) -> list[dict]:
    """Every record of a JSONL artifact; a blank line is skipped, a bad one raises."""
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not a JSON record ({exc})") from exc
    if not records:
        raise ValueError(f"{path}: the artifact holds no records")
    return records


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or None when nothing was measured -- never a fabricated 0."""
    return round(numerator / denominator, 4) if denominator else None


def judge_remark(record: dict, catalogue: RemarkCatalogue) -> dict:
    """What the catalogue says about one turn's FINAL parameters.

    Judged offline against the catalogue the run used, so the red arm -- which
    ran with no catalogue and no validation -- is scored by the same rule.
    """
    params = record.get("parameters") or {}
    is_act = record.get("action") == ACT
    named = record.get("remark_named") or ""
    declared = catalogue.declared_kinds(params) if is_act else []
    agrees = False
    problem = None
    if not is_act:
        problem = f"chose {record.get('action')!r}, not a turn"
    else:
        try:
            catalogue.check_turn(params)
            agrees = True
        except RemarkElementError as exc:
            problem = exc.problem
    return {
        "is_act": is_act,
        "named_in_catalogue": is_act and catalogue.get(named) is not None,
        "agrees": agrees,
        "apt": is_act and named in record.get("apt_keys", []),
        "kind_apt": is_act and declared == ([record["apt_kind"]] if record.get("apt_kind") else []),
        "problem": problem,
    }


def judge_references(record: dict) -> list[dict]:
    """How each reference parameter the turn set fared against the case's real ids.

    ``resolved``: the id the wire carries is one of the case's real ids of that
    domain. ``apt``: it is one of the ids an apt turn would name, or None when the
    case defines none (so an unexpected-but-real reference is not scored either
    way). A row whose domain this scorer has no id set for is unscorable and is
    dropped rather than counted as a pass or a fabrication.
    """
    rows = []
    for ref in record.get("references") or []:
        sets = _REFERENCE_ID_SETS.get(ref.get("domain"))
        if sets is None:
            continue
        known, expected = (set(record.get(key) or []) for key in sets)
        wire_value = ref.get("wire_value")
        rows.append(
            {
                "resolved": wire_value in known,
                "apt": (wire_value in expected) if expected else None,
            }
        )
    return rows


# --- (a) validity -------------------------------------------------------------


def validity_report(records: Iterable[dict], catalogue: RemarkCatalogue) -> dict:
    """Per arm: how often a turn named a valid, agreeing and apt remark."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_arm[record["arm"]].append(record)

    report = {}
    for arm, rows in sorted(by_arm.items()):
        judged = [judge_remark(r, catalogue) for r in rows]
        acts = [j for j in judged if j["is_act"]]
        retries = [r.get("retries", 0) for r in rows]
        references = [ref for r in rows for ref in judge_references(r)]
        scored_apt = [ref["apt"] for ref in references if ref["apt"] is not None]
        report[arm] = {
            "turns": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "act_rate": _rate(len(acts), len(rows)),
            "valid_key_rate": _rate(sum(j["named_in_catalogue"] for j in acts), len(acts)),
            "agreement_rate": _rate(sum(j["agrees"] for j in acts), len(acts)),
            "apt_rate": _rate(sum(j["apt"] for j in acts), len(acts)),
            "kind_apt_rate": _rate(sum(j["kind_apt"] for j in acts), len(acts)),
            "first_try_rate": _rate(
                sum(1 for r in rows if not r.get("validation_failures")), len(rows)
            ),
            "mean_retries": round(statistics.fmean(retries), 4) if retries else None,
            # Why attempts failed: the remark rule (among the attempt's errors), or
            # only other errors (JSON, field types, another action validator). One
            # pydantic error can carry both; it counts once, as a remark failure.
            "remark_failures": sum(
                1 for r in rows for f in r.get("validation_failures") or [] if REMARK_FAILURE in f
            ),
            # A refused handle (NPC-2021), counted apart from remark failures. A
            # failure that is a remark failure counts as that, once.
            "reference_failures": sum(
                1
                for r in rows
                for f in r.get("validation_failures") or []
                if REMARK_FAILURE not in f and REFERENCE_FAILURE in f
            ),
            "other_failures": sum(
                1
                for r in rows
                for f in r.get("validation_failures") or []
                if REMARK_FAILURE not in f and REFERENCE_FAILURE not in f
            ),
            # Of the reference values the FINAL actions set: the share the wire
            # carries as a real id of the case. A value that is not one is a
            # fabrication the game would drop.
            "reference_values": len(references),
            "reference_resolved_rate": _rate(
                sum(ref["resolved"] for ref in references), len(references)
            ),
            "fabricated_reference_count": sum(1 for ref in references if not ref["resolved"]),
            "reference_apt_rate": _rate(sum(scored_apt), len(scored_apt)),
            "max_retries": max(retries) if retries else None,
            "salvage_rate": _rate(sum(1 for r in rows if r.get("salvaged")), len(rows)),
            "by_case": _by_case(rows, judged),
        }
    return report


def _by_case(rows: list[dict], judged: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for row, verdict in zip(rows, judged, strict=True):
        case = out.setdefault(row["case_id"], {"turns": 0, "apt": 0, "agrees": 0, "named": []})
        case["turns"] += 1
        case["apt"] += int(verdict["apt"])
        case["agrees"] += int(verdict["agrees"])
        case["named"].append(row.get("remark_named") or "")
    return out


# The rates a floor guards: measured-arm rates must not fall below their first
# measured value; the red arm's agreement must stay below the measured arm's,
# or the red arm has stopped proving the catalogue does anything.
FLOOR_METRICS = ("valid_key_rate", "agreement_rate", "apt_rate", "first_try_rate")


def derive_floors(report: dict, *, source: str) -> dict:
    """Floors from a MEASURED run (plan section 2.5: derived, never guessed)."""
    measured = report.get(ARM_CATALOGUE)
    if measured is None:
        raise ValueError("cannot derive floors: the artifact has no catalogue-arm turns")
    floors = {m: measured[m] for m in FLOOR_METRICS if measured.get(m) is not None}
    return {"source": source, "floors": floors}


def check_floors(report: dict, floors: dict) -> list[str]:
    """Violations of a floors sidecar; an empty list is a pass. Missing data fails."""
    violations = []
    measured = report.get(ARM_CATALOGUE)
    if measured is None:
        return ["no catalogue-arm turns to check"]
    for metric, floor in floors["floors"].items():
        value = measured.get(metric)
        if value is None:
            violations.append(f"{metric}: not measured (floor {floor})")
        elif value < floor:
            violations.append(f"{metric}: {value} below floor {floor}")
    red = report.get(ARM_RED)
    if red is None:
        violations.append("no red-arm turns: the run cannot show the catalogue matters")
    elif (red.get("agreement_rate") or 0) >= (measured.get("agreement_rate") or 0):
        violations.append(
            f"red arm agreement {red.get('agreement_rate')} is not below the measured "
            f"arm's {measured.get('agreement_rate')}: the red arm failed to fail"
        )
    return violations


# --- (b) NPC-1929 dataset -----------------------------------------------------


def export_dataset(
    records: Iterable[dict], catalogue: RemarkCatalogue, *, include_invalid: bool = False
) -> list[dict]:
    """Labelled ``words -> remark key`` rows, with provenance.

    Only catalogue-arm turns: the red arm's labels were named blind. Only
    turns whose remark agrees with their declaration, unless asked: a label is
    worth training on only if it passed the rule the game enforces. Agreement
    is not truthfulness -- the judge pass measures that (plan section 5).
    """
    rows = []
    for record in records:
        if record["arm"] != ARM_CATALOGUE:
            continue
        verdict = judge_remark(record, catalogue)
        if not verdict["is_act"] or not (record.get("words") or "").strip():
            continue
        if not verdict["agrees"] and not include_invalid:
            continue
        named = record.get("remark_named") or ""
        entry = catalogue.get(named)
        rows.append(
            {
                "text": record["words"],
                "label": named,
                "declares": entry.declares if entry is not None else None,
                "agrees": verdict["agrees"],
                "apt": verdict["apt"],
                "case_id": record["case_id"],
                "persona": record["persona"],
                "model": record["model"],
                "backend": record["backend"],
                "run_id": record["run_id"],
                "game_sha": record["game_sha"],
            }
        )
    return rows


# --- (c) judge pass -------------------------------------------------------------

ON_TOPIC_PROMPT = """You are judging one turn of a conversation between villagers in a game.

Conversation so far:
{transcript}

The reply being judged, by {speaker}:
"{words}"

Score how well the reply responds to the conversation so far: 5 = directly on topic and
natural, 3 = loosely related, 1 = ignores or contradicts it. Respond with ONLY a JSON object:
{{"score": <1-5>, "reason": "<one short sentence>"}}"""

DISTINCT_PROMPT = """Two different villagers were each given the same moment in a conversation.

Villager A ({persona_a}) replied: "{words_a}"
Villager B ({persona_b}) replied: "{words_b}"

Score how clearly the two replies sound like DIFFERENT people (voice, attitude, word choice),
not whether either is good: 5 = unmistakably different people, 1 = interchangeable. Respond
with ONLY a JSON object: {{"score": <1-5>, "reason": "<one short sentence>"}}"""

_SCORE = re.compile(r'"score"\s*:\s*([1-5])')


def parse_score(text: str) -> int | None:
    """The judge's 1-5 score, or None when it did not give one -- never a default."""
    match = _SCORE.search(text or "")
    return int(match.group(1)) if match else None


def _transcript(record: dict) -> str:
    """The partner-visible transcript, taken from the first prompt actually sent."""
    attempts = record.get("attempts") or []
    sent = attempts[0]["sent"] if attempts else ""
    marker = "### Active Conversation Transcript\n"
    if marker not in sent:
        return "(transcript unavailable)"
    body = sent.split(marker, 1)[1]
    return body.split("\n\n###", 1)[0].strip()


def judge_pass(records: Iterable[dict], judge: Callable[[str], str]) -> dict:
    """On-topic per turn and distinctness per persona pair, catalogue arm only.

    ``judge`` takes a prompt and returns the model's text. An unparseable
    answer is counted under ``unscored`` and excluded from the means.
    """
    rows = [r for r in records if r["arm"] == ARM_CATALOGUE and (r.get("words") or "").strip()]
    on_topic = []
    for record in rows:
        prompt = ON_TOPIC_PROMPT.format(
            transcript=_transcript(record), speaker=record["persona"], words=record["words"]
        )
        on_topic.append(
            {
                "case_id": record["case_id"],
                "persona": record["persona"],
                "rep": record.get("rep", 0),
                "score": parse_score(judge(prompt)),
            }
        )

    by_case_rep: dict[tuple, dict[str, str]] = defaultdict(dict)
    for record in rows:
        by_case_rep[(record["case_id"], record.get("rep", 0))][record["persona"]] = record["words"]
    distinct = []
    for (case_id, rep), replies in sorted(by_case_rep.items()):
        for persona_a, persona_b in combinations(sorted(replies), 2):
            prompt = DISTINCT_PROMPT.format(
                persona_a=persona_a,
                persona_b=persona_b,
                words_a=replies[persona_a],
                words_b=replies[persona_b],
            )
            distinct.append(
                {
                    "case_id": case_id,
                    "rep": rep,
                    "pair": f"{persona_a}|{persona_b}",
                    "score": parse_score(judge(prompt)),
                }
            )

    def summary(items: list[dict]) -> dict:
        scores = [i["score"] for i in items if i["score"] is not None]
        return {
            "judged": len(items),
            "unscored": len(items) - len(scores),
            "mean": round(statistics.fmean(scores), 3) if scores else None,
        }

    return {
        "on_topic": {**summary(on_topic), "items": on_topic},
        "persona_distinctness": {**summary(distinct), "items": distinct},
    }


# --- (d) cost -------------------------------------------------------------------


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def cost_report(records: Iterable[dict], catalogue: RemarkCatalogue) -> dict:
    """Per arm: tokens per turn, retry overhead, latency, throughput; plus catalogue size."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_arm[record["arm"]].append(record)

    report: dict = {}
    for arm, rows in sorted(by_arm.items()):
        usages = [r["usage"] for r in rows if r.get("usage")]
        attempts = [a for r in rows for a in r.get("attempts") or []]
        retry_attempts = [a for r in rows for a in (r.get("attempts") or [])[1:]]
        latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        completion = [a["completion_tokens"] for a in attempts if a.get("completion_tokens")]
        call_ms = [a["latency_ms"] for a in attempts if a.get("completion_tokens")]

        def mean_of(key: str, source: list[dict] = usages) -> float | None:
            values = [u.get(key) for u in source if u.get(key) is not None]
            return round(statistics.fmean(values), 1) if values else None

        report[arm] = {
            "turns": len(rows),
            "model_calls": len(attempts),
            "calls_per_turn": _rate(len(attempts), len(rows)),
            "prompt_tokens_per_turn": mean_of("prompt_tokens"),
            "completion_tokens_per_turn": mean_of("completion_tokens"),
            "total_tokens_per_turn": mean_of("total_tokens"),
            "retry_calls": len(retry_attempts),
            "retry_tokens": sum(
                (a.get("prompt_tokens") or 0) + (a.get("completion_tokens") or 0)
                for a in retry_attempts
            ),
            "latency_ms_mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "latency_ms_p50": _percentile(latencies, 0.5),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "effective_completion_tokens_per_second": (
                round(sum(completion) / (sum(call_ms) / 1000), 1) if sum(call_ms) else None
            ),
        }

    block = catalogue.prompt_block()
    report["catalogue_block"] = {
        "remarks": len(catalogue.remarks),
        "block_chars": len(block),
        # ~4 characters per token for English prose; labelled as an estimate.
        "block_tokens_estimate": round(len(block) / 4),
    }
    return report
