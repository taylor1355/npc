"""Live measurement of what one decision cycle costs (NPC-1318).

Runs the real cognitive pipeline over committed fixtures and reports the
``DecisionTelemetry`` distribution — the same record ``decide_action`` ships to
the simulation, where it becomes a ``DecisionCost`` on the per-decision ledger.

WHY THIS IS LIVE-ONLY. Four telemetry fields — ``completion_tokens``,
``cached_prompt_tokens``, ``cache_write_tokens``, ``cache_reporting`` — plus
``model_calls``, ``unreported_calls``, ``provenance`` and ``server_ms`` cannot
be produced offline at all. A token counter can count a prompt string; it
cannot say whether the provider served the static prefix from its cache, how
many round-trips a retry burned, or what the model wrote back. The offline
arm that CAN be answered without an LLM is the recent-events rendering A/B,
and that already exists as ``tools/measure_recent_events_rendering.py``.

WHAT THIS DOES NOT DO. It does not compare against the 5,367 tok/cycle figure
recorded on 2026-08-20. That harness was never committed, the fixtures it ran
on carried no events (so its buffer was empty), the node set has since changed
from three LLM steps to two, the reflection prompt has gained a cache
breakpoint, and the event rendering was rewritten. Subtracting across those is
arithmetic, not measurement. This produces a NEW baseline.

Usage — from the ``mind`` project root, inside WSL:

    export UV_PROJECT_ENVIRONMENT=/home/taylor/.local/share/npc-simulation/venvs/mind
    PYTHONPATH=$PWD uv run --frozen python tools/measure_decision_cycle.py --dry-run
    PYTHONPATH=$PWD uv run --frozen python tools/measure_decision_cycle.py --reps 3 \
        --json /tmp/decision_cycle_raw.json

``PYTHONPATH=$PWD`` because the matrix lives under ``tests/fixtures``, which is
not part of the installed wheel. Run ``--dry-run`` FIRST: it builds every
pipeline state and prints the buffer and action counts without a single
provider round-trip, so an empty-buffer scenario is caught before any token is
spent — which is exactly the defect that made the predecessor number unusable.

The logic this depends on lives under ``tests/fixtures/measurement.py`` (inside
ruff's and pytest's scope, pinned by ``tests/unit/test_measurement_fixtures.py``);
this file is argument parsing and printing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mind import constants
from mind.interfaces.mcp.mind import Mind
from mind.interfaces.mcp.models import DecisionTelemetry
from tests.fixtures.measurement import (
    MeasurementScenario,
    apply_cycle_inputs,
    build_measurement_scenarios,
    scenario_config,
)

# LLM nodes per cycle. memory_retrieval makes no provider call, so a clean run
# burns exactly this many round-trips per cycle; anything above it is retries.
EXPECTED_MODEL_CALLS_PER_CYCLE = 2


def _git_prefix() -> list[str]:
    """``git``, plus a ``--git-dir`` when the repo is a Windows-hosted worktree.

    A LINKED worktree's ``.git`` is a file holding an absolute ``gitdir:`` path.
    When the checkout lives on a Windows drive and this runs under WSL, that
    path is a ``C:/...`` string Linux git cannot follow, and every git call
    fails with "not a git repository" — so the harness reports ``unknown`` for
    its own commit and the numbers travel with no scope at all. Translating the
    drive prefix is what keeps a figure measured here re-checkable elsewhere.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    marker = repo_root / ".git"
    if not marker.is_file():
        return ["git"]
    text = marker.read_text().strip()
    if not text.startswith("gitdir:"):
        return ["git"]
    raw = text.split(":", 1)[1].strip()
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if not match:
        return ["git"]
    drive, rest = match.groups()
    git_dir = "/mnt/" + drive.lower() + "/" + rest.replace("\\", "/")
    # --work-tree too: with only --git-dir, git takes the CWD as the work tree,
    # and this runs from mind/ — so `status` would describe a subdirectory and
    # call a dirty tree clean whenever the edits sat outside it.
    return ["git", f"--git-dir={git_dir}", f"--work-tree={repo_root}"]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            [*_git_prefix(), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _resolved_commit() -> str:
    """The commit these numbers describe, so the figure carries its own scope."""
    return _git("rev-parse", "--short", "HEAD") or "unknown"


def _dirty() -> str:
    """Whether the tree differs from that commit. A measurement taken on a
    dirty tree describes no commit anybody else can check out — and "unknown"
    is reported as itself rather than as "clean", because a check that could
    not run must never be read as a check that passed."""
    out = _git("status", "--porcelain")
    if out is None:
        return "unknown"
    return "dirty" if out else "clean"


def _build_mind(scenario: MeasurementScenario, storage_path: str, model: str) -> Mind:
    config = scenario_config(scenario, storage_path)
    if model != config.llm_model:
        config = config.model_copy(update={"llm_model": model})
    mind = Mind.from_config(
        mind_id=f"measure_{scenario.id}",
        entity_id=scenario.entity_id,
        config=config,
    )
    apply_cycle_inputs(mind, scenario)
    return mind


def _print_header(args, mind: Mind, scenario_count: int) -> None:
    node_names = sorted(mind.pipeline.graph.nodes)
    reflection = mind.pipeline.reflection_node
    print(
        f"commit={_resolved_commit()} ({_dirty()})  model={args.model}  temperature=0  "
        f"reps={args.reps}  scenarios={scenario_count}"
    )
    print(
        f"nodes={node_names}  "
        f"reflection_static_prefix_chars={len(reflection.static_prefix)}  "
        f"cache_breakpoint={'on' if reflection._cache_breakpoint_enabled() else 'off'}"
    )


def _print_structure(rows: list[dict]) -> None:
    print()
    print(f"{'scenario':<26}{'events':>8}{'actions':>9}{'memories':>10}  why")
    for row in rows:
        print(
            f"{row['scenario']:<26}{row['n_events']:>8}{row['n_actions']:>9}"
            f"{row['n_seed_memories']:>10}  {row['why']}"
        )
    empty = [row["scenario"] for row in rows if row["n_events"] == 0]
    if empty:
        print(f"\nEMPTY EVENT BUFFER in {empty} — do not spend tokens on this matrix.")
    else:
        print("\nEvery scenario has a populated event buffer.")


def _stats(values: list[float]) -> str:
    if not values:
        return f"{'-':>9}{'-':>9}{'-':>9}{'-':>9}"
    return (
        f"{statistics.mean(values):>9.1f}{statistics.median(values):>9.1f}"
        f"{min(values):>9.0f}{max(values):>9.0f}"
    )


def _report(records: list[dict], reps: int, scenario_count: int) -> None:
    failed = [r for r in records if "failed" in r]
    ok = [r for r in records if "failed" not in r]
    metered = [r for r in ok if r["telemetry"]["provenance"] == "metered"]
    unreported = [r for r in ok if r["telemetry"]["provenance"] == "unreported"]

    print(f"\ncycles attempted={reps * scenario_count}  completed={len(ok)}  failed={len(failed)}")
    for record in failed:
        print(f"  FAILED {record['scenario']} rep {record['rep']}: {record['failed']}")

    print(f"provenance: metered={len(metered)}  unreported={len(unreported)}")
    if unreported:
        print(
            "  Unreported cycles are EXCLUDED from every mean below. A provider that "
            "reported nothing is unknown cost, never zero cost."
        )
    if not metered:
        print("\nNo metered cycles — nothing to average.")
        return

    # --- Per cycle -----------------------------------------------------------
    #
    # TWO populations, both published, because they answer different questions
    # and neither substitutes for the other. ALL CYCLES is what the provider
    # actually bills — retried round-trips are real spend. RETRY-FREE is the
    # cost of the cognitive work itself, and it is what a per-cycle cost model
    # should be fitted on: a retry multiplies one node's tokens without
    # producing anything extra, so a mean over a mixed population is neither
    # figure. Publishing only the first hides the cognition; publishing only
    # the second hides the bill.
    retry_free = [
        r for r in metered if r["telemetry"]["model_calls"] == EXPECTED_MODEL_CALLS_PER_CYCLE
    ]
    retry_free_keys = {(r["scenario"], r["rep"]) for r in retry_free}
    fields = ("total_tokens", "prompt_tokens", "completion_tokens", "cached_prompt_tokens")
    for title, population in (
        ("per cycle, ALL", metered),
        ("per cycle, RETRY-FREE", retry_free),
    ):
        label = f"{title} (n={len(population)})"
        print(f"\n{label:<26}{'mean':>9}{'median':>9}{'min':>9}{'max':>9}")
        for field in fields:
            print(f"{field:<26}{_stats([float(r['telemetry'][field]) for r in population])}")
        print(
            f"{'server_ms':<26}{_stats([float(r['telemetry']['server_ms']) for r in population])}"
        )

    # --- Per node ------------------------------------------------------------
    # Split the same two ways as per-cycle, and for the same reason: a retry
    # multiplies ONE node's tokens, so an all-cycles node mean says as much
    # about the provider's failure rate as about the node. Step names are
    # enumerated from the records, never hardcoded, so a future node merge or
    # split changes this table instead of silently mislabelling it.
    step_names = sorted({name for r in metered for name in r["telemetry"]["per_step"]})
    for title, population in (("per node, ALL", metered), ("per node, RETRY-FREE", retry_free)):
        print(
            f"\n{title:<26}{'prompt':>9}{'compl':>9}{'total':>9}"
            f"{'cached':>9}{'cache_wr':>10}{'calls':>8}{'ms':>8}"
        )
        for name in step_names:
            steps = [
                r["telemetry"]["per_step"][name]
                for r in population
                if name in r["telemetry"]["per_step"]
            ]
            if not steps:
                continue
            ms = [r["telemetry"]["per_step_ms"].get(name, 0) for r in population]
            print(
                f"{name:<26}"
                f"{statistics.mean(s['prompt_tokens'] for s in steps):>9.1f}"
                f"{statistics.mean(s['completion_tokens'] for s in steps):>9.1f}"
                f"{statistics.mean(s['total_tokens'] for s in steps):>9.1f}"
                f"{statistics.mean(s['cached_prompt_tokens'] for s in steps):>9.1f}"
                f"{statistics.mean(s['cache_write_tokens'] for s in steps):>10.1f}"
                f"{sum(s['model_calls'] for s in steps):>8}"
                f"{statistics.mean(ms):>8.0f}"
            )

    # --- Round-trip accounting ----------------------------------------------
    total_calls = sum(r["telemetry"]["model_calls"] for r in ok)
    expected_calls = EXPECTED_MODEL_CALLS_PER_CYCLE * len(ok)
    unreported_calls = sum(r["telemetry"]["unreported_calls"] for r in ok)
    print(f"\nmodel_calls={total_calls} (expected {expected_calls} = 2 x {len(ok)} cycles)")
    if total_calls > expected_calls:
        print(
            f"  CONTAMINATED: {total_calls - expected_calls} extra round-trip(s) means retries "
            f"fired. Retried tokens are real spend with no extra cognitive product, so the "
            f"means above are inflated for the affected node — read them as an upper bound."
        )
    elif total_calls < expected_calls:
        print("  FEWER calls than cycles x 2 — a node did not run. Investigate before publishing.")
    print(f"unreported_calls={unreported_calls}")

    reporting = sum(1 for r in ok if r["telemetry"]["cache_reporting"])
    print(
        f"cache_reporting={reporting}/{len(ok)} cycles carried real provider cache accounting "
        f"(0 cached tokens means 'no hits' only where this is true; elsewhere it means "
        f"'nobody told us')"
    )

    # --- Per scenario --------------------------------------------------------
    #
    # Retry-free only, and the `retried` column says how many cycles were left
    # out. A scenario whose retries were averaged in reports a token cost that
    # is about the provider's failure rate rather than about the scenario, and
    # the ordering between scenarios inverts — which is the whole reason the
    # axes columns beside it (events/actions/retrvd) are worth reading.
    print(
        f"\n{'per scenario (retry-free)':<26}{'n':>4}{'total':>9}{'prompt':>9}{'compl':>9}"
        f"{'events':>8}{'actions':>9}{'retrvd':>8}{'retried':>9}"
    )
    for scenario_id in sorted({r["scenario"] for r in metered}):
        all_rows = [r for r in metered if r["scenario"] == scenario_id]
        rows = [r for r in all_rows if (r["scenario"], r["rep"]) in retry_free_keys]
        excluded = len(all_rows) - len(rows)
        if not rows:
            print(f"{scenario_id:<26}{0:>4}{'  every cycle retried':<45}{excluded:>9}")
            continue
        print(
            f"{scenario_id:<26}{len(rows):>4}"
            f"{statistics.mean(r['telemetry']['total_tokens'] for r in rows):>9.1f}"
            f"{statistics.mean(r['telemetry']['prompt_tokens'] for r in rows):>9.1f}"
            f"{statistics.mean(r['telemetry']['completion_tokens'] for r in rows):>9.1f}"
            f"{rows[0]['n_events']:>8}{rows[0]['n_actions']:>9}"
            f"{statistics.mean(r['n_retrieved'] for r in rows):>8.1f}"
            f"{excluded:>9}"
        )

    print("\nActions chosen: " + ", ".join(f"{r['scenario']}/{r['rep']}={r['action']}" for r in ok))


def replay(path: str) -> int:
    """Re-render the report from a saved raw dump, spending nothing.

    A summary cannot be re-analysed; raw records can. This is what makes the
    ``--json`` dump worth writing: when the report grows a new breakdown, an
    earlier run's numbers can be re-read through it instead of being re-bought.
    """
    with open(path) as handle:
        payload = json.load(handle)
    records = payload["records"]
    scenario_count = len({r["scenario"] for r in records}) or 1
    print(
        f"REPLAY of {path}: commit={payload.get('commit')} ({payload.get('tree')})  "
        f"model={payload.get('model')}  reps={payload.get('reps')}"
    )
    _report(records, payload.get("reps", 0), scenario_count)
    return 0


async def run(args) -> int:
    scenarios = build_measurement_scenarios()

    # Structural pass first, always — even on a live run. Building the states
    # costs nothing and an empty buffer invalidates everything downstream.
    structure: list[dict] = []
    header_printed = False
    for scenario in scenarios:
        with TemporaryDirectory() as tmp:
            mind = _build_mind(scenario, tmp, args.model)
            if not header_printed:
                _print_header(args, mind, len(scenarios))
                header_printed = True
            state = mind.build_pipeline_state(scenario.observation)
            structure.append(
                {
                    "scenario": scenario.id,
                    "why": scenario.why,
                    "n_events": len(state.recent_events),
                    "n_actions": len(state.available_actions),
                    "n_seed_memories": len(scenario.config.initial_long_term_memories),
                }
            )
    _print_structure(structure)

    if any(row["n_events"] == 0 for row in structure):
        return 1
    if args.dry_run:
        print("\n--dry-run: no provider round-trips were made.")
        return 0

    records: list[dict] = []
    for rep in range(args.reps):
        for scenario in scenarios:
            # Fresh store per repetition: a shared one lets rep 2 retrieve what
            # rep 1 wrote, so the reps stop being independent samples.
            with TemporaryDirectory() as tmp:
                mind = _build_mind(scenario, tmp, args.model)
                state = mind.build_pipeline_state(scenario.observation)
                try:
                    result = await mind.pipeline.process(state)
                except Exception as error:  # noqa: BLE001 - a failed cycle must be recorded, not raised
                    print(f"  cycle failed: {scenario.id} rep {rep}: {error!r}", file=sys.stderr)
                    records.append({"scenario": scenario.id, "rep": rep, "failed": repr(error)})
                    continue

                telemetry = DecisionTelemetry.from_pipeline_state(result, mind.llm_model)
                records.append(
                    {
                        "scenario": scenario.id,
                        "rep": rep,
                        "telemetry": telemetry.model_dump(),
                        "n_events": len(state.recent_events),
                        "n_actions": len(state.available_actions),
                        "n_retrieved": len(result.retrieved_memories),
                        "action": result.chosen_action.action if result.chosen_action else None,
                    }
                )
            print(f"  ran {scenario.id} rep {rep}", file=sys.stderr)

    _report(records, args.reps, len(scenarios))

    if args.json:
        payload = {
            "commit": _resolved_commit(),
            "tree": _dirty(),
            "model": args.model,
            "reps": args.reps,
            "records": records,
        }
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nRaw per-cycle records written to {args.json}")

    return 0 if all("failed" not in r for r in records) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=3, help="repetitions per scenario")
    parser.add_argument(
        "--model", default=constants.DEFAULT_SMALL_MODEL, help="OpenRouter model slug"
    )
    parser.add_argument("--json", help="write the raw per-cycle records here")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build every pipeline state and print its shape, making no provider call",
    )
    parser.add_argument(
        "--replay",
        help="re-render the report from a previous --json dump; makes no provider call",
    )
    args = parser.parse_args()
    if args.replay:
        sys.exit(replay(args.replay))
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
