"""The MCP experiment tier: run the real ReflectionNode over the remark matrix (NPC-1916).

An on-demand testing rung ABOVE the simulation's behavioural suite
(``run_tests.sh --behavioral-only``): expensive, live-only, run by an
orchestrator when the mind's conversational output needs measuring, not on
every PR. It answers Decision 4's questions from ONE recorded run:

  (a) remark-naming validity  -> tools/mcp_experiment_validity.py
  (b) NPC-1929 labelled data  -> tools/mcp_experiment_dataset.py
  (c) reply-quality judging   -> tools/mcp_experiment_judge.py
  (d) prompt and token cost   -> tools/mcp_experiment_cost.py

WHAT IT RUNS. Every (case, persona) of ``tests/fixtures/mcp_experiment.py::CASES``
x ``PERSONAS``, through the real ``ReflectionNode`` and its real retry budget,
in two arms: ``catalogue`` (the remark catalogue present, so naming a remark is
required and validated) and ``no_catalogue`` (the red arm: the same states with
the catalogue removed; naming validity must collapse there, or the measured arm
proves nothing). Only the reflection node runs -- memory query and retrieval are
replaced by the case's committed memories, so a run measures the remark
behaviour and nothing about the vector store.

WHY LIVE-ONLY. Whether a model names an agreeing remark, how often it needs a
retry, and what that costs exist only in a model's answers. The offline parts --
the matrix, the record shape, all four analyses -- are pinned by
``tests/unit/test_mcp_experiment.py`` instead.

ORCHESTRATOR ENTRY POINT. From the ``mind`` project root:

    # Spend nothing: build every state and print prompt sizes.
    PYTHONPATH=$PWD uv run --frozen python tools/run_mcp_experiment.py --dry-run

    # Local model (the target backend; Ollama must be serving):
    PYTHONPATH=$PWD uv run --frozen python tools/run_mcp_experiment.py \\
        --backend ollama --model qwen3:8b --reasoning-effort none \\
        --out runs/qwen3-8b.jsonl

    # Reference backend (needs OPENROUTER_API_KEY in $MIND_HOME/credentials/api_keys.cfg):
    PYTHONPATH=$PWD uv run --frozen python tools/run_mcp_experiment.py \\
        --backend openrouter --model google/gemini-2.5-flash-lite --out runs/flash-lite.jsonl

    # Then the analyses, as often as wanted, over the stored artifact:
    PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_validity.py runs/qwen3-8b.jsonl
    PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_cost.py runs/qwen3-8b.jsonl
    PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_dataset.py runs/qwen3-8b.jsonl \\
        --out runs/qwen3-8b.npc1929.jsonl
    PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_judge.py runs/qwen3-8b.jsonl \\
        --backend ollama --model qwen3:8b --reasoning-effort none

``PYTHONPATH=$PWD`` because the matrix lives under ``tests/fixtures``, which the
wheel does not carry. Each run writes ``<out>.meta.json`` beside the artifact:
the mind commit, whether the tree was dirty, the arguments, and (for Ollama)
``ollama ps`` -- the resident model's VRAM and context length.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mind.apis.langchain_llm import OLLAMA_DEFAULT_BASE_URL, get_llm, get_local_llm
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from tests.fixtures.mcp_experiment import (
    ARMS,
    CASES,
    PERSONAS,
    RecordingLLM,
    matrix_game_sha,
    run_turn,
)

BACKENDS = ("ollama", "openrouter")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _ollama_ps() -> str | None:
    """``ollama ps`` text (resident models, VRAM size, context), or None."""
    exe = shutil.which("ollama") or str(
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    )
    try:
        return subprocess.run(
            [exe, "ps"], capture_output=True, text=True, check=True, timeout=30
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def build_llm(args):
    if args.backend == "ollama":
        return get_local_llm(
            args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
    return get_llm(args.model, temperature=args.temperature)


def _selected(args):
    cases = [c for c in CASES if not args.cases or c.id in args.cases]
    personas = [p for p in PERSONAS if not args.personas or p.id in args.personas]
    unknown = set(args.cases or []) - {c.id for c in CASES}
    unknown |= set(args.personas or []) - {p.id for p in PERSONAS}
    if unknown:
        sys.exit(f"unknown case or persona ids: {sorted(unknown)}")
    return cases, personas


def dry_run(args) -> int:
    cases, personas = _selected(args)
    for arm in args.arms:
        for case in cases:
            for persona in personas:
                state = case.state(persona, with_catalogue=(arm == "catalogue"))
                print(
                    f"{arm:13} {case.id:28} {persona.id:6} "
                    f"actions={len(state.available_actions)} "
                    f"catalogue={'yes' if state.remark_catalogue else 'no'} "
                    f"apt={','.join(case.apt_keys)[:60]}"
                )
    print(f"\n{len(cases) * len(personas) * len(args.arms) * args.reps} turns would run.")
    return 0


async def run(args) -> int:
    cases, personas = _selected(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recorder = RecordingLLM(build_llm(args))
    node = ReflectionNode(recorder)
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{args.model.replace('/', '_').replace(':', '_')}"
    total = len(cases) * len(personas) * len(args.arms) * args.reps
    done = 0
    started = time.time()
    with out.open("w", encoding="utf-8") as handle:
        for rep in range(args.reps):
            for arm in args.arms:
                for case in cases:
                    for persona in personas:
                        record = await run_turn(
                            node,
                            recorder,
                            case,
                            persona,
                            arm,
                            rep=rep,
                            run_id=run_id,
                            backend=args.backend,
                            model=args.model,
                        )
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        done += 1
                        print(
                            f"[{done}/{total}] {arm:13} {case.id:28} {persona.id:6} "
                            f"named={record['remark_named'] or '-':38} "
                            f"retries={record['retries']} "
                            f"{'SALVAGED ' if record['salvaged'] else ''}"
                            f"{'ERROR ' + record['error'] if record['error'] else ''}",
                            flush=True,
                        )
    meta = {
        "run_id": run_id,
        "game_sha": matrix_game_sha(),
        "mind_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "mind_tree": (
            "unknown"
            if (status := _git("status", "--porcelain")) is None
            else ("dirty" if status else "clean")
        ),
        "args": vars(args),
        "turns": total,
        "wall_seconds": round(time.time() - started, 1),
        "ollama_ps": _ollama_ps() if args.backend == "ollama" else None,
    }
    Path(f"{out}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out} ({total} records) and {out}.meta.json")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=BACKENDS, default="ollama")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default=OLLAMA_DEFAULT_BASE_URL)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Ollama only: 'none' turns off a hybrid model's thinking (qwen3).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--cases", nargs="*", help="case ids (default: all)")
    parser.add_argument("--personas", nargs="*", help="persona ids (default: all)")
    parser.add_argument("--out", default="runs/mcp_experiment.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.reps > 1 and args.temperature == 0:
        # At temperature 0 every rep replays the same turns, so --reps would
        # multiply the record count without adding a single sample.
        parser.error(
            f"--reps {args.reps} with --temperature 0 would replay identical samples; "
            "raise --temperature (for example 0.7) or use --reps 1"
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
