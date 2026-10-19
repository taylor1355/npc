"""Analysis (c) over an MCP experiment artifact: LLM-as-judge reply quality (NPC-1916).

Scores every catalogue-arm reply for being on topic (1-5), and every persona
pair on the same case for sounding like different people (1-5). The first
automated rung for #1002's open reply-quality question. Live: it calls the judge
model once per reply and once per pair. Entry point: tools/run_mcp_experiment.py.

    PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_judge.py RUN.jsonl \\
        --backend ollama --model qwen3:8b --reasoning-effort none [--out JUDGE.json]

A judge from the same family as the model under test grades its own kind; for a
number anyone should act on, prefer a different, stronger judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mind.apis.langchain_llm import OLLAMA_DEFAULT_BASE_URL, get_llm, get_local_llm
from tests.fixtures.mcp_experiment_analysis import judge_pass, load_artifact


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact")
    parser.add_argument("--backend", choices=("ollama", "openrouter"), default="ollama")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default=OLLAMA_DEFAULT_BASE_URL)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--out", help="write the full report (with per-item scores) as JSON")
    args = parser.parse_args(argv)

    if args.backend == "ollama":
        llm = get_local_llm(
            args.model, base_url=args.base_url, reasoning_effort=args.reasoning_effort
        )
    else:
        llm = get_llm(args.model)

    def judge(prompt: str) -> str:
        return str(llm.invoke(prompt).content)

    report = judge_pass(load_artifact(args.artifact), judge)
    report["judge"] = {"backend": args.backend, "model": args.model}
    for name in ("on_topic", "persona_distinctness"):
        section = report[name]
        print(
            f"{name:22} mean {section['mean']}  judged {section['judged']}  "
            f"unscored {section['unscored']}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
