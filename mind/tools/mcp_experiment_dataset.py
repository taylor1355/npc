"""Analysis (b) over an MCP experiment artifact: the NPC-1929 labelled dataset (NPC-1916).

Writes ``words -> remark key`` rows (JSONL) with provenance, from catalogue-arm
turns whose remark agreed with their declaration. ``--include-invalid`` keeps
the disagreeing ones too, flagged. Entry point: tools/run_mcp_experiment.py.

    PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_dataset.py RUN.jsonl \\
        --out RUN.npc1929.jsonl [--include-invalid]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tests.fixtures.mcp_experiment_analysis import export_dataset, load_artifact
from tests.fixtures.remark_catalogue import catalogue


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact")
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-invalid", action="store_true")
    args = parser.parse_args(argv)

    rows = export_dataset(
        load_artifact(args.artifact), catalogue(), include_invalid=args.include_invalid
    )
    Path(args.out).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    labels = Counter(row["label"] for row in rows)
    print(f"wrote {len(rows)} rows to {args.out}; {len(labels)} distinct labels")
    for label, count in labels.most_common():
        print(f"  {count:4}  {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
