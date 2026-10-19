"""Analysis (d) over an MCP experiment artifact: prompt and token cost (NPC-1916).

Per arm: tokens per turn, model calls per turn, retry calls and the tokens they
burned, latency (mean, p50, p95) and completion throughput; plus the catalogue
block's size. Entry point: tools/run_mcp_experiment.py.

    PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_cost.py RUN.jsonl [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

from tests.fixtures.mcp_experiment_analysis import cost_report, load_artifact
from tests.fixtures.remark_catalogue import catalogue


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = cost_report(load_artifact(args.artifact), catalogue())
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    for section, row in report.items():
        print(f"[{section}]")
        for key, value in row.items():
            print(f"  {key:30} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
