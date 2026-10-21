"""Analysis (a) over an MCP experiment artifact: remark-naming validity (NPC-1916).

Valid-key, agreement, apt and kind-apt rates, first-try rate, retries per turn
and failure causes, per arm. Floors are derived from a measured run, never
guessed: ``--write-floors`` records this artifact's catalogue-arm rates as the
floors sidecar; ``--check-floors`` exits 1 on any violation, including a red arm
that failed to fail. Entry point and the run itself: tools/run_mcp_experiment.py.

    PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_validity.py RUN.jsonl \\
        [--write-floors FLOORS.json | --check-floors FLOORS.json] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tests.fixtures.mcp_experiment_analysis import (
    check_floors,
    derive_floors,
    load_artifact,
    validity_report,
)
from tests.fixtures.remark_catalogue import catalogue

HEADLINE = (
    "turns",
    "errors",
    "act_rate",
    "valid_key_rate",
    "agreement_rate",
    "apt_rate",
    "kind_apt_rate",
    "first_try_rate",
    "mean_retries",
    "remark_failures",
    "reference_failures",
    "reference_values",
    "reference_resolved_rate",
    "reference_apt_rate",
    "fabricated_reference_count",
    "other_failures",
    "salvage_rate",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write-floors")
    group.add_argument("--check-floors")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)

    report = validity_report(load_artifact(args.artifact), catalogue())
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for arm, row in report.items():
            print(f"[{arm}]")
            for key in HEADLINE:
                print(f"  {key:16} {row[key]}")
            for case_id, case in row["by_case"].items():
                print(f"    {case_id:28} apt {case['apt']}/{case['turns']}  named {case['named']}")

    if args.write_floors:
        floors = derive_floors(report, source=str(args.artifact))
        Path(args.write_floors).write_text(json.dumps(floors, indent=2) + "\n", encoding="utf-8")
        print(f"wrote floors to {args.write_floors}")
    if args.check_floors:
        floors = json.loads(Path(args.check_floors).read_text(encoding="utf-8"))
        violations = check_floors(report, floors)
        for violation in violations:
            print(f"FLOOR VIOLATION: {violation}")
        print("floors: FAIL" if violations else "floors: PASS")
        return 1 if violations else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
