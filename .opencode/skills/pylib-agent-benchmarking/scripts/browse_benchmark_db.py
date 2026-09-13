#!/usr/bin/env python3
"""Browse local agent benchmark SQLite results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_results import BenchmarkResultIndex, rows_to_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse local agent benchmark SQLite results.")
    parser.add_argument("--db-path", type=Path, default=Path("agent_benchmark_results/agent_benchmarks.sqlite"))
    parser.add_argument("--json", action="store_true", help="emit JSON instead of tabular text")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recent = subparsers.add_parser("recent", help="show recent runs")
    recent.add_argument("--limit", type=int, default=20)
    recent.add_argument("--status", default=None, help="filter by run status, e.g. failed or passed")

    subparsers.add_parser("remediation", help="summarize remediation actions")

    artifacts = subparsers.add_parser("artifacts", help="list artifact paths for one run")
    artifacts.add_argument("run_id")

    result = subparsers.add_parser("result", help="print stored result JSON for one run")
    result.add_argument("run_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with BenchmarkResultIndex(args.db_path) as index:
        if args.command == "recent":
            rows = index.recent_runs(limit=args.limit, status=args.status)
            if args.json:
                print(rows_to_json(rows))
                return 0
            for row in rows:
                commit = row.repo_commit_hash[:8] if row.repo_commit_hash else ""
                tokens = row.tokens_total if row.tokens_total is not None else ""
                print(f"{row.run_started_at}\t{row.status}\t{row.benchmark_id}\t{row.model}\t{tokens}\t{row.tool_call_count}\t{commit}\t{row.run_id}")
                if row.analysis_summary:
                    print(f"  analysis: {row.analysis_summary}")
        elif args.command == "remediation":
            rows = index.remediation_summary()
            if args.json:
                print(rows_to_json(rows))
                return 0
            for row in rows:
                print(f"{row.occurrences}\t{row.action}\t[{row.benchmarks or ''}]")
        elif args.command == "artifacts":
            rows = index.artifacts_for_run(args.run_id)
            if args.json:
                print(rows_to_json(rows))
                return 0
            for row in rows:
                print(f"{row.kind}\t{row.path}")
        elif args.command == "result":
            result = index.result_json(args.run_id)
            if result is None:
                print(f"unknown run_id: {args.run_id}", file=sys.stderr)
                return 1
            print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
