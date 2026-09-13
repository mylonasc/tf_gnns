#!/usr/bin/env python3
"""Regenerate trace analysis for existing benchmark result directories."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_agent_benchmarks import (  # noqa: E402
    BenchmarkResult,
    JudgeResult,
    TraceAnalysis,
    ValidationResult,
    analyze_trace,
    load_trace_rules,
    write_trace_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect existing opencode agent benchmark traces.")
    parser.add_argument("results_dir", type=Path, help="benchmark output directory containing runs/")
    parser.add_argument("--trace-rules", type=Path, default=None, help="optional JSON trace rules")
    args = parser.parse_args(argv)
    rules = load_trace_rules(args.trace_rules)
    rows: list[dict[str, object]] = []
    for result_path in sorted((args.results_dir / "runs").glob("*/result.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        result = result_from_dict(data)
        analysis = analyze_trace(result_path.parent, result, rules)
        result.analysis = analysis
        write_trace_bundle(result_path.parent, result)
        result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        rows.append({"run_id": result.run_id, "benchmark_id": result.benchmark_id, "analysis": asdict(analysis)})
    aggregate = {
        "runs": rows,
        "remediation_actions": sorted({action for row in rows for action in row["analysis"].get("remediation_actions", [])}),
    }
    (args.results_dir / "trace-inspection-summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(f"inspected {len(rows)} run(s)")
    return 0


def result_from_dict(data: dict[str, object]) -> BenchmarkResult:
    payload = dict(data)
    validation_data = payload.pop("validation", {"status": "not_run"})
    judge_data = payload.pop("judge", {"enabled": False, "status": "not_run"})
    analysis_data = payload.pop("analysis", None)
    validation = ValidationResult(**validation_data) if isinstance(validation_data, dict) else ValidationResult(status="not_run")
    judge = JudgeResult(**judge_data) if isinstance(judge_data, dict) else JudgeResult(enabled=False, status="not_run")
    result = BenchmarkResult(**payload, validation=validation, judge=judge)
    if isinstance(analysis_data, dict):
        result.analysis = TraceAnalysis(**analysis_data)
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
