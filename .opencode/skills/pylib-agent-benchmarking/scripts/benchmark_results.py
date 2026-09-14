#!/usr/bin/env python3
"""Read-only result inspection helpers for agent benchmark SQLite databases.

This module is deliberately separate from the benchmark runner. Replace it, wrap
it, or build a richer UI on top of the same SQLite tables and report-data.json
without changing benchmark execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
from typing import Any


@dataclass(frozen=True)
class RecentRun:
    run_started_at: str
    run_id: str
    suite_name: str
    library_name: str
    benchmark_id: str
    status: str
    model: str
    repo_commit_hash: str
    tokens_total: int | None
    tool_call_count: int
    analysis_summary: str


@dataclass(frozen=True)
class RemediationAction:
    action: str
    occurrences: int
    benchmarks: str


@dataclass(frozen=True)
class RunArtifact:
    kind: str
    path: str


class BenchmarkResultIndex:
    """Read-only SQLite facade for benchmark result inspection."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "BenchmarkResultIndex":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def recent_runs(self, limit: int = 20, status: str | None = None) -> list[RecentRun]:
        where = "WHERE status = ?" if status else ""
        params: tuple[object, ...] = (status, limit) if status else (limit,)
        rows = self.conn.execute(
            f"""
            SELECT run_started_at, run_id, suite_name, library_name, benchmark_id, status,
                   model, repo_commit_hash, tokens_total, tool_call_count, analysis_summary
            FROM benchmark_runs
            {where}
            ORDER BY COALESCE(run_started_at, ingested_at) DESC
            LIMIT ?
            """,
            params,
        )
        return [RecentRun(**dict(row)) for row in rows]

    def remediation_summary(self) -> list[RemediationAction]:
        rows = self.conn.execute(
            """
            SELECT action, COUNT(*) AS occurrences, GROUP_CONCAT(DISTINCT benchmark_id) AS benchmarks
            FROM remediation_actions
            JOIN benchmark_runs USING (run_id)
            GROUP BY action
            ORDER BY occurrences DESC, action ASC
            """
        )
        return [RemediationAction(**dict(row)) for row in rows]

    def artifacts_for_run(self, run_id: str) -> list[RunArtifact]:
        rows = self.conn.execute("SELECT kind, path FROM run_artifacts WHERE run_id = ? ORDER BY kind", (run_id,))
        return [RunArtifact(**dict(row)) for row in rows]

    def result_json(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT result_json FROM benchmark_runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return json.loads(row["result_json"])


def rows_to_json(rows: list[object]) -> str:
    return json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True)
