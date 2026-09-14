#!/usr/bin/env python3
"""Run generic opencode agentic coding benchmarks for Python-library docs skills."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import argparse
import csv
import fnmatch
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable
import uuid


@dataclass(frozen=True)
class SuiteConfig:
    suite_name: str
    library_name: str
    suite_root: Path
    benchmark_dir: Path
    output_dir: Path
    solution_dir: str = "agent_benchmark_solutions"
    db_path: Path | None = None
    trace_rules_path: Path | None = None
    user_skill_path: Path | None = None
    user_skill_install_command: str = ""
    agent_name: str = "pylib-benchmark-agent"
    judge_agent_name: str = "pylib-benchmark-judge"
    agent_prompt: str = ""
    judge_prompt: str = ""

    @property
    def effective_db_path(self) -> Path:
        return self.db_path or (self.output_dir / "agent_benchmarks.sqlite")


@dataclass(frozen=True)
class RunnerConfig:
    repo_root: Path
    skill_root: Path
    suite: SuiteConfig
    model: str
    judge_model: str | None = None
    benchmarks: list[str] | None = None
    repetitions: int = 1
    timeout_seconds: int | None = None
    parallelism: str = "sequential"
    max_workers: int = 1
    auto: bool = True
    judge_enabled: bool = True
    html: bool = False
    dry_run: bool = False
    preserve_worktrees: bool = False
    worktree_base: Path | None = None

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model

    @property
    def skill_version(self) -> str:
        version_file = self.skill_root / "VERSION"
        if not version_file.exists():
            return "unknown"
        return version_file.read_text(encoding="utf-8").strip() or "unknown"


@dataclass(frozen=True)
class BenchmarkTask:
    benchmark_id: str
    name: str
    description: str
    prompt: str
    validation_commands: list[str]
    setup_commands: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    judge_rubric: str = ""
    minimum_judge_score: int = 4
    timeout_seconds: int = 900
    source_path: Path | None = None

    @property
    def task_hash(self) -> str:
        source = self.source_path.read_bytes() if self.source_path else repr(self).encode("utf-8")
        return hashlib.sha256(source).hexdigest()[:16]

    @classmethod
    def from_file(cls, path: Path) -> "BenchmarkTask":
        data = load_simple_yaml(path)
        return cls(
            benchmark_id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            description=str(data.get("description", "")),
            prompt=str(data["prompt"]),
            validation_commands=[str(item) for item in data.get("validation_commands", [])],
            setup_commands=[str(item) for item in data.get("setup_commands", [])],
            allowed_paths=[str(item) for item in data.get("allowed_paths", [])],
            tags=[str(item) for item in data.get("tags", [])],
            judge_rubric=str(data.get("judge_rubric", "")),
            minimum_judge_score=int(data.get("minimum_judge_score", 4)),
            timeout_seconds=int(data.get("timeout_seconds", 900)),
            source_path=path,
        )


@dataclass
class ValidationResult:
    status: str
    command: str = ""
    returncode: int | None = None
    stdout_path: str = ""
    stderr_path: str = ""
    duration_seconds: float | None = None


@dataclass
class JudgeResult:
    enabled: bool
    status: str
    score: int | None = None
    max_score: int = 5
    passed: bool | None = None
    reasoning: str = ""
    raw_output_path: str = ""


@dataclass
class TraceAnalysis:
    status: str = "not_run"
    summary: str = ""
    self_correction_count: int = 0
    suspected_error_patterns: list[str] = field(default_factory=list)
    library_misuse_signals: list[str] = field(default_factory=list)
    invalid_code_signals: list[str] = field(default_factory=list)
    documentation_gaps: list[str] = field(default_factory=list)
    remediation_actions: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    run_id: str
    suite_name: str
    library_name: str
    benchmark_id: str
    benchmark_name: str
    repetition: int
    status: str
    model: str
    judge_model: str
    benchmark_skill_version: str
    benchmark_task_hash: str
    repo_commit_hash: str
    repo_dirty: bool
    opencode_version: str
    parallelism_mode: str
    auto_approve: bool
    run_started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_finished_at: str = ""
    wall_seconds: float | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    tokens_total: int | None = None
    model_context_configured: int | None = None
    model_context_available: int | None = None
    cost_usd: float | None = None
    tool_call_count: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    tool_call_trace: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    validation: ValidationResult = field(default_factory=lambda: ValidationResult(status="not_run"))
    judge: JudgeResult = field(default_factory=lambda: JudgeResult(enabled=False, status="not_run"))
    run_dir: str = ""
    worktree_path: str = ""
    patch_path: str = ""
    session_export_path: str = ""
    events_path: str = ""
    trace_path: str = ""
    analysis: TraceAnalysis = field(default_factory=TraceAnalysis)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpencodeRunResult:
    returncode: int
    wall_seconds: float
    events: list[dict[str, Any]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    session_id: str = ""
    tokens_input: int | None = None
    tokens_output: int | None = None
    tokens_total: int | None = None
    cost_usd: float | None = None
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    tool_call_trace: list[dict[str, Any]] = field(default_factory=list)
    timed_out: bool = False

    @property
    def tool_call_count(self) -> int:
        return sum(self.tool_call_counts.values())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    script_path = Path(__file__).resolve()
    skill_root = script_path.parents[1]
    repo_root = args.repo_root.resolve() if args.repo_root else Path.cwd().resolve()
    suite = load_suite_config(args.suite_config.resolve())
    if args.output_dir:
        suite = dataclass_replace(suite, output_dir=args.output_dir.resolve())
    if args.db_path:
        suite = dataclass_replace(suite, db_path=args.db_path.resolve())
    max_workers = args.max_workers if args.parallelism == "parallel" else 1
    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")
    if max_workers < 1:
        parser.error("--max-workers must be >= 1")
    config = RunnerConfig(
        repo_root=repo_root,
        skill_root=skill_root,
        suite=suite,
        model=args.model,
        judge_model=args.judge_model,
        benchmarks=args.benchmarks,
        repetitions=args.repetitions,
        timeout_seconds=args.timeout_seconds,
        parallelism=args.parallelism,
        max_workers=max_workers,
        auto=args.auto,
        judge_enabled=args.judge_enabled,
        html=args.html,
        dry_run=args.dry_run,
        preserve_worktrees=args.preserve_worktrees,
        worktree_base=args.worktree_base.resolve() if args.worktree_base else None,
    )
    results = BenchmarkRunner(config).run()
    passed = sum(1 for result in results if result.status == "passed")
    print(f"agent benchmarks complete: {passed}/{len(results)} passed")
    print(f"results: {suite.output_dir}")
    print(f"sqlite db: {suite.effective_db_path}")
    return 0 if passed == len(results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run generic Python-library opencode agent benchmarks.")
    parser.add_argument("--suite-config", type=Path, required=True, help="suite YAML config")
    parser.add_argument("--repo-root", type=Path, default=None, help="repository to benchmark; defaults to current directory")
    parser.add_argument("--model", default="ollama/gemma4:26b", help="opencode model id for coding agents")
    parser.add_argument("--judge-model", default=None, help="model id for LLM-as-judge; defaults to --model")
    parser.add_argument("--benchmarks", nargs="*", default=None, help="benchmark ids to run; defaults to all")
    parser.add_argument("--repetitions", type=int, default=1, help="number of repetitions per benchmark")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="override per-task opencode timeout")
    parser.add_argument("--parallelism", choices=("sequential", "parallel"), default="sequential")
    parser.add_argument("--max-workers", type=int, default=1, help="parallel worker count when --parallelism parallel")
    parser.add_argument("--output-dir", type=Path, default=None, help="override suite output_dir")
    parser.add_argument("--db-path", type=Path, default=None, help="override SQLite database path")
    parser.add_argument("--worktree-base", type=Path, default=None, help="directory for temporary git worktrees")
    parser.add_argument("--preserve-worktrees", action="store_true", help="keep worktrees after each run")
    parser.add_argument("--html", action="store_true", help="render index.html report")
    parser.add_argument("--dry-run", action="store_true", help="exercise runner/reporting without calling opencode")
    parser.add_argument("--no-auto", dest="auto", action="store_false", help="do not pass --auto to opencode run")
    parser.set_defaults(auto=True)
    parser.add_argument("--no-judge", dest="judge_enabled", action="store_false", help="disable LLM-as-judge")
    parser.set_defaults(judge_enabled=True)
    return parser


class BenchmarkRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.trace_rules = load_trace_rules(config.suite.trace_rules_path)

    def run(self) -> list[BenchmarkResult]:
        benchmarks = load_benchmarks(self.config.suite.benchmark_dir, self.config.benchmarks)
        jobs = [(benchmark, repetition) for benchmark in benchmarks for repetition in range(1, self.config.repetitions + 1)]
        self.config.suite.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.parallelism == "parallel" and self.config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                futures = [pool.submit(self._run_one, benchmark, repetition) for benchmark, repetition in jobs]
                results = [future.result() for future in as_completed(futures)]
        else:
            results = [self._run_one(benchmark, repetition) for benchmark, repetition in jobs]
        results.sort(key=lambda item: (item.benchmark_id, item.repetition, item.run_id))
        write_results(self.config.suite.output_dir, results)
        summary = summarize_results(results)
        summary["sqlite_db_path"] = str(self.config.suite.effective_db_path)
        write_report_data(self.config.suite.output_dir, results, summary)
        (self.config.suite.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        with BenchmarkDB(self.config.suite.effective_db_path) as db:
            db.upsert_results(results)
        if self.config.html:
            render_html_report(self.config.suite.output_dir, results, summary)
        return results

    def _run_one(self, benchmark: BenchmarkTask, repetition: int) -> BenchmarkResult:
        started_at = datetime.now(timezone.utc).isoformat()
        run_id = f"{benchmark.benchmark_id}-r{repetition}-{uuid.uuid4().hex[:8]}"
        run_dir = self.config.suite.output_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._base_metadata(benchmark, repetition, run_id, run_dir)
        metadata["run_started_at"] = started_at
        worktree: Path | None = None
        try:
            worktree = self._prepare_worktree(run_id)
            metadata["worktree_path"] = str(worktree if self.config.preserve_worktrees else "")
            self._write_opencode_config(worktree)
            self._install_user_skill_if_configured(worktree, run_dir)
            if self.config.dry_run:
                return self._run_dry(metadata, run_dir)
            self._run_setup(benchmark, worktree, run_dir)
            events_path = run_dir / "opencode-events.jsonl"
            opencode_result = run_opencode_task(
                benchmark=benchmark,
                worktree=worktree,
                model=self.config.model,
                agent_name=self.config.suite.agent_name,
                auto=self.config.auto,
                timeout_seconds=self.config.timeout_seconds or benchmark.timeout_seconds,
                events_path=events_path,
            )
            session_export_path = run_dir / "session-export.json"
            if opencode_result.session_id:
                export_session(opencode_result.session_id, worktree, session_export_path)
            patch_path = run_dir / "patch.diff"
            patch_text = self._write_patch(worktree, patch_path)
            if opencode_result.timed_out:
                validation = ValidationResult(status="not_run", command="skipped after opencode timeout")
                judge = JudgeResult(enabled=self.config.judge_enabled, status="skipped", passed=False if self.config.judge_enabled else None)
            else:
                validation = self._run_validation(benchmark, worktree, run_dir)
                judge = self._run_judge_if_enabled(benchmark, worktree, run_dir, patch_text, validation)
            status = combined_status(opencode_result.returncode, validation, judge)
            result = BenchmarkResult(
                **metadata,
                status=status,
                wall_seconds=opencode_result.wall_seconds,
                tokens_input=opencode_result.tokens_input,
                tokens_output=opencode_result.tokens_output,
                tokens_total=opencode_result.tokens_total,
                cost_usd=opencode_result.cost_usd,
                tool_call_count=opencode_result.tool_call_count,
                tool_call_counts=opencode_result.tool_call_counts,
                tool_call_trace=opencode_result.tool_call_trace,
                session_id=opencode_result.session_id,
                validation=validation,
                judge=judge,
                patch_path=str(patch_path),
                session_export_path=str(session_export_path if session_export_path.exists() else ""),
                events_path=str(events_path),
                error="opencode timed out" if opencode_result.timed_out else "",
            )
            return self._finalize_result(run_dir, result)
        except Exception as exc:  # noqa: BLE001 - benchmark runner records failures.
            return self._finalize_result(run_dir, BenchmarkResult(**metadata, status="error", error=str(exc)))
        finally:
            if worktree and not self.config.preserve_worktrees:
                self._remove_worktree(worktree)

    def _base_metadata(self, benchmark: BenchmarkTask, repetition: int, run_id: str, run_dir: Path) -> dict[str, object]:
        return {
            "run_id": run_id,
            "suite_name": self.config.suite.suite_name,
            "library_name": self.config.suite.library_name,
            "benchmark_id": benchmark.benchmark_id,
            "benchmark_name": benchmark.name,
            "repetition": repetition,
            "model": self.config.model,
            "judge_model": self.config.effective_judge_model,
            "benchmark_skill_version": self.config.skill_version,
            "benchmark_task_hash": benchmark.task_hash,
            "repo_commit_hash": command_text(["git", "rev-parse", "HEAD"], self.config.repo_root),
            "repo_dirty": bool(command_text(["git", "status", "--porcelain"], self.config.repo_root)),
            "opencode_version": command_text(["opencode", "--version"], self.config.repo_root),
            "parallelism_mode": self.config.parallelism,
            "auto_approve": self.config.auto,
            "run_dir": str(run_dir),
            "model_context_configured": configured_context_limit(self.config.model),
            "model_context_available": available_context_limit(self.config.model),
        }

    def _run_dry(self, metadata: dict[str, object], run_dir: Path) -> BenchmarkResult:
        events_path = run_dir / "opencode-events.jsonl"
        events_path.write_text(
            json.dumps({"type": "tool.execute.before", "tool": "write", "status": "started"}) + "\n"
            + json.dumps({"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}) + "\n",
            encoding="utf-8",
        )
        parsed = parse_opencode_events(events_path.read_text(encoding="utf-8"))
        patch_path = run_dir / "patch.diff"
        patch_path.write_text("", encoding="utf-8")
        judge = dry_run_judge(run_dir / "judge.json") if self.config.judge_enabled else JudgeResult(enabled=False, status="disabled")
        result = BenchmarkResult(
            **metadata,
            status="passed",
            wall_seconds=0.0,
            tokens_input=parsed.tokens_input,
            tokens_output=parsed.tokens_output,
            tokens_total=parsed.tokens_total,
            tool_call_count=parsed.tool_call_count,
            tool_call_counts=parsed.tool_call_counts,
            tool_call_trace=parsed.tool_call_trace,
            validation=ValidationResult(status="passed", command="dry-run", returncode=0),
            judge=judge,
            patch_path=str(patch_path),
            events_path=str(events_path),
        )
        return self._finalize_result(run_dir, result)

    def _finalize_result(self, run_dir: Path, result: BenchmarkResult) -> BenchmarkResult:
        result.run_finished_at = datetime.now(timezone.utc).isoformat()
        result.trace_path = str(write_trace_bundle(run_dir, result))
        result.analysis = analyze_trace(run_dir, result, self.trace_rules)
        write_trace_bundle(run_dir, result)
        (run_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return result

    def _prepare_worktree(self, run_id: str) -> Path:
        base = self.config.worktree_base or (self.config.suite.output_dir / "worktrees")
        base.mkdir(parents=True, exist_ok=True)
        worktree = base / run_id
        if self.config.dry_run:
            worktree.mkdir(parents=True, exist_ok=True)
            return worktree
        subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=self.config.repo_root, text=True, capture_output=True, check=True)
        return worktree

    def _remove_worktree(self, worktree: Path) -> None:
        if self.config.dry_run:
            shutil.rmtree(worktree, ignore_errors=True)
            return
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=self.config.repo_root, text=True, capture_output=True, check=False)

    def _write_opencode_config(self, worktree: Path) -> None:
        config_dir = worktree / ".opencode"
        config_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "$schema": "https://opencode.ai/config.json",
            "model": self.config.model,
            "provider": ollama_provider_config(),
            "agent": {
                self.config.suite.agent_name: {
                    "description": f"Runs isolated {self.config.suite.library_name} library-usage benchmark tasks.",
                    "mode": "primary",
                    "model": self.config.model,
                    "permission": benchmark_permissions(),
                    "prompt": self.config.suite.agent_prompt or default_agent_prompt(self.config.suite.library_name),
                },
                self.config.suite.judge_agent_name: {
                    "description": f"Judges {self.config.suite.library_name} benchmark patches with a fixed rubric.",
                    "mode": "primary",
                    "model": self.config.effective_judge_model,
                    "permission": {"read": "allow", "glob": "allow", "grep": "allow", "list": "allow", "edit": "deny", "bash": "deny", "external_directory": "deny"},
                    "prompt": self.config.suite.judge_prompt or "You are an impartial evaluator. Return only the requested structured JSON judgement.",
                },
            },
        }
        (config_dir / "opencode.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _install_user_skill_if_configured(self, worktree: Path, run_dir: Path) -> None:
        if self.config.suite.user_skill_path:
            source = self.config.suite.user_skill_path
            target = worktree / ".opencode" / "skills" / source.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        if self.config.suite.user_skill_install_command:
            self._run_shell(self.config.suite.user_skill_install_command, worktree, run_dir / "user-skill-install.stdout", run_dir / "user-skill-install.stderr")

    def _run_setup(self, benchmark: BenchmarkTask, worktree: Path, run_dir: Path) -> None:
        for index, command in enumerate(benchmark.setup_commands, 1):
            self._run_shell(command, worktree, run_dir / f"setup-{index}.stdout", run_dir / f"setup-{index}.stderr")

    def _run_validation(self, benchmark: BenchmarkTask, worktree: Path, run_dir: Path) -> ValidationResult:
        if not benchmark.validation_commands:
            return self._apply_allowed_path_check(benchmark, worktree, run_dir, ValidationResult(status="not_run"))
        command = " && ".join(benchmark.validation_commands)
        stdout = run_dir / "validation.stdout"
        stderr = run_dir / "validation.stderr"
        started = time.monotonic()
        proc = self._run_shell(command, worktree, stdout, stderr, check=False)
        result = ValidationResult(status="passed" if proc.returncode == 0 else "failed", command=command, returncode=proc.returncode, stdout_path=str(stdout), stderr_path=str(stderr), duration_seconds=time.monotonic() - started)
        return self._apply_allowed_path_check(benchmark, worktree, run_dir, result)

    def _apply_allowed_path_check(self, benchmark: BenchmarkTask, worktree: Path, run_dir: Path, result: ValidationResult) -> ValidationResult:
        if self.config.dry_run or not benchmark.allowed_paths:
            return result
        changed = [path for path in changed_paths(worktree) if not path.startswith(".opencode/")]
        violations = [path for path in changed if not any(fnmatch.fnmatch(path, pattern) for pattern in benchmark.allowed_paths)]
        if not violations:
            return result
        result.status = "failed"
        stderr_path = Path(result.stderr_path) if result.stderr_path else run_dir / "validation.stderr"
        with stderr_path.open("a", encoding="utf-8") as stderr_file:
            stderr_file.write("\nChanged files outside allowed_paths:\n" + "\n".join(f"- {path}" for path in violations) + "\n")
        result.stderr_path = str(stderr_path)
        return result

    def _run_judge_if_enabled(self, benchmark: BenchmarkTask, worktree: Path, run_dir: Path, patch_text: str, validation: ValidationResult) -> JudgeResult:
        if not self.config.judge_enabled:
            return JudgeResult(enabled=False, status="disabled")
        return run_judge(benchmark, worktree, self.config.effective_judge_model, self.config.suite.judge_agent_name, self.config.auto, patch_text, validation, run_dir / "judge.json", self.config.suite.library_name)

    def _write_patch(self, worktree: Path, patch_path: Path) -> str:
        if self.config.dry_run:
            patch_path.write_text("", encoding="utf-8")
            return ""
        proc = subprocess.run(["git", "diff", "--binary"], cwd=worktree, text=True, capture_output=True, check=False)
        patch_path.write_text(proc.stdout, encoding="utf-8")
        return proc.stdout

    def _run_shell(self, command: str, cwd: Path, stdout_path: Path, stderr_path: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(command, cwd=cwd, shell=True, text=True, capture_output=True, check=False)
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command, proc.stdout, proc.stderr)
        return proc


def load_suite_config(path: Path) -> SuiteConfig:
    data = load_simple_yaml(path)
    root = path.parent
    return SuiteConfig(
        suite_name=str(data.get("suite_name", "Python Library Agent Benchmarks")),
        library_name=str(data.get("library_name", "Python library")),
        suite_root=root,
        benchmark_dir=resolve_config_path(root, str(data.get("benchmark_dir", "benchmarks"))),
        output_dir=resolve_config_path(root, str(data.get("output_dir", "agent_benchmark_results"))),
        solution_dir=str(data.get("solution_dir", "agent_benchmark_solutions")),
        db_path=resolve_optional_path(root, data.get("db_path")),
        trace_rules_path=resolve_optional_path(root, data.get("trace_rules_path")),
        user_skill_path=resolve_optional_path(root, data.get("user_skill_path")),
        user_skill_install_command=str(data.get("user_skill_install_command", "")),
        agent_name=str(data.get("agent_name", "pylib-benchmark-agent")),
        judge_agent_name=str(data.get("judge_agent_name", "pylib-benchmark-judge")),
        agent_prompt=str(data.get("agent_prompt", "")),
        judge_prompt=str(data.get("judge_prompt", "")),
    )


def resolve_config_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_optional_path(root: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_config_path(root, str(value))


def dataclass_replace(suite: SuiteConfig, **changes: Any) -> SuiteConfig:
    data = asdict(suite)
    data.update(changes)
    return SuiteConfig(**data)


def load_benchmarks(directory: Path, selected: list[str] | None = None) -> list[BenchmarkTask]:
    wanted = set(selected or [])
    benchmarks = [BenchmarkTask.from_file(path) for path in sorted(directory.glob("*.yaml"))]
    if wanted:
        benchmarks = [benchmark for benchmark in benchmarks if benchmark.benchmark_id in wanted]
        missing = wanted - {benchmark.benchmark_id for benchmark in benchmarks}
        if missing:
            raise ValueError(f"unknown benchmark id(s): {', '.join(sorted(missing))}")
    if not benchmarks:
        raise ValueError(f"no benchmark YAML files found in {directory}")
    return benchmarks


def load_simple_yaml(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw = lines[index]
        if not raw.strip() or raw.lstrip().startswith("#"):
            index += 1
            continue
        if raw.startswith(" "):
            raise ValueError(f"unexpected indentation in {path}: {raw!r}")
        if ":" not in raw:
            raise ValueError(f"expected key/value in {path}: {raw!r}")
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "|":
            block: list[str] = []
            index += 1
            while index < len(lines):
                block_line = lines[index]
                if block_line and not block_line.startswith(" "):
                    break
                block.append(block_line[2:] if block_line.startswith("  ") else "")
                index += 1
            result[key] = "\n".join(block).rstrip() + "\n"
            continue
        if value == "[]":
            result[key] = []
            index += 1
            continue
        if value == "":
            items: list[str] = []
            index += 1
            while index < len(lines):
                item_line = lines[index]
                if item_line.startswith("  - "):
                    items.append(item_line[4:].strip())
                    index += 1
                    continue
                if not item_line.strip():
                    index += 1
                    continue
                break
            result[key] = items
            continue
        result[key] = parse_scalar(value)
        index += 1
    return result


def parse_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value.strip('"\'')


def run_opencode_task(benchmark: BenchmarkTask, worktree: Path, model: str, agent_name: str, auto: bool, timeout_seconds: int, events_path: Path) -> OpencodeRunResult:
    cmd = ["opencode", "run", "--format", "json", "--model", model, "--agent", agent_name, "--dir", str(worktree)]
    if auto:
        cmd.append("--auto")
    cmd.append(benchmark.prompt)
    started = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=worktree, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    events_path.write_text(stdout or "", encoding="utf-8")
    events_path.with_suffix(".stderr").write_text(stderr or "", encoding="utf-8")
    parsed = parse_opencode_events(stdout or "")
    parsed.returncode = 124 if timed_out else int(proc.returncode or 0)
    parsed.wall_seconds = time.monotonic() - started
    parsed.stdout = stdout or ""
    parsed.stderr = stderr or ""
    parsed.timed_out = timed_out
    return parsed


def export_session(session_id: str, worktree: Path, output_path: Path) -> bool:
    proc = subprocess.run(["opencode", "export", session_id], cwd=worktree, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        output_path.write_text(proc.stderr, encoding="utf-8")
        return False
    output_path.write_text(proc.stdout, encoding="utf-8")
    return True


def parse_opencode_events(output: str) -> OpencodeRunResult:
    events: list[dict[str, Any]] = []
    for line in output.splitlines():
        try:
            value = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    result = OpencodeRunResult(returncode=0, wall_seconds=0.0, events=events)
    for event in events:
        collect_session_id(event, result)
        collect_usage(event, result)
        collect_tool_call(event, result)
    if result.tokens_total is None and result.tokens_input is not None and result.tokens_output is not None:
        result.tokens_total = result.tokens_input + result.tokens_output
    return result


def collect_session_id(event: dict[str, Any], result: OpencodeRunResult) -> None:
    for key in ("sessionID", "session_id", "sessionId", "id"):
        value = event.get(key)
        if isinstance(value, str) and ("session" in key.lower() or event.get("type") == "session"):
            result.session_id = value
    nested = event.get("session")
    if isinstance(nested, dict):
        value = nested.get("id") or nested.get("sessionID") or nested.get("session_id")
        if isinstance(value, str):
            result.session_id = value


def collect_usage(event: dict[str, Any], result: OpencodeRunResult) -> None:
    usage = event.get("usage")
    if not isinstance(usage, dict):
        message = event.get("message")
        usage = message.get("usage") if isinstance(message, dict) else None
    if not isinstance(usage, dict):
        part = event.get("part")
        usage = part.get("tokens") if isinstance(part, dict) else None
    if not isinstance(usage, dict):
        return
    input_tokens = first_int(usage, "input", "input_tokens", "prompt_tokens", "prompt")
    output_tokens = first_int(usage, "output", "output_tokens", "completion_tokens", "completion")
    total_tokens = first_int(usage, "total", "total_tokens", "tokens")
    cost = first_float(usage, "cost", "cost_usd")
    if input_tokens is not None:
        result.tokens_input = (result.tokens_input or 0) + input_tokens
    if output_tokens is not None:
        result.tokens_output = (result.tokens_output or 0) + output_tokens
    if total_tokens is not None:
        result.tokens_total = (result.tokens_total or 0) + total_tokens
    if cost is not None:
        result.cost_usd = (result.cost_usd or 0.0) + cost


def collect_tool_call(event: dict[str, Any], result: OpencodeRunResult) -> None:
    tool = None
    status = str(event.get("status", "")) or "unknown"
    if isinstance(event.get("tool"), str):
        tool = event["tool"]
    elif isinstance(event.get("tool"), dict):
        tool = event["tool"].get("name") or event["tool"].get("id")
    elif isinstance(event.get("part"), dict):
        part = event["part"]
        if part.get("type") == "tool":
            tool = part.get("tool") or part.get("name")
            state = part.get("state")
            if isinstance(state, dict):
                status = str(state.get("status", status))
    event_type = str(event.get("type", ""))
    if not tool and "tool" not in event_type.lower():
        return
    tool = str(tool or event_type or "unknown")
    if looks_like_tool_start(event):
        result.tool_call_counts[tool] = result.tool_call_counts.get(tool, 0) + 1
        result.tool_call_trace.append({"tool": tool, "status": status, "event_type": event_type})


def looks_like_tool_start(event: dict[str, Any]) -> bool:
    status = str(event.get("status", "")).lower()
    event_type = str(event.get("type", "")).lower()
    phase = str(event.get("phase", "")).lower()
    if any(marker in event_type for marker in ("tool.execute", "tool_call", "tool.call", "tool_use")):
        return not any(done in event_type for done in ("after", "result", "complete", "finish"))
    return status in {"started", "running", "pending"} or phase in {"start", "before"}


def run_judge(benchmark: BenchmarkTask, worktree: Path, model: str, judge_agent_name: str, auto: bool, patch_text: str, validation: ValidationResult, output_path: Path, library_name: str) -> JudgeResult:
    prompt = build_judge_prompt(benchmark, patch_text, validation, library_name)
    cmd = ["opencode", "run", "--format", "json", "--model", model, "--agent", judge_agent_name, "--dir", str(worktree)]
    if auto:
        cmd.append("--auto")
    cmd.append(prompt)
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=worktree, text=True, capture_output=True, timeout=600, check=False)
    duration = time.monotonic() - started
    raw = proc.stdout + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
    output_path.write_text(raw, encoding="utf-8")
    if proc.returncode != 0:
        return JudgeResult(enabled=True, status="error", passed=False, reasoning=f"judge command failed with return code {proc.returncode} after {duration:.2f}s", raw_output_path=str(output_path))
    parsed = extract_judge_json(raw)
    score = parsed.get("score") if isinstance(parsed.get("score"), int) else None
    passed = parsed.get("passed")
    if not isinstance(passed, bool):
        passed = score is not None and score >= benchmark.minimum_judge_score
    return JudgeResult(enabled=True, status="passed" if passed else "failed", score=score, max_score=int(parsed.get("max_score", 5)) if parsed else 5, passed=passed, reasoning=str(parsed.get("reasoning", "")).strip() if parsed else raw[-2000:], raw_output_path=str(output_path))


def build_judge_prompt(benchmark: BenchmarkTask, patch_text: str, validation: ValidationResult, library_name: str) -> str:
    return f"""
You are judging the result of a {library_name} agentic coding benchmark.

Benchmark ID: {benchmark.benchmark_id}
Benchmark name: {benchmark.name}

Original coding task:
{benchmark.prompt}

Rubric:
{benchmark.judge_rubric}

Deterministic validation status: {validation.status}
Validation command: {validation.command}
Validation return code: {validation.returncode}

Patch diff:
```diff
{patch_text[:60000]}
```

Return a single JSON object with this exact shape:
{{"score": 1, "max_score": 5, "passed": false, "reasoning": "brief explanation"}}
Use score >= {benchmark.minimum_judge_score} for passed unless deterministic validation reveals a serious correctness issue.
""".strip()


def dry_run_judge(output_path: Path) -> JudgeResult:
    payload = {"score": 5, "max_score": 5, "passed": True, "reasoning": "Dry-run judge result. No model was called."}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return JudgeResult(enabled=True, status="passed", score=5, max_score=5, passed=True, reasoning=payload["reasoning"], raw_output_path=str(output_path))


def extract_judge_json(raw: str) -> dict[str, object]:
    candidates: list[str] = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            part = event.get("part")
            text = part.get("text") if isinstance(part, dict) else event.get("text")
            if isinstance(text, str):
                candidates.extend(json_candidates_from_text(text))
    candidates.extend(json_candidates_from_text(raw))
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "score" in value:
            return value
    return {}


def json_candidates_from_text(text: str) -> list[str]:
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            _, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        candidates.append(text[match.start() : match.start() + end])
    return candidates


SELF_CORRECTION_PATTERNS = (r"\bfix(ed|ing)?\b", r"\btry again\b", r"\bmistake\b", r"\berror\b", r"\bfailed\b", r"\btraceback\b", r"\bincorrect\b", r"\bnot supported\b")


def load_trace_rules(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"trace rules must be a JSON list: {path}")
    return [rule for rule in data if isinstance(rule, dict)]


def write_trace_bundle(run_dir: Path, result: BenchmarkResult) -> Path:
    trace_path = run_dir / "trace.json"
    payload = {
        "result": result.to_dict(),
        "events": read_jsonl(Path(result.events_path)) if result.events_path else [],
        "session_export": read_json(Path(result.session_export_path)) if result.session_export_path else None,
        "patch": read_text(Path(result.patch_path)) if result.patch_path else "",
        "validation_stdout": read_text(Path(result.validation.stdout_path)) if result.validation.stdout_path else "",
        "validation_stderr": read_text(Path(result.validation.stderr_path)) if result.validation.stderr_path else "",
        "judge_raw": read_text(Path(result.judge.raw_output_path)) if result.judge.raw_output_path else "",
    }
    trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return trace_path


def analyze_trace(run_dir: Path, result: BenchmarkResult, rules: list[dict[str, str]]) -> TraceAnalysis:
    session_export = read_json(Path(result.session_export_path)) if result.session_export_path else None
    assistant_corpus = "\n".join([assistant_text_from_events(read_jsonl(Path(result.events_path))) if result.events_path else "", assistant_text_from_session(session_export)])
    patch_text = read_text(Path(result.patch_path)) if result.patch_path else ""
    solution_corpus = solution_patch(patch_text)
    validation_corpus = "\n".join([read_text(Path(result.validation.stdout_path)) if result.validation.stdout_path else "", read_text(Path(result.validation.stderr_path)) if result.validation.stderr_path else ""])
    all_corpus = "\n".join([assistant_corpus, patch_text, validation_corpus, result.error])
    analysis = TraceAnalysis(status="ok")
    analysis.self_correction_count = sum(len(re.findall(pattern, all_corpus, flags=re.IGNORECASE)) for pattern in SELF_CORRECTION_PATTERNS)
    if result.status == "timeout":
        analysis.suspected_error_patterns.append("agent timed out before producing a valid benchmark result")
        analysis.remediation_actions.append("Reduce task ambiguity or add more direct examples for the relevant public API.")
    if "Traceback" in all_corpus or "AssertionError" in all_corpus:
        analysis.invalid_code_signals.append("generated code failed at runtime or assertion time")
    if "Unexpected server error" in all_corpus:
        analysis.suspected_error_patterns.append("opencode/provider failed before agent work could be evaluated")
    corpora = {"assistant": assistant_corpus, "solution": solution_corpus, "validation": validation_corpus, "all": all_corpus}
    for rule in rules:
        label = str(rule.get("label", "unnamed trace rule"))
        pattern = str(rule.get("regex", ""))
        remediation = str(rule.get("remediation", ""))
        corpus_name = str(rule.get("corpus", "all"))
        if pattern and re.search(pattern, corpora.get(corpus_name, all_corpus), flags=re.IGNORECASE | re.DOTALL):
            analysis.library_misuse_signals.append(label)
            if remediation and remediation not in analysis.remediation_actions:
                analysis.remediation_actions.append(remediation)
            analysis.evidence.append({"pattern": label, "regex": pattern, "corpus": corpus_name})
    if not analysis.suspected_error_patterns and not analysis.library_misuse_signals and not analysis.invalid_code_signals:
        analysis.summary = "No deterministic trace issues were detected. Inspect trace.json for qualitative review."
    else:
        parts = []
        if analysis.suspected_error_patterns:
            parts.append("; ".join(analysis.suspected_error_patterns))
        if analysis.library_misuse_signals:
            parts.append("library misuse: " + "; ".join(analysis.library_misuse_signals))
        if analysis.invalid_code_signals:
            parts.append("invalid code: " + "; ".join(analysis.invalid_code_signals))
        analysis.summary = " | ".join(parts)
    analysis.documentation_gaps = sorted({action.split(" should ", 1)[0] for action in analysis.remediation_actions if " should " in action})
    (run_dir / "trace-analysis.json").write_text(json.dumps(asdict(analysis), indent=2, sort_keys=True), encoding="utf-8")
    return analysis


def solution_patch(patch_text: str) -> str:
    hunks = re.split(r"(?m)^(?=diff --git )", patch_text)
    kept = [hunk for hunk in hunks if not re.match(r"diff --git a/\.opencode/", hunk)]
    return "".join(kept)


def write_results(output_dir: Path, results: Iterable[BenchmarkResult]) -> None:
    rows = [result.to_dict() for result in results]
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")
    fields = ["run_id", "suite_name", "library_name", "benchmark_id", "benchmark_name", "repetition", "status", "model", "judge_model", "benchmark_skill_version", "benchmark_task_hash", "repo_commit_hash", "repo_dirty", "opencode_version", "parallelism_mode", "auto_approve", "wall_seconds", "tokens_input", "tokens_output", "tokens_total", "model_context_configured", "model_context_available", "cost_usd", "tool_call_count", "session_id", "run_dir", "patch_path", "session_export_path", "events_path", "trace_path", "error"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(results: list[BenchmarkResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total_runs": len(results), "passed_runs": sum(1 for r in results if r.status == "passed"), "failed_runs": sum(1 for r in results if r.status != "passed"), "by_benchmark": {}, "remediation_actions": []}
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.benchmark_id, []).append(result)
    for benchmark_id, group in grouped.items():
        wall = [r.wall_seconds for r in group if isinstance(r.wall_seconds, (int, float))]
        tokens = [r.tokens_total for r in group if isinstance(r.tokens_total, int)]
        summary["by_benchmark"][benchmark_id] = {"runs": len(group), "passed": sum(1 for r in group if r.status == "passed"), "failed": sum(1 for r in group if r.status != "passed"), "wall_seconds_mean": statistics.mean(wall) if wall else None, "tokens_total_mean": statistics.mean(tokens) if tokens else None}
    summary["remediation_actions"] = sorted({action for result in results for action in result.analysis.remediation_actions})
    return summary


def render_html_report(output_dir: Path, results: list[BenchmarkResult], summary: dict[str, Any]) -> Path:
    rows = "\n".join(
        f"<tr><td><code>{esc(r.run_id)}</code></td><td>{esc(r.benchmark_name)}</td><td class='{esc(r.status)}'>{esc(r.status)}</td><td><code>{esc(r.model)}</code></td><td>{r.wall_seconds or ''}</td><td>{r.tokens_total or ''}</td><td>{esc(r.analysis.summary)}</td><td><code>{esc(r.trace_path)}</code></td></tr>"
        for r in results
    )
    remediation = "".join(f"<li>{esc(action)}</li>" for action in summary.get("remediation_actions", [])) or "<li>No remediation actions detected.</li>"
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Benchmark Report</title><style>:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{margin:2rem;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #9995;padding:.45rem;text-align:left;vertical-align:top}}th{{background:#7772}}.passed{{color:#087f23;font-weight:700}}.failed,.error,.timeout,.judge_failed{{color:#b00020;font-weight:700}}code{{font-family:ui-monospace,monospace}}</style></head>
<body><h1>Agent Benchmark Report</h1><p>Total runs: {summary['total_runs']}. Passed: {summary['passed_runs']}. Failed: {summary['failed_runs']}.</p><p>SQLite index: <code>{esc(str(summary.get('sqlite_db_path', '')))}</code></p>
<h2>Runs</h2><table><thead><tr><th>Run</th><th>Benchmark</th><th>Status</th><th>Model</th><th>Seconds</th><th>Tokens</th><th>Trace Analysis</th><th>Trace</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Remediation Actions</h2><ul>{remediation}</ul></body></html>"""
    output_path = output_dir / "index.html"
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def write_report_data(output_dir: Path, results: list[BenchmarkResult], summary: dict[str, Any]) -> Path:
    """Write frontend-neutral report data for custom dashboards.

    The bundled HTML is intentionally simple. Higher-quality frontends can ignore
    it and consume this stable JSON payload instead.
    """
    output_path = output_dir / "report-data.json"
    payload = {"summary": summary, "results": [result.to_dict() for result in results]}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


class BenchmarkDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def __enter__(self) -> "BenchmarkDB":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            run_id TEXT PRIMARY KEY, suite_name TEXT, library_name TEXT, benchmark_id TEXT NOT NULL,
            benchmark_name TEXT NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL, model TEXT NOT NULL,
            judge_model TEXT NOT NULL, benchmark_skill_version TEXT NOT NULL, benchmark_task_hash TEXT NOT NULL,
            repo_commit_hash TEXT NOT NULL, repo_dirty INTEGER NOT NULL, opencode_version TEXT NOT NULL,
            parallelism_mode TEXT NOT NULL, auto_approve INTEGER NOT NULL, run_started_at TEXT, run_finished_at TEXT,
            ingested_at TEXT NOT NULL, wall_seconds REAL, tokens_input INTEGER, tokens_output INTEGER,
            tokens_total INTEGER, cost_usd REAL, tool_call_count INTEGER NOT NULL, tool_call_counts_json TEXT NOT NULL,
            validation_status TEXT, validation_command TEXT, validation_returncode INTEGER, judge_status TEXT,
            judge_score INTEGER, judge_passed INTEGER, analysis_summary TEXT, analysis_json TEXT NOT NULL,
            result_json TEXT NOT NULL, error TEXT
        );
        CREATE TABLE IF NOT EXISTS run_artifacts (run_id TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY (run_id, kind));
        CREATE TABLE IF NOT EXISTS remediation_actions (run_id TEXT NOT NULL, action TEXT NOT NULL, documentation_gap TEXT, PRIMARY KEY (run_id, action));
        """)
        self.conn.commit()

    def upsert_results(self, results: Iterable[BenchmarkResult]) -> None:
        for result in results:
            self.upsert_result(result, commit=False)
        self.conn.commit()

    def upsert_result(self, result: BenchmarkResult, commit: bool = True) -> None:
        data = result.to_dict()
        self.conn.execute(
            """INSERT OR REPLACE INTO benchmark_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result.run_id, result.suite_name, result.library_name, result.benchmark_id, result.benchmark_name, result.repetition, result.status, result.model, result.judge_model, result.benchmark_skill_version, result.benchmark_task_hash, result.repo_commit_hash, int(result.repo_dirty), result.opencode_version, result.parallelism_mode, int(result.auto_approve), result.run_started_at, result.run_finished_at, datetime.now(timezone.utc).isoformat(), result.wall_seconds, result.tokens_input, result.tokens_output, result.tokens_total, result.cost_usd, result.tool_call_count, json.dumps(result.tool_call_counts, sort_keys=True), result.validation.status, result.validation.command, result.validation.returncode, result.judge.status, result.judge.score, bool_or_none(result.judge.passed), result.analysis.summary, json.dumps(asdict(result.analysis), sort_keys=True), json.dumps(data, sort_keys=True), result.error),
        )
        self.conn.execute("DELETE FROM run_artifacts WHERE run_id = ?", (result.run_id,))
        artifacts = {
            "run_dir": result.run_dir,
            "worktree": result.worktree_path,
            "patch": result.patch_path,
            "events": result.events_path,
            "session_export": result.session_export_path,
            "trace": result.trace_path,
            "validation_stdout": result.validation.stdout_path,
            "validation_stderr": result.validation.stderr_path,
            "judge_raw": result.judge.raw_output_path,
        }
        for kind, path in artifacts.items():
            if path:
                self.conn.execute("INSERT OR REPLACE INTO run_artifacts(run_id, kind, path) VALUES (?, ?, ?)", (result.run_id, kind, path))
        self.conn.execute("DELETE FROM remediation_actions WHERE run_id = ?", (result.run_id,))
        for action in result.analysis.remediation_actions:
            gap = next((gap for gap in result.analysis.documentation_gaps if action.startswith(gap)), "")
            self.conn.execute("INSERT OR REPLACE INTO remediation_actions(run_id, action, documentation_gap) VALUES (?, ?, ?)", (result.run_id, action, gap))
        if commit:
            self.conn.commit()

    def recent_runs(self, limit: int = 20, status: str | None = None) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        where = "WHERE status = ?" if status else ""
        params: tuple[object, ...] = (status, limit) if status else (limit,)
        return list(
            self.conn.execute(
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
        )

    def remediation_summary(self) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        return list(
            self.conn.execute(
                """
                SELECT action, COUNT(*) AS occurrences, GROUP_CONCAT(DISTINCT benchmark_id) AS benchmarks
                FROM remediation_actions
                JOIN benchmark_runs USING (run_id)
                GROUP BY action
                ORDER BY occurrences DESC, action ASC
                """
            )
        )

    def artifacts_for_run(self, run_id: str) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        return list(self.conn.execute("SELECT kind, path FROM run_artifacts WHERE run_id = ? ORDER BY kind", (run_id,)))


def combined_status(opencode_returncode: int, validation: ValidationResult, judge: JudgeResult) -> str:
    if opencode_returncode == 124:
        return "timeout"
    if opencode_returncode != 0:
        return "error"
    if validation.status not in {"passed", "not_run"}:
        return "failed"
    if judge.enabled and judge.passed is False:
        return "judge_failed"
    return "passed"


def changed_paths(worktree: Path) -> list[str]:
    proc = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=worktree, text=True, capture_output=True, check=False)
    paths: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def benchmark_permissions() -> dict[str, object]:
    return {"read": "allow", "glob": "allow", "grep": "allow", "list": "allow", "edit": "allow", "external_directory": "deny", "bash": {"*": "ask", "uv run pytest *": "allow", "uv run python *": "allow", "python *": "allow", "python -m pytest *": "allow", "git diff*": "allow", "git status*": "allow", "git rev-parse*": "allow", "ls *": "allow", "rm *": "deny", "git reset *": "deny", "git checkout *": "deny", "git clean *": "deny", "git worktree *": "deny"}}


def ollama_provider_config() -> dict[str, object]:
    models: dict[str, object] = {}
    for model_id, name, family, context, output in (("gemma4:latest", "Gemma 4", "gemma4", 8192, 4096), ("gemma4:26b", "Gemma 4 26B", "gemma4", 8192, 4096), ("gemma4:31b", "Gemma 4 31B", "gemma4", 8192, 4096), ("qwen3.8:latest", "Qwen 3 27B", "qwen3", 262144, 32768)):
        models[model_id] = {"id": model_id, "name": name, "family": family, "status": "active", "reasoning": True, "tool_call": True, "temperature": True, "cost": {"input": 0, "output": 0}, "limit": {"context": context, "output": output}}
    return {"ollama": {"models": models}}


def configured_context_limit(model: str) -> int | None:
    provider, _, model_id = model.partition("/")
    if provider != "ollama" or not model_id:
        return None
    model_config = ollama_provider_config()["ollama"]["models"].get(model_id)  # type: ignore[index]
    if not isinstance(model_config, dict):
        return None
    limit = model_config.get("limit")
    return limit.get("context") if isinstance(limit, dict) and isinstance(limit.get("context"), int) else None


def available_context_limit(model: str) -> int | None:
    provider, _, model_id = model.partition("/")
    if provider != "ollama" or not model_id:
        return None
    proc = subprocess.run(["ollama", "show", model_id], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        fields = line.strip().split()
        if len(fields) >= 3 and fields[0] == "context" and fields[1] == "length":
            try:
                return int(fields[2])
            except ValueError:
                return None
    return None


def default_agent_prompt(library_name: str) -> str:
    return f"You are a careful benchmarked coding agent using {library_name} as an application developer. Write the required solution file first, validate with the requested commands, iterate on failures, prefer user-facing docs over source internals, and keep changes minimal."


def command_text(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    return (proc.stdout or proc.stderr).strip()


def first_int(mapping: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def bool_or_none(value: bool | None) -> int | None:
    return None if value is None else int(value)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line in read_text(path).splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


def assistant_text_from_events(events: list[Any]) -> str:
    chunks: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", ""))
        part = event.get("part")
        if event_type == "text" and isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("synthetic"):
            chunks.append(part["text"])
        if "tool" in event_type.lower() or event_type == "error":
            chunks.append(json.dumps(event, sort_keys=True))
    return "\n".join(chunks)


def assistant_text_from_session(session_export: Any) -> str:
    if not isinstance(session_export, dict):
        return ""
    chunks: list[str] = []
    for message in session_export.get("messages", []):
        if not isinstance(message, dict):
            continue
        info = message.get("info")
        if not isinstance(info, dict) or info.get("role") == "user":
            continue
        for part in message.get("parts", []):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(chunks)


def esc(value: str) -> str:
    return html.escape(value, quote=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
