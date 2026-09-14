---
name: pylib-agent-benchmarking
description: Use when benchmarking opencode agents on Python-library usage tasks, especially to measure whether agent-facing docs or skills improve correctness, tool use, and token efficiency.
---

# Python Library Agent Benchmarking

Source: copied from `git@github.com:mylonasc/opencode-agent-pylib-doc-metaskill.git` at commit `88db0be1c385bd6de65ce9378a11e8a9846446c6`.

Use this skill to create and run repeatable agentic coding benchmarks for Python libraries. The benchmark definitions are part of the skill workflow: if the current library is not yet customized with a suite config, task YAMLs, and trace rules, create those benchmark configurations first from the current session's repository discovery. Only after concrete benchmark definitions exist should benchmark runner agents execute them. The benchmarked agent should act as a library user: create a small standalone application/example file, validate it, and avoid modifying the library internals. The runner creates isolated git worktrees, invokes `opencode run`, records token/tool metadata when available, validates the patch, optionally runs an LLM judge, analyzes traces with configurable rules, and writes JSONL/CSV/HTML reports.

During suite bootstrapping, keep the user informed interactively. Tell the user that this skill is being customized for the current library so the benchmarks capture meaningful library-specific workflows rather than generic coding tasks. After discovering candidate workflows and before treating the suite as ready to run, present how the benchmark material was discovered and request the user's approval for each proposed benchmark.

## Quick Start

If the repository already has library-specific benchmark definitions, run a dry run first:

```bash
python .opencode/skills/pylib-agent-benchmarking/scripts/run_agent_benchmarks.py \
  --suite-config .opencode/skills/pylib-agent-benchmarking/suite.example.yaml \
  --dry-run \
  --html
```

Run real benchmarks with a local or remote opencode model:

```bash
python .opencode/skills/pylib-agent-benchmarking/scripts/run_agent_benchmarks.py \
  --suite-config path/to/suite.yaml \
  --model anthropic/claude-sonnet-4-6 \
  --parallelism parallel \
  --max-workers 4 \
  --repetitions 3 \
  --html
```

## Suite Config

The suite config is intentionally small and portable:

```yaml
suite_name: MyLib API Usage Benchmarks
library_name: MyLib
benchmark_dir: benchmarks
output_dir: agent_benchmark_results
solution_dir: agent_benchmark_solutions
user_skill_path: .opencode/skills/mylib-user-guide
trace_rules_path: trace-rules.json
agent_prompt: |
  You are a careful benchmarked coding agent. Write the required solution file first,
  then validate with the requested commands and iterate on failures. Prefer the
  library's user-facing docs and opencode skill over reading source internals.
judge_prompt: |
  You are an impartial evaluator. Return only the requested structured JSON judgement.
```

Paths are resolved relative to the suite config file unless absolute.
For library-specific suites, place the suite config and related task files inside this skill directory under `.opencode/skills/pylib-agent-benchmarking/agent_benchmarks/<library>/`. Because paths are resolved relative to the suite config, point outputs back to the repository workspace when needed, for example `output_dir: ../../../../agent_benchmark_results`.

## Bootstrap A Suite

When no benchmark suite exists yet, creating it is the first step of using this skill. First discover the library's real user workflows, then scaffold benchmark definitions around those workflows. Do this inside this benchmarking skill rather than creating a separate bootstrap skill or asking smaller benchmark-running agents to invent tasks on the fly.

The output of this bootstrap step should be a library-specific benchmark configuration that runner agents can consume directly:

- A suite YAML under this skill's library-specific benchmark directory, usually `.opencode/skills/pylib-agent-benchmarking/agent_benchmarks/<library>/<library>.suite.yaml`.
- One task YAML per documented user workflow, under the suite's `benchmark_dir`.
- A trace-rules JSON file for known deterministic failure signals.
- A configured `user_skill_path` or `user_skill_install_command` so isolated benchmark worktrees have the library's user-facing docs/skill available.

Do not run a real benchmark before this configuration exists. If the user asks to benchmark a library and no suite is present, bootstrap the suite in the current session, verify it with `--dry-run --html`, and then report the suite path and runnable command.

During bootstrapping:

- Inform the user before discovery that the skill will customize benchmark definitions for the current library's documented workflows, public APIs, and common failure modes.
- Store library-specific benchmark definitions inside this skill directory under `agent_benchmarks/<library>/`, not at the repository root, unless the user explicitly requests a different location.
- Discover benchmark material from README, docs, examples, notebooks, tests that exercise public APIs, public exports, docstrings, packaged user skills, and issue-like troubleshooting notes.
- After discovery, summarize the sources used and the workflows found before finalizing the suite.
- Present the proposed benchmark list with one concise entry per benchmark: id, covered workflow, source evidence, difficulty, validation command, and why it is meaningful.
- Request explicit user approval per benchmark. Use an interactive question when available; otherwise ask the user to approve, reject, or revise each benchmark in a numbered list.
- Create or finalize only the benchmarks the user approves. If the user asks for revisions, update the benchmark proposal and ask again for the affected benchmark.
- Run `--dry-run --html` only after the approved benchmark definitions exist.

Read `references/bootstrap-suite.md` before creating or revising a suite. In short:

- Inventory use cases from README quickstarts, `docs/` or Sphinx pages, notebooks, examples, tests, public docstrings, package exports, tutorials, and issue-like troubleshooting notes.
- Prefer workflows that a real user would implement in a small standalone script: construct data, configure the library, call public APIs, inspect outputs, save/load, train/evaluate, or integrate with common optional dependencies.
- Create one deterministic benchmark task per documented workflow the user skill claims to teach.
- Make every task produce one file under `solution_dir`, validate with runnable commands, and restrict `allowed_paths` to the expected solution files.
- Configure the suite to install the packaged user skill with `user_skill_path` or `user_skill_install_command`, and add trace rules for known failure modes such as wrong imports, source spelunking, deprecated APIs, or common shape/type mistakes.

## Task YAML

Each file in `benchmark_dir` defines one benchmark:

```yaml
id: quickstart-usage-script
name: Quickstart Usage Script
description: Create a small script that uses the documented public API.
timeout_seconds: 300
tags:
  - quickstart
  - library-usage
allowed_paths:
  - agent_benchmark_solutions/quickstart_usage.py
setup_commands: []
validation_commands:
  - python agent_benchmark_solutions/quickstart_usage.py
prompt: |
  You are working as a user of MyLib. Create a standalone script at
  agent_benchmark_solutions/quickstart_usage.py using only documented public APIs.
judge_rubric: |
  Score from 1 to 5. Award high scores for runnable public API usage, meaningful
  assertions, and no library source edits.
minimum_judge_score: 4
```

## Metrics Collected

- Wall-clock duration.
- Input, output, total tokens, and cost when provider metadata exposes them.
- Tool call count, counts by tool, and ordered tool call trace.
- Model, judge model, opencode version, suite version, task hash, commit hash, dirty status, and runner configuration.
- Validation stdout/stderr, opencode event stream, optional session export, patch diff, trace bundle, trace analysis, and judge output.
- JSONL, CSV, `summary.json`, `report-data.json`, optional `index.html`, and a local SQLite index.

## Inspect Results

Start with the generated summary and HTML report:

```bash
python -m json.tool agent_benchmark_results/summary.json
open agent_benchmark_results/index.html
```

Browse the SQLite index with the bundled helper:

```bash
python .opencode/skills/pylib-agent-benchmarking/scripts/browse_benchmark_db.py \
  --db-path agent_benchmark_results/agent_benchmarks.sqlite recent --limit 20
python .opencode/skills/pylib-agent-benchmarking/scripts/browse_benchmark_db.py \
  --db-path agent_benchmark_results/agent_benchmarks.sqlite remediation
python .opencode/skills/pylib-agent-benchmarking/scripts/browse_benchmark_db.py \
  --db-path agent_benchmark_results/agent_benchmarks.sqlite artifacts <run-id>
```

Regenerate trace analysis after changing trace rules:

```bash
python .opencode/skills/pylib-agent-benchmarking/scripts/inspect_agent_traces.py \
  agent_benchmark_results \
  --trace-rules path/to/trace-rules.json
```

Each run stores `opencode-events.jsonl`, optional `session-export.json`, `patch.diff`, validation output, `trace.json`, `trace-analysis.json`, and `judge.json`. Use `browse_benchmark_db.py artifacts <run-id>` to locate them.

## Trace Rules

Use `trace_rules_path` to add library-specific deterministic failure signals without editing the runner:

```json
[
  {
    "label": "package-root import used for non-exported class",
    "regex": "from\\s+mylib\\s+import\\s+.*\\bClient\\b",
    "corpus": "solution",
    "remediation": "Document exact import paths in the user skill and quickstart examples."
  }
]
```

Supported `corpus` values are `solution`, `assistant`, `validation`, and `all`.

## Modular Extension Points

- Benchmark execution lives in `scripts/run_agent_benchmarks.py`.
- Read-only result inspection lives in `scripts/benchmark_results.py`; swap or wrap this module for richer analytics without changing execution.
- The SQLite browser CLI is `scripts/browse_benchmark_db.py`; replace it with a dashboard or notebook if needed.
- The trace re-analysis CLI is `scripts/inspect_agent_traces.py`; trace rules are external JSON so library-specific signals stay out of runner code.
- `report-data.json` is the stable frontend payload. The bundled `index.html` is a minimal default and can be replaced by a higher-quality frontend that consumes the same JSON.

## Workflow

- Check whether the library already has a suite config and task YAMLs under `.opencode/skills/pylib-agent-benchmarking/agent_benchmarks/<library>/`. If not, bootstrap them from README/docs/examples/tests/public exports before running benchmarks.
- Before bootstrapping, inform the user that benchmark definitions will be customized to the library's specifics and that they will be asked to approve each benchmark.
- After discovery, present the discovery summary and proposed benchmark list for approval before finalizing or running the suite.
- Start with one deterministic task that validates with `python path/to/script.py` or the repository's available Python executable.
- Keep generated solutions under one directory and put only that directory in `allowed_paths`.
- Add one benchmark for every public API workflow that the user skill claims to teach.
- Keep benchmark definitions stable, reviewable files in the repository; benchmark runner agents should read these definitions rather than generating their own tasks.
- Run at least one repetition with `--preserve-worktrees` while designing a new task.
- Use trace analysis to convert repeated failure modes into docs examples, import rules, and unsupported-boundary notes.

## Safety

The runner uses `opencode run --auto` by default, but every run happens in a detached temporary git worktree with a restricted benchmark-specific opencode config. Use `--no-auto` for interactive debugging.

## Verification

After changing a suite or this skill, run:

```bash
python .opencode/skills/pylib-agent-benchmarking/scripts/run_agent_benchmarks.py --suite-config path/to/suite.yaml --dry-run --html
```
