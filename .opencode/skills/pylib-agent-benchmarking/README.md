# Python Library Agent Benchmarking

Standalone opencode skill for creating and running benchmarks of agentic coding performance on Python-library usage tasks.

The benchmark runner is generic: tasks, docs-skill installation, output paths, agent prompts, and trace-analysis rules are configured by suite files rather than hard-coded for one library. If a library does not yet have those suite files, the first step is to bootstrap them from the current repository's docs, examples, tests, public exports, and packaged user skill before invoking benchmark-running agents.

Suite bootstrapping is interactive by design. The assistant should tell the user that it is customizing this skill's benchmark definitions for the current library, summarize the discovery sources and candidate workflows, and request user approval for each proposed benchmark before finalizing or running the suite.

Library-specific benchmark definitions should live inside this skill directory under `agent_benchmarks/<library>/`, with task YAMLs under that suite's `benchmark_dir`. Keep generated benchmark outputs outside the skill when appropriate by using relative `output_dir` paths in the suite config.

Result inspection is modular. `scripts/benchmark_results.py` exposes read-only SQLite helpers, `scripts/browse_benchmark_db.py` provides a small CLI, `scripts/inspect_agent_traces.py` regenerates trace analysis, and `report-data.json` is emitted for custom dashboards or richer frontends.

See `SKILL.md` for usage, `references/bootstrap-suite.md` for use-case discovery and suite bootstrapping guidance, and `suite.example.yaml` plus `benchmarks/example-task.yaml` for a minimal suite.
