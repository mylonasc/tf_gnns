---
name: pylib-user-skill
description: Use when shipping a user-assistant skill with a Python library so opencode assistants learn the public API from runnable examples, paged retrieval, and an auto-generated API index.
---

# Python Library User Skill

Source: copied from `git@github.com:mylonasc/opencode-agent-pylib-doc-metaskill.git` at commit `6539a4abdae4fd39d2ffd4f79744aeecceadc170`.

Use this skill when adding a user-assistant skill to a Python library: a packaged skill directory, a zero-dependency retrieval CLI, and an auto-generated API index, verified by tests, drift checks, and agentic benchmarks.

## Design Principle

Agent runs are reasoning-dominated (~18k tokens per agent step in our measurements). Tool calls are cheap; thinking is expensive. Every decision below serves one goal: **first-try-correct code**. An agent that copies a working example costs ~50k tokens; one that debugs an undocumented gotcha costs ~450k.

## Fast Retrieval

This repo holds skill source, not a package. The canonical layout for the shipped skill is:

```text
src/<pkg>/agent_skill/
├── SKILL.md            # index and router, not a manual
├── api_index.json      # auto-generated structured API (see references/api-index.md)
└── references/
    ├── quickstart.md
    ├── backends.md     # persistence, backends, inspection
    ├── indexing.md
    ├── ingestion.md
    ├── query.md        # query language, if the library has one
    └── sampling.md
```

The retrieval CLI (`src/<pkg>/agent_docs.py`) exposes `list`, `get <topic> [--examples|--rules]`, `search <regex> [--limit --page --context --examples-only]`, and an installer that copies the skill to `.opencode/skills/<name>/`.

## References

- `skill-layout`: topic granularity, the example-coverage rule, shapes/gotchas/boundaries documentation.
- `retrieval-cli`: paged search, API cards, example-aware ranking, `importlib.resources`, project-local installer.
- `api-index`: AST generator with heuristic topic tagging, freshness guard, package data.
- `verification-loop`: tests, drift checks, example execution, benchmark-driven docs loop.
- `harness-and-release`: context-limit discipline, harness details, release workflow.
