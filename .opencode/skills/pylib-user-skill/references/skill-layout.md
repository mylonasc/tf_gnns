# Skill Layout And Content Rules

## SKILL.md Is An Index And Router, Not A Manual

Keep it short (~100 lines). Required sections:

- Valid skill frontmatter: `name` plus a one-sentence `description` covering *what* and *when*.
- "Fast Retrieval" with the exact CLI commands (agents copy-paste these verbatim).
- "Available topics": one line per topic stating **when to read it**.
- "Agent Workflow": *write the solution file first, then validate and iterate; prefer the retrieval CLI over reading library source*. Evidence: a "inspect the repository before editing" system prompt measurably caused over-exploration loops where agents read source for dozens of steps without writing code.
- "Import Rules" with exact submodule import lines, including what is deliberately *not* exported from the package root.
- "Core Usage Rules" as bullets: resource cleanup, ID conventions, index discipline.
- A "Boundaries" section stating what is **unsupported** — as important as what works, otherwise agents hallucinate it.
- One "Minimal Runnable Pattern" (full lifecycle in ~25 lines).

## Topic Granularity

Split `references/<topic>.md` by agent task, not by module (e.g. quickstart, backends/persistence, indexing, ingestion, query, sampling). Each file keeps the same shape: a one-line "read this when…" header, `## Rules` (terse, assertion-style bullets), and `## <Name> Example` blocks. If a capability matters and has no topic whose *when-to-read* line names it, agents will not discover it — we learned this with database open/inspect APIs.

## The Example-Coverage Rule (Highest ROI)

**Every public API an agent may be asked to use must have a copy-pasteable example.** Evidence: one task burned ~200k tokens on source spelunking purely because the docs example covered a sibling API but not the one under test. After adding the exact example, that task went 257k → 56k tokens (−78%).

## Document Shapes, Not Just Names

For every API, document return shapes exactly: result key names, bytes-vs-string IDs, tuple-vs-list. A single list-vs-tuple mismatch once cost an entire benchmark run. State ordering guarantees explicitly ("order is not guaranteed; compare sets") — never leave it implied by example style. Put decoding helpers next to the API that needs them, not in a distant section.

## Document The Gotchas

Maintain a "debug-loop" section per topic: each entry prevents a ~200k-token debug loop. Typical entries: silent dtype coercions in ingestion paths, formats without nested columns, ID-encoding conventions, classes that are not context managers (show `try/finally`), enumeration APIs that do not exist (show the supported listing pattern instead). **Operating rule:** whenever an agent trace shows more than ~3 tool calls of debugging, that session becomes a docs entry.
