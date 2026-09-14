# Verification Loop

All of the following are required before release:

1. **Unit tests**: topics/skill present, examples extractable, search flags and API cards, installer behavior (including refuse-without-`--force`), API-index freshness.
2. **Drift checker**: package-root exports, backend/serializer-style registries, method signatures, snippet AST/import validity, referenced user-docs existence, API-index freshness.
3. **Execute every packaged example** in isolation. A skill example that doesn't run is worse than none — agents copy it verbatim, including its bugs.
4. **Benchmark dry-run** (task loading, report rendering), then the full agentic suite recording per-run tokens, context metadata, and compactions.

## Benchmark-Driven Documentation Loop

This is the operating process, not a one-time step:

1. Run the agentic suite (usage tasks: write small standalone scripts against the public API, deterministically validated).
2. Inspect traces in a trace viewer that shows **configured vs. advertised context and cumulative tokens per event**.
3. Find where agents spelunk source or debug: each such loop is a missing example, rule, or API card.
4. Add it, re-run, compare tokens per task.

Reference results from applying this loop: 5/5 tasks passing with sampling −78%, ingestion −56%, persistence −52% in tokens, zero compactions.
