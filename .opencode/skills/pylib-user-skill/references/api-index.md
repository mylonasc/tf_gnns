# Auto-Generated API Index

Hand-written docs drift. Generate structure from the code:

- A generator script AST-parses the public surface (e.g. public methods of the main class) into a shipped JSON sidecar: name, full signature (including keyword-only args with defaults), docstring summary, `Returns:` note, source line, and per-method example coverage computed by scanning the packaged examples.
- **Topic tagging is heuristic**: ordered name-based rules, first match wins (e.g. `query` → query topic, `sample_*`/adjacency → sampling, `ingest_*` → ingestion, `create`/`open`/`manifest` → backends, default → quickstart). No hand curation, so the index stays in sync with the code.
- Ship the JSON as package data and read it through the retrieval CLI's search (API cards).
- Add a **freshness guard**: a docs-consistency check plus a unit test that fail when the committed index differs from a fresh build, with the regeneration command in the error message.
