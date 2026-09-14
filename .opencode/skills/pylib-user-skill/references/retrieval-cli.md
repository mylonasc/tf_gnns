# Retrieval CLI

Ship a zero-extra-dependency module (e.g. `python -m mypkg.agent_docs`) with:

- `list` — topics with titles and when-to-read lines.
- `get <topic> [--examples|--rules]` — full topic, only runnable examples, or prose only.
- `search <regex> [--limit --page --context --examples-only]` — see below.
- An installer (e.g. `install-opencode-skill`) copying the skill to `.opencode/skills/<name>/`, refusing to overwrite without `--force`.

## Search Must Be Bounded, Paged, And Answer-Shaped

- Default to ≤10 hits per page with a small context window and `page N/M` navigation. Unbounded output once flooded a run.
- Print **structured API cards above snippet hits**: signature, one-line summary, returns note, owning topic, and the covering example (from the API index). This turned a `sample_neighbors` search from one useless bullet line into a signature plus the complete runnable example.
- Rank hits inside runnable examples first and attach the full example block; `--examples-only` shows just those.
- Read packaged resources via `importlib.resources`, never repository-relative paths — installed users have no repo checkout.

## Project-Local Installer

Evals and users must load the skill as a *real skill*. The harness should install it into every agent worktree automatically. Evidence: traces showed agents never touching the docs at all until installation was automated; after that, skill reads appear in every passing trace.
