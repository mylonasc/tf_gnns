# Harness Discipline And Release

## Context-Limit Discipline

Model cards advertise maximum context; harness config imposes actual context. We observed 22 compactions under an 8k configured cap on a model advertising 262k, producing loop-like behavior that looked like agent confusion. Always compare **advertised vs. configured context vs. trace compactions before blaming the docs**, record all three per run, and set harness limits to the model's plausible maximum.

## Harness Details That Matter

- Benchmark agent system prompt must say *write the file first, then validate and iterate*.
- `setup_commands` must pre-install optional dependencies into the worktree (agents otherwise burn runs proving deps exist elsewhere); keep build dirs git-ignored so setup doesn't trip scope checks.
- Scope checks must ignore harness-owned paths (e.g. `.opencode/`); trace analyzers must scan solution code, not harness config — keyword signals otherwise fire on config prose (e.g. a glob `"*"` or the word "with" in JSON).
- Judge-output parsers must handle structured JSON embedded in event streams, not just raw braces.

## Release

Patch-bump version plus changelog entry, full test/docs/build/distribution-check gate, branch → PR to main with CI green → merge → tagged release targeting main, letting Trusted Publishing do the upload. Never upload by hand.
