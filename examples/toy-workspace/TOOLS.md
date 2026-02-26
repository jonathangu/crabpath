# Tool Usage Notes

This workspace treats tools as explicit collaborators. Use the minimum required set of tools to complete each task safely.

Primary tool categories:

- Version control: `git` for diff, status, commit provenance.
- Search: `rg` for fast symbol and string discovery.
- File IO: `sed`, `cat`, and `Path` helpers for reading and writing local files.
- Runtime checks: `python` scripts and `pytest` for scoped validation.
- Data helpers: JSON and JSONL readers for deterministic parsing and replay.

Execution style:

- Convert `~` and relative paths to absolute paths before persistence.
- Validate file and directory existence before reading.
- Log command intent and outputs in short lines.
- Keep outputs structured; prefer JSON where tooling expects machine-readable formats.

When extracting sessions or migration logs:

- Scan only the designated directory.
- Ignore non-user messages and scheduler/system entries.
- Preserve order where possible to maintain temporal context.

Modeling workflow:

1. Initialize context from AGENTS, SOUL, TOOLS, MEMORY, USER.
2. Generate deterministic nodes and edges.
3. Run health checks after bootstrap.
4. Capture migration summary and next actionable steps.

Quality gates:

- If a result looks suspicious, capture a short debug trace.
- Prefer explicit failures and clear error payloads.
- If required flags are missing, fail with an actionable message.

Use-case examples:

- Re-run bootstrap when workspace files change.
- Run `extract-sessions` before long query tuning.
- Compare health snapshots over time to confirm graph maturity.
