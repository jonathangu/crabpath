# Acme Bot Workspace

This folder contains a compact Acme Bot operations dataset used by the quickstart guide.

Acme Bot is a fictional customer-facing operations assistant that triages incidents,
collects run context, and suggests safe recovery actions.

The workspace is intentionally small but realistic:

- `architecture.md` describes the memory and control boundaries.
- `runbook.md` lists the day-to-day operational playbook.
- `troubleshooting.md` captures recurring failure patterns.
- `api-reference.md` maps high-level APIs that the bot can call.

Read this file first, then follow the architecture to understand cross-file references.
The runbook points to guardrails in `troubleshooting.md`, and that file links back to
`api-reference.md` when operators need command-level support.

Use this workspace as input for bootstrap and then ask about release safety,
incident response, or adapter wiring.

For example, the query: "How do we handle a stale cache during deployment?" should
surface notes in `runbook.md` and `troubleshooting.md`, while deployment questions
should also touch `architecture.md` sections on rollout control.

This dataset is deterministic: same folder, same bootstrap output when run in isolation.
