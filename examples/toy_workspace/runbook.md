# Acme Bot Operations Runbook

This runbook captures common procedures for running and recovering the bot service.

## 1) Start the Service

1. Pull latest config and lock image tags.
2. Load the bootstrap graph once at startup.
3. Run a smoke query from `README.md` examples.

If memory loading fails, check `api-reference.md` for adapter flags and persistence
file locations.

## 2) Incident Triage Procedure

- Acknowledge alerts in the incident channel.
- Collect latest failing log slice and mark relevant nodes manually.
- Execute rollback checks before rerouting traffic.
- Escalate only after three failed restart attempts.

Detailed edge cases for repeated restarts are documented in `troubleshooting.md`.

## 3) Deployment Window

- Freeze non-critical changes at T-20.
- Verify `architecture.md` for dependency constraints.
- Run validation queries against key runbook procedures.
- Preserve guardrails: never bypass a rollback step.

## 4) Post-incident Review

Document root cause, affected nodes, and remediation actions.
Update sections in this runbook and `README.md` so the next on-call operator has current
context.

Cross-references: rollout risk details -> `architecture.md`; API failures -> `troubleshooting.md`.
