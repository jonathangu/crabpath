# Architecture Plan: Out-of-the-Box Tool Awareness + Secret Pointers + Learning Loops

Status: Draft (generated from operator learnings).  
Owner: OpenClawBrain / OpenClaw integration.  
Primary objective: make a new host **just work** (no key pasting, no per-workspace drift, no re-explaining what tools exist), while preserving strict secret hygiene.

## 0) What we are solving

We want all OpenClaw agents (across many workspaces) to:

1. **Know what tools/APIs exist** and when to use them.
2. **Know where secrets live** (pointers) without ever storing/printing values.
3. **Prove capability availability** with boolean checks (present/missing), not leakage.
4. **Learn in real time** (same-turn feedback) without prompt bloat.
5. Avoid per-workspace manual wiring and configuration drift.

## 1) Principles

- **No secret values** in: workspace files, brain state, prompts, logs, launchd/systemd config.
- Store only: key names, pointer paths, and presence booleans.
- Keep prompts tight: retrieval via `[BRAIN_CONTEXT]` only.
- Learning must be idempotent: dedup by message-id/dedup-key.
- Single source of truth: host-level registry + workspace symlinks.

## 2) Current building blocks (already shipped)

- `openclawbrain.openclaw_adapter.query_brain --format prompt` (retrieval)
- `openclawbrain.openclaw_adapter.capture_feedback` (canonical learning/inject + dedup)
- `openclawbrain.ops.harvest_secret_pointers` (pointer registry generator)
- `openclawbrain.ops.audit_secret_leaks --strict` (leak detector)
- `openclawbrain.ops.sync_registry` (host-level registry + workspace symlink refresh)

## 3) Target end-state (one-button OOTB)

A fresh machine should require **one command** to become correct.

### 3.1 One-shot bootstrap (new)
Propose new op: `python3 -m openclawbrain.ops.bootstrap_host` (name TBD).

Responsibilities:

1. Ensure `~/.openclaw/credentials/{env,registry}` exists.
2. Migrate/symlink `.env` into `~/.openclaw/credentials/env/<project>.env` **without printing values**.
3. Run `sync_registry` (which regenerates `registry/secret-pointers.md` + `registry/capabilities.md`).
4. Refresh workspace symlinks (`docs/secret-pointers.md`, `docs/capabilities.md`).
5. (macOS) Write/patch launchd plists so daemons source the centralized env file.
6. Run `audit_secret_leaks --strict` at end.
7. Emit a *machine-readable summary* (`--json`) for CI/ops.

### 3.2 Workspace discovery (no flags)
- `sync_registry` should discover workspaces from `~/.openclaw/openclaw.json` and fallback to glob scan.

### 3.3 Capability matrix completeness
- Capability mapping should include:
  - Search: Perplexity / web_search
  - Market data: Polygon
  - LLMs: OpenAI/Anthropic/Gemini
  - Messaging: Telegram / WhatsApp
  - Storage: DATABASE_URL
  - Optional: X/Twitter
- Unknown keys should appear in “Unmapped Keys” table.

### 3.4 Learn loop / memory consistency
- Use `capture_feedback` for same-turn corrections/teachings/directives.
- Use host registry docs as the durable “tool awareness” substrate.

## 4) Implementation plan

### Phase A: make `sync_registry` OOTB
- [ ] Auto-discover workspaces from openclaw.json.
- [ ] Optional fallback to process env when no centralized env files exist (configurable).

### Phase B: bootstrap_host
- [ ] Add `openclawbrain/ops/bootstrap_host.py`.
- [ ] Add safe env migration helper (copy + chmod 600 + symlink).
- [ ] Add launchd writer that uses `bash -lc 'set -a; source <env>; exec openclawbrain serve ...'`.
- [ ] Add systemd template writer for Linux.
- [ ] Add end-to-end tests (tmp dirs; ensure no secret substrings appear).

### Phase C: docs + operator guide
- [ ] Add “one command” quickstart.
- [ ] Add “accounting loop” (when rotate keys → run bootstrap/sync + audit).

## 5) Security checklist

- Ensure tools never output secret values.
- Ensure audit tool reports only `path:line`.
- Ensure generated docs contain no token-like substrings.
- Ensure launchd/systemd configs contain no secrets.

## 6) Open questions

- Should centralized env storage be required or optional? (Recommendation: required for OOTB correctness.)
- Should bootstrap_host modify user repos by default or require `--apply`?

