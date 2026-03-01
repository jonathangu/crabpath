# SOP: Create a new OpenClaw agent + dedicated OpenClawBrain (best-practice)

Canonical operator flow for a new profile-style setup:

- one OpenClaw agent workspace
- one dedicated OpenClawBrain `state.json`
- one dedicated `openclawbrain serve` daemon
- one explicit OpenClaw routing binding

## A) Decide IDs and paths

```bash
agentId="pelican"
agentName="Pelican"
workspaceDir="$HOME/.openclaw/workspace-pelican"
brainDir="$HOME/.openclawbrain/$agentId"
statePath="$brainDir/state.json"
```

Conventions:

- `agentId` is lowercase and stable (`main`, `pelican`, `bountiful`, ...)
- workspace is per-agent (do not share one workspace across agents)
- brain path is `~/.openclawbrain/<agentId>/state.json`

## B) Create workspace skeleton

Create a minimal OpenClaw-ready workspace skeleton:

```bash
mkdir -p "$workspaceDir/memory"

: > "$workspaceDir/AGENTS.md"
: > "$workspaceDir/SOUL.md"
: > "$workspaceDir/USER.md"
: > "$workspaceDir/MEMORY.md"
: > "$workspaceDir/IDENTITY.md"
: > "$workspaceDir/TOOLS.md"
: > "$workspaceDir/active-tasks.md"
: > "$workspaceDir/memory/$(date +%F).md"
```

Add your agent-specific instructions in `AGENTS.md` and persona/policy files before production traffic.

## C) Init brain (`openclawbrain init`)

```bash
mkdir -p "$brainDir"
openclawbrain init --workspace "$workspaceDir" --output "$brainDir"
openclawbrain doctor --state "$statePath"
```

Expected result: `statePath` exists and doctor output is healthy.

## D) Start daemon in production (launchd)

1. Generate a filled plist from the packaged template writer:

```bash
mkdir -p "$HOME/Library/LaunchAgents"
python3 -m openclawbrain.ops.write_launchd_plist \
  --agent-id "$agentId" \
  --state-path "$statePath" \
  --out-path "$HOME/Library/LaunchAgents/com.openclawbrain.${agentId}.plist" \
  --log-path "$HOME/.openclawbrain/${agentId}/daemon.log"
```

2. Load and verify:

```bash
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.openclawbrain.${agentId}.plist"
launchctl kickstart -k "gui/$(id -u)/com.openclawbrain.${agentId}"
launchctl print "gui/$(id -u)/com.openclawbrain.${agentId}"
```

Security note: if plist contains secrets, set `chmod 600 "$HOME/Library/LaunchAgents/com.openclawbrain.${agentId}.plist"` and prefer secure key-loading over inline secrets.

## E) Patch OpenClaw routing config (safe, idempotent)

Use the provided ops utility to patch `~/.openclaw/openclaw.json` with the real schema safely:

- `agents.list` is a list of dicts.
- `bindings` is a list of dicts like:
  - `{"agentId":"family","match":{"channel":"telegram","accountId":"family"}}`
- Telegram accounts live at `channels.telegram.accounts[accountId]`.

Create a stable token file (avoid `/tmp`):

```bash
mkdir -p "$HOME/.openclaw/tokens"
# paste token into file and lock permissions
printf '%s' "PASTE_TELEGRAM_BOT_TOKEN_HERE" > "$HOME/.openclaw/tokens/telegram-${agentId}.token"
chmod 600 "$HOME/.openclaw/tokens/telegram-${agentId}.token"
```

Then run:

```bash
python3 -m openclawbrain.ops.patch_openclaw_config \
  --agent-id "$agentId" \
  --agent-name "$agentName" \
  --workspace "$workspaceDir" \
  --telegram-account-id "$agentId" \
  --telegram-token-file "$HOME/.openclaw/tokens/telegram-${agentId}.token"
```

Optional allow-list (repeat `--allow-from`):

```bash
python3 -m openclawbrain.ops.patch_openclaw_config \
  --agent-id "$agentId" \
  --agent-name "$agentName" \
  --workspace "$workspaceDir" \
  --telegram-account-id "$agentId" \
  --telegram-token-file "$HOME/.openclaw/tokens/telegram-${agentId}.token" \
  --allow-from "telegram:111111" \
  --allow-from "telegram:222222"
```

Short inline schema snippet (reference only):

```json
{
  "agents": {
    "list": [
      {"id": "pelican", "name": "Pelican", "workspace": "~/.openclaw/workspace-pelican"}
    ]
  },
  "channels": {
    "telegram": {
      "accounts": {
        "pelican": {
          "tokenFile": "~/.openclaw/tokens/telegram-pelican.token",
          "enabled": true,
          "dmPolicy": "pairing",
          "groupPolicy": "disabled"
        }
      }
    }
  },
  "bindings": [
    {"agentId": "pelican", "match": {"channel": "telegram", "accountId": "pelican"}}
  ]
}
```

## Secrets: pointers only + run harvest

Secrets policy:

- Never store secret values in workspace memory files, prompts, or brain state.
- Store only pointers: env key names and local `tokenFile` paths.
- Recommended local pattern on Mac mini:
  - Keep secret values in `~/.openclaw/credentials/env/<project>.env` (`chmod 600`).
  - Keep repo/workspace `.env` as a symlink to that centralized file.
  - Keep token files in `~/.openclaw/credentials/*.token` (`chmod 600`).
- Verify only with boolean presence checks.

Run capability/secret-pointer harvest after setup:

```bash
python3 -m openclawbrain.ops.harvest_secret_pointers \
  --workspace "$workspaceDir"
```

Recommended for host-wide multi-agent setups:

```bash
python3 -m openclawbrain.ops.sync_registry
```

This keeps one host-level registry in `~/.openclaw/credentials/registry` and wires each workspace docs file as a symlink, so capability/secret-pointer docs stay in sync across agents.

Optional strict leak audit:

```bash
python3 -m openclawbrain.ops.audit_secret_leaks \
  --workspace "$workspaceDir" \
  --strict
```

## F) Pin the query command in workspace `AGENTS.md`

Use packaged CLI module invocation (no `~/openclawbrain` clone required):

```bash
python3 -m openclawbrain.openclaw_adapter.query_brain ~/.openclawbrain/<agentId>/state.json '<summary>' --chat-id '<chat_id>' --format prompt --exclude-bootstrap --max-prompt-context-chars 12000
```

Learn command (no fired node IDs in prompt payload):

```bash
python3 -m openclawbrain.openclaw_adapter.learn_by_chat_id --state ~/.openclawbrain/<agentId>/state.json --chat-id '<chat_id>' --outcome 1.0 --lookback 1 --json
```

Canonical feedback command:

```bash
python3 -m openclawbrain.openclaw_adapter.capture_feedback --state ~/.openclawbrain/<agentId>/state.json --chat-id '<chat_id>' --kind CORRECTION --content '<correction text>' --lookback 1 --message-id '<stable-message-id>' --json
```

## Always-on self-learning policy (recommended)

Copy/paste into workspace `SOUL.md`:

```md
## Always-on self-learning (default)

- Treat clear user corrections as immediate memory updates: run `capture_feedback --kind CORRECTION` in the same turn with `--chat-id '<chat_id>' --lookback 1`.
- Treat durable user teachings (rules/facts) as memory updates: run `capture_feedback --kind TEACHING` (or update canonical files and `sync`).
- If correction vs teaching is unclear, ask one clarifying question before writing memory.
- Never log secrets (tokens, passwords, private keys, sensitive personal data).
- Use `--dedup-key` (or `--message-id`) whenever available so replay/harvester retries do not double-inject.
- Keep prompts tight: use `query_brain --format prompt` for retrieval; use `capture_feedback`/`learn_by_chat_id` keyed by `chat_id`; do not add `fired_nodes` to prompts.
```

## G) Verification checklist

- `openclawbrain status --state ~/.openclawbrain/<agentId>/state.json` shows running socket.
- Adapter prompt output check:

```bash
python3 -m openclawbrain.openclaw_adapter.query_brain \
  ~/.openclawbrain/<agentId>/state.json "test query" \
  --chat-id "smoke-test" --format prompt --exclude-bootstrap
```

Verify output is a `[BRAIN_CONTEXT v1]...[/BRAIN_CONTEXT]` block.

- `openclaw channels status --probe` is healthy.
- Send `/start` to the new bot and confirm route goes to the new agent.

## H) Common footguns

- Using `/tmp` token files (lost on reboot/cleanup).
- Forgetting daemon restart after `openclawbrain` upgrade.
- Prompt bloat when `--exclude-bootstrap` is omitted.
- World-readable secrets (`tokenFile`, plist, config).
