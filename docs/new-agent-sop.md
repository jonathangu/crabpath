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

1. Copy template from this repo:

```bash
mkdir -p "$HOME/Library/LaunchAgents"
cp examples/ops/com.openclawbrain.AGENT.plist.template \
  "$HOME/Library/LaunchAgents/com.openclawbrain.${agentId}.plist"
```

2. Replace placeholders in copied plist:

- `AGENT_ID` -> your `agentId`
- `STATE_PATH` -> `~/.openclawbrain/<agentId>/state.json`
- `LOG_PATH` -> stable log path (example: `~/.openclawbrain/<agentId>/daemon.log`)

3. Load and verify:

```bash
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.openclawbrain.${agentId}.plist"
launchctl kickstart -k "gui/$(id -u)/com.openclawbrain.${agentId}"
launchctl print "gui/$(id -u)/com.openclawbrain.${agentId}"
```

Security note: if plist contains secrets, set `chmod 600` on that plist and prefer secure key-loading over inline secrets.

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
python3 examples/ops/patch_openclaw_config.py \
  --agent-id "$agentId" \
  --agent-name "$agentName" \
  --workspace "$workspaceDir" \
  --telegram-account-id "$agentId" \
  --telegram-token-file "$HOME/.openclaw/tokens/telegram-${agentId}.token"
```

Optional allow-list (repeat `--allow-from`):

```bash
python3 examples/ops/patch_openclaw_config.py \
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

## F) Pin the query command in workspace `AGENTS.md`

Use packaged CLI module invocation (no `~/openclawbrain` clone required):

```bash
python3 -m openclawbrain.openclaw_adapter.query_brain ~/.openclawbrain/<agentId>/state.json '<summary>' --chat-id '<chat_id>' --json --compact --no-include-node-ids --exclude-bootstrap --max-prompt-context-chars 12000
```

Correction command:

```bash
python3 -m openclawbrain.openclaw_adapter.learn_correction --state ~/.openclawbrain/<agentId>/state.json --chat-id '<chat_id>' --outcome -1.0 --lookback 1 --content '<correction text>'
```

Back-compat shims remain available:

- `python3 examples/openclaw_adapter/query_brain.py ...`
- `python3 examples/openclaw_adapter/learn_correction.py ...`

## G) Verification checklist

- `openclawbrain status --state ~/.openclawbrain/<agentId>/state.json` shows running socket.
- Adapter compact output check:

```bash
python3 -m openclawbrain.openclaw_adapter.query_brain \
  ~/.openclawbrain/<agentId>/state.json "test query" \
  --chat-id "smoke-test" --json --compact --exclude-bootstrap
```

In compact mode, verify JSON has only 4 top-level keys by default: `state`, `query`, `fired_nodes`, `prompt_context`.

- `openclaw channels status --probe` is healthy.
- Send `/start` to the new bot and confirm route goes to the new agent.

## H) Common footguns

- Using `/tmp` token files (lost on reboot/cleanup).
- Forgetting daemon restart after `openclawbrain` upgrade.
- Prompt bloat when `--exclude-bootstrap` is omitted.
- World-readable secrets (`tokenFile`, plist, config).
