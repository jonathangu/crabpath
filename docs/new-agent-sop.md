# SOP: Create a new OpenClaw agent + dedicated OpenClawBrain (best-practice)

This is the canonical operator flow for creating a new OpenClaw profile-style setup:

- one OpenClaw agent workspace
- one dedicated OpenClawBrain `state.json`
- one dedicated `openclawbrain serve` daemon

## A) Decide IDs/paths (agentId, workspace dir, brain dir)

Choose stable names first:

```bash
agentId="pelican"
workspaceDir="$HOME/.openclaw/workspace-pelican"
brainDir="$HOME/.openclawbrain/$agentId"
statePath="$brainDir/state.json"
```

Recommended conventions:

- `agentId` is lowercase and stable (`main`, `pelican`, `bountiful`, etc.)
- workspace is per-agent (do not share one workspace across multiple agents)
- brain path is `~/.openclawbrain/<agentId>/state.json`

## B) Create workspace skeleton (files to create)

Create a minimal OpenClaw-ready workspace skeleton:

```bash
mkdir -p "$workspaceDir/memory"

: > "$workspaceDir/AGENTS.md"
: > "$workspaceDir/SOUL.md"
: > "$workspaceDir/USER.md"
: > "$workspaceDir/MEMORY.md"
: > "$workspaceDir/active-tasks.md"
: > "$workspaceDir/memory/today.md"
```

Add your agent-specific instructions in `AGENTS.md` and persona/policy files before production traffic.

## C) Init brain (`openclawbrain init`)

```bash
mkdir -p "$brainDir"
openclawbrain init --workspace "$workspaceDir" --output "$brainDir"
openclawbrain doctor --state "$statePath"
```

Expected result: `statePath` exists and doctor output is healthy.

## D) Start daemon in production

### launchd template + commands (bootstrap/kickstart/print)

1. Copy the template from this repo:

```bash
mkdir -p "$HOME/Library/LaunchAgents"
cp ~/openclawbrain/examples/ops/com.openclawbrain.AGENT.plist.template \
  "$HOME/Library/LaunchAgents/com.openclawbrain.${agentId}.plist"
```

2. Replace placeholders in the copied plist:

- `AGENT_ID` -> your `agentId`
- `STATE_PATH` -> `~/.openclawbrain/<agentId>/state.json`
- `LOG_PATH` -> a stable log path (example: `~/.openclawbrain/<agentId>/daemon.log`)

3. Load and verify with launchctl:

```bash
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.openclawbrain.${agentId}.plist"
launchctl kickstart -k "gui/$(id -u)/com.openclawbrain.${agentId}"
launchctl print "gui/$(id -u)/com.openclawbrain.${agentId}"
```

### Security notes

- If a plist contains secrets (for example inline `OPENAI_API_KEY`), restrict permissions:

```bash
chmod 600 "$HOME/Library/LaunchAgents/com.openclawbrain.${agentId}.plist"
```

- Prefer environment files or secure key-loading patterns over inline secrets.

## E) Wire OpenClaw routing (Telegram bot optional)

If you use Telegram:

1. Add Telegram `accountId` (stable account identifier).
2. Store bot token in a separate `tokenFile`, not in command history or docs.
3. Lock token file permissions:

```bash
tokenFile="$HOME/.openclaw/tokens/telegram-${agentId}.token"
mkdir -p "$(dirname "$tokenFile")"
chmod 700 "$(dirname "$tokenFile")"
chmod 600 "$tokenFile"
```

Use this Python fallback to patch `~/.openclaw/openclaw.json` safely. It adds:

- an agent entry under `agents.list`
- a Telegram binding from `accountId` -> `agentId`

It does not print token values.

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--agent-id", required=True)
parser.add_argument("--workspace", required=True)
parser.add_argument("--brain-dir", required=True)
parser.add_argument("--account-id", required=True)
parser.add_argument("--token-file", required=True)
args = parser.parse_args()

cfg_path = Path.home() / ".openclaw" / "openclaw.json"
cfg_path.parent.mkdir(parents=True, exist_ok=True)

if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8") or "{}")
else:
    cfg = {}

agents = cfg.setdefault("agents", {})
agent_list = agents.setdefault("list", [])
if not isinstance(agent_list, list):
    agent_list = []
    agents["list"] = agent_list

agent_entry = {
    "id": args.agent_id,
    "workspace": args.workspace,
    "brain": {
        "state": f"{args.brain_dir}/state.json"
    }
}

updated = False
for i, entry in enumerate(agent_list):
    if isinstance(entry, dict) and entry.get("id") == args.agent_id:
        agent_list[i] = {**entry, **agent_entry}
        updated = True
        break
if not updated:
    agent_list.append(agent_entry)

bindings = cfg.setdefault("bindings", {})
if isinstance(bindings, dict):
    telegram = bindings.setdefault("telegram", {})
    if not isinstance(telegram, dict):
        telegram = {}
        bindings["telegram"] = telegram
    telegram[args.account_id] = {
        "agentId": args.agent_id,
        "tokenFile": args.token_file,
    }
else:
    cfg["bindings"] = {
        "telegram": {
            args.account_id: {
                "agentId": args.agent_id,
                "tokenFile": args.token_file,
            }
        }
    }

cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
print(f"Patched {cfg_path}")
```

Example run:

```bash
python3 /tmp/patch_openclaw_config.py \
  --agent-id "$agentId" \
  --workspace "$workspaceDir" \
  --brain-dir "$brainDir" \
  --account-id "telegram:123456789" \
  --token-file "$tokenFile"
```

Some OpenClaw builds may not support `openclaw agents bind` or `openclaw agents add --bind` yet. The patch snippet above is the reliable fallback.

## F) Pin the tight query line in that workspace’s `AGENTS.md`

Use this exact line (replace `<agentId>`, `<summary>`, `<chat_id>`):

```bash
python3 ~/openclawbrain/examples/openclaw_adapter/query_brain.py ~/.openclawbrain/<agentId>/state.json '<summary>' --chat-id '<chat_id>' --json --compact --no-include-node-ids --exclude-bootstrap --max-prompt-context-chars 12000
```

## G) Verification checklist

- `openclawbrain status --state ~/.openclawbrain/<agentId>/state.json` shows `daemon.sock` running.
- Adapter compact output check:

```bash
python3 ~/openclawbrain/examples/openclaw_adapter/query_brain.py \
  ~/.openclawbrain/<agentId>/state.json "test query" \
  --chat-id "smoke-test" --json --compact --exclude-bootstrap
```

In compact mode, verify JSON contains only 4 top-level keys: `state`, `query`, `fired_nodes`, `prompt_context`.

- `openclaw channels status --probe` is healthy.
- Send `/start` to the new bot and confirm it routes to the new agent.

## H) Common footguns

- Using `/tmp` token files (they vanish on reboot or cleanup jobs).
- Forgetting to restart the daemon after upgrading `openclawbrain`.
- Prompt bloat from bootstrap duplication when `--exclude-bootstrap` is not enabled.
- Secrets stored world-readable (`tokenFile`, plist, or config with broad permissions).
