#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if isinstance(value, dict):
        return value
    replacement: dict[str, Any] = {}
    parent[key] = replacement
    return replacement


def _ensure_list(parent: dict[str, Any], key: str) -> list[Any]:
    value = parent.get(key)
    if isinstance(value, list):
        return value
    replacement: list[Any] = []
    parent[key] = replacement
    return replacement


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _upsert_agent(cfg: dict[str, Any], agent_id: str, agent_name: str, workspace: str) -> None:
    agents = _ensure_dict(cfg, "agents")
    agent_list = _ensure_list(agents, "list")

    new_entry = {"id": agent_id, "name": agent_name, "workspace": workspace}
    for idx, entry in enumerate(agent_list):
        if isinstance(entry, dict) and entry.get("id") == agent_id:
            merged = dict(entry)
            merged.update(new_entry)
            agent_list[idx] = merged
            return
    agent_list.append(new_entry)


def _upsert_telegram_account(
    cfg: dict[str, Any],
    account_id: str,
    token_file: str,
    allow_from: list[str] | None,
) -> None:
    channels = _ensure_dict(cfg, "channels")
    telegram = _ensure_dict(channels, "telegram")
    accounts = _ensure_dict(telegram, "accounts")

    current = accounts.get(account_id)
    account = dict(current) if isinstance(current, dict) else {}
    account["tokenFile"] = token_file
    account["enabled"] = True
    account["dmPolicy"] = "pairing"
    account["groupPolicy"] = "disabled"
    if allow_from is not None:
        account["allowFrom"] = _dedupe_strings(allow_from)

    accounts[account_id] = account


def _upsert_binding(cfg: dict[str, Any], agent_id: str, account_id: str) -> None:
    bindings = _ensure_list(cfg, "bindings")
    expected_match = {"channel": "telegram", "accountId": account_id}

    for idx, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            continue
        match = binding.get("match")
        if not isinstance(match, dict):
            continue
        if match.get("channel") == "telegram" and match.get("accountId") == account_id:
            merged = dict(binding)
            merged["agentId"] = agent_id
            merged["match"] = {**match, **expected_match}
            bindings[idx] = merged
            return

    bindings.append({"agentId": agent_id, "match": expected_match})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely patch ~/.openclaw/openclaw.json")
    parser.add_argument("--agent-id", required=True, help="OpenClaw agent id")
    parser.add_argument("--agent-name", required=True, help="Human-readable OpenClaw agent name")
    parser.add_argument("--workspace", required=True, help="Agent workspace path")
    parser.add_argument("--telegram-account-id", help="Telegram accountId for this agent")
    parser.add_argument("--telegram-token-file", help="Path to Telegram token file")
    parser.add_argument(
        "--allow-from",
        action="append",
        default=None,
        help="Optional allowFrom entry for Telegram account (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if bool(args.telegram_account_id) ^ bool(args.telegram_token_file):
        raise SystemExit("--telegram-account-id and --telegram-token-file must be provided together")

    cfg_path = Path.home() / ".openclaw" / "openclaw.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSON at {cfg_path}: {exc}") from exc
        if not isinstance(cfg, dict):
            cfg = {}
    else:
        cfg = {}

    _upsert_agent(cfg, args.agent_id, args.agent_name, str(Path(args.workspace).expanduser()))

    if args.telegram_account_id and args.telegram_token_file:
        _upsert_telegram_account(
            cfg,
            args.telegram_account_id,
            str(Path(args.telegram_token_file).expanduser()),
            args.allow_from,
        )
        _upsert_binding(cfg, args.agent_id, args.telegram_account_id)

    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Patched {cfg_path}")


if __name__ == "__main__":
    main()

