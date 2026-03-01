#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from openclawbrain.ops import harvest_secret_pointers

DEFAULT_WORKSPACES: tuple[str, ...] = (
    "~/.openclaw/workspace",
    "~/.openclaw/workspace-pelican",
    "~/.openclaw/workspace-bountiful",
    "~/.openclaw/workspace-family",
)


@dataclass(frozen=True)
class SymlinkAction:
    workspace: str
    link: str
    target: str
    action: str


@dataclass(frozen=True)
class RunSummary:
    credentials_dir: str
    env_dir: str
    registry_dir: str
    openclaw_config: str
    workspaces: list[str]
    anchor_workspace: str
    env_files: list[str]
    secret_pointers_path: str
    capabilities_path: str
    symlink_actions: list[SymlinkAction]
    dry_run: bool


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate host-level capability/secret-pointer registry and sync workspace docs symlinks."
    )
    parser.add_argument(
        "--credentials-dir",
        default="~/.openclaw/credentials",
        help="OpenClaw credentials root (default: ~/.openclaw/credentials)",
    )
    parser.add_argument(
        "--env-dir",
        help="Directory with centralized env files (default: <credentials-dir>/env)",
    )
    parser.add_argument(
        "--registry-dir",
        help="Registry directory (default: <credentials-dir>/registry)",
    )
    parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        help="Workspace root to receive docs symlinks (repeatable)",
    )
    parser.add_argument(
        "--openclaw-config",
        default="~/.openclaw/openclaw.json",
        help="OpenClaw config path for tokenFile pointer harvest",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing files")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary")
    return parser.parse_args(argv)


def _resolve_workspaces(raw_workspaces: list[str]) -> list[Path]:
    if raw_workspaces:
        return [Path(p).expanduser() for p in raw_workspaces]
    return [Path(p).expanduser() for p in DEFAULT_WORKSPACES]


def _choose_anchor_workspace(workspaces: list[Path]) -> Path:
    for workspace in workspaces:
        if workspace.exists():
            return workspace
    return workspaces[0]


def _iter_env_files(env_dir: Path) -> list[Path]:
    if not env_dir.exists() or not env_dir.is_dir():
        return []
    env_files = [path for path in sorted(env_dir.rglob("*")) if path.is_file() and path.suffix.lower() == ".env"]
    return env_files


def _resolve_env_hits(env_files: list[Path]) -> dict[str, list[tuple[str, bool]]]:
    hits: dict[str, list[tuple[str, bool]]] = {}
    for env_file in env_files:
        parsed = harvest_secret_pointers._parse_env_file(env_file)
        for key_name, present in parsed.items():
            hits.setdefault(key_name, []).append((str(env_file), present))
    return hits


def _verify_template(required_keys: tuple[str, ...], env_file: str | None) -> str:
    source_target = env_file or "<envfile>"
    source_expr = source_target if source_target == "<envfile>" else f'"{source_target}"'
    if not required_keys:
        return f"( source {source_expr} && true )"
    checks = " && ".join([f'test -n "${{{key}:-}}"' for key in required_keys])
    return f"( source {source_expr} && {checks} )"


def _capability_row(
    capability: harvest_secret_pointers.Capability,
    env_hits: dict[str, list[tuple[str, bool]]],
) -> tuple[str, str, str, str, str, str]:
    key_names = ", ".join(capability.required_keys) if capability.required_keys else "(none)"
    if not capability.required_keys:
        return (
            capability.service,
            capability.capability,
            key_names,
            "true",
            "(public endpoint)",
            _verify_template(capability.required_keys, "<envfile>"),
        )

    present = True
    env_files: set[str] = set()
    for key_name in capability.required_keys:
        key_hits = env_hits.get(key_name, [])
        if not any(hit_present for _, hit_present in key_hits):
            present = False
        for env_file, _ in key_hits:
            env_files.add(env_file)
    env_file_display = ", ".join(sorted(env_files)) if env_files else "(not found)"
    verify_envfile = sorted(env_files)[0] if env_files else None
    return (
        capability.service,
        capability.capability,
        key_names,
        str(present).lower(),
        env_file_display,
        _verify_template(capability.required_keys, verify_envfile),
    )


def _render_capabilities_markdown(env_hits: dict[str, list[tuple[str, bool]]]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    host = socket.gethostname()

    out: list[str] = []
    out.append("# Global Capabilities Registry")
    out.append("")
    out.append("Generated by `python3 -m openclawbrain.ops.sync_registry`.")
    out.append("")
    out.append(f"- Timestamp (UTC): `{now}`")
    out.append(f"- Host: `{host}`")
    out.append("- Secret values are intentionally never printed or stored here.")
    out.append("")
    out.append("## Rules")
    out.append("")
    out.append("- Store and share only key names, pointer paths, and presence booleans.")
    out.append("- Never print secret values in docs, logs, or tool output.")
    out.append("- Verify capability availability with env presence checks only.")
    out.append("")
    out.append("## Capability Matrix")
    out.append("")
    out.append("| service | capability | key names | present | env pointers | verify template |")
    out.append("|---|---|---|---|---|---|")

    known_keys: set[str] = set()
    for capability in harvest_secret_pointers.CAPABILITIES:
        for key_name in capability.required_keys:
            known_keys.add(key_name)
        service, capability_name, key_names, present, pointers, verify = _capability_row(capability, env_hits)
        out.append(
            f"| {service} | {capability_name} | `{key_names}` | `{present}` | `{pointers}` | `{verify}` |"
        )

    unknown_keys = sorted([key for key in env_hits if key not in known_keys])
    if unknown_keys:
        out.append("")
        out.append("## Unmapped Keys")
        out.append("")
        out.append("| key name | present | env pointers | verify template |")
        out.append("|---|---|---|---|")
        for key_name in unknown_keys:
            hits = env_hits[key_name]
            present = any(hit_present for _, hit_present in hits)
            env_pointers = ", ".join(sorted({env_file for env_file, _ in hits}))
            verify_env = sorted({env_file for env_file, _ in hits})[0] if hits else None
            verify = _verify_template((key_name,), verify_env)
            out.append(f"| `{key_name}` | `{str(present).lower()}` | `{env_pointers}` | `{verify}` |")

    return "\n".join(out).rstrip() + "\n"


def _update_symlink(source: Path, target: Path, dry_run: bool) -> str:
    if source.is_symlink() and source.resolve(strict=False) == target.resolve(strict=False):
        return "unchanged"
    if source.exists() or source.is_symlink():
        if source.is_dir() and not source.is_symlink():
            raise RuntimeError(f"Cannot replace directory with symlink: {source}")
        if not dry_run:
            source.unlink()
    if not dry_run:
        source.symlink_to(target)
    return "updated"


def _sync_workspace_links(workspaces: list[Path], registry_dir: Path, dry_run: bool) -> list[SymlinkAction]:
    actions: list[SymlinkAction] = []
    targets = (
        ("secret-pointers.md", registry_dir / "secret-pointers.md"),
        ("capabilities.md", registry_dir / "capabilities.md"),
    )
    for workspace in workspaces:
        docs_dir = workspace / "docs"
        if not dry_run:
            docs_dir.mkdir(parents=True, exist_ok=True)
        for name, target in targets:
            link_path = docs_dir / name
            action = _update_symlink(link_path, target, dry_run=dry_run)
            actions.append(
                SymlinkAction(
                    workspace=str(workspace),
                    link=str(link_path),
                    target=str(target),
                    action=action,
                )
            )
    return actions


def _run_harvest_secret_pointers(
    anchor_workspace: Path,
    registry_secret_pointers: Path,
    openclaw_config: Path,
    credentials_dir: Path,
    env_files: list[Path],
    dry_run: bool,
) -> None:
    if dry_run:
        return
    harvest_argv = [
        "--workspace",
        str(anchor_workspace),
        "--out",
        str(registry_secret_pointers),
        "--openclaw-config",
        str(openclaw_config),
        "--credentials-dir",
        str(credentials_dir),
    ]
    for env_file in env_files:
        harvest_argv.extend(["--extra-env-file", str(env_file)])
    code = harvest_secret_pointers.main(harvest_argv)
    if code != 0:
        raise RuntimeError("harvest_secret_pointers failed")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    credentials_dir = Path(args.credentials_dir).expanduser()
    env_dir = Path(args.env_dir).expanduser() if args.env_dir else credentials_dir / "env"
    registry_dir = Path(args.registry_dir).expanduser() if args.registry_dir else credentials_dir / "registry"
    openclaw_config = Path(args.openclaw_config).expanduser()
    workspaces = _resolve_workspaces(args.workspace)
    anchor_workspace = _choose_anchor_workspace(workspaces)
    env_files = _iter_env_files(env_dir)
    env_hits = _resolve_env_hits(env_files)

    registry_secret_pointers = registry_dir / "secret-pointers.md"
    registry_capabilities = registry_dir / "capabilities.md"

    if not args.dry_run:
        registry_dir.mkdir(parents=True, exist_ok=True)

    _run_harvest_secret_pointers(
        anchor_workspace=anchor_workspace,
        registry_secret_pointers=registry_secret_pointers,
        openclaw_config=openclaw_config,
        credentials_dir=credentials_dir,
        env_files=env_files,
        dry_run=args.dry_run,
    )

    capabilities_markdown = _render_capabilities_markdown(env_hits)
    if not args.dry_run:
        registry_capabilities.write_text(capabilities_markdown, encoding="utf-8")

    symlink_actions = _sync_workspace_links(workspaces, registry_dir, dry_run=args.dry_run)

    summary = RunSummary(
        credentials_dir=str(credentials_dir),
        env_dir=str(env_dir),
        registry_dir=str(registry_dir),
        openclaw_config=str(openclaw_config),
        workspaces=[str(workspace) for workspace in workspaces],
        anchor_workspace=str(anchor_workspace),
        env_files=[str(path) for path in env_files],
        secret_pointers_path=str(registry_secret_pointers),
        capabilities_path=str(registry_capabilities),
        symlink_actions=symlink_actions,
        dry_run=args.dry_run,
    )

    if args.json:
        payload = asdict(summary)
        print(json.dumps(payload, indent=2))
    else:
        if args.dry_run:
            print("Dry run: no files written.")
        else:
            print(f"Wrote {registry_secret_pointers}")
            print(f"Wrote {registry_capabilities}")
        for action in symlink_actions:
            print(f"{action.action}: {action.link} -> {action.target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
