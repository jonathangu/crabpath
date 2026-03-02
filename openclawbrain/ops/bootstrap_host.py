#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import plistlib
import re
import shlex
import shutil
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openclawbrain.ops import audit_secret_leaks, sync_registry


@dataclass(frozen=True)
class DirectoryAction:
    path: str
    action: str


@dataclass(frozen=True)
class EnvMigration:
    repo_env_file: str
    centralized_env_file: str
    transfer_action: str
    symlink_action: str
    chmod_action: str


@dataclass(frozen=True)
class LaunchdAction:
    plist_path: str
    action: str


@dataclass(frozen=True)
class AuditResult:
    workspace: str
    exit_code: int


@dataclass(frozen=True)
class BootstrapSummary:
    dry_run: bool
    repo_root: str
    credentials_dir: str
    env_dir: str
    registry_dir: str
    repo_env_file: str
    centralized_env_file: str
    directory_actions: list[DirectoryAction]
    migration: EnvMigration
    sync_registry_exit_code: int
    sync_registry_workspaces: list[str]
    launchd: LaunchdAction | None
    audit_results: list[AuditResult]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap host-level OpenClaw secret pointers, centralized env files, and workspace registry links."
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    parser.add_argument("--repo-root", default=".", help="Repo/workspace root (default: current directory)")
    parser.add_argument(
        "--repo-env-file",
        help="Repo env file path (default: <repo-root>/.env)",
    )
    parser.add_argument(
        "--credentials-dir",
        default="~/.openclaw/credentials",
        help="Credentials root (default: ~/.openclaw/credentials)",
    )
    parser.add_argument(
        "--env-dir",
        help="Centralized env dir (default: <credentials-dir>/env)",
    )
    parser.add_argument(
        "--registry-dir",
        help="Registry dir (default: <credentials-dir>/registry)",
    )
    parser.add_argument(
        "--centralized-env-file",
        help="Centralized env path (default: <env-dir>/<repo-name>.env)",
    )
    parser.add_argument(
        "--move-source",
        action="store_true",
        help="Move repo env file into centralized env file instead of copying",
    )
    parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        help="Workspace override passed through to sync_registry (repeatable)",
    )
    parser.add_argument(
        "--openclaw-config",
        default="~/.openclaw/openclaw.json",
        help="OpenClaw config path",
    )
    parser.add_argument("--launchd-plist-out", help="Optional launchd plist output path (macOS)")
    parser.add_argument(
        "--launchd-label",
        default="com.openclawbrain.main",
        help="launchd label when writing/patching plist",
    )
    parser.add_argument(
        "--launchd-state-path",
        default="~/.openclawbrain/main/state.json",
        help="State path for launchd ProgramArguments",
    )
    parser.add_argument(
        "--launchd-log-path",
        default="~/.openclawbrain/main/serve.log",
        help="Log path for launchd stdout/stderr",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary")
    return parser.parse_args(argv)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "workspace"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _resolve_repo_root(raw_repo_root: str) -> Path:
    repo_root = Path(raw_repo_root).expanduser()
    if not repo_root.is_absolute():
        repo_root = Path.cwd() / repo_root
    return repo_root.resolve(strict=False)


def _resolve_repo_env_file(repo_root: Path, raw_repo_env_file: str | None) -> Path:
    if not raw_repo_env_file:
        return repo_root / ".env"
    candidate = Path(raw_repo_env_file).expanduser()
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve(strict=False)


def _resolve_centralized_env_file(repo_root: Path, env_dir: Path, raw_centralized_env_file: str | None) -> Path:
    if raw_centralized_env_file:
        candidate = Path(raw_centralized_env_file).expanduser()
        if candidate.is_absolute():
            return candidate
        return (env_dir / candidate).resolve(strict=False)
    return env_dir / f"{_slugify(repo_root.name)}.env"


def _ensure_directory(path: Path, dry_run: bool) -> DirectoryAction:
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Expected directory path but found non-directory: {path}")
    if path.exists():
        return DirectoryAction(path=str(path), action="exists")
    if dry_run:
        return DirectoryAction(path=str(path), action="would-create")
    path.mkdir(parents=True, exist_ok=True)
    return DirectoryAction(path=str(path), action="created")


def _files_equal(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.read_bytes() == path_b.read_bytes()
    except OSError:
        return False


def _migrate_repo_env(
    repo_env_file: Path,
    centralized_env_file: Path,
    *,
    move_source: bool,
    dry_run: bool,
) -> EnvMigration:
    repo_exists = repo_env_file.exists() or repo_env_file.is_symlink()
    central_exists = centralized_env_file.exists()

    if centralized_env_file.exists() and centralized_env_file.is_dir():
        raise RuntimeError(f"Centralized env path is a directory: {centralized_env_file}")
    if repo_exists and repo_env_file.is_dir() and not repo_env_file.is_symlink():
        raise RuntimeError(f"Repo env path is a directory: {repo_env_file}")

    transfer_action = "none"
    if repo_env_file.is_symlink():
        current_target = repo_env_file.resolve(strict=False)
        desired_target = centralized_env_file.resolve(strict=False)
        if current_target == desired_target:
            transfer_action = "already-symlinked"
        elif not central_exists:
            if not current_target.exists():
                raise RuntimeError(f"Repo env symlink target does not exist: {current_target}")
            transfer_action = "would-copy-from-symlink-target" if dry_run else "copied-from-symlink-target"
            if not dry_run:
                centralized_env_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current_target, centralized_env_file)
                central_exists = True
        else:
            transfer_action = "reused-centralized"
    elif repo_exists:
        if not central_exists:
            transfer_action = "would-move" if dry_run and move_source else "would-copy" if dry_run else "moved" if move_source else "copied"
            if not dry_run:
                centralized_env_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo_env_file, centralized_env_file)
                central_exists = True
                if move_source:
                    repo_env_file.unlink()
                    repo_exists = False
        else:
            if not _files_equal(repo_env_file, centralized_env_file):
                raise RuntimeError(
                    f"Refusing to overwrite existing centralized env with different content: {centralized_env_file}"
                )
            transfer_action = "reused-centralized"
    elif not central_exists:
        raise RuntimeError(
            f"Neither repo env file nor centralized env file exists: {repo_env_file} / {centralized_env_file}"
        )
    else:
        transfer_action = "reused-centralized"

    chmod_action = "unchanged"
    if dry_run:
        chmod_action = "would-chmod-600"
    elif centralized_env_file.exists():
        os.chmod(centralized_env_file, 0o600)
        chmod_action = "chmod-600"

    desired_link = centralized_env_file.resolve(strict=False)
    if repo_env_file.is_symlink() and repo_env_file.resolve(strict=False) == desired_link:
        symlink_action = "unchanged"
    elif dry_run:
        symlink_action = "would-link"
    else:
        repo_env_file.parent.mkdir(parents=True, exist_ok=True)
        if repo_env_file.exists() or repo_env_file.is_symlink():
            if repo_env_file.is_dir() and not repo_env_file.is_symlink():
                raise RuntimeError(f"Cannot replace directory with symlink: {repo_env_file}")
            repo_env_file.unlink()
        repo_env_file.symlink_to(centralized_env_file)
        symlink_action = "linked"

    return EnvMigration(
        repo_env_file=str(repo_env_file),
        centralized_env_file=str(centralized_env_file),
        transfer_action=transfer_action,
        symlink_action=symlink_action,
        chmod_action=chmod_action,
    )


def _invoke_cli(fn: Any, argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = fn(argv)
    return int(code), out.getvalue()


def _run_sync_registry(
    *,
    credentials_dir: Path,
    env_dir: Path,
    registry_dir: Path,
    openclaw_config: Path,
    workspace_overrides: list[Path],
    dry_run: bool,
) -> tuple[int, dict[str, Any] | None]:
    argv = [
        "--credentials-dir",
        str(credentials_dir),
        "--env-dir",
        str(env_dir),
        "--registry-dir",
        str(registry_dir),
        "--openclaw-config",
        str(openclaw_config),
        "--json",
    ]
    for workspace in workspace_overrides:
        argv.extend(["--workspace", str(workspace)])
    if dry_run:
        argv.append("--dry-run")

    code, output = _invoke_cli(sync_registry.main, argv)
    payload: dict[str, Any] | None = None
    raw = output.strip()
    if raw:
        lines = raw.splitlines()
        for idx, line in enumerate(lines):
            if not line.lstrip().startswith("{"):
                continue
            candidate = "\n".join(lines[idx:])
            try:
                maybe_payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(maybe_payload, dict):
                payload = maybe_payload
                break
    return code, payload


def _select_audit_workspaces(
    workspace_overrides: list[Path], sync_payload: dict[str, Any] | None, repo_root: Path
) -> list[Path]:
    if workspace_overrides:
        return _dedupe_paths(workspace_overrides)
    if sync_payload and isinstance(sync_payload.get("workspaces"), list):
        discovered = [
            Path(item).expanduser()
            for item in sync_payload["workspaces"]
            if isinstance(item, str) and item.strip()
        ]
        if discovered:
            return _dedupe_paths(discovered)
    return [repo_root]


def _run_audit(workspace: Path) -> AuditResult:
    argv = ["--workspace", str(workspace), "--strict"]
    code, _ = _invoke_cli(audit_secret_leaks.main, argv)
    return AuditResult(workspace=str(workspace), exit_code=code)


def _write_launchd_plist(
    *,
    plist_path: Path,
    label: str,
    state_path: Path,
    log_path: Path,
    env_file: Path,
) -> LaunchdAction:
    payload: dict[str, Any]
    action = "written"
    if plist_path.exists():
        try:
            current = plistlib.loads(plist_path.read_bytes())
            payload = dict(current) if isinstance(current, dict) else {}
        except Exception:
            payload = {}
        action = "patched"
    else:
        payload = {}

    command = (
        "set -a; "
        f"source {shlex.quote(str(env_file))}; "
        "set +a; "
        f"exec openclawbrain serve --state {shlex.quote(str(state_path))}"
    )
    payload["Label"] = label
    payload["ProgramArguments"] = ["/bin/bash", "-lc", command]
    payload["RunAtLoad"] = True
    payload["KeepAlive"] = True
    payload["StandardOutPath"] = str(log_path)
    payload["StandardErrorPath"] = str(log_path)

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    with plist_path.open("wb") as f:
        plistlib.dump(payload, f, sort_keys=False)

    return LaunchdAction(plist_path=str(plist_path), action=action)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dry_run = not args.apply

    repo_root = _resolve_repo_root(args.repo_root)
    credentials_dir = Path(args.credentials_dir).expanduser()
    env_dir = Path(args.env_dir).expanduser() if args.env_dir else credentials_dir / "env"
    registry_dir = Path(args.registry_dir).expanduser() if args.registry_dir else credentials_dir / "registry"
    repo_env_file = _resolve_repo_env_file(repo_root, args.repo_env_file)
    centralized_env_file = _resolve_centralized_env_file(repo_root, env_dir, args.centralized_env_file)
    openclaw_config = Path(args.openclaw_config).expanduser()
    workspace_overrides = [Path(path).expanduser() for path in args.workspace]

    directory_actions = [
        _ensure_directory(credentials_dir, dry_run=dry_run),
        _ensure_directory(env_dir, dry_run=dry_run),
        _ensure_directory(registry_dir, dry_run=dry_run),
    ]
    migration = _migrate_repo_env(
        repo_env_file,
        centralized_env_file,
        move_source=args.move_source,
        dry_run=dry_run,
    )

    sync_code, sync_payload = _run_sync_registry(
        credentials_dir=credentials_dir,
        env_dir=env_dir,
        registry_dir=registry_dir,
        openclaw_config=openclaw_config,
        workspace_overrides=workspace_overrides,
        dry_run=dry_run,
    )

    launchd_action: LaunchdAction | None = None
    if args.launchd_plist_out:
        plist_path = Path(args.launchd_plist_out).expanduser()
        if dry_run:
            launchd_action = LaunchdAction(plist_path=str(plist_path), action="skipped-dry-run")
        else:
            launchd_action = _write_launchd_plist(
                plist_path=plist_path,
                label=args.launchd_label,
                state_path=Path(args.launchd_state_path).expanduser(),
                log_path=Path(args.launchd_log_path).expanduser(),
                env_file=centralized_env_file,
            )

    audit_workspaces = _select_audit_workspaces(workspace_overrides, sync_payload, repo_root)
    audit_results = [_run_audit(workspace) for workspace in audit_workspaces]

    summary = BootstrapSummary(
        dry_run=dry_run,
        repo_root=str(repo_root),
        credentials_dir=str(credentials_dir),
        env_dir=str(env_dir),
        registry_dir=str(registry_dir),
        repo_env_file=str(repo_env_file),
        centralized_env_file=str(centralized_env_file),
        directory_actions=directory_actions,
        migration=migration,
        sync_registry_exit_code=sync_code,
        sync_registry_workspaces=[
            str(item)
            for item in (sync_payload.get("workspaces", []) if isinstance(sync_payload, dict) else [])
            if isinstance(item, str)
        ],
        launchd=launchd_action,
        audit_results=audit_results,
    )

    if args.json:
        print(json.dumps(asdict(summary), indent=2))
    else:
        if dry_run:
            print("Dry run: no files written.")
        for action in directory_actions:
            print(f"{action.action}: {action.path}")
        print(
            f"env migration: transfer={migration.transfer_action}, symlink={migration.symlink_action}, "
            f"chmod={migration.chmod_action}"
        )
        print(f"sync_registry exit={sync_code}")
        if launchd_action is not None:
            print(f"launchd plist {launchd_action.action}: {launchd_action.plist_path}")
        for result in audit_results:
            print(f"audit exit={result.exit_code}: {result.workspace}")

    if sync_code != 0:
        return 1
    if any(result.exit_code != 0 for result in audit_results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
