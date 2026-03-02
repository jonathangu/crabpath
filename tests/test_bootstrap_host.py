from __future__ import annotations

import json
import plistlib
from pathlib import Path

from openclawbrain.ops import bootstrap_host


def test_bootstrap_host_dry_run_invokes_sync_and_audit_without_writes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = "sk-DRYRUNSHOULDNEVERAPPEAR12345"
    repo_env_file = repo / ".env"
    repo_env_file.write_text(f"OPENAI_API_KEY={secret}\n", encoding="utf-8")

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    calls: dict[str, list[list[str]]] = {"sync": [], "audit": []}

    def _fake_sync_registry(argv: list[str] | None = None) -> int:
        argv = list(argv or [])
        calls["sync"].append(argv)
        print(json.dumps({"workspaces": [str(repo)]}))
        return 0

    def _fake_audit(argv: list[str] | None = None) -> int:
        calls["audit"].append(list(argv or []))
        return 0

    monkeypatch.setattr(bootstrap_host.sync_registry, "main", _fake_sync_registry)
    monkeypatch.setattr(bootstrap_host.audit_secret_leaks, "main", _fake_audit)

    launchd_plist = tmp_path / "com.openclawbrain.main.plist"
    code = bootstrap_host.main(
        [
            "--repo-root",
            str(repo),
            "--workspace",
            str(repo),
            "--launchd-plist-out",
            str(launchd_plist),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert calls["sync"]
    assert calls["audit"]
    assert "--dry-run" in calls["sync"][0]
    assert "--json" in calls["sync"][0]
    assert "--strict" in calls["audit"][0]
    assert str(repo) in calls["audit"][0]

    assert not (home / ".openclaw" / "credentials").exists()
    assert repo_env_file.exists()
    assert not repo_env_file.is_symlink()
    assert not launchd_plist.exists()
    assert secret not in out


def test_bootstrap_host_apply_migrates_env_syncs_registry_and_writes_launchd(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = "sk-APPLYSHOULDNEVERAPPEAR12345"
    repo_env_file = repo / ".env"
    repo_env_file.write_text(f"OPENAI_API_KEY={secret}\nUNMAPPED_FOO=top-secret-value\n", encoding="utf-8")

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    launchd_plist = tmp_path / "launchd" / "com.openclawbrain.main.plist"
    code = bootstrap_host.main(
        [
            "--apply",
            "--repo-root",
            str(repo),
            "--workspace",
            str(repo),
            "--launchd-plist-out",
            str(launchd_plist),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0

    credentials_dir = home / ".openclaw" / "credentials"
    centralized_env = credentials_dir / "env" / "repo.env"
    registry_dir = credentials_dir / "registry"

    assert centralized_env.exists()
    assert (centralized_env.stat().st_mode & 0o777) == 0o600
    assert repo_env_file.is_symlink()
    assert repo_env_file.resolve(strict=False) == centralized_env.resolve(strict=False)

    secret_pointers = registry_dir / "secret-pointers.md"
    capabilities = registry_dir / "capabilities.md"
    assert secret_pointers.exists()
    assert capabilities.exists()

    workspace_secret_link = repo / "docs" / "secret-pointers.md"
    workspace_cap_link = repo / "docs" / "capabilities.md"
    assert workspace_secret_link.is_symlink()
    assert workspace_cap_link.is_symlink()
    assert workspace_secret_link.resolve(strict=False) == secret_pointers.resolve(strict=False)
    assert workspace_cap_link.resolve(strict=False) == capabilities.resolve(strict=False)

    assert launchd_plist.exists()
    plist_payload = plistlib.loads(launchd_plist.read_bytes())
    assert plist_payload["ProgramArguments"][0] == "/bin/bash"
    assert plist_payload["ProgramArguments"][1] == "-lc"
    assert str(centralized_env) in plist_payload["ProgramArguments"][2]
    assert "openclawbrain serve --state" in plist_payload["ProgramArguments"][2]

    assert "audit exit=0" in out
    assert secret not in out
    assert "top-secret-value" not in out
    assert secret not in secret_pointers.read_text(encoding="utf-8")
    assert secret not in capabilities.read_text(encoding="utf-8")
