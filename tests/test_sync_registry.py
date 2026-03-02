from __future__ import annotations

import json
from pathlib import Path

from openclawbrain.ops import sync_registry


def test_sync_registry_creates_global_registry_and_workspace_symlinks(tmp_path: Path) -> None:
    credentials_dir = tmp_path / "credentials"
    env_dir = credentials_dir / "env"
    registry_dir = credentials_dir / "registry"
    env_dir.mkdir(parents=True)

    env_file = env_dir / "alpha.env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-dummy-should-not-appear\nUNMAPPED_FOO=top-secret-value\n",
        encoding="utf-8",
    )

    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()

    code = sync_registry.main(
        [
            "--credentials-dir",
            str(credentials_dir),
            "--workspace",
            str(workspace_a),
            "--workspace",
            str(workspace_b),
            "--openclaw-config",
            str(tmp_path / "openclaw.json"),
        ]
    )
    assert code == 0

    secret_pointers = registry_dir / "secret-pointers.md"
    capabilities = registry_dir / "capabilities.md"
    assert secret_pointers.exists()
    assert capabilities.exists()

    link_a_secret = workspace_a / "docs" / "secret-pointers.md"
    link_a_cap = workspace_a / "docs" / "capabilities.md"
    link_b_secret = workspace_b / "docs" / "secret-pointers.md"
    link_b_cap = workspace_b / "docs" / "capabilities.md"

    assert link_a_secret.is_symlink()
    assert link_a_cap.is_symlink()
    assert link_b_secret.is_symlink()
    assert link_b_cap.is_symlink()

    assert link_a_secret.resolve(strict=False) == secret_pointers.resolve(strict=False)
    assert link_a_cap.resolve(strict=False) == capabilities.resolve(strict=False)
    assert link_b_secret.resolve(strict=False) == secret_pointers.resolve(strict=False)
    assert link_b_cap.resolve(strict=False) == capabilities.resolve(strict=False)

    secret_report = secret_pointers.read_text(encoding="utf-8")
    capabilities_report = capabilities.read_text(encoding="utf-8")

    assert "OPENAI_API_KEY" in secret_report
    assert "OPENAI_API_KEY" in capabilities_report
    assert "UNMAPPED_FOO" in capabilities_report
    assert "sk-dummy-should-not-appear" not in secret_report
    assert "sk-dummy-should-not-appear" not in capabilities_report
    assert "top-secret-value" not in secret_report
    assert "top-secret-value" not in capabilities_report


def test_sync_registry_discovers_workspaces_from_openclaw_config(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()

    openclaw_config = tmp_path / "openclaw.json"
    openclaw_config.write_text(
        json.dumps(
            {
                "agents": {
                    "list": [
                        {"id": "a", "workspace": str(workspace_a)},
                        {"id": "b", "workspace": str(workspace_b)},
                        {"id": "dup", "workspace": str(workspace_a)},
                        {"id": "invalid"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    workspaces = sync_registry._resolve_workspaces([], openclaw_config=openclaw_config)
    assert [str(path) for path in workspaces] == [str(workspace_a), str(workspace_b)]


def test_sync_registry_fallbacks_to_workspace_glob(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    openclaw_root = home / ".openclaw"
    openclaw_root.mkdir(parents=True)
    workspace_b = openclaw_root / "workspace-b"
    workspace_a = openclaw_root / "workspace-a"
    workspace_b.mkdir()
    workspace_a.mkdir()

    monkeypatch.setenv("HOME", str(home))
    workspaces = sync_registry._resolve_workspaces([], openclaw_config=openclaw_root / "missing-openclaw.json")

    assert [str(path) for path in workspaces] == [str(workspace_a), str(workspace_b)]
