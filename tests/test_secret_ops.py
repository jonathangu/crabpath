from __future__ import annotations

from pathlib import Path

from openclawbrain.ops import audit_secret_leaks, harvest_secret_pointers


def test_harvest_secret_pointers_markdown_has_no_secret_values(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    docs_dir = workspace / "docs"
    docs_dir.mkdir(parents=True)
    (workspace / ".env").write_text(
        "OPENAI_API_KEY=sk-this-should-never-appear\nPPLX_API_KEY=\n",
        encoding="utf-8",
    )
    out_path = docs_dir / "secret-pointers.md"

    code = harvest_secret_pointers.main(
        [
            "--workspace",
            str(workspace),
            "--out",
            str(out_path),
        ]
    )
    assert code == 0

    report = out_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-this-should-never-appear" not in report
    assert "sk-this-should-never-appear" not in report
    assert "OPENAI_API_KEY" in report
    assert "| `OPENAI_API_KEY` | `true` |" in report


def test_audit_secret_leaks_reports_path_and_line_without_echoing_token(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_file = workspace / "notes.md"
    token = "sk-ABCDEF1234567890XYZ"
    leak_file.write_text(f"This line has a leak: {token}\n", encoding="utf-8")

    code = audit_secret_leaks.main(["--workspace", str(workspace), "--strict"])
    out = capsys.readouterr().out

    assert code == 1
    assert f"{leak_file}:1" in out
    assert token not in out
