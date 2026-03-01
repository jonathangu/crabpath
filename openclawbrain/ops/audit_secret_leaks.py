#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bpk_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bxoxb-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{10,}\b"),
    re.compile(r"\bPPLX-[A-Za-z0-9]{8,}\b", re.IGNORECASE),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit workspace files for accidental secret leaks.")
    parser.add_argument("--workspace", required=True, help="Workspace root path")
    parser.add_argument("--state", help="Optional state.json path to scan in addition to workspace")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if matches are found")
    return parser.parse_args(argv)


def _iter_scan_files(workspace: Path, state_path: Path | None) -> list[Path]:
    scan_files: list[Path] = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            scan_files.append(path)
    if state_path is not None and state_path.exists():
        scan_files.append(state_path)
    return scan_files


def _file_hits(path: Path) -> list[int]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    hits: list[int] = []
    for idx, line in enumerate(content.splitlines(), start=1):
        for pattern in PATTERNS:
            if pattern.search(line):
                hits.append(idx)
                break
    return hits


def _scan(workspace: Path, state_path: Path | None) -> list[tuple[Path, int]]:
    findings: list[tuple[Path, int]] = []
    for file_path in _iter_scan_files(workspace, state_path):
        for lineno in _file_hits(file_path):
            findings.append((file_path, lineno))
    return findings


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace = Path(args.workspace).expanduser()
    state_path = Path(args.state).expanduser() if args.state else None

    findings = _scan(workspace, state_path)
    if findings:
        print("Potential secret leaks found:")
        for path, lineno in findings:
            print(f"{path}:{lineno}")
    else:
        print("No potential secret leaks found.")

    if args.strict and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
