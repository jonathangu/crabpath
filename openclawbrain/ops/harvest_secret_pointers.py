#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ENV_ASSIGN_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


@dataclass(frozen=True)
class Capability:
    service: str
    capability: str
    required_keys: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class PointerRow:
    service: str
    key_name: str
    present: bool
    pointer: str
    used_for: str
    verify: str


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        service="Mapbox",
        capability="Maps, geocoding, tiles, and routing",
        required_keys=("VITE_MAPBOX_TOKEN", "MAPBOX_API_KEY", "MAPBOX_SECRET_TOKEN"),
        notes="Prefer public token for browser maps; keep secret token server-side and rotate regularly.",
    ),
    Capability(
        service="Perplexity",
        capability="Web-grounded LLM search and answer generation",
        required_keys=("PPLX_API_KEY",),
        notes="Treat key as secret and rotate when service access changes.",
    ),
    Capability(
        service="Polygon",
        capability="Market data and aggregates",
        required_keys=("POLYGON_API_KEY",),
        notes="Keep key scoped to required market-data products.",
    ),
    Capability(
        service="NewsAPI",
        capability="News search and top-headlines fetches",
        required_keys=("NEWSAPI_KEY",),
        notes="Respect plan limits and rotate on exposure.",
    ),
    Capability(
        service="OpenAI",
        capability="LLM and embedding API access",
        required_keys=("OPENAI_API_KEY",),
        notes="Required for OpenAI-backed embeddings/LLM flows.",
    ),
    Capability(
        service="SEC/EDGAR",
        capability="Public filings data access",
        required_keys=(),
        notes="No API key required; follow SEC fair-access and User-Agent guidance.",
    ),
    Capability(
        service="ClinicalTrials.gov",
        capability="Public clinical trial registry queries",
        required_keys=(),
        notes="No API key required for standard public endpoints.",
    ),
    Capability(
        service="FDA open data",
        capability="OpenFDA public datasets and query endpoints",
        required_keys=(),
        notes="No API key required for standard public endpoints.",
    ),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest secret pointers and capability key presence.")
    parser.add_argument("--workspace", required=True, help="Workspace root path")
    parser.add_argument(
        "--out",
        help="Output markdown path (default: <workspace>/docs/secret-pointers.md)",
    )
    parser.add_argument(
        "--extra-env-file",
        action="append",
        default=[],
        help="Additional .env-like file to inspect for key names (repeatable)",
    )
    parser.add_argument(
        "--openclaw-config",
        help="Path to openclaw.json (default: ~/.openclaw/openclaw.json if it exists)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of writing markdown")
    return parser.parse_args(argv)


def _strip_inline_comment(raw: str) -> str:
    in_single = False
    in_double = False
    out: list[str] = []
    for ch in raw:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()


def _parse_env_file(path: Path) -> dict[str, bool]:
    keys: dict[str, bool] = {}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return keys
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ENV_ASSIGN_RE.match(stripped)
        if not match:
            continue
        key = match.group(1)
        raw_value = _strip_inline_comment(match.group(2))
        value = raw_value
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        keys[key] = bool(value.strip())
    return keys


def _iter_default_env_files(workspace: Path) -> list[Path]:
    return [
        workspace / ".env",
        workspace / ".env.local",
        workspace / ".env.development",
        workspace / ".env.production",
    ]


def _collect_env_key_presence(workspace: Path, extra_env_files: list[str]) -> dict[str, list[tuple[str, bool]]]:
    results: dict[str, list[tuple[str, bool]]] = {}
    env_files = _iter_default_env_files(workspace)
    env_files.extend(Path(p).expanduser() for p in extra_env_files)
    for env_file in env_files:
        if not env_file.exists():
            continue
        parsed = _parse_env_file(env_file)
        for key, present in parsed.items():
            results.setdefault(key, []).append((str(env_file), present))
    return results


def _iter_tokenfile_pointers(obj: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    pointers: list[tuple[tuple[str, ...], str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = (*path, str(key))
            if key == "tokenFile" and isinstance(value, str):
                pointers.append((new_path, value))
            pointers.extend(_iter_tokenfile_pointers(value, new_path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            pointers.extend(_iter_tokenfile_pointers(value, (*path, str(idx))))
    return pointers


def _infer_token_service(path: tuple[str, ...]) -> str:
    lowered = {segment.lower() for segment in path}
    if "telegram" in lowered:
        return "OpenClaw Telegram"
    return "OpenClaw Config"


def _infer_token_used_for(path: tuple[str, ...]) -> str:
    parts = [p for p in path if p != "tokenFile"]
    if "accounts" in parts:
        try:
            account_id = parts[parts.index("accounts") + 1]
            return f"OpenClaw account token ({account_id})"
        except (IndexError, ValueError):
            pass
    return "OpenClaw tokenFile credential"


def _collect_tokenfile_rows(openclaw_config: Path | None) -> list[PointerRow]:
    if openclaw_config is None or not openclaw_config.exists():
        return []
    try:
        payload = json.loads(openclaw_config.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[PointerRow] = []
    for path, raw_pointer in _iter_tokenfile_pointers(payload):
        pointer_path = Path(raw_pointer).expanduser()
        rows.append(
            PointerRow(
                service=_infer_token_service(path),
                key_name="tokenFile",
                present=pointer_path.exists(),
                pointer=str(pointer_path),
                used_for=_infer_token_used_for(path),
                verify=f"file exists at {pointer_path}",
            )
        )
    return rows


def _resolve_env_pointer(
    key_name: str,
    env_presence: dict[str, list[tuple[str, bool]]],
) -> tuple[bool, str, str]:
    hits = env_presence.get(key_name, [])
    for pointer, present in hits:
        if present:
            return True, pointer, f"non-empty assignment in {pointer}"
    if hits:
        return False, hits[0][0], f"non-empty assignment in {hits[0][0]}"
    process_present = bool(os.getenv(key_name, "").strip())
    if process_present:
        return True, "process environment", f"os.environ has non-empty {key_name}"
    return False, "(not found)", f"os.environ has non-empty {key_name}"


def _build_rows(
    env_presence: dict[str, list[tuple[str, bool]]],
    token_rows: list[PointerRow],
) -> list[PointerRow]:
    rows: list[PointerRow] = []
    for cap in CAPABILITIES:
        if not cap.required_keys:
            rows.append(
                PointerRow(
                    service=cap.service,
                    key_name="(none)",
                    present=True,
                    pointer="(public endpoint)",
                    used_for=cap.capability,
                    verify="true (no key required)",
                )
            )
            continue
        for key_name in cap.required_keys:
            present, pointer, verify = _resolve_env_pointer(key_name, env_presence)
            rows.append(
                PointerRow(
                    service=cap.service,
                    key_name=key_name,
                    present=present,
                    pointer=pointer,
                    used_for=cap.capability,
                    verify=verify,
                )
            )
    rows.extend(token_rows)
    return rows


def _render_markdown(workspace: Path, rows: list[PointerRow]) -> str:
    out: list[str] = []
    out.append("# Secret Pointers Registry")
    out.append("")
    out.append("Generated by `python3 -m openclawbrain.ops.harvest_secret_pointers`.")
    out.append("")
    out.append(f"- Workspace: `{workspace}`")
    out.append("- Secret values are intentionally never printed or stored here.")
    out.append("")
    out.append("## Registry")
    out.append("")

    grouped: dict[str, list[PointerRow]] = {}
    for row in rows:
        grouped.setdefault(row.service, []).append(row)

    for service in sorted(grouped):
        out.append(f"### {service}")
        out.append("")
        out.append("| key name | present | pointer | used for | verify |")
        out.append("|---|---|---|---|---|")
        for row in grouped[service]:
            out.append(
                f"| `{row.key_name}` | `{str(row.present).lower()}` | `{row.pointer}` | {row.used_for} | {row.verify} |"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _default_openclaw_config() -> Path | None:
    candidate = Path.home() / ".openclaw" / "openclaw.json"
    return candidate if candidate.exists() else None


def _resolve_openclaw_config(raw_path: str | None) -> Path | None:
    if raw_path:
        return Path(raw_path).expanduser()
    return _default_openclaw_config()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace = Path(args.workspace).expanduser()
    out_path = Path(args.out).expanduser() if args.out else workspace / "docs" / "secret-pointers.md"
    openclaw_config = _resolve_openclaw_config(args.openclaw_config)

    env_presence = _collect_env_key_presence(workspace, args.extra_env_file)
    token_rows = _collect_tokenfile_rows(openclaw_config)
    rows = _build_rows(env_presence, token_rows)

    if args.json:
        payload = {
            "workspace": str(workspace),
            "openclaw_config": str(openclaw_config) if openclaw_config else None,
            "rows": [asdict(row) for row in rows],
            "capabilities": [asdict(cap) for cap in CAPABILITIES],
        }
        print(json.dumps(payload, indent=2))
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_markdown(workspace, rows), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
