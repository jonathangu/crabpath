#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.openclawbrain.AGENT_ID</string>

  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/env</string>
    <string>openclawbrain</string>
    <string>serve</string>
    <string>--state</string>
    <string>STATE_PATH</string>
  </array>

  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>LOG_PATH</string>
  <key>StandardErrorPath</key>
  <string>LOG_PATH</string>

  <!-- Optional: only add EnvironmentVariables when your setup requires it.
       Avoid storing secrets inline when possible. If you do, chmod 600 this plist. -->
  <!--
  <key>EnvironmentVariables</key>
  <dict>
    <key>OPENAI_API_KEY</key>
    <string>REPLACE_ME</string>
  </dict>
  -->
</dict>
</plist>
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a launchd plist for openclawbrain serve")
    parser.add_argument("--agent-id", required=True, help="Agent id used in launchd label")
    parser.add_argument("--state-path", required=True, help="Path to state.json")
    parser.add_argument("--out-path", required=True, help="Output plist path")
    parser.add_argument("--log-path", required=True, help="Path used for stdout/stderr logs")
    return parser.parse_args(argv)


def _render(agent_id: str, state_path: str, log_path: str) -> str:
    return (
        PLIST_TEMPLATE
        .replace("AGENT_ID", agent_id)
        .replace("STATE_PATH", state_path)
        .replace("LOG_PATH", log_path)
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    state_path = str(Path(args.state_path).expanduser())
    out_path = Path(args.out_path).expanduser()
    log_path = str(Path(args.log_path).expanduser())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(args.agent_id, state_path, log_path), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

