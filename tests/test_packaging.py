from __future__ import annotations

import tomllib
from pathlib import Path

from crabpath import __version__


def test_version_string_matches_pyproject() -> None:
    with Path("pyproject.toml").open("rb") as stream:
        data = tomllib.load(stream)

    assert __version__ == data["project"]["version"]


def test_console_entrypoint_configured() -> None:
    with Path("pyproject.toml").open("rb") as stream:
        data = tomllib.load(stream)

    assert data["project"]["scripts"]["crabpath"] == "crabpath.cli:main"
