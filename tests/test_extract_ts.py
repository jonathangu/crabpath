from __future__ import annotations

from openclawbrain.full_learning import _extract_ts


def test_extract_ts_parses_iso_z_timestamp() -> None:
    payload = {"timestamp": "2026-03-01T04:30:00.074Z"}
    ts = _extract_ts(payload)
    assert ts is not None
    assert ts > 1_700_000_000


def test_extract_ts_parses_nested_message_timestamp() -> None:
    payload = {"message": {"timestamp": "2026-03-01T04:30:00.074Z"}}
    ts = _extract_ts(payload)
    assert ts is not None
    assert ts > 1_700_000_000
