from __future__ import annotations

import importlib


def test_packaged_openclaw_adapter_cli_modules_importable() -> None:
    query_mod = importlib.import_module("openclawbrain.openclaw_adapter.query_brain")
    learn_mod = importlib.import_module("openclawbrain.openclaw_adapter.learn_correction")

    assert callable(query_mod.main)
    assert callable(learn_mod.main)
