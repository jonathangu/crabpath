from __future__ import annotations

import importlib


def test_packaged_openclaw_adapter_cli_modules_importable() -> None:
    query_mod = importlib.import_module("openclawbrain.openclaw_adapter.query_brain")
    learn_mod = importlib.import_module("openclawbrain.openclaw_adapter.learn_correction")
    learn_chat_mod = importlib.import_module("openclawbrain.openclaw_adapter.learn_by_chat_id")
    feedback_mod = importlib.import_module("openclawbrain.openclaw_adapter.capture_feedback")

    assert callable(query_mod.main)
    assert callable(learn_mod.main)
    assert callable(learn_chat_mod.main)
    assert callable(feedback_mod.main)


def test_packaged_ops_modules_importable() -> None:
    patch_mod = importlib.import_module("openclawbrain.ops.patch_openclaw_config")
    launchd_mod = importlib.import_module("openclawbrain.ops.write_launchd_plist")

    assert callable(patch_mod.main)
    assert callable(launchd_mod.main)
