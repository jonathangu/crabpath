from __future__ import annotations

from openclawbrain import socket_client as socket_client_module


def test_socket_client_query_method_includes_route_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, socket_path: str, timeout: float = 30.0) -> None:
            captured["socket_path"] = socket_path
            captured["timeout"] = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def request(self, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
            captured["method"] = method
            captured["params"] = dict(params or {})
            return {"ok": True}

    monkeypatch.setattr(socket_client_module, "OCBClient", FakeClient)

    code = socket_client_module._main(
        [
            "--socket",
            "/tmp/daemon.sock",
            "--method",
            "query",
            "--params",
            '{"query":"alpha","top_k":3}',
            "--route-mode",
            "edge+sim",
            "--route-top-k",
            "8",
            "--route-alpha-sim",
            "0.2",
            "--no-route-use-relevance",
        ]
    )

    assert code == 0
    assert captured["socket_path"] == "/tmp/daemon.sock"
    assert captured["method"] == "query"
    assert captured["params"] == {
        "query": "alpha",
        "top_k": 3,
        "route_mode": "edge+sim",
        "route_top_k": 8,
        "route_alpha_sim": 0.2,
        "route_use_relevance": False,
    }


def test_socket_client_non_query_method_does_not_add_route_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, socket_path: str, timeout: float = 30.0) -> None:
            captured["socket_path"] = socket_path
            captured["timeout"] = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def request(self, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
            captured["method"] = method
            captured["params"] = dict(params or {})
            return {"status": "ok"}

    monkeypatch.setattr(socket_client_module, "OCBClient", FakeClient)

    code = socket_client_module._main(
        [
            "--socket",
            "/tmp/daemon.sock",
            "--method",
            "health",
            "--params",
            "{}",
        ]
    )

    assert code == 0
    assert captured["socket_path"] == "/tmp/daemon.sock"
    assert captured["method"] == "health"
    assert captured["params"] == {}
