"""Tests for MCP server handlers and request routing."""

from __future__ import annotations

import json

from crabpath import Edge, Graph, Node, mcp_server


def _build_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node(id="query-root", content="runbook query root"))
    graph.add_node(Node(id="query-answer", content="runbook answer"))
    graph.add_edge(Edge(source="query-root", target="query-answer", weight=0.85))
    return graph


class _MockIndex:
    def __init__(self) -> None:
        self.vectors = {"query-root": [0.1, 0.2]}

    def seed(self, query: str, top_k: int | None = None, embed_fn=None) -> dict[str, float]:
        return {"query-root": 2.0}


def _tool_names() -> list[str]:
    return [tool["name"] for tool in mcp_server.TOOLS]


def _handle_tools_call(name: str, arguments: dict, monkeypatch) -> dict:
    captured: list[dict] = []
    monkeypatch.setattr(mcp_server, "_emit", lambda payload: captured.append(payload))

    request = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    mcp_server._handle_request(request)
    assert captured and "result" in captured[-1]
    return json.loads(captured[-1]["result"]["content"][0]["text"])


def test_tools_list_is_complete(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(mcp_server, "_emit", lambda payload: captured.append(payload))
    mcp_server._handle_tools_list("tools-list")

    assert captured and "result" in captured[-1]
    names = {item["name"] for item in captured[-1]["result"]["tools"]}

    assert names == {
        "query",
        "migrate",
        "learn",
        "stats",
        "split",
        "add",
        "remove",
        "consolidate",
        "health",
        "evolve",
        "sim",
        "snapshot",
        "feedback",
    }


def test_query_handler_roundtrip(monkeypatch):
    graph = _build_graph()
    monkeypatch.setattr(mcp_server, "_load_graph", lambda _path: graph)
    monkeypatch.setattr(mcp_server, "_load_index", lambda _path: _MockIndex())
    monkeypatch.setattr(
        mcp_server,
        "_safe_openai_embed_fn",
        lambda: (lambda texts: [[0.0] for _ in texts]),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    payload = _handle_tools_call(
        "query",
        {"query": "runbook query root", "graph": "ignored", "index": "ignored"},
        monkeypatch,
    )

    assert "fired" in payload
    assert isinstance(payload["fired"], list)
    assert payload["fired"]
    assert payload["fired"][0]["id"] == "query-root"


def test_learn_handler_roundtrip(monkeypatch):
    graph = _build_graph()
    before = graph.get_edge("query-root", "query-answer").weight
    monkeypatch.setattr(mcp_server, "_load_graph", lambda _path: graph)

    payload = _handle_tools_call(
        "learn",
        {"graph": "ignored", "fired_ids": "query-root,query-answer", "outcome": 1.0},
        monkeypatch,
    )

    assert payload["ok"] is True
    assert payload["edges_updated"] >= 1
    assert graph.get_edge("query-root", "query-answer").weight != before


def test_health_and_stats_handlers(monkeypatch):
    graph = _build_graph()
    graph.add_node(Node(id="isolated", content="isolated"))
    graph.add_node(Node(id="other", content="other"))
    graph.add_edge(Edge(source="isolated", target="other", weight=0.25))
    monkeypatch.setattr(mcp_server, "_load_graph", lambda _path: graph)

    health = _handle_tools_call("health", {"graph": "ignored"}, monkeypatch)
    assert health["ok"] is True
    assert health["query_stats_provided"] is False
    assert any(item["metric"] == "avg_nodes_fired_per_query" for item in health["metrics"])

    stats = _handle_tools_call("stats", {"graph": "ignored"}, monkeypatch)
    assert stats["nodes"] == 4
    assert stats["edges"] == 2
    assert stats["top_hubs"][0] in {"query-root", "query-answer", "isolated", "other"}


def test_tools_call_missing_graph_emits_error(monkeypatch, tmp_path):
    captured: list[dict] = []
    monkeypatch.setattr(mcp_server, "_emit", lambda payload: captured.append(payload))

    request = {
        "jsonrpc": "2.0",
        "id": "missing-graph",
        "method": "tools/call",
        "params": {
            "name": "query",
            "arguments": {
                "query": "find this",
                "graph": str(tmp_path / "does-not-exist.json"),
                "index": str(tmp_path / "does-not-exist.index"),
            },
        },
    }
    mcp_server._handle_request(request)

    assert captured and "error" in captured[-1]
    assert captured[-1]["error"]["code"] == -32602
    assert "graph file not found" in captured[-1]["error"]["message"]
