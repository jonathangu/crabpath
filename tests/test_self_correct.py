"""Tests for self_correct daemon method and supporting code."""

from __future__ import annotations

import json
from pathlib import Path

from openclawbrain.graph import Edge, Graph, Node
from openclawbrain.index import VectorIndex
from openclawbrain.daemon import _handle_self_learn, _DaemonState
from openclawbrain.hasher import HashEmbedder
from openclawbrain.socket_client import OCBClient


def _make_daemon_state(graph: Graph, index: VectorIndex | None = None) -> "_DaemonState":
    """Minimal daemon state for testing."""
    from collections import deque
    return _DaemonState(
        graph=graph,
        index=index or VectorIndex(),
        meta={},
        fired_log={},
    )


def _simple_graph() -> tuple[Graph, VectorIndex]:
    g = Graph()
    g.add_node(Node("a", "alpha content about deployment"))
    g.add_node(Node("b", "beta content about cleanup"))
    g.add_node(Node("c", "gamma content about monitoring"))
    g.add_edge(Edge("a", "b", weight=0.5))
    g.add_edge(Edge("b", "c", weight=0.4))
    idx = VectorIndex()
    embed = HashEmbedder().embed
    idx.upsert("a", embed("alpha content about deployment"))
    idx.upsert("b", embed("beta content about cleanup"))
    idx.upsert("c", embed("gamma content about monitoring"))
    return g, idx


def test_self_correct_injects_correction_node() -> None:
    """Self-correct with CORRECTION type creates node + inhibitory edges."""
    g, idx = _simple_graph()
    ds = _make_daemon_state(g, idx)

    payload, should_write = _handle_self_learn(
        daemon_state=ds,
        graph=g,
        index=idx,
        embed_fn=HashEmbedder().embed,
        state_path="/tmp/fake_state.json",
        params={
            "content": "Always download artifacts before terminating instances",
            "fired_ids": ["a", "b"],
            "outcome": -1.0,
            "node_type": "CORRECTION",
        },
    )

    assert should_write is True
    assert payload["node_injected"] is True
    node_id = payload["node_id"]
    assert g.get_node(node_id) is not None
    assert g.get_node(node_id).metadata["source"] == "self"
    assert g.get_node(node_id).metadata["type"] == "CORRECTION"

    # Should have inhibitory edges to fired_ids
    for target in ["a", "b"]:
        edge = g._edges.get(node_id, {}).get(target)
        assert edge is not None
        assert edge.weight < 0, f"Expected inhibitory edge to {target}"


def test_self_correct_teaching_no_inhibitory() -> None:
    """Self-correct with TEACHING type creates node WITHOUT inhibitory edges."""
    g, idx = _simple_graph()
    ds = _make_daemon_state(g, idx)

    payload, should_write = _handle_self_learn(
        daemon_state=ds,
        graph=g,
        index=idx,
        embed_fn=HashEmbedder().embed,
        state_path="/tmp/fake_state.json",
        params={
            "content": "GBM training takes about 40 minutes on g5.xlarge",
            "fired_ids": ["a"],
            "outcome": 0.0,  # neutral — don't penalize
            "node_type": "TEACHING",
        },
    )

    assert payload["node_injected"] is True
    node_id = payload["node_id"]
    node = g.get_node(node_id)
    assert node is not None
    assert node.metadata["type"] == "TEACHING"

    # No inhibitory edges
    outgoing = list(g.outgoing(node_id))
    for _target, edge in outgoing:
        assert edge.weight >= 0, "TEACHING should not create inhibitory edges"


def test_self_correct_without_fired_ids() -> None:
    """Self-correct without fired_ids just injects the node."""
    g, idx = _simple_graph()
    ds = _make_daemon_state(g, idx)

    payload, should_write = _handle_self_learn(
        daemon_state=ds,
        graph=g,
        index=idx,
        embed_fn=HashEmbedder().embed,
        state_path="/tmp/fake_state.json",
        params={
            "content": "Never forget to snapshot EBS volumes",
        },
    )

    assert payload["node_injected"] is True
    assert payload["fired_ids_penalized"] == []
    assert payload["edges_updated"] == 0


def test_self_correct_penalizes_fired_ids() -> None:
    """Self-correct with fired_ids and negative outcome changes edge weights."""
    g, idx = _simple_graph()
    ds = _make_daemon_state(g, idx)

    w_before = g._edges["a"]["b"].weight

    _handle_self_learn(
        daemon_state=ds,
        graph=g,
        index=idx,
        embed_fn=HashEmbedder().embed,
        state_path="/tmp/fake_state.json",
        params={
            "content": "Check instance status before terminating",
            "fired_ids": ["a", "b"],
            "outcome": -1.0,
        },
    )

    w_after = g._edges["a"]["b"].weight
    assert w_after != w_before, "Edge weight should have changed after negative outcome"


def test_self_learn_positive_reinforcement() -> None:
    """Positive outcome + TEACHING reinforces the path without inhibition."""
    g, idx = _simple_graph()
    ds = _make_daemon_state(g, idx)

    w_before = g._edges["a"]["b"].weight

    payload, should_write = _handle_self_learn(
        daemon_state=ds,
        graph=g,
        index=idx,
        embed_fn=HashEmbedder().embed,
        state_path="/tmp/fake_state.json",
        params={
            "content": "Downloading artifacts before terminating works reliably",
            "fired_ids": ["a", "b"],
            "outcome": 1.0,
            "node_type": "TEACHING",
        },
    )

    assert should_write is True
    assert payload["node_injected"] is True
    node_id = payload["node_id"]
    assert g.get_node(node_id).metadata["type"] == "TEACHING"

    # No inhibitory edges from TEACHING
    for _, edge in g.outgoing(node_id):
        assert edge.weight >= 0

    # Positive outcome should have changed edge weights (reinforcement)
    w_after = g._edges["a"]["b"].weight
    assert w_after != w_before


def test_self_learn_client_methods_exist() -> None:
    """OCBClient has both self_learn and self_correct methods."""
    assert hasattr(OCBClient, "self_learn")
    assert hasattr(OCBClient, "self_correct")
    import inspect
    sig = inspect.signature(OCBClient.self_learn)
    params = list(sig.parameters.keys())
    assert "content" in params
    assert "fired_ids" in params
    assert "outcome" in params
    assert "node_type" in params


def test_self_correct_client_method_exists() -> None:
    """OCBClient has a self_correct method with correct signature."""
    assert hasattr(OCBClient, "self_correct")
    import inspect
    sig = inspect.signature(OCBClient.self_correct)
    params = list(sig.parameters.keys())
    assert "content" in params
    assert "fired_ids" in params
    assert "outcome" in params
    assert "node_type" in params
