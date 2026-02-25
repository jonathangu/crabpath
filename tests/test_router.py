from __future__ import annotations

import pytest

from crabpath.router import Router, RouterConfig, RouterError
from crabpath.graph import Graph, Node, Edge


def test_fallback_picks_highest_weight():
    router = Router()
    decision = router.fallback(
        [
            ("low", 0.15),
            ("high", 0.94),
            ("mid", 0.70),
        ],
        "reflex",
    )

    assert decision.chosen_target == "high"
    assert decision.tier == "reflex"
    assert decision.confidence > 0.9
    assert decision.alternatives[0][0] == "mid"


def test_parse_json_valid():
    router = Router()
    raw = (
        '{"target":"node-1","confidence":0.81,"rationale":"best match",'
        '"alternatives":[["node-2",0.2]]}'
    )
    payload = router.parse_json(raw)

    assert payload["target"] == "node-1"
    assert payload["confidence"] == 0.81
    assert payload["rationale"] == "best match"


def test_parse_json_invalid_raises():
    router = Router()

    with pytest.raises(RouterError):
        router.parse_json("not json")

    with pytest.raises(RouterError):
        router.parse_json('{"target":123,"confidence":2.0,"rationale":7}')


def test_build_prompt_under_budget():
    router = Router()
    query = (
        "How do I deploy a hotfix for memory pressure in production under high latency conditions?"
    )
    candidates = [(f"node-{i}", float(i) / 10.0) for i in range(30)]
    context = {"node_summary": "Production incident runbook notes with multiple rollout steps."}

    prompt = router.build_prompt(query, candidates, context, budget=140)

    assert len(prompt.split()) < 200
    assert len(prompt.split()) <= 140
    assert "System:" in prompt
    assert "Query:" in prompt


def test_router_config_defaults():
    cfg = RouterConfig()

    assert cfg.model == "gpt-5-mini"
    assert cfg.temperature is None  # Use model default
    assert cfg.timeout_s == 8.0
    assert cfg.max_retries == 2
    assert cfg.fallback_behavior == "heuristic"
    assert cfg.max_select == 5


# ---------------------------------------------------------------------------
# Tests: select_nodes (0, 1, or N)
# ---------------------------------------------------------------------------


def test_select_fallback_returns_top_candidates():
    router = Router()
    candidates = [
        ("node-a", 0.9, "identity rules"),
        ("node-b", 0.7, "tool config"),
        ("node-c", 0.1, "unrelated"),
    ]
    selected = router.select_nodes("what are the identity rules?", candidates)
    assert "node-a" in selected
    assert "node-b" in selected
    assert "node-c" not in selected  # Below threshold


def test_select_fallback_uses_relative_threshold():
    router = Router()
    candidates = [
        ("node-a", 1.0, "best"),
        ("node-b", 0.61, "strong"),
        ("node-c", 0.60, "boundary"),
        ("node-d", 0.59, "below boundary"),
    ]
    selected = router.select_nodes("topic", candidates)
    assert selected == ["node-a", "node-b", "node-c"]


def test_select_fallback_applies_max_select():
    router = Router(config=RouterConfig(max_select=2, fallback_behavior="heuristic"))
    candidates = [(f"node-{i}", 1.0 - (i * 0.05), "summary") for i in range(10)]
    selected = router.select_nodes("topic", candidates)
    assert selected == ["node-0", "node-1"]


def test_select_fallback_returns_empty_for_trivial():
    router = Router()
    candidates = [
        ("node-a", 0.5, "something"),
    ]
    selected = router.select_nodes("hello", candidates)
    assert selected == []


def test_select_fallback_returns_empty_for_no_candidates():
    router = Router()
    selected = router.select_nodes("anything", [])
    assert selected == []


def test_select_fallback_returns_top_one_if_marginal():
    router = Router()
    candidates = [
        ("node-a", 0.25, "somewhat relevant"),
        ("node-b", 0.10, "barely relevant"),
    ]
    selected = router.select_nodes("obscure topic", candidates)
    assert selected == ["node-a"]


def test_select_with_llm_client():
    """Test select_nodes with a mock LLM client that returns JSON."""

    def mock_client(messages):
        return '{"selected": ["node-a", "node-c"], "rationale": "both needed"}'

    router = Router(config=RouterConfig(fallback_behavior="llm"), client=mock_client)
    candidates = [
        ("node-a", 0.9, "identity"),
        ("node-b", 0.7, "tools"),
        ("node-c", 0.5, "safety"),
    ]
    selected = router.select_nodes("identity and safety rules", candidates)
    assert selected == ["node-a", "node-c"]


def test_select_with_llm_respects_max_select():
    """LLM output should be capped by router max_select."""

    def mock_client(messages):
        return (
            '{"selected": ["node-a", "node-b", "node-c", "node-d", "node-e", "node-f"],'
            '"rationale": "too many"}'
        )

    router = Router(config=RouterConfig(fallback_behavior="llm", max_select=4), client=mock_client)
    candidates = [
        ("node-a", 0.9, "identity"),
        ("node-b", 0.8, "tools"),
        ("node-c", 0.7, "safety"),
        ("node-d", 0.6, "notes"),
        ("node-e", 0.5, "extra"),
        ("node-f", 0.4, "more"),
    ]
    selected = router.select_nodes("identity and safety rules", candidates)
    assert selected == ["node-a", "node-b", "node-c", "node-d"]


def test_select_with_llm_returns_zero():
    """LLM can return empty selection."""

    def mock_client(messages):
        return '{"selected": [], "rationale": "trivial greeting"}'

    router = Router(config=RouterConfig(fallback_behavior="llm"), client=mock_client)
    candidates = [("node-a", 0.5, "stuff")]
    selected = router.select_nodes("hi", candidates)
    assert selected == []


def test_select_with_llm_filters_invalid_ids():
    """LLM returns an ID that's not in candidates — filter it out."""

    def mock_client(messages):
        return '{"selected": ["node-a", "node-FAKE"], "rationale": "test"}'

    router = Router(config=RouterConfig(fallback_behavior="llm"), client=mock_client)
    candidates = [("node-a", 0.9, "real node")]
    selected = router.select_nodes("test", candidates)
    assert selected == ["node-a"]


def test_select_fallback_filters_negative_edges():
    graph = Graph()
    graph.add_node(Node(id="current", content="Current context"))
    graph.add_node(Node(id="avoid", content="Avoided node"))
    graph.add_node(Node(id="safe", content="Safe node"))
    graph.add_edge(Edge(source="current", target="avoid", weight=-0.2))
    graph.add_edge(Edge(source="current", target="safe", weight=0.4))

    router = Router()
    candidates = [("avoid", 0.9, "should be skipped"), ("safe", 0.8, "safe pick")]

    selected = router.select_nodes("query", candidates, current_node_id="current", graph=graph)

    assert selected == ["safe"]


def test_select_falls_back_on_error():
    """If LLM errors, falls back to heuristic."""

    def bad_client(messages):
        raise RuntimeError("API down")

    router = Router(
        config=RouterConfig(fallback_behavior="heuristic", max_retries=0),
        client=bad_client,
    )
    candidates = [("node-a", 0.9, "relevant")]
    selected = router.select_nodes("query", candidates)
    assert "node-a" in selected  # Fallback should still return it


def test_parse_select_json():
    router = Router()
    raw = '{"selected": ["a", "b"], "rationale": "both needed"}'
    payload = router.parse_select_json(raw)
    assert payload["selected"] == ["a", "b"]


def test_parse_select_json_markdown_wrapped():
    router = Router()
    raw = '```json\n{"selected": ["a"], "rationale": "ok"}\n```'
    payload = router.parse_select_json(raw)
    assert payload["selected"] == ["a"]
