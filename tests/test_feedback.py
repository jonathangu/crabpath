from __future__ import annotations

import builtins

import pytest
from crabpath import Edge, Graph, Node, SynaptogenesisConfig, SynaptogenesisState
from crabpath.feedback import (
    auto_feedback,
    detect_correction,
    no_reward_on_missing_signal,
    score_retrieval,
)


def test_detect_correction_strong_and_mild_patterns() -> None:
    assert detect_correction("No, that's wrong", "answer") == -1.0
    assert detect_correction("actually, should be B", "answer") == -1.0
    assert detect_correction("don't do that", "answer") == -0.5


def test_score_retrieval_parses_llm_json() -> None:
    nodes = [
        ("node-a", "Alpha node content"),
        ("node-b", "Beta node content"),
    ]

    def llm(prompt: str, system: str) -> str:
        return (
            '{"scores": {"node-a": 1.0, "node-b": -0.5}, "overall": 0.75}'
        )

    result = score_retrieval(
        "What is alpha?",
        nodes,
        "Here is the answer.",
        llm,
    )
    assert result["scores"] == {"node-a": 1.0, "node-b": -0.5}
    assert result["overall"] == 0.75


def test_score_retrieval_default_on_invalid_response() -> None:
    def llm(prompt: str, system: str) -> str:
        return "not json"

    result = score_retrieval(
        "query",
        [("x", "x"), ("y", "y")],
        "answer",
        llm,
    )
    assert result["scores"] == {"x": 0.0, "y": 0.0}
    assert result["overall"] == 0.0


def test_score_retrieval_requires_openai_when_scoring(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "openai":
            raise ImportError("no module named 'openai'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(
        ImportError,
        match="pip install openai required for LLM scoring. Use --no-score to skip\\.",
    ):
        score_retrieval("query", [("n", "n content")], "answer")


def test_auto_feedback_applies_negative_correction() -> None:
    graph = Graph()
    graph.add_node(Node(id="a", content="a"))
    graph.add_node(Node(id="b", content="b"))
    graph.add_edge(Edge(source="a", target="b", weight=1.0))

    result = auto_feedback(
        query="query",
        user_followup="No, that's not right",
        trajectory=["a", "b"],
        graph=graph,
        syn_state=SynaptogenesisState(),
        config=SynaptogenesisConfig(correction_decay=0.5),
    )

    assert result["action"] == "record_correction"
    assert graph.get_edge("a", "b").weight == 0.5


def test_auto_feedback_applies_implicit_positive() -> None:
    graph = Graph()
    graph.add_node(Node(id="a", content="a"))
    graph.add_node(Node(id="b", content="b"))
    graph.add_edge(Edge(source="a", target="b", weight=1.0))

    result = auto_feedback(
        query="query",
        user_followup="looks good to me",
        trajectory=["a", "b"],
        graph=graph,
        syn_state=SynaptogenesisState(),
    )

    assert result["action"] == "record_cofiring"
    assert result["implicit_reward"] is None
    assert graph.get_edge("a", "b").weight > 1.0


def test_no_reward_on_missing_signal() -> None:
    assert no_reward_on_missing_signal(correction=0.0, retrieval_helpfulness=None) is None
    assert no_reward_on_missing_signal(
        correction=0.0, retrieval_helpfulness={"a": 0.29}, min_helpfulness=0.3
    ) is None
    assert no_reward_on_missing_signal(
        correction=0.0, retrieval_helpfulness={"a": 0.3}, min_helpfulness=0.3
    ) == 0.3  # marginal positive
    assert no_reward_on_missing_signal(
        correction=0.0, retrieval_helpfulness={"a": 0.6}, min_helpfulness=0.3
    ) == 0.6  # max helpful node
    assert no_reward_on_missing_signal(
        correction=0.0, retrieval_helpfulness={"a": 0.6, "b": 0.2}, min_helpfulness=0.3
    ) == 0.6  # max score gate on best chunk
    assert no_reward_on_missing_signal(
        correction=0.0, retrieval_helpfulness=0.4, min_helpfulness=0.3
    ) == 0.4  # scalar input returns actual score when above gate
    # Negative scores flow through
    assert no_reward_on_missing_signal(
        correction=0.0, retrieval_helpfulness={"good": 1.0, "bad": -1.0}
    ) == -1.0  # harmful node takes priority
    assert no_reward_on_missing_signal(
        correction=0.0, retrieval_helpfulness={"a": -0.8, "b": 0.3}
    ) == -0.8  # harmful dominates
    assert no_reward_on_missing_signal(correction=-1.0, retrieval_helpfulness=0.9) == -1.0
