"""Tests for `_structural_utils` helpers."""

from __future__ import annotations

import json

import pytest

from crabpath import Edge, Graph, Node
from crabpath._structural_utils import (
    classify_edge_tier,
    count_cross_file_edges,
    node_file_id,
    parse_markdown_json,
    split_fallback_sections,
)


def test_parse_markdown_json_accepts_valid_json():
    payload = parse_markdown_json('{"mode": "known"}')
    assert payload == {"mode": "known"}


def test_parse_markdown_json_handles_fenced_markdown():
    payload = parse_markdown_json("```json\n{\"ok\": true}\n```")
    assert payload == {"ok": True}


def test_parse_markdown_json_raises_on_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        parse_markdown_json("{oops}")


def test_parse_markdown_json_fails_on_mixed_markdown():
    with pytest.raises(json.JSONDecodeError):
        parse_markdown_json("```json\n{\"ok\": true}\n```\nextra text")


def test_split_fallback_sections_prefers_headers():
    content = """
## One
first paragraph

## Two
second paragraph
"""
    sections = split_fallback_sections(content)
    assert sections == ["## One\nfirst paragraph", "## Two\nsecond paragraph"]


def test_split_fallback_sections_fallback_to_paragraphs():
    content = "A short paragraph.\n\nA second paragraph."
    sections = split_fallback_sections(content, merge_short_paragraphs=200)
    assert sections == ["A short paragraph.", "A second paragraph."]


def test_split_fallback_sections_merges_short_paragraphs():
    content = "A\n\nB\n\nC\n\nD"
    sections = split_fallback_sections(content, merge_short_paragraphs=3)
    assert sections == ["A\n\nB", "C\n\nD"]
    assert sections[0] == "A\n\nB"


def test_node_file_id_basic_and_fallback():
    assert node_file_id("foo.py::123") == "foo.py"
    assert node_file_id("single_node") == "single_node"


def test_count_cross_file_edges_counts_only_cross_file():
    graph = Graph()
    graph.add_node(Node(id="a.py::alpha", content="a"))
    graph.add_node(Node(id="a.py::beta", content="b"))
    graph.add_node(Node(id="b.py::gamma", content="c"))
    graph.add_edge(Edge(source="a.py::alpha", target="a.py::beta", weight=0.8))
    graph.add_edge(Edge(source="a.py::alpha", target="b.py::gamma", weight=0.8))

    assert count_cross_file_edges(graph) == 1


def test_count_cross_file_edges_for_small_graph_is_zero():
    assert count_cross_file_edges(Graph()) == 0


def test_classify_edge_tier_uses_thresholds():
    assert classify_edge_tier(0.8) == "reflex"
    assert classify_edge_tier(0.79, reflex_threshold=0.8) == "habitual"
    assert classify_edge_tier(0.2) == "dormant"
    assert classify_edge_tier(0.45, reflex_threshold=0.9, dormant_threshold=0.2) == "habitual"
