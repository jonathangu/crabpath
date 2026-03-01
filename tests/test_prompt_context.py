from __future__ import annotations

from openclawbrain.graph import Graph, Node
from openclawbrain.prompt_context import (
    build_prompt_context,
    build_prompt_context_ranked_with_stats,
    build_prompt_context_with_stats,
)


def test_build_prompt_context_is_deterministic_across_input_order() -> None:
    """Prompt block is stable regardless of incoming node_ids order."""
    graph = Graph()
    graph.add_node(
        Node(
            "z",
            "zeta",
            metadata={"file": "docs/a.md", "start_line": 10, "end_line": 20},
        )
    )
    graph.add_node(
        Node(
            "a",
            "alpha",
            metadata={"file": "docs/a.md", "start_line": 2, "end_line": 5},
        )
    )
    graph.add_node(Node("m", "mu"))

    rendered_one = build_prompt_context(graph=graph, node_ids=["z", "m", "a"])
    rendered_two = build_prompt_context(graph=graph, node_ids=["a", "z", "m"])

    assert rendered_one == rendered_two

    first_entry_idx = rendered_one.find("- node: a")
    second_entry_idx = rendered_one.find("- node: z")
    third_entry_idx = rendered_one.find("- node: m")
    assert first_entry_idx < second_entry_idx < third_entry_idx


def test_build_prompt_context_includes_citation_with_source_lines() -> None:
    """Citation line uses source path and line span when metadata is present."""
    graph = Graph()
    graph.add_node(
        Node(
            "deploy.md::0",
            "Deploy with CI checks enabled.",
            metadata={"path": "workspace/deploy.md", "start_line": 41, "end_line": 63},
        )
    )

    rendered = build_prompt_context(graph=graph, node_ids=["deploy.md::0"])

    assert "[BRAIN_CONTEXT v1]" in rendered
    assert "- node: deploy.md::0" in rendered
    assert "  source: workspace/deploy.md#L41-L63" in rendered
    assert "  Deploy with CI checks enabled." in rendered
    assert rendered.endswith("[/BRAIN_CONTEXT]")


def test_build_prompt_context_with_stats_reports_trimmed_and_dropped_ids() -> None:
    """Stats include deterministic included/dropped ids when budget trims context."""
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"file": "docs/a.md", "start_line": 1}))
    graph.add_node(Node("b", "beta", metadata={"file": "docs/a.md", "start_line": 2}))
    graph.add_node(Node("c", "gamma", metadata={"file": "docs/a.md", "start_line": 3}))

    rendered, stats = build_prompt_context_with_stats(
        graph=graph,
        node_ids=["c", "a", "b"],
        max_chars=110,
    )

    assert rendered.startswith("[BRAIN_CONTEXT v1]")
    assert stats["prompt_context_trimmed"] is True
    assert stats["prompt_context_max_chars"] == 110
    assert stats["prompt_context_len"] == len(rendered)
    assert stats["prompt_context_included_node_ids"] == ["a"]
    assert stats["prompt_context_dropped_node_ids"] == ["b", "c"]
    assert stats["prompt_context_dropped_count"] == 2
    assert stats["prompt_context_dropped_node_ids_truncated"] is False


def test_ranked_prompt_context_prefers_authority_then_score_then_source_order() -> None:
    """Ranked builder prioritizes authority and score before source/path order."""
    graph = Graph()
    graph.add_node(
        Node(
            "overlay-high",
            "overlay high",
            metadata={"authority": "overlay", "file": "docs/a.md", "start_line": 1},
        )
    )
    graph.add_node(
        Node(
            "canonical-low",
            "canonical low",
            metadata={"authority": "canonical", "file": "docs/z.md", "start_line": 99},
        )
    )
    graph.add_node(
        Node(
            "canonical-high",
            "canonical high",
            metadata={"authority": "canonical", "file": "docs/y.md", "start_line": 50},
        )
    )
    graph.add_node(
        Node(
            "constitutional",
            "constitutional top",
            metadata={"authority": "constitutional", "file": "docs/z.md", "start_line": 200},
        )
    )

    rendered, stats = build_prompt_context_ranked_with_stats(
        graph=graph,
        node_ids=["overlay-high", "canonical-low", "canonical-high", "constitutional"],
        node_scores={
            "overlay-high": 0.99,
            "canonical-low": 0.2,
            "canonical-high": 0.9,
            "constitutional": 0.1,
        },
        max_chars=240,
    )

    # Authority outranks score and source ordering.
    assert stats["prompt_context_included_node_ids"] == ["constitutional", "canonical-high", "canonical-low"]
    assert stats["prompt_context_dropped_node_ids"] == ["overlay-high"]
    assert stats["prompt_context_dropped_authority_counts"] == {"overlay": 1}
    assert rendered.find("- node: constitutional") < rendered.find("- node: canonical-high")
