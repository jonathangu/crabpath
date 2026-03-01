from __future__ import annotations

from openclawbrain.graph import Graph, Node
from openclawbrain.prompt_context import build_prompt_context


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
