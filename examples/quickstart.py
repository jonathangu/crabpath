"""Quickstart for bootstrapping and querying the Acme Bot toy workspace."""

from pathlib import Path

from crabpath import Graph, MemoryController
from crabpath.mitosis import bootstrap_workspace


def main() -> None:
    workspace = Path(__file__).parent / "toy_workspace"
    graph_path = Path(__file__).with_name("toy_workspace_graph.json")
    bootstrap_workspace(workspace, graph_path)

    graph = Graph.load(str(graph_path))
    controller = MemoryController(graph)
    queries = [
        "How do we recover from a deployment timeout?",
        "What are Acme Bot safety guardrails during rollback?",
        "Which API calls are available for operators?",
    ]
    for q in queries:
        result = controller.query(q)
        print(f"query={q}")
        print("selected", result.selected_nodes)
        print("context:\n", result.context)


if __name__ == "__main__":
    main()
