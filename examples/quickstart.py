"""Quickstart for bootstrapping and querying the Acme Bot toy workspace."""

from pathlib import Path

from crabpath import Graph, MemoryController
from crabpath.migrate import fallback_llm_split, gather_files
from crabpath.mitosis import MitosisState, bootstrap_workspace

WORKSPACE_PATH = Path(__file__).parent / "toy_workspace"
GRAPH_PATH = Path(__file__).with_name("toy_workspace_graph.json")


def build_toy_graph(workspace: Path, graph_path: Path) -> Graph:
    workspace_files = gather_files(workspace)
    graph = Graph()
    bootstrap_workspace(
        graph=graph,
        workspace_files=workspace_files,
        llm_call=fallback_llm_split,
        state=MitosisState(),
    )
    graph.save(str(graph_path))
    return graph


def main() -> None:
    workspace = WORKSPACE_PATH
    graph_path = GRAPH_PATH

    if graph_path.exists():
        graph = Graph.load(str(graph_path))
    else:
        graph = build_toy_graph(workspace, graph_path)

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
