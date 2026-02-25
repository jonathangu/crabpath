from __future__ import annotations

from crabpath.graph import Edge, Graph, Node


def build_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node(id="query-root", content="How do I open the door?"))
    graph.add_node(Node(id="unsafe", content="Do not open the door unless checked."))
    graph.add_node(Node(id="safe", content="Open the door safely and verify constraints."))
    graph.add_node(Node(id="notes", content="Use the safe door-open workflow before testing."))

    graph.add_edge(Edge(source="query-root", target="unsafe", weight=0.70))
    graph.add_edge(Edge(source="query-root", target="safe", weight=0.68))
    graph.add_edge(Edge(source="safe", target="notes", weight=0.60))
    return graph


def main() -> None:
    build_graph()


if __name__ == "__main__":
    main()
