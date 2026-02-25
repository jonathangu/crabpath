from __future__ import annotations

from crabpath.graph import Edge, Graph, Node


def build_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node(id="query-root", content="What is the codeword?"))
    graph.add_node(Node(id="giraffe", content="The codeword is giraffe."))
    graph.add_node(Node(id="elephant", content="The codeword is elephant."))
    graph.add_node(Node(id="guard", content="Remember the current codeword exactly."))

    graph.add_edge(Edge(source="query-root", target="giraffe", weight=0.70))
    graph.add_edge(Edge(source="query-root", target="elephant", weight=0.68))
    graph.add_edge(Edge(source="giraffe", target="guard", weight=0.55))
    graph.add_edge(Edge(source="elephant", target="guard", weight=0.55))
    return graph


def main() -> None:
    build_graph()


if __name__ == "__main__":
    main()
