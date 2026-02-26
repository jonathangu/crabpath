"""The smallest possible shadow-mode loop."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from crabpath import Edge, Graph, MemoryController, Node

GRAPH_PATH = Path("shadow_example_graph.json")
SHADOW_LOG = Path.home() / ".crabpath" / "shadow.log"


def main() -> None:
    if GRAPH_PATH.exists():
        graph = Graph.load(str(GRAPH_PATH))
    else:
        graph = Graph()
        graph.add_node(Node(id="check", content="Check deployment logs"))
        graph.add_node(Node(id="roll", content="Rollback if needed"))
        graph.add_edge(Edge("check", "roll", 0.6))
        graph.save(str(GRAPH_PATH))

    controller = MemoryController(graph)
    query = " ".join(sys.argv[1:]) or "deployment issue"
    result = controller.query(query)

    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SHADOW_LOG.open("a", encoding="utf-8") as f:
        payload = {"ts": time.time(), "query": query, "selected_nodes": result.selected_nodes}
        f.write(json.dumps(payload) + "\n")

    controller.learn(result, 1.0)
    graph.save(str(GRAPH_PATH))
    print(result.context)


if __name__ == "__main__":
    main()
