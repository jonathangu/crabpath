"""LangChain adapter examples for CrabPath.

Note: LangChain is NOT a dependency of this package. Install it separately to run
this file (for example, `pip install langchain`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
else:
    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    except Exception:  # pragma: no cover - optional dependency path
        class Document:
            def __init__(self, page_content: str, metadata: dict[str, Any] | None = None) -> None:
                self.page_content = page_content
                self.metadata = metadata or {}

        class BaseRetriever:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "Install langchain-core to use CrabPathRetriever as a retriever."
                )

            def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
                raise RuntimeError("langchain-core is required for this adapter.")

        BaseRetriever.__name__ = "BaseRetriever"
        Document.__name__ = "Document"

from crabpath import Edge, Graph, MemoryController, Node


class CrabPathRetriever(BaseRetriever):
    """Minimal LangChain `BaseRetriever` backed by `MemoryController`."""

    def __init__(self, graph_path: str, index_path: str, top_k: int = 8) -> None:
        self.graph_path = graph_path
        self.index_path = index_path
        self.top_k = top_k
        self.graph = Graph.load(graph_path) if Path(graph_path).exists() else Graph()
        self.controller = MemoryController(self.graph)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        del run_manager
        if not isinstance(self.graph, Graph):
            raise RuntimeError(
                "langchain-core is required for CrabPathRetriever runtime usage."
            )
        result = self.controller.query(query)
        docs: list[Document] = []

        for node_id in result.selected_nodes[: self.top_k]:
            node = self.graph.get_node(node_id)
            if node is None:
                continue
            docs.append(
                Document(
                    page_content=node.content,
                    metadata={"node_id": node_id, "query": query},
                )
            )
        return docs


def main() -> None:
    graph_path = "examples_memory_graph.json"
    index_path = "examples_memory_embeddings.json"

    if not Path(graph_path).exists():
        bootstrap = Graph()
        bootstrap.add_node(Node(id="deploy", content="Deploy the service safely"))
        bootstrap.add_node(Node(id="rollback", content="Rollback if needed"))
        bootstrap.add_edge(Edge(source="deploy", target="rollback", weight=0.8))
        bootstrap.save(graph_path)

    retriever = CrabPathRetriever(graph_path=graph_path, index_path=index_path, top_k=3)

    try:
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install langchain and an LLM provider to run this demo.") from exc

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    print(qa.run("What should I check when deploy fails?"))


if __name__ == "__main__":
    main()
