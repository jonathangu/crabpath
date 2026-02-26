from __future__ import annotations

import json
from pathlib import Path

from crabpath.cli import main
from crabpath.decay import DecayConfig, apply_decay
from crabpath.graph import Edge, Graph, Node
from crabpath.learn import apply_outcome
from crabpath.split import split_workspace
from crabpath.traverse import TraversalConfig, traverse


def test_full_cycle_init_query_learn_query(tmp_path, capsys) -> None:
    workspace = tmp_path / "docs"
    workspace.mkdir()
    (workspace / "note.md").write_text(
        "## Alpha\nalpha content\n\n## Beta\nbeta content",
        encoding="utf-8",
    )
    graph_path = tmp_path / "state.json"

    assert main(["init", "--workspace", str(workspace), "--output", str(graph_path)]) == 0
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert len(payload["graph"]["nodes"]) == 2
    capsys.readouterr()

    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "note.md::0": [1.0, 0.0],
                "note.md::1": [0.0, 1.0],
            },
        )
    )

    # first query
    code = main(
        [
            "query",
            "seed",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "2",
            "--json",
            "--query-vector",
            "1,0",
        ]
    )
    assert code == 0
    first = json.loads(capsys.readouterr().out.strip())
    first_fired = first["fired"]

    before = payload["graph"]["edges"][0]["weight"]

    code = main(["learn", "--graph", str(graph_path), "--outcome", "1.0", "--fired-ids", ",".join(first_fired[:2]), "--json"])
    assert code == 0

    updated = json.loads(graph_path.read_text(encoding="utf-8"))
    after = updated["graph"]["edges"][0]["weight"]
    assert after > before

    capsys.readouterr()
    code = main(
        [
            "query",
            "seed",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "2",
            "--json",
            "--query-vector",
            "1,0",
        ]
    )
    assert code == 0
    second = json.loads(capsys.readouterr().out.strip())
    assert second["fired"][0] == "note.md::0"


def test_edge_damping_simulation_converges() -> None:
    g = Graph()
    g.add_node(Node("a", "A"))
    g.add_node(Node("b", "B"))
    g.add_edge(Edge("a", "b", 1.0))
    g.add_edge(Edge("b", "a", 1.0))

    result = traverse(g, [("a", 1.0)], config=TraversalConfig(max_hops=100, beam_width=1, edge_damping=0.5))
    assert len(result.steps) == 100
    assert all(step.effective_weight > 0 for step in result.steps)
    assert result.steps[-1].effective_weight < 1e-9


def test_negation_learning_suppresses_bad_node() -> None:
    g = Graph()
    g.add_node(Node("good", "good"))
    g.add_node(Node("bad", "bad"))
    g.add_edge(Edge("good", "bad", 0.05))

    apply_outcome(g, ["good", "bad"], outcome=-1.0)
    assert g._edges["good"]["bad"].kind == "inhibitory"
    assert g._edges["good"]["bad"].weight <= 0.06


def test_decay_and_learning_interaction() -> None:
    g = Graph()
    g.add_node(Node("a", "A"))
    g.add_node(Node("b", "B"))
    g.add_edge(Edge("a", "b", 0.5))

    apply_outcome(g, ["a", "b"], outcome=1.0)
    learned = g._edges["a"]["b"].weight
    changed = apply_decay(g, config=DecayConfig(half_life=1, min_weight=0.0))

    decayed = g._edges["a"]["b"].weight
    apply_outcome(g, ["a", "b"], outcome=1.0)
    final = g._edges["a"]["b"].weight

    assert changed == 1
    assert decayed < final < learned
    assert final < learned + 1


def test_large_workspace_pipeline_end_to_end(tmp_path, capsys) -> None:
    workspace = tmp_path / "docs"
    workspace.mkdir()

    for file_idx in range(50):
        chunks = []
        for chunk_idx in range(10):
            chunks.append(f"## Section {chunk_idx}\nFile {file_idx} chunk {chunk_idx}")
        (workspace / f"doc_{file_idx}.md").write_text("\n\n".join(chunks), encoding="utf-8")

    payload_path = tmp_path / "state.json"
    assert main(["init", "--workspace", str(workspace), "--output", str(payload_path)]) == 0

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert len(payload["graph"]["nodes"]) >= 500
    capsys.readouterr()

    index_path = tmp_path / "index.json"
    index_payload = {
        node["id"]: [float(idx % 2), float((idx + 1) % 2)]
        for idx, node in enumerate(payload["graph"]["nodes"])
    }
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")

    code = main(
        [
            "query",
            "search",
            "--graph",
            str(payload_path),
            "--index",
            str(index_path),
            "--top",
            "5",
            "--json",
            "--query-vector",
            "1,0",
        ]
    )
    assert code == 0
    query_out = json.loads(capsys.readouterr().out.strip())
    assert query_out["fired"]

    first_ids = ",".join(query_out["fired"][:2])
    learn_code = main(["learn", "--graph", str(payload_path), "--outcome", "1.0", "--fired-ids", first_ids])
    assert learn_code == 0

    health_code = main(["health", "--graph", str(payload_path), "--json"])
    assert health_code == 0
