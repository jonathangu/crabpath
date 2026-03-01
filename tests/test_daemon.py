from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import importlib.util

from openclawbrain import Edge, Graph, HashEmbedder, Node, VectorIndex, save_state
from openclawbrain import daemon as daemon_module
from openclawbrain.journal import read_journal
from openclawbrain.store import load_state


def _write_state(path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta", metadata={"file": "b.md"}))
    graph.add_edge(Edge("a", "b", weight=0.5, kind="sibling", metadata={"source": "unit"}))

    embedder = HashEmbedder()
    index = VectorIndex()
    index.upsert("a", embedder.embed("alpha"))
    index.upsert("b", embedder.embed("beta"))

    save_state(
        graph=graph,
        index=index,
        path=path,
        meta={"embedder_name": "hash-v1", "embedder_dim": embedder.dim},
    )


def _start_daemon(state_path: Path, auto_save_interval: int = 10) -> subprocess.Popen:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openclawbrain",
            "daemon",
            "--state",
            str(state_path),
            "--auto-save-interval",
            str(auto_save_interval),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )


def _call(proc: subprocess.Popen, method: str, params: dict | None = None, req_id: str = "1") -> dict:
    assert proc.stdin is not None
    assert proc.stdout is not None
    req = {"id": req_id, "method": method, "params": params or {}}
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    response = proc.stdout.readline()
    assert response
    return json.loads(response)


def _shutdown_daemon(proc: subprocess.Popen) -> dict:
    if proc.poll() is not None:
        return {"result": {"shutdown": True}}
    response = _call(proc, "shutdown", {}, req_id="shutdown")
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    return response


def _load_query_brain_module():
    spec = importlib.util.spec_from_file_location("oc_query_brain", Path("examples/openclaw_adapter/query_brain.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load query_brain module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_daemon_responds_to_health(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    proc = _start_daemon(state_path)
    try:
        response = _call(proc, "health", req_id="health-1")
        assert response["id"] == "health-1"
        assert "result" in response
        assert "dormant_pct" in response["result"]
        assert "nodes" in response["result"]
        assert response["result"]["nodes"] == 2
    finally:
        _shutdown_daemon(proc)


def test_daemon_responds_to_info(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    proc = _start_daemon(state_path)
    try:
        response = _call(proc, "info", req_id="info-1")
        assert response["id"] == "info-1"
        result = response["result"]
        assert result["nodes"] == 2
        assert result["edges"] == 1
        assert result["embedder"] == "hash-v1"
    finally:
        _shutdown_daemon(proc)


def test_daemon_query_returns_fired_nodes(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    proc = _start_daemon(state_path)
    try:
        response = _call(proc, "query", {"query": "alpha", "top_k": 2}, req_id="query-1")
        result = response["result"]
        assert response["id"] == "query-1"
        assert "fired_nodes" in result
        assert result["fired_nodes"]
        assert "a" in result["fired_nodes"]
        assert result["seeds"]

        # Deterministic appendix block for prompt caching.
        assert "prompt_context" in result
        assert result["prompt_context"].startswith("[BRAIN_CONTEXT v1]")
        assert "- node:" in result["prompt_context"]
        assert "alpha" in result["prompt_context"]
        assert result["prompt_context_len"] == len(result["prompt_context"])
        assert result["prompt_context_max_chars"] == 30000
        assert isinstance(result["prompt_context_trimmed"], bool)
        assert isinstance(result["prompt_context_included_node_ids"], list)
        assert isinstance(result["prompt_context_dropped_node_ids"], list)
        assert isinstance(result["prompt_context_dropped_count"], int)

        assert isinstance(result["embed_query_ms"], float)
        assert isinstance(result["traverse_ms"], float)
        assert isinstance(result["total_ms"], float)
    finally:
        _shutdown_daemon(proc)

    journal_entries = read_journal(journal_path=str(tmp_path / "journal.jsonl"))
    query_entries = [entry for entry in journal_entries if entry.get("type") == "query"]
    assert query_entries
    metadata = query_entries[-1]["metadata"]
    assert metadata["prompt_context_len"] == len(result["prompt_context"])
    assert metadata["prompt_context_max_chars"] == 30000
    assert isinstance(metadata["prompt_context_trimmed"], bool)


def test_daemon_unknown_method_returns_error(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    proc = _start_daemon(state_path)
    try:
        response = _call(proc, "no-such-method", req_id="bad-1")
        assert "error" in response
        assert response["error"]["code"] == -32601
    finally:
        _shutdown_daemon(proc)


def test_daemon_shutdown_saves_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    graph, _, _ = load_state(str(state_path))
    before_weight = graph._edges["a"]["b"].weight

    proc = _start_daemon(state_path)
    try:
        learn = _call(proc, "learn", {"fired_nodes": ["a", "b"], "outcome": 1.0}, req_id="learn-1")
        assert learn["result"]["edges_updated"] == 1
        shutdown = _shutdown_daemon(proc)
        assert shutdown["result"]["shutdown"]

    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=1)

    graph_after, _, _ = load_state(str(state_path))
    after_weight = graph_after._edges["a"]["b"].weight
    assert after_weight > before_weight


def test_daemon_inject_creates_node_and_persists(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    proc = _start_daemon(state_path)
    try:
        response = _call(
            proc,
            "inject",
            {"id": "c", "type": "TEACHING", "content": "gamma", "metadata": {"source": "unit-test"}},
            req_id="inject-1",
        )
        assert response["id"] == "inject-1"
        assert response["result"]["injected"] is True
        assert response["result"]["node_id"] == "c"
    finally:
        _shutdown_daemon(proc)

    graph, _, _ = load_state(str(state_path))
    node = graph.get_node("c")
    assert node is not None
    assert node.metadata["type"] == "TEACHING"


def test_daemon_correction_penalizes_fired_nodes_and_injects(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    proc = _start_daemon(state_path)
    try:
        query_response = _call(
            proc,
            "query",
            {"query": "alpha", "top_k": 2, "chat_id": "chat-1"},
            req_id="query-corr",
        )["result"]
        assert query_response["fired_nodes"]

        correction = _call(
            proc,
            "correction",
            {
                "chat_id": "chat-1",
                "outcome": -1.0,
                "content": "Avoid alpha in this context.",
                "lookback": 1,
            },
            req_id="corr-1",
        )["result"]
        assert correction["correction_injected"] is True
        assert correction["fired_ids_penalized"] == query_response["fired_nodes"]
        assert correction["edges_updated"] >= 1
    finally:
        _shutdown_daemon(proc)

    graph, _, _ = load_state(str(state_path))
    correction_node_id = [node.id for node in graph.nodes() if node.id.startswith("correction::") and node.metadata.get("type") == "CORRECTION"]
    assert len(correction_node_id) == 1
    corr_node = graph.get_node(correction_node_id[0])
    assert corr_node is not None
    inhibitory_targets = {edge.target for _target, edge in graph.outgoing(corr_node.id) if edge.kind == "inhibitory"}
    assert set(query_response["fired_nodes"]).intersection(inhibitory_targets)


def test_daemon_query_populates_fired_history(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    proc = _start_daemon(state_path)
    try:
        query_response = _call(proc, "query", {"query": "alpha", "top_k": 2, "chat_id": "chat-history"}, req_id="query-1")["result"]
        correction_response = _call(
            proc,
            "correction",
            {"chat_id": "chat-history", "outcome": -1.0, "lookback": 3},
            req_id="corr-1",
        )["result"]
        assert correction_response["fired_ids_penalized"] == query_response["fired_nodes"]
    finally:
        _shutdown_daemon(proc)


def test_daemon_fired_log_ring_buffer_tracks_per_chat_id() -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"file": "a.md"}))
    index = VectorIndex()
    hash_embedder = HashEmbedder()
    index.upsert("a", hash_embedder.embed("alpha"))
    daemon_state = daemon_module._DaemonState(
        graph=graph,
        index=index,
        meta={"embedder_name": "hash-v1", "embedder_dim": hash_embedder.dim},
        fired_log={},
    )

    for count in range(105):
        daemon_module._append_fired_history(
            daemon_state=daemon_state,
            chat_id="chat-1",
            query=f"q-{count}",
            fired_nodes=["a"],
            timestamp=float(count),
        )
    assert len(daemon_state.fired_log["chat-1"]) == 100
    assert daemon_state.fired_log["chat-1"][0]["query"] == "q-5"

    for count in range(1205):
        daemon_module._append_fired_history(
            daemon_state=daemon_state,
            chat_id=f"chat-{count % 20}",
            query=f"q-global-{count}",
            fired_nodes=["a"],
            timestamp=float(count + 200),
        )
    assert daemon_module._fired_history_size(daemon_state) == 1000


def test_query_brain_uses_state_embedder_for_embeddings(monkeypatch) -> None:
    query_module = _load_query_brain_module()

    hash_vector = query_module._embed_fn_from_state(
        {"embedder_name": "hash-v1", "embedder_dim": HashEmbedder().dim}
    )[0]("probe")
    assert len(hash_vector) == HashEmbedder().dim

    class FakeOpenAIResponse:
        data = [type("item", (), {"embedding": [0.1, 0.2, 0.3]})()]

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        class embeddings:  # type: ignore
            @staticmethod
            def create(model: str, input: list[str]) -> FakeOpenAIResponse:
                return FakeOpenAIResponse()

    import openai as openai_mod
    monkeypatch.setattr(openai_mod, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(query_module, "require_api_key", lambda: "test-key")

    openai_vector, embedder_name = query_module._embed_fn_from_state({"embedder_name": "text-embedding-3-small"})
    assert openai_vector("probe") == [0.1, 0.2, 0.3]
    assert embedder_name == "text-embedding-3-small"
