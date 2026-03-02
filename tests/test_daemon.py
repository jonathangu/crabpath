from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
import importlib

from openclawbrain import Edge, Graph, HashEmbedder, Node, VectorIndex, save_state
from openclawbrain.traverse import TraversalResult
import openclawbrain.daemon as daemon_module
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
    return importlib.import_module("openclawbrain.openclaw_adapter.query_brain")


def test_resolve_embed_fn_auto_uses_hash_for_hash_state() -> None:
    resolved = daemon_module._resolve_embed_fn(
        "auto",
        {"embedder_name": "hash-v1", "embedder_dim": 1024},
    )
    assert resolved is None


def test_resolve_embed_fn_auto_does_not_use_openai_for_openai_state() -> None:
    resolved = daemon_module._resolve_embed_fn(
        "auto",
        {"embedder_name": "openai-text-embedding-3-small", "embedder_dim": 1536},
    )
    assert resolved is None


def test_resolve_embed_fn_auto_uses_local_for_local_state(monkeypatch) -> None:
    fake_embed = lambda _text: [0.1, 0.2]
    seen_models: list[str] = []

    def _fake_make_local_embed_fn(model_name: str):
        seen_models.append(model_name)
        return fake_embed

    monkeypatch.setattr(daemon_module, "_make_local_embed_fn", _fake_make_local_embed_fn)
    resolved = daemon_module._resolve_embed_fn(
        "auto",
        {"embedder_name": "local:bge-small-en-v1.5", "embedder_dim": 384},
    )
    assert callable(resolved)
    assert resolved is fake_embed
    assert seen_models == ["BAAI/bge-small-en-v1.5"]


def test_daemon_parser_defaults_route_mode_learned() -> None:
    parser = daemon_module._build_parser()
    args = parser.parse_args(["--state", "/tmp/state.json"])
    assert args.route_mode == "learned"


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
        assert result["max_fired_nodes"] == 30
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
    assert metadata["max_fired_nodes"] == 30
    assert isinstance(metadata["prompt_context_trimmed"], bool)


def test_query_route_fn_scoring_is_deterministic_with_target_tiebreak() -> None:
    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])
    index.upsert("b", [1.0, 0.0])
    index.upsert("c", [0.0, 1.0])

    route_fn = daemon_module._build_query_route_fn(
        route_mode="edge+sim",
        route_top_k=2,
        route_alpha_sim=0.5,
        route_use_relevance=True,
        query_vector=[1.0, 0.0],
        index=index,
    )
    assert route_fn is not None

    candidates = [
        Edge("src", "b", weight=0.4, metadata={"relevance": 0.0}),
        Edge("src", "a", weight=0.4, metadata={"relevance": 0.0}),
        Edge("src", "c", weight=0.4, metadata={"relevance": 0.0}),
    ]
    chosen_first = route_fn("src", candidates, "q")
    chosen_second = route_fn("src", list(reversed(candidates)), "q")

    # a/b tie on score; target_id breaks tie deterministically.
    assert chosen_first == ["a", "b"]
    assert chosen_second == ["a", "b"]


def test_daemon_query_route_mode_edge_sim_selects_expected_habitual_target(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("seed", "seed content", metadata={"file": "seed.md"}))
    graph.add_node(Node("good", "good answer", metadata={"file": "good.md"}))
    graph.add_node(Node("bad", "bad answer", metadata={"file": "bad.md"}))
    graph.add_edge(Edge("seed", "good", weight=0.4, metadata={"relevance": 0.0}))
    graph.add_edge(Edge("seed", "bad", weight=0.4, metadata={"relevance": 0.0}))

    index = VectorIndex()
    index.upsert("seed", [1.0, 0.0])
    index.upsert("good", [1.0, 0.0])
    index.upsert("bad", [0.0, 1.0])
    meta = {"embedder_name": "hash-v1", "embedder_dim": 2}

    response = daemon_module._handle_query(
        graph=graph,
        index=index,
        meta=meta,
        embed_fn=lambda _q: [1.0, 0.0],
        params={
            "query": "route with similarity",
            "top_k": 1,
            "route_mode": "edge+sim",
            "route_top_k": 1,
            "route_alpha_sim": 0.5,
            "route_use_relevance": True,
        },
        state_path=str(tmp_path / "state.json"),
    )

    assert "seed" in response["fired_nodes"]
    assert "good" in response["fired_nodes"]
    assert "bad" not in response["fired_nodes"]


def test_daemon_query_route_mode_learned_falls_back_to_edge_sim_without_model(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("seed", "seed content", metadata={"file": "seed.md"}))
    graph.add_node(Node("good", "good answer", metadata={"file": "good.md"}))
    graph.add_node(Node("bad", "bad answer", metadata={"file": "bad.md"}))
    graph.add_edge(Edge("seed", "good", weight=0.4, metadata={"relevance": 0.0}))
    graph.add_edge(Edge("seed", "bad", weight=0.4, metadata={"relevance": 0.0}))

    index = VectorIndex()
    index.upsert("seed", [1.0, 0.0])
    index.upsert("good", [1.0, 0.0])
    index.upsert("bad", [0.0, 1.0])
    meta = {"embedder_name": "hash-v1", "embedder_dim": 2}

    response = daemon_module._handle_query(
        graph=graph,
        index=index,
        meta=meta,
        embed_fn=lambda _q: [1.0, 0.0],
        params={
            "query": "route learned fallback",
            "top_k": 1,
            "route_mode": "learned",
            "route_top_k": 1,
            "route_alpha_sim": 0.5,
            "route_use_relevance": True,
        },
        state_path=str(tmp_path / "state.json"),
        learned_model=None,
    )

    assert "seed" in response["fired_nodes"]
    assert "good" in response["fired_nodes"]
    assert "bad" not in response["fired_nodes"]


def test_daemon_query_ranked_prompt_context_uses_authority_and_scores(monkeypatch, tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("overlay", "overlay text", metadata={"authority": "overlay", "file": "docs/a.md", "start_line": 1}))
    graph.add_node(
        Node(
            "canonical",
            "canonical text",
            metadata={"authority": "canonical", "file": "docs/z.md", "start_line": 100},
        )
    )
    graph.add_node(
        Node(
            "constitutional",
            "constitutional text",
            metadata={"authority": "constitutional", "file": "docs/z.md", "start_line": 200},
        )
    )

    index = VectorIndex()
    index.upsert("overlay", [1.0, 0.0])
    index.upsert("canonical", [0.9, 0.1])
    index.upsert("constitutional", [0.8, 0.2])
    meta = {"embedder_name": "hash-v1", "embedder_dim": 2}

    def _fake_traverse(*_args, **_kwargs):
        return TraversalResult(
            fired=["overlay", "canonical", "constitutional"],
            steps=[],
            context="",
            fired_scores={"overlay": 0.95, "canonical": 0.7, "constitutional": 0.1},
        )

    monkeypatch.setattr(daemon_module, "traverse", _fake_traverse)

    response = daemon_module._handle_query(
        graph=graph,
        index=index,
        meta=meta,
        embed_fn=lambda _q: [1.0, 0.0],
        params={
            "query": "anything",
            "top_k": 3,
            "max_prompt_context_chars": 170,
            "max_fired_nodes": 7,
            "chat_id": "chat-ranked",
        },
        state_path=str(tmp_path / "state.json"),
    )

    assert response["max_fired_nodes"] == 7
    # Authority dominates score: constitutional before canonical before overlay.
    assert response["prompt_context_included_node_ids"] == ["constitutional", "canonical"]
    assert response["prompt_context_dropped_node_ids"] == ["overlay"]
    assert response["prompt_context_dropped_authority_counts"] == {"overlay": 1}


def test_daemon_query_can_omit_prompt_context_node_id_lines(monkeypatch, tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"authority": "canonical", "file": "docs/a.md", "start_line": 1}))

    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])
    meta = {"embedder_name": "hash-v1", "embedder_dim": 2}

    def _fake_traverse(*_args, **_kwargs):
        return TraversalResult(
            fired=["a"],
            steps=[],
            context="",
            fired_scores={"a": 0.95},
        )

    monkeypatch.setattr(daemon_module, "traverse", _fake_traverse)

    response = daemon_module._handle_query(
        graph=graph,
        index=index,
        meta=meta,
        embed_fn=lambda _q: [1.0, 0.0],
        params={
            "query": "anything",
            "top_k": 1,
            "prompt_context_include_node_ids": False,
        },
        state_path=str(tmp_path / "state.json"),
    )

    assert "alpha" in response["prompt_context"]
    assert "- node:" not in response["prompt_context"]


def test_daemon_query_exclusions_affect_only_prompt_context(monkeypatch, tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node("bootstrap", "bootstrap policy", metadata={"file": "AGENTS.md", "authority": "canonical"}))
    graph.add_node(Node("work", "workspace context", metadata={"file": "docs/work.md", "authority": "canonical"}))

    index = VectorIndex()
    index.upsert("bootstrap", [1.0, 0.0])
    index.upsert("work", [0.9, 0.1])
    meta = {"embedder_name": "hash-v1", "embedder_dim": 2}

    def _fake_traverse(*_args, **_kwargs):
        return TraversalResult(
            fired=["bootstrap", "work"],
            steps=[],
            context="raw context body",
            fired_scores={"bootstrap": 0.95, "work": 0.7},
        )

    monkeypatch.setattr(daemon_module, "traverse", _fake_traverse)

    response = daemon_module._handle_query(
        graph=graph,
        index=index,
        meta=meta,
        embed_fn=lambda _q: [1.0, 0.0],
        params={
            "query": "anything",
            "top_k": 2,
            "exclude_files": ["AGENTS.md"],
            "max_prompt_context_chars": 500,
        },
        state_path=str(tmp_path / "state.json"),
    )

    assert response["fired_nodes"] == ["bootstrap", "work"]
    assert response["context"] == "raw context body"
    assert "bootstrap policy" not in response["prompt_context"]
    assert "workspace context" in response["prompt_context"]
    assert response["prompt_context_excluded_files_count"] == 1
    assert response["prompt_context_excluded_node_ids_count"] == 1


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


def test_daemon_last_fired_and_learn_by_chat_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    proc = _start_daemon(state_path)
    try:
        query_response = _call(
            proc,
            "query",
            {"query": "alpha", "top_k": 2, "chat_id": "chat-learn"},
            req_id="query-1",
        )["result"]
        assert query_response["fired_nodes"]

        last_fired = _call(
            proc,
            "last_fired",
            {"chat_id": "chat-learn", "lookback": 1},
            req_id="last-fired-1",
        )["result"]
        assert last_fired["chat_id"] == "chat-learn"
        assert last_fired["lookback"] == 1
        assert last_fired["fired_nodes"] == query_response["fired_nodes"]

        learned = _call(
            proc,
            "learn_by_chat_id",
            {"chat_id": "chat-learn", "outcome": -1.0, "lookback": 1},
            req_id="learn-chat-1",
        )["result"]
        assert learned["fired_ids_penalized"] == query_response["fired_nodes"]
        assert learned["edges_updated"] >= 1
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
        feedback_dedup_keys=set(),
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


def test_daemon_capture_feedback_injects_and_learns(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    graph_before, _index_before, _meta_before = load_state(str(state_path))
    before_weight = graph_before._edges["a"]["b"].weight

    proc = _start_daemon(state_path)
    try:
        query_response = _call(
            proc,
            "query",
            {"query": "alpha", "top_k": 2, "chat_id": "chat-feedback"},
            req_id="query-feedback-1",
        )["result"]
        assert query_response["fired_nodes"]

        feedback = _call(
            proc,
            "capture_feedback",
            {
                "chat_id": "chat-feedback",
                "kind": "CORRECTION",
                "content": "Do not use alpha for this task.",
                "lookback": 1,
            },
            req_id="capture-feedback-1",
        )["result"]
        assert feedback["deduped"] is False
        assert feedback["fired_ids_used"] == query_response["fired_nodes"]
        assert feedback["edges_updated"] >= 1
        assert feedback["outcome_used"] == -1.0
        assert str(feedback["injected_node_id"]).startswith("correction::")
    finally:
        _shutdown_daemon(proc)

    graph_after, _index_after, _meta_after = load_state(str(state_path))
    node_id = str(feedback["injected_node_id"])
    node = graph_after.get_node(node_id)
    assert node is not None
    assert node.metadata.get("type") == "CORRECTION"
    assert node.metadata.get("source") == "capture_feedback"
    assert graph_after._edges["a"]["b"].weight != before_weight


def test_daemon_capture_feedback_dedups_by_key_and_message_id_alias(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    proc = _start_daemon(state_path)
    try:
        _call(
            proc,
            "query",
            {"query": "alpha", "top_k": 2, "chat_id": "chat-dedup"},
            req_id="query-dedup-1",
        )
        first = _call(
            proc,
            "capture_feedback",
            {
                "chat_id": "chat-dedup",
                "kind": "TEACHING",
                "content": "Prefer explicit assertions in tests.",
                "outcome": 1.0,
                "lookback": 1,
                "dedup_key": "event-123",
            },
            req_id="capture-feedback-first",
        )["result"]
        assert first["deduped"] is False
        assert first["dedup_key_used"] == "event-123"
        assert str(first["injected_node_id"]).startswith("teaching::")

        second = _call(
            proc,
            "capture_feedback",
            {
                "chat_id": "chat-dedup",
                "kind": "TEACHING",
                "content": "Prefer explicit assertions in tests.",
                "outcome": 1.0,
                "lookback": 1,
                "dedup_key": "event-123",
            },
            req_id="capture-feedback-second",
        )["result"]
        assert second["deduped"] is True
        assert second["dedup_key_used"] == "event-123"
        assert second["edges_updated"] == 0
        assert "injected_node_id" not in second

        third = _call(
            proc,
            "capture_feedback",
            {
                "chat_id": "chat-dedup",
                "kind": "TEACHING",
                "content": "Prefer explicit assertions in tests.",
                "lookback": 1,
                "message_id": "event-123",
            },
            req_id="capture-feedback-third",
        )["result"]
        assert third["deduped"] is True
        assert third["dedup_key_used"] == "event-123"
    finally:
        _shutdown_daemon(proc)

    proc = _start_daemon(state_path)
    try:
        after_restart = _call(
            proc,
            "capture_feedback",
            {
                "chat_id": "chat-dedup",
                "kind": "TEACHING",
                "content": "Prefer explicit assertions in tests.",
                "dedup_key": "event-123",
            },
            req_id="capture-feedback-restart",
        )["result"]
        assert after_restart["deduped"] is True
        assert after_restart["dedup_key_used"] == "event-123"
    finally:
        _shutdown_daemon(proc)

    graph_after, _index_after, _meta_after = load_state(str(state_path))
    teaching_nodes = [
        node.id
        for node in graph_after.nodes()
        if node.metadata.get("source") == "capture_feedback" and node.metadata.get("type") == "TEACHING"
    ]
    assert len(teaching_nodes) == 1

    dedup_log = tmp_path / "injected_feedback.jsonl"
    rows = [line for line in dedup_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1


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

    try:
        import openai as openai_mod
    except ImportError:
        openai_mod = types.ModuleType("openai")
        monkeypatch.setitem(sys.modules, "openai", openai_mod)
    monkeypatch.setattr(openai_mod, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(query_module, "require_api_key", lambda: "test-key")

    openai_vector, embedder_name = query_module._embed_fn_from_state({"embedder_name": "text-embedding-3-small"})
    assert openai_vector("probe") == [0.1, 0.2, 0.3]
    assert embedder_name == "text-embedding-3-small"


def test_query_brain_json_compact_omits_context(tmp_path: Path, capsys, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    query_module = _load_query_brain_module()
    monkeypatch.setattr(
        query_module,
        "_load_query_via_socket",
        lambda *args, **kwargs: {
            "fired_nodes": ["a"],
            "context": "non-deterministic context",
            "prompt_context": "[BRAIN_CONTEXT v1]\n- node: a\n  alpha\n[/BRAIN_CONTEXT]",
            "prompt_context_len": 54,
            "prompt_context_max_chars": 12000,
            "prompt_context_trimmed": False,
            "embed_query_ms": 1.0,
            "traverse_ms": 2.0,
            "total_ms": 3.0,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["query_brain.py", str(state_path), "alpha", "--socket", str(tmp_path / "daemon.sock"), "--json"],
    )
    query_module.main()

    raw_output = capsys.readouterr().out.strip()
    output = json.loads(raw_output)
    assert output["state"] == str(state_path)
    assert output["fired_nodes"] == ["a"]
    assert "prompt_context" in output
    assert "context" not in output
    assert "seeds" not in output
    assert "prompt_context_included_node_ids" not in output
    assert "\n" not in raw_output


def test_query_brain_socket_node_id_default_follows_compact_mode(tmp_path: Path, capsys, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    query_module = _load_query_brain_module()

    captured: list[bool] = []

    def _fake_socket(*args, **_kwargs):
        captured.append(bool(args[-1]))
        return {
            "fired_nodes": ["a"],
            "context": "ctx",
            "seeds": [["a", 1.0]],
            "prompt_context": "[BRAIN_CONTEXT v1]\nalpha\n[/BRAIN_CONTEXT]",
        }

    monkeypatch.setattr(query_module, "_load_query_via_socket", _fake_socket)

    monkeypatch.setattr(
        sys,
        "argv",
        ["query_brain.py", str(state_path), "alpha", "--socket", str(tmp_path / "daemon.sock"), "--json"],
    )
    query_module.main()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        ["query_brain.py", str(state_path), "alpha", "--socket", str(tmp_path / "daemon.sock"), "--json", "--no-compact"],
    )
    query_module.main()
    capsys.readouterr()

    assert captured == [False, True]


def test_query_brain_compact_include_stats_uses_slim_subset(tmp_path: Path, capsys, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    query_module = _load_query_brain_module()
    monkeypatch.setattr(
        query_module,
        "_load_query_via_socket",
        lambda *args, **kwargs: {
            "fired_nodes": ["a"],
            "context": "non-deterministic context",
            "prompt_context": "[BRAIN_CONTEXT v1]\nalpha\n[/BRAIN_CONTEXT]",
            "prompt_context_len": 54,
            "prompt_context_max_chars": 12000,
            "prompt_context_trimmed": False,
            "prompt_context_dropped_count": 1,
            "prompt_context_dropped_authority_counts": {"overlay": 1},
            "prompt_context_excluded_files_count": 2,
            "prompt_context_excluded_node_ids_count": 2,
            "prompt_context_included_node_ids": ["a"],
            "prompt_context_dropped_node_ids": ["b"],
            "embed_query_ms": 1.0,
            "traverse_ms": 2.0,
            "total_ms": 3.0,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query_brain.py",
            str(state_path),
            "alpha",
            "--socket",
            str(tmp_path / "daemon.sock"),
            "--json",
            "--include-stats",
        ],
    )
    query_module.main()

    output = json.loads(capsys.readouterr().out.strip())
    assert output["state"] == str(state_path)
    assert output["fired_nodes"] == ["a"]
    assert output["prompt_context_len"] == 54
    assert output["prompt_context_dropped_authority_counts"] == {"overlay": 1}
    assert output["embed_query_ms"] == 1.0
    assert "prompt_context_included_node_ids" not in output
    assert "prompt_context_dropped_node_ids" not in output
    assert "context" not in output
    assert "seeds" not in output


def test_query_brain_socket_passes_route_params(tmp_path: Path, capsys, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    query_module = _load_query_brain_module()

    captured: dict[str, object] = {}

    def _fake_socket(
        socket_path: str | None,
        query_text: str,
        chat_id: str | None,
        top: int,
        route_mode: str,
        route_top_k: int,
        route_alpha_sim: float,
        route_use_relevance: bool,
        max_prompt_context_chars: int,
        exclude_files: list[str],
        exclude_file_prefixes: list[str],
        prompt_context_include_node_ids: bool,
    ) -> dict[str, object]:
        captured.update(
            {
                "socket_path": socket_path,
                "query_text": query_text,
                "chat_id": chat_id,
                "top": top,
                "route_mode": route_mode,
                "route_top_k": route_top_k,
                "route_alpha_sim": route_alpha_sim,
                "route_use_relevance": route_use_relevance,
                "max_prompt_context_chars": max_prompt_context_chars,
                "exclude_files": exclude_files,
                "exclude_file_prefixes": exclude_file_prefixes,
                "prompt_context_include_node_ids": prompt_context_include_node_ids,
            }
        )
        return {
            "fired_nodes": ["a"],
            "context": "ctx",
            "prompt_context": "[BRAIN_CONTEXT v1]\nalpha\n[/BRAIN_CONTEXT]",
        }

    monkeypatch.setattr(query_module, "_load_query_via_socket", _fake_socket)
    socket_path = str(tmp_path / "daemon.sock")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query_brain.py",
            str(state_path),
            "alpha",
            "--chat-id",
            "chat-123",
            "--socket",
            socket_path,
            "--top",
            "6",
            "--route-mode",
            "edge+sim",
            "--route-top-k",
            "7",
            "--route-alpha-sim",
            "0.25",
            "--no-route-use-relevance",
            "--json",
        ],
    )
    query_module.main()
    capsys.readouterr()

    assert captured["socket_path"] == socket_path
    assert captured["query_text"] == "alpha"
    assert captured["chat_id"] == "chat-123"
    assert captured["top"] == 6
    assert captured["route_mode"] == "edge+sim"
    assert captured["route_top_k"] == 7
    assert captured["route_alpha_sim"] == 0.25
    assert captured["route_use_relevance"] is False
