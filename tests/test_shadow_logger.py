from __future__ import annotations

from pathlib import Path

from crabpath import Node, ShadowLog


def test_shadow_log_query_and_tail(tmp_path: Path) -> None:
    log = ShadowLog(tmp_path / "shadow.jsonl")
    record = log.log_query(
        query="deploy broke after config change and service restarted multiple times after a bad merge " * 2,
        selected_nodes=[Node(id="deploy_config", content="a" * 60), "check_logs"],
        scores={"scores": {"deploy_config": 1.0}, "overall": 0.88},
        reward=0.88,
        trajectory=[("deploy_config", "check_logs"), ("check_logs", "restart_service")],
        tiers={"reflex": 3, "habitual": 2, "dormant": 1, "inhibitory": 0, "proto_edge_count": 5},
    )

    assert record["event"] == "query"
    assert len(record["query"]) == 100
    assert record["selected_node_ids"] == ["deploy_config", "check_logs"]
    assert record["selected_node_snippets"][0]["id"] == "deploy_config"
    assert len(record["selected_node_snippets"][0]["snippet"]) == 50
    assert record["trajectory_edges"] == [["deploy_config", "check_logs"], ["check_logs", "restart_service"]]
    assert record["reward_source"] == "scoring"
    assert record["proto_edge_count"] == 5

    log.log_health({"avg_nodes_fired_per_query": 2.1})
    tail = log.tail(2)
    assert len(tail) == 2
    assert tail[0]["event"] == "query"
    assert tail[1]["event"] == "health"


def test_shadow_log_tune_and_summary(tmp_path: Path) -> None:
    log = ShadowLog(tmp_path / "shadow.jsonl")
    log.log_query(
        query="first query",
        selected_nodes=[Node(id="a", content="alpha")],
        scores=None,
        reward=-1.0,
        trajectory=[{"from_node": "a", "to_node": "b"}],
        tiers={"reflex": 1, "habitual": 2, "dormant": 3, "inhibitory": 0},
    )
    log.log_query(
        query="second query",
        selected_nodes=[Node(id="x", content="x"), Node(id="y", content="y"), Node(id="z", content="z")],
        scores=None,
        reward=None,
        trajectory=[],
        tiers={"reflex": 0, "habitual": 1, "dormant": 2, "inhibitory": 1},
    )
    tune_record = log.log_tune([{"parameter": "decay_half_life", "value": 72}], {"status": "ok"})

    assert tune_record["event"] == "tune"
    summary = log.summary(last_n=10)
    assert summary["queries"] == 2
    assert summary["avg_selected"] == 2.0
    assert summary["avg_reward"] == -1.0
    assert summary["tier_trends"]["reflex"] == 0.5
    assert summary["tier_trends"]["inhibitory"] == 0.5
