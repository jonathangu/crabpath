from __future__ import annotations

from openclawbrain.reward import RewardSource
from openclawbrain.trace import RouteCandidate, RouteDecisionPoint, RouteTrace, route_trace_to_json


def test_route_trace_serialization_is_deterministic() -> None:
    trace = RouteTrace(
        query_id="q-1",
        ts=123.0,
        chat_id="chat-1",
        query_text="alpha",
        seeds=[["seed", 1.0]],
        fired_nodes=["seed", "target_a"],
        traversal_config={"max_hops": 30},
        route_policy={"route_mode": "off"},
        decision_points=[
            RouteDecisionPoint(
                query_text="alpha",
                source_id="seed",
                source_preview="alpha source",
                router_entropy=0.2,
                router_conf=0.8,
                router_margin=0.6,
                relevance_entropy=0.3,
                relevance_conf=0.7,
                policy_disagreement=0.1,
                candidates=[
                    RouteCandidate(
                        target_id="target_b",
                        edge_weight=0.3,
                        edge_relevance=0.0,
                        target_preview="b",
                        graph_prior_score=0.2,
                        router_score_raw=0.1,
                        final_score=0.18,
                    ),
                    RouteCandidate(
                        target_id="target_a",
                        edge_weight=0.3,
                        edge_relevance=0.0,
                        target_preview="a",
                        graph_prior_score=0.3,
                        router_score_raw=0.6,
                        final_score=0.54,
                    ),
                ],
                teacher_choose=[],
                teacher_scores={},
                ts=123.0,
                reward_source=RewardSource.TEACHER,
            )
        ],
    )

    first = route_trace_to_json(trace)
    second = route_trace_to_json(trace)
    assert first == second
    # tie-breaker on target_id for same edge weight
    assert first.index("target_a") < first.index("target_b")
