from __future__ import annotations

from openclawbrain import Edge, VectorIndex
from openclawbrain.policy import RoutingPolicy, make_runtime_route_fn


def test_make_runtime_route_fn_off_returns_none() -> None:
    index = VectorIndex()
    route_fn = make_runtime_route_fn(policy=RoutingPolicy(route_mode="off"), query_vector=[1.0], index=index)
    assert route_fn is None


def test_make_runtime_route_fn_edge_sim_is_deterministic_with_tiebreak() -> None:
    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])
    index.upsert("b", [1.0, 0.0])
    index.upsert("c", [0.0, 1.0])

    route_fn = make_runtime_route_fn(
        policy=RoutingPolicy(route_mode="edge+sim", top_k=2, alpha_sim=0.5, use_relevance=True),
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

    assert chosen_first == ["a", "b"]
    assert chosen_second == ["a", "b"]


def test_make_runtime_route_fn_edge_mode_ignores_similarity() -> None:
    index = VectorIndex()
    index.upsert("high-sim", [1.0, 0.0])
    index.upsert("low-sim", [0.0, 1.0])

    route_fn = make_runtime_route_fn(
        policy=RoutingPolicy(route_mode="edge", top_k=1, alpha_sim=100.0, use_relevance=False),
        query_vector=[1.0, 0.0],
        index=index,
    )
    assert route_fn is not None

    candidates = [
        Edge("src", "high-sim", weight=0.1),
        Edge("src", "low-sim", weight=0.9),
    ]
    assert route_fn("src", candidates, "q") == ["low-sim"]
