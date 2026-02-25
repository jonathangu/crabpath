"""Tests for CrabPath autotuning and health measurement helpers."""

from crabpath.autotune import (
    suggest_config,
    DEFAULTS,
    GraphHealth,
    HEALTH_TARGETS,
    Adjustment,
    measure_health,
    autotune,
    apply_adjustments,
    self_tune,
)
from crabpath.graph import Edge, Graph, Node
from crabpath.mitosis import MitosisState
from crabpath.decay import DecayConfig
from crabpath.synaptogenesis import SynaptogenesisConfig
from crabpath.mitosis import MitosisConfig


def _workspace_with_sizes(file_count: int, char_count: int) -> dict[str, str]:
    return {f"file-{i}": "x" * char_count for i in range(file_count)}


def _metric(adjustments: list[Adjustment], metric: str) -> Adjustment | None:
    for item in adjustments:
        if item.metric == metric:
            return item
    return None


def test_suggest_config_small_workspace():
    files = _workspace_with_sizes(5, 9_000)
    config = suggest_config(files)

    assert config["sibling_weight"] == DEFAULTS["small"]["sibling_weight"]
    assert config["promotion_threshold"] < DEFAULTS["medium"]["promotion_threshold"]
    assert config["decay_half_life"] < DEFAULTS["medium"]["decay_half_life"]


def test_suggest_config_medium_workspace():
    files = _workspace_with_sizes(20, 100_000)
    config = suggest_config(files)

    assert config == DEFAULTS["medium"]


def test_suggest_config_large_workspace():
    files = _workspace_with_sizes(60, 600_000)
    config = suggest_config(files)

    assert config["promotion_threshold"] > DEFAULTS["medium"]["promotion_threshold"]
    assert config["decay_half_life"] > DEFAULTS["medium"]["decay_half_life"]


def test_measure_health_uses_graph_state_and_query_stats():
    graph = Graph()
    nodes = [
        Node(id="file_a::a", content="alpha " * 20),
        Node(id="file_a::b", content="beta " * 20),
        Node(id="file_b::c", content="gamma " * 20),
    ]
    for node in nodes:
        graph.add_node(node)

    # 1 dormant, 1 habitual, 1 reflex
    graph.add_edge(Edge(source="file_a::a", target="file_a::b", weight=0.2))
    graph.add_edge(Edge(source="file_a::b", target="file_b::c", weight=0.5))
    graph.add_edge(Edge(source="file_b::c", target="file_a::a", weight=0.95))

    state = MitosisState(families={
        "parent::x": ["file_a::a", "file_a::b"],
        "parent::y": ["file_a::a", "file_b::c"],
    })
    query_stats = {
        "fired_counts": [[1, 2, 3, 4], [1, 2, 3]],
        "chars": [40, 60],
        "promotions": 1,
        "proto_created": 10,
        "reconverged_families": 1,
    }

    health = measure_health(graph, state, query_stats)

    assert health.avg_nodes_fired_per_query == 3.5
    assert health.cross_file_edge_pct == (2 / 3) * 100
    assert health.dormant_pct == (1 / 3) * 100
    assert health.reflex_pct == (1 / 3) * 100
    assert health.context_compression == ((40 + 60) / 2) / sum(len(n.content) for n in nodes) * 100
    assert health.proto_promotion_rate == 10.0
    assert health.reconvergence_rate == 50.0
    assert health.orphan_nodes == 0


def test_autotune_for_avg_nodes_recommendations():
    health = GraphHealth(
        avg_nodes_fired_per_query=10,
        cross_file_edge_pct=10,
        dormant_pct=75,
        reflex_pct=2,
        context_compression=12,
        proto_promotion_rate=8,
        reconvergence_rate=0,
        orphan_nodes=0,
    )
    adjustments = autotune(Graph(), health)
    assert _metric(adjustments, "avg_nodes_fired_per_query") is not None
    item = _metric(adjustments, "avg_nodes_fired_per_query")
    assert item and item.suggested_change["decay_half_life"] == "decrease"


def test_autotune_for_low_avg_nodes_and_fast_growth():
    health = GraphHealth(
        avg_nodes_fired_per_query=1,
        cross_file_edge_pct=10,
        dormant_pct=75,
        reflex_pct=2,
        context_compression=12,
        proto_promotion_rate=8,
        reconvergence_rate=0,
        orphan_nodes=0,
    )
    adjustments = autotune(Graph(), health)
    item = _metric(adjustments, "avg_nodes_fired_per_query")
    assert item is not None
    assert item.suggested_change["decay_half_life"] == "increase"
    assert item.suggested_change["promotion_threshold"] == "decrease"


def test_autotune_low_cross_file_and_dormant_no_conflict():
    """When both cross_file and dormant are low, cross-file wins — dormant decay suppressed."""
    health = GraphHealth(
        avg_nodes_fired_per_query=5,
        cross_file_edge_pct=1,
        dormant_pct=40,
        reflex_pct=2,
        context_compression=12,
        proto_promotion_rate=8,
        reconvergence_rate=0,
        orphan_nodes=0,
    )
    adjustments = autotune(Graph(), health)
    cross_file = _metric(adjustments, "cross_file_edge_pct")
    dormant = _metric(adjustments, "dormant_pct")
    assert cross_file is not None
    assert cross_file.suggested_change["promotion_threshold"] == "decrease"
    # Dormant adjustment suppressed to avoid conflicting with cross-file discovery
    assert dormant is None


def test_autotune_high_reflex_and_promotion_rates():
    health = GraphHealth(
        avg_nodes_fired_per_query=5,
        cross_file_edge_pct=10,
        dormant_pct=75,
        reflex_pct=8,
        context_compression=12,
        proto_promotion_rate=3,
        reconvergence_rate=0,
        orphan_nodes=0,
    )
    adjustments = autotune(Graph(), health)
    reflex = _metric(adjustments, "reflex_pct")
    proto_low = _metric(adjustments, "proto_promotion_rate")
    assert reflex is not None
    assert reflex.suggested_change["decay_half_life"] == "decrease"
    assert proto_low is not None
    assert proto_low.suggested_change["promotion_threshold"] == "decrease"


def test_autotune_orphan_nodes_flags_investigation():
    health = GraphHealth(
        avg_nodes_fired_per_query=5,
        cross_file_edge_pct=10,
        dormant_pct=75,
        reflex_pct=2,
        context_compression=12,
        proto_promotion_rate=8,
        reconvergence_rate=0,
        orphan_nodes=2,
    )
    adjustments = autotune(Graph(), health)
    orphan = _metric(adjustments, "orphan_nodes")
    assert orphan is not None
    assert orphan.suggested_change["investigation"] == "trace why nodes have no edges"


def test_autotune_targets_constant_shape():
    assert len(HEALTH_TARGETS) == 8
    assert HEALTH_TARGETS["avg_nodes_fired_per_query"] == (3.0, 8.0)


def test_apply_adjustments_changes_expected_ranges_and_returns_deltas():
    syn_config = SynaptogenesisConfig(
        promotion_threshold=4,
        reflex_threshold=0.9,
        skip_factor=0.92,
    )
    decay_config = DecayConfig(half_life_turns=80)
    mitosis_config = MitosisConfig(sibling_weight=0.65)

    adjustments = [
        Adjustment(
            metric="avg_nodes_fired_per_query",
            current=10.0,
            target_range=(3.0, 8.0),
            suggested_change={"decay_half_life": "decrease", "promotion_threshold": "increase"},
            reason="spread too wide",
        ),
        Adjustment(
            metric="reflex_pct",
            current=8.0,
            target_range=(1.0, 5.0),
            suggested_change={"reflex_threshold": "increase"},
            reason="too many reflex edges",
        ),
        Adjustment(
            metric="cross_file_edge_pct",
            current=1.0,
            target_range=(5.0, 20.0),
            suggested_change={"skip_factor": "decrease"},
            reason="skip factor adjustment",
        ),
    ]

    changes = apply_adjustments(adjustments, syn_config, decay_config, mitosis_config)

    assert decay_config.half_life_turns == 60
    assert syn_config.promotion_threshold == 5
    assert syn_config.reflex_threshold == 0.95
    assert syn_config.skip_factor == 0.9

    assert changes["decay_half_life"]["before"] == 80
    assert changes["decay_half_life"]["after"] == 60
    assert changes["decay_half_life"]["delta"] == -20
    assert changes["promotion_threshold"]["before"] == 4
    assert changes["promotion_threshold"]["after"] == 5
    assert changes["reflex_threshold"]["before"] == 0.9
    assert changes["reflex_threshold"]["after"] == 0.95
    assert changes["skip_factor"]["before"] == 0.92
    assert changes["skip_factor"]["after"] == 0.9
    assert mitosis_config.sibling_weight == 0.65


def test_self_tune_runs_apply_adjustments_cycle(monkeypatch):
    graph = Graph()
    graph.add_node(Node(id="file::a", content="alpha"))
    graph.add_node(Node(id="file::b", content="beta"))
    syn_config = SynaptogenesisConfig(
        promotion_threshold=4,
        reflex_threshold=0.9,
        skip_factor=0.9,
    )
    decay_config = DecayConfig(half_life_turns=40)
    mitosis_config = MitosisConfig()

    def fake_measure(*_args, **_kwargs):
        return GraphHealth(
            avg_nodes_fired_per_query=10.0,
            cross_file_edge_pct=1.0,
            dormant_pct=50.0,
            reflex_pct=8.0,
            context_compression=0.0,
            proto_promotion_rate=1.0,
            reconvergence_rate=0.0,
            orphan_nodes=0,
        )

    def fake_autotune(*_args, **_kwargs):
        return [
            Adjustment(
                metric="avg_nodes_fired_per_query",
                current=10.0,
                target_range=(3.0, 8.0),
                suggested_change={"decay_half_life": "decrease"},
                reason="high spread",
            ),
            Adjustment(
                metric="cross_file_edge_pct",
                current=1.0,
                target_range=(5.0, 20.0),
                suggested_change={"promotion_threshold": "decrease"},
                reason="cross-file too low",
            ),
        ]

    import importlib

    autotune_module = importlib.import_module("crabpath.autotune")

    monkeypatch.setattr(autotune_module, "measure_health", fake_measure)
    monkeypatch.setattr(autotune_module, "autotune", fake_autotune)

    health, adjustments, changes = self_tune(
        graph,
        MitosisState(),
        {"fired_counts": [3, 4]},
        syn_config,
        decay_config,
        mitosis_config,
    )

    assert health.avg_nodes_fired_per_query == 10.0
    assert len(adjustments) == 2
    assert decay_config.half_life_turns == 30
    assert syn_config.promotion_threshold == 3
    assert changes["decay_half_life"]["delta"] == -10
    assert changes["promotion_threshold"]["delta"] == -1
