"""Tests for CrabPath neuron-style activation."""

from crabpath import Graph, Node, Edge, activate, learn


def _simple_graph() -> Graph:
    """A -> B -> C, with A -inhibits-> D.

    Default thresholds (1.0), so a seed of 1.0 on A will fire A,
    which sends energy to B (excitatory) and D (inhibitory).
    """
    g = Graph()
    g.add_node(Node(id="a", content="start node"))
    g.add_node(Node(id="b", content="middle node"))
    g.add_node(Node(id="c", content="end node"))
    g.add_node(Node(id="d", content="bad node"))

    g.add_edge(Edge(source="a", target="b", weight=1.5))   # excitatory
    g.add_edge(Edge(source="b", target="c", weight=1.2))   # excitatory
    g.add_edge(Edge(source="a", target="d", weight=-1.0))  # inhibitory
    return g


# ---- Basic firing ----

def test_basic_firing():
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})

    assert len(result.fired) > 0
    assert result.steps > 0


def test_activation_propagates():
    """A fires → sends 1.5 energy to B → B fires (1.5 >= 1.0)."""
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})

    fired_ids = [n.id for n, _ in result.fired]
    assert "a" in fired_ids
    assert "b" in fired_ids


def test_chain_propagation():
    """A → B → C. Energy should chain through all three."""
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0}, max_steps=5)

    fired_ids = [n.id for n, _ in result.fired]
    assert "a" in fired_ids
    assert "b" in fired_ids
    assert "c" in fired_ids


def test_inhibition():
    """A has negative edge to D. D should be inhibited."""
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})

    assert "d" in result.inhibited
    fired_ids = [n.id for n, _ in result.fired]
    assert "d" not in fired_ids


def test_threshold_gating():
    """Node with high threshold shouldn't fire on low energy."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B", threshold=10.0))  # high threshold
    g.add_edge(Edge(source="a", target="b", weight=1.0))

    result = activate(g, seeds={"a": 1.0})

    fired_ids = [n.id for n, _ in result.fired]
    assert "a" in fired_ids
    assert "b" not in fired_ids  # 1.0 energy < 10.0 threshold


def test_low_threshold_fires_easily():
    """Node with low threshold fires on small energy."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B", threshold=0.1))
    g.add_edge(Edge(source="a", target="b", weight=0.5))

    result = activate(g, seeds={"a": 1.0})

    fired_ids = [n.id for n, _ in result.fired]
    assert "b" in fired_ids


def test_empty_graph():
    g = Graph()
    result = activate(g, seeds={})
    assert len(result.fired) == 0


def test_decay():
    """With high decay, subthreshold energy should vanish."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B", threshold=1.0))
    g.add_edge(Edge(source="a", target="b", weight=0.8))  # 0.8 < 1.0 threshold

    result = activate(g, seeds={"a": 1.0}, decay=0.9)
    fired_ids = [n.id for n, _ in result.fired]
    assert "b" not in fired_ids


def test_convergent_firing():
    """Two nodes both connect to a target — their energy should sum."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C", threshold=1.5))
    g.add_edge(Edge(source="a", target="c", weight=0.9))
    g.add_edge(Edge(source="b", target="c", weight=0.9))

    # Each sends 0.9, total 1.8 >= 1.5
    result = activate(g, seeds={"a": 1.0, "b": 1.0})
    fired_ids = [n.id for n, _ in result.fired]
    assert "c" in fired_ids


def test_top_k():
    g = Graph()
    for i in range(20):
        g.add_node(Node(id=f"n{i}", content=f"node {i}", threshold=0.1))
    for i in range(1, 20):
        g.add_edge(Edge(source="n0", target=f"n{i}", weight=0.5))

    result = activate(g, seeds={"n0": 1.0}, top_k=5)
    assert len(result.fired) <= 5


def test_refractory():
    """A node that fired shouldn't fire again in the same activation pass."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=2.0))
    g.add_edge(Edge(source="b", target="a", weight=2.0))  # cycle

    result = activate(g, seeds={"a": 1.0}, max_steps=5)

    # A should fire exactly once even though there's a cycle back
    a_count = sum(1 for n, _ in result.fired if n.id == "a")
    assert a_count == 1


def test_energy_at_firing():
    """The score should be the node's potential at time of firing."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    result = activate(g, seeds={"a": 2.5})

    assert len(result.fired) == 1
    node, score = result.fired[0]
    assert node.id == "a"
    assert score == 2.5


# ---- fired_at timing ----

def test_fired_at_records_steps():
    """fired_at should record which step each node fired at."""
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0}, max_steps=5)

    assert "a" in result.fired_at
    assert "b" in result.fired_at
    assert "c" in result.fired_at
    # A fires first (step 0), B second (step 1), C third (step 2)
    assert result.fired_at["a"] < result.fired_at["b"]
    assert result.fired_at["b"] < result.fired_at["c"]


def test_fired_at_step_zero_for_seeds():
    """Seed nodes that meet threshold fire at step 0."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    result = activate(g, seeds={"a": 1.0})
    assert result.fired_at["a"] == 0


def test_fired_at_inhibited_not_present():
    """Inhibited nodes should not appear in fired_at."""
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})
    assert "d" not in result.fired_at


# ---- Traces ----

def test_trace_set_on_firing():
    """A node's trace should be set to its firing energy."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    activate(g, seeds={"a": 2.0})
    assert g.get_node("a").trace == 2.0


def test_trace_decays_across_calls():
    """Trace should decay each time activate() is called."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))

    # First call: fire A
    activate(g, seeds={"a": 1.0}, trace_decay=0.5)
    assert g.get_node("a").trace == 1.0

    # Second call: don't fire A, trace should decay
    activate(g, seeds={}, trace_decay=0.5)
    assert abs(g.get_node("a").trace - 0.5) < 0.01


def test_trace_refreshes_on_refire():
    """If a node fires again, its trace should refresh."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))

    activate(g, seeds={"a": 1.0}, trace_decay=0.5)
    assert g.get_node("a").trace == 1.0

    # Fire again with more energy
    activate(g, seeds={"a": 3.0}, trace_decay=0.5)
    # Trace decays to 0.5, then refreshes to 3.0 on firing
    assert g.get_node("a").trace == 3.0


def test_warm_nodes():
    """warm_nodes() should return recently-fired nodes."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C"))

    activate(g, seeds={"a": 1.0})

    warm = g.warm_nodes()
    warm_ids = [n.id for n, _ in warm]
    assert "a" in warm_ids
    assert "b" not in warm_ids  # never fired


# ---- Persistent state (reset=False) ----

def test_persistent_state():
    """With reset=False, potential should carry over between calls."""
    g = Graph()
    g.add_node(Node(id="a", content="A", threshold=2.0))

    # First call: inject 1.0, doesn't fire (1.0 < 2.0)
    r1 = activate(g, seeds={"a": 1.0}, reset=True)
    assert len(r1.fired) == 0

    # Manually set potential (simulating persistent state)
    g.get_node("a").potential = 1.0

    # Second call with reset=False: inject 1.0 more, total 2.0, fires!
    r2 = activate(g, seeds={"a": 1.0}, reset=False)
    fired_ids = [n.id for n, _ in r2.fired]
    assert "a" in fired_ids


def test_persistent_warmth():
    """Energy from first call lingers and helps second call fire."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B", threshold=2.0))
    g.add_edge(Edge(source="a", target="b", weight=1.5))

    # First call: A fires, B gets 1.5 (below 2.0 threshold)
    r1 = activate(g, seeds={"a": 1.0}, reset=True)
    fired_1 = [n.id for n, _ in r1.fired]
    assert "b" not in fired_1

    # B should have leftover potential (after decay)
    assert g.get_node("b").potential > 0

    # Second call with reset=False: A fires again, B gets more energy
    r2 = activate(g, seeds={"a": 1.0}, reset=False)
    fired_2 = [n.id for n, _ in r2.fired]
    assert "b" in fired_2


# ---- STDP-aware learning ----

def test_learning_strengthens():
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})

    old_weight = g.get_edge("a", "b").weight
    learn(g, result, outcome=1.0)
    new_weight = g.get_edge("a", "b").weight

    assert new_weight > old_weight


def test_learning_weakens():
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})

    old_weight = g.get_edge("a", "b").weight
    learn(g, result, outcome=-1.0)
    new_weight = g.get_edge("a", "b").weight

    assert new_weight < old_weight


def test_learning_only_affects_cofired_edges():
    """Learning should only touch edges between nodes that both fired."""
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0}, max_steps=1)

    old_inhibit_weight = g.get_edge("a", "d").weight
    learn(g, result, outcome=1.0)
    assert g.get_edge("a", "d").weight == old_inhibit_weight


def test_stdp_causal_stronger_than_anticausal():
    """Causal edges (src fires before tgt) should get more credit than anti-causal."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=1.5))  # causal direction
    g.add_edge(Edge(source="b", target="a", weight=1.5))  # anti-causal direction

    result = activate(g, seeds={"a": 1.0}, max_steps=3)

    # A fires first (step 0), B fires second (step 1)
    assert result.fired_at["a"] < result.fired_at["b"]

    ab_before = g.get_edge("a", "b").weight
    ba_before = g.get_edge("b", "a").weight

    learn(g, result, outcome=1.0, rate=0.1)

    ab_delta = g.get_edge("a", "b").weight - ab_before
    ba_delta = g.get_edge("b", "a").weight - ba_before

    # Causal edge (a→b) should strengthen
    assert ab_delta > 0
    # Anti-causal edge (b→a) should weaken
    assert ba_delta < 0
    # Causal gets more magnitude than anti-causal
    assert abs(ab_delta) > abs(ba_delta)


def test_stdp_simultaneous_partial_credit():
    """Nodes firing at the same step get partial credit."""
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=1.0))

    # Both fire at step 0 if both seeded above threshold
    result = activate(g, seeds={"a": 1.0, "b": 1.0})

    assert result.fired_at.get("a") == result.fired_at.get("b")  # same step

    old = g.get_edge("a", "b").weight
    learn(g, result, outcome=1.0, rate=0.1)
    new = g.get_edge("a", "b").weight

    # Should strengthen with partial credit (0.5 factor)
    assert new > old
    delta = new - old
    assert abs(delta - 0.1 * 1.0 * 0.5) < 0.001  # rate * outcome * 0.5
