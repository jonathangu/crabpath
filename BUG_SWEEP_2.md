# Bug Sweep #2 Findings

## Scope
Deep dive requested across learning, traversal, inhibition, synaptogenesis, mitosis, graph ops, and integration behavior.

## Findings and fixes

### 1) Synaptogenesis can silently discard proto-edges on rejected promotion
- **File**: `crabpath/synaptogenesis.py`
- **Severity**: 🟡
- **Issue**: `record_cofiring()` removed a proto-edge from `state.proto_edges` unconditionally whenever the promotion condition was met, even when `_add_edge_with_competition()` rejected insertion due to outgoing-cap constraints.
- **Impact**: Proto-edge evidence could be lost even though no real edge was created, weakening future learning and masking a valid co-firing relationship.
- **Fix**: Guard proto-edge deletion with a success check.
  - In `record_cofiring`, proto removal and `promoted += 1` now happen only when `_add_edge_with_competition(...)` returns `True`.
  - `_add_edge_with_competition()` now returns `bool` to indicate whether an edge was added/replaced.

### 2) Graph safety note for thread safety
- **File**: `crabpath/graph.py`
- **Severity**: 🟢
- **Issue**: Add/remove operations are not synchronized for concurrent callers.
- **Fix**: Documented explicitly in `Graph` class docstring that mutation methods are not thread-safe and should be externally serialized.

### 3) Validation coverage additions (no code defect change beyond #1)
The following edge cases were validated with new/updated tests:
- Learning: reward=0 leaves edge weights unchanged when advantage is zero.
- Controller/traversal: damping factors (1.0 and 0.0), visit-penalty=0 identity, no-edge single-node query, empty `QueryResult` learning path.
- Inhibition: competition with protected outgoing targets does not evict when at cap and cannot add a new inhibitory edge.
- Synaptogenesis: proto-edge with full competition can still be rejected without being dropped.
- Graph: duplicate node id replacement behavior and self-loop edge support.
- Traversal: disconnected start node path remains singleton.

## Tests added
- `tests/test_synaptogenesis.py`
  - `test_promotion_not_dropped_when_competition_rejects`
- `tests/test_inhibition.py`
  - `test_competition_respects_protected_targets`
- `tests/test_edge_damping.py`
  - `test_damping_factor_one_is_identity`
  - `test_damping_factor_zero_fully_suppresses_after_first_use`
  - `test_visit_penalty_zero_is_identity`
- `tests/test_controller.py`
  - `test_query_single_node_no_edges`
  - `test_learn_empty_query_result_no_updates`
- `tests/test_graph.py`
  - `test_duplicate_node_id_replaces_node`
  - `test_self_loop_edges_are_supported`
- `tests/test_learning.py`
  - `test_zero_reward_does_not_change_weights`
- `tests/test_traversal.py`
  - `test_disconnected_start_has_singleton_visit_order`

## Test run
Executed:

```bash
python3 -m pytest --tb=short -q
```

Result: `327 passed`.
