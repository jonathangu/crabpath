> **Note:** This design doc is historical. The implementation lives in crabpath/*.py. See ARCHITECTURE_REVIEW.md for current architecture.

# CrabPath v2 Simulation Experiment Suite

This suite is designed to test the claim that CrabPath v2 reduces context per turn while improving retrieval/selection quality versus static context and top-k RAG, and that the Gu-corrected update rule materially improves correction behavior.

## Scope and simulation baseline

I used the existing simulator stack to define these runs:
- `crabpath/traversal.py` performs 2-3 hop weighted traversal; it already tracks `context_nodes` and `raw_context` in `TraversalTrajectory`.
- `crabpath/simulator.py` reads JSONL scenarios and records trajectory metrics per episode.
- `crabpath/learning.py` currently uses Gu-corrected advantage and weight updates in `gu_corrected_advantage(...)`.
- `crabpath/router.py` includes a deterministic fallback path but supports habitual/fallback selection for ambiguity.

### Supporting findings from prior run
The v2 vision and results in `/Users/guclaw/crabpath-private/docs/v2-research` already demonstrate:
- Negation handling is immediate when correction text is explicit.
- With the giraffe→elephant setup, 8 episodes moved weights to convergence (`elephant` surpasses `giraffe` between eps 7-8).

This experiment set uses that behavior as the expected convergence target for correction-focused scenarios.

## Run command template

For each experiment, first build artifacts:

```bash
python3 experiments/build_<experiment>_graph.py \
  --graph experiments/<experiment>_graph.json \
  --scenario scenarios/<experiment>.jsonl
```

Baseline simulation run:

```bash
python3 -m crabpath.simulator \
  --graph experiments/<experiment>_graph.json \
  --scenarios scenarios/<experiment>.jsonl \
  --output experiments/<experiment>_crabpath.json
```

For fair comparison, collect three systems with the same scenario order:
1. `STATIC`: seed nodes = all graph nodes each turn.
2. `RAG top-k`: seed nodes = top-k by embedding similarity (k=8).
3. `CRABPATH`: default simulator behavior using `traverse(...)` from existing code.

Metrics collection should be done from simulator outputs plus a small wrapper around traversal that logs:
- `len(trajectory.context_nodes)`
- token count of `trajectory.raw_context` (split-by-whitespace)
- whether expected fragments appear in `raw_context` (for quality checks)
- edge updates from simulator metric `edges_updated`

---

## 1) CONTEXT BLOAT EXPERIMENT

### Graph structure
- Script: `experiments/build_context_bloat_graph.py`
- Nodes: 50 contextual nodes (`fact`, `procedure`, `guardrail`, `tool_call`) plus one root and one query-entry node per turn.
- Expected mix (from script):
  - facts: many entries
  - procedures: many entries
  - guardrails: 10 entries
  - tool calls: 10 entries
- Query nodes: 20 (`cb_query_01` ... `cb_query_20`)
- For each query node: 3–4 relevant strong edges (0.77+), then weak distractor edges.

### Scenario
- File: `scenarios/context_bloat.jsonl`
- 20 queries
- Each turn requests narrow operational behavior and each expected answer only needs 3–5 relevant nodes.
- Example metric target per row: `expected_answer_fragments` list.

### What to measure
- `loaded_nodes = len(trajectory.context_nodes)`
- `loaded_tokens = len(trajectory.raw_context.split())`
- `answer_coverage = fraction(expected_answer_fragments subset in raw_context)`
- `context_precision = hit_count / loaded_nodes`

### Expected results

| Method | Avg loaded nodes | Avg tokens/turn | Answer coverage | Notes |
|---|---:|---:|---:|---|
| Static | 50+ (all nodes) | High | High (because all present) | no adaptation, but maximal bloat |
| RAG top-8 | 8 | ~1/6 of context | Medium | may miss 1-2 required nodes |
| CrabPath | 3–5 | Lowest sustained | High after learning | should keep only fired nodes |

### How to run
1. Build graph+scenario:
```bash
python3 experiments/build_context_bloat_graph.py \
  --graph experiments/context_bloat_graph.json \
  --scenario scenarios/context_bloat.jsonl
```
2. Run three strategies (static, RAG top-8, default crabpath) and compare per-turn node and token histograms.

---

## 2) STALE CONTEXT EXPERIMENT

### Graph structure
- Script: `experiments/build_stale_context_graph.py`
- Nodes:
  - Stable policy nodes for coordinator, region, retry, plus `old` and `new` values for each.
  - Guardrail nodes to suppress stale alternatives.
  - 1 shared runbook tool node.
- Topic entry nodes for coordinator/region/retry with mixed old/new pointers.
- Inhibitory edges from new nodes to old nodes are included explicitly.

### Scenario
- File: `scenarios/stale_context.jsonl`
- 30 turns:
  - 10 establishment turns (old facts)
  - 5 correction turns (explicit corrections)
  - 15 recall turns
- Recall turns include one intentionally noisy stale expectation to measure recovery behavior.

### What to measure
- `accuracy_by_turn`
- `precision_after_correction@k`
- `stale_load_rate` = share of turns including old fact nodes after corrections
- `convergence_turn` = first turn where recall accuracy stays above 90% for 3 consecutive turns

### Expected results

| Method | Post-correction behavior | Convergence turns | Why |
|---|---|---:|---|
| Static | Old facts never decay unless manual edit | N/A | stale data persists |
| RAG top-8 | Ambiguous: both old/new often similar | 10+ | both often retrieved in overlap |
| Gu-corrected CrabPath | stale edges decrease; new edges rise | 2–6 after correction wave | inhibitory updates + positive reinforcement |

### How to run
1. Build:
```bash
python3 experiments/build_stale_context_graph.py \
  --graph experiments/stale_context_graph.json \
  --scenario scenarios/stale_context.jsonl
```
2. Run simulator for 30 turns and compute convergence/accuracy stats.
3. Re-run with a myopic update variant (no baseline correction) for delta comparison.

---

## 3) GATE BLOAT EXPERIMENT

### Graph structure
- Script: `experiments/build_gate_bloat_graph.py`
- 130 `guardrail` nodes across 5 semantic categories (coding, deployment, memory, security, review).
- 20 query nodes (diverse tasks).
- For each query node, strong edges target top relevant category gates, plus 2 weak cross-category distractors.

### Scenario
- File: `scenarios/gate_bloat.jsonl`
- 20 diverse prompts including coding, deployment, and memory themes.
- Each has three expected fragments indicating which category cluster should dominate.

### What to measure
- `loaded_nodes` and `loaded_tokens`
- `relevant_gate_precision` = relevant gates fired / total gates fired
- `category_precision` = fired gates in expected category / total fired gates
- `first_wrong_category_load_rate`

### Expected results

| Method | Avg loaded nodes | Avg tokens | Gate precision |
|---|---:|---:|---:|
| Static | 130 | Highest | Low (diluted by irrelevant gates) |
| RAG top-8 | 8 | Lower than static but variable | Moderate |
| CrabPath | 2–4 relevant gates | Lowest | High (category-aware firing over time) |

### How to run
1. Build:
```bash
python3 experiments/build_gate_bloat_graph.py \
  --graph experiments/gate_bloat_graph.json \
  --scenario scenarios/gate_bloat.jsonl
```
2. Simulate default crabpath and compare node precision with static/top-8 baselines.

---

## 4) NEGATION EXPERIMENT

### Graph structure
- Script: `experiments/build_negation_graph.py`
- Competing facts:
  - `ng_do_run_migrations_now` (A)
  - `ng_do_not_run_migrations_in_prod` (B)
- Corrective/sequence nodes for staging, windowing, and feature flag.
- Explicit inhibitory edge between A and B directions.

### Scenario
- File: `scenarios/negation.jsonl`
- 3 pre-correction turns, 1 explicit correction, then reinforcement and probing queries.
- Rewards include `0.0/-1.0/1.0` to force explicit correction.

### What to measure
- `turn_wrong_answer` = first turn with wrong node still leading context
- `wrong_node_prevalence_by_turn` = count of visits/weights for wrong policy node
- `turns_to_correction_lock` = first turn where correct policy appears without wrong policy in top-1 step

### Expected results

| Method | Negation handling | Turn-to-correct | Token cost trend |
|---|---|---:|---|
| Static | both policies always present | Never | Always high |
| RAG top-8 | both likely retrieved (semantic overlap) | Variable | Moderate |
| Gu-corrected CrabPath | wrong node suppressed after explicit correction | 1–2 turns after correction | declines quickly |

### How to run
1. Build:
```bash
python3 experiments/build_negation_graph.py \
  --graph experiments/negation_graph.json \
  --scenario scenarios/negation.jsonl
```
2. Check turn-by-turn node trajectory from traversal to confirm inhibitory suppression.

---

## 5) PROCEDURE LEARNING EXPERIMENT

### Graph structure
- Script: `experiments/build_procedure_graph.py`
- 4 true steps (`check-logs`, `read-config`, `fix-code`, `run-tests`) and decoys.
- Weak chain edges encode partial correct order; cross-links produce tempting distractors.
- Inhibitory edges from final step to decoys.

### Scenario
- File: `scenarios/procedure.jsonl`
- 10 episodes of production-debug style queries.
- Positive feedback increases when full ordered sequence is present in expected fragments.

### What to measure
- `order_f1` against canonical order `check-logs -> read-config -> fix-code -> run-tests`
- `episodes_to_stable_order` = turns until order_f1 >= 1.0 for 3 consecutive turns
- `loaded_nodes` and token spend

### Expected results

| Method | First episodes | Episode 10 | Notes |
|---|---|---|---|
| Static | high token spend, no ordering | none | deterministic ordering not inferred |
| RAG top-8 | retrieves all four steps but no order confidence | weak | order signal absent |
| CrabPath | weak at start (distractors) | stable order by ~6–8 | inhibitory updates keep decoys down |

### How to run
1. Build:
```bash
python3 experiments/build_procedure_graph.py \
  --graph experiments/procedure_graph.json \
  --scenario scenarios/procedure.jsonl
```
2. Simulate and plot `order_f1` + `loaded_tokens` by episode.

---

## Artifact checklist

Generated here:
- `experiments/build_context_bloat_graph.py` + `scenarios/context_bloat.jsonl`
- `experiments/build_stale_context_graph.py` + `scenarios/stale_context.jsonl`
- `experiments/build_gate_bloat_graph.py` + `scenarios/gate_bloat.jsonl`
- `experiments/build_negation_graph.py` + `scenarios/negation.jsonl`
- `experiments/build_procedure_graph.py` + `scenarios/procedure.jsonl`
- Graph outputs:
  - `experiments/context_bloat_graph.json`
  - `experiments/stale_context_graph.json`
  - `experiments/gate_bloat_graph.json`
  - `experiments/negation_graph.json`
  - `experiments/procedure_graph.json`

## Reproducibility notes

- Run each experiment twice:
  1. default `CrabPath` (`make_learning_step` as currently implemented with Gu-corrected update).
  2. myopic control variant (manual update rule in a custom harness where each trajectory edge reward ignores baseline/discount and uses one-step reward only).
- Compare both to verify correction quality and weight convergence.
- Use the giraffe result trend as sanity check: the correction-only flip should settle by ~episode 7-8 under comparable settings.
