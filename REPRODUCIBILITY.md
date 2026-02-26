# Reproducibility Manifest

Every quantitative claim in the paper maps to a script, seed, and expected output.

## Environment

- Python 3.10+
- Zero pip dependencies for core runtime (stdlib only)
- Optional: `OPENAI_API_KEY` for LLM-based routing/scoring (mock router used in ablation)
- Hardware: Apple M4 Pro, 64 GB RAM (any modern machine will work)

## Exact reproduction workflow

```bash
# 1) Create a clean environment
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# 2) Verify Python and dependency baselines
python --version
python - <<'PY'
import sys
print("python", sys.version)
print("python_requires", ">=3.10")
PY
```

## Commands to reproduce all paper outputs

```bash
# Full benchmark suite used by the paper
cd experiments
python run_all.py

# Deploy simulation and comparison outputs
python run_deploy_sim.py
python run_comparison.py
python ../scripts/external_benchmark.py

# Ablation and phase-transition figures (deterministic/stochastic split as noted)
cd ../scripts
python ablation_study.py
python phase_transition_plot.py
python sparsity_scale_experiment.py

# Additional diagnostics
python replay_shadow_queries.py
python bootstrap_from_workspace.py
```

## Seeds and deterministic configuration

- All ablation and phase-transition scripts use seeded RNGs (`SEED = 2026`).
- Bootstrap CIs in ablation use `bootstrap_ci(..., seed=2026)`.
- Run `cd scripts` before calling script-level entrypoints above to match default relative paths.

## Expected experiment output

- `run_all.py` and comparators write JSON artifacts under `experiments/results/*.json`.
- `ablation_study.py` prints per-arm accuracies, token counts, CIs, and a JSON-like report.
- `phase_transition_plot.py` writes plot artifacts with trajectory summaries.
- Always archive the raw output directory and commit it for exact comparisons.

## Quickstart before reproducing paper scripts

Use `examples/quickstart.py` as the canonical starter path before full benchmark runs:

```bash
python examples/quickstart.py
```

## Paper Claims → Scripts

| Paper Claim | Section | Script | Seed | Expected Output |
|---|---|---|---|---|
| Context Bloat: 95% reduction (6,066 → 297 tok/turn) | §7.1, Table 3 | `experiments/run_all.py` | deterministic | `context_bloat: static=6066, crabpath=297` |
| Gate Bloat: 99% reduction (8,163 → 89 tok/turn) | §7.1, Table 3 | `experiments/run_all.py` | deterministic | `gate_bloat: static=8163, crabpath=89` |
| Stale Context: 90% reduction | §7.1, Table 3 | `experiments/run_all.py` | deterministic | `stale_context: static=895, crabpath=88` |
| Negation: 84% reduction | §7.1, Table 3 | `experiments/run_all.py` | deterministic | `negation: static=546, crabpath=87` |
| Procedure: 63% reduction | §7.1, Table 3 | `experiments/run_all.py` | deterministic | `procedure: static=548, crabpath=205` |
| Two-tier cost: $0.004/turn vs $0.091 static | §7.2, Table 4 | `experiments/run_all.py` | deterministic | Cost computed from token counts × published rates |
| Giraffe Test: crossover by episode 8 | §7.3, Table 5 | `experiments/build_giraffe_test.py` | deterministic | `w(elephant) > w(giraffe)` by episode 8 |
| Deploy Pipeline: safe path reflex by ep 10 | §7.4, Table 6 | `experiments/run_deploy_sim.py` | deterministic | `w(check_tests) > 0.9` by episode 10 |
| RAG = Static on deploy pipeline (525 tok) | §7.4, Table 7 | `experiments/run_comparison.py` | deterministic | `deploy: rag=525, static=525` |
| Ablation Arm 1 accuracy: 0.742 [0.700, 0.782] | §7.5, Table 8 | `scripts/ablation_study.py` | 2026 | `arm_1: accuracy=0.742, ci=[0.700, 0.782]` |
| BM25 baseline: 0.737 [0.695, 0.779] | §7.5, Table 8 | `scripts/ablation_study.py` | 2026 | `arm_0: accuracy=0.737, ci=[0.695, 0.779]` |
| BM25 negation: 0.000 vs CrabPath 1.000 | §7.5, Table 8 | `scripts/ablation_study.py` | 2026 | `arm_0: negation=0.000, arm_1: negation=1.000` |
| Corrected PG +11pp over myopic | §7.5, Table 8 | `scripts/ablation_study.py` | 2026 | `arm_1=0.742 vs arm_3=0.632, CIs non-overlapping` |
| No synaptogenesis worst (0.211) | §7.5, Table 8 | `scripts/ablation_study.py` | 2026 | `arm_5: accuracy=0.211` |
| Phase transition at Q~100 | §7.6, Table 9 | `scripts/phase_transition_plot.py` | 2026 | Entropy drops, gradient spikes between Q80-Q120 |
| Shadow mode: 235 queries, reward 0.99 | §C, Table 12 | `scripts/replay_shadow_queries.py` | N/A (real queries) | `queries=235, avg_reward=0.99, reflex=4, cross_file=182` |
| Production bootstrap: 3,667 nodes, 32,698 edges | §C, Table 11 | `scripts/bootstrap_from_workspace.py` | N/A (real workspace) | `nodes=3667, edges=32698` |

## Running the Full Suite

```bash
# All benchmark experiments (deterministic, no API key needed)
cd experiments && python run_all.py

# Ablation study (deterministic mock router, seed=2026)
python scripts/ablation_study.py

# Phase transition diagnostic
python scripts/phase_transition_plot.py

# Sparsity-scale crossover experiment
python scripts/sparsity_scale_experiment.py

# NIAH (multi-needle) + scaling curves
python scripts/niah_scaling_benchmark.py

# Context utilization + noise sensitivity + temporal drift
python scripts/context_noise_drift_benchmark.py

# Downstream accuracy + RULER multi-fact + NarrativeQA/MS MARCO stubs
python scripts/downstream_accuracy_benchmark.py

# Deploy pipeline simulation
python experiments/run_deploy_sim.py

# Shadow mode replay (requires graph + embeddings from production)
python scripts/replay_shadow_queries.py
```

## Bootstrap CIs

All confidence intervals use 10,000 bootstrap resamples with `seed=2026`. The bootstrap procedure:

1. Run all 190 queries through each ablation arm
2. Collect per-query binary accuracy scores
3. Resample with replacement 10,000 times
4. Report 2.5th and 97.5th percentiles

## Deterministic vs Stochastic

- **Deterministic scripts** (experiments/): use mock routers with keyword matching. Results are exact.
- **Stochastic scripts** (ablation, phase transition): use seeded random generators (`seed=2026`). Results reproduce within bootstrap CI bounds.
- **Production scripts** (shadow replay, bootstrap): depend on real workspace/query data. Results are specific to the production environment but methodology is reproducible.

## Test Suite

```bash
# Run all 305 tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_learning.py -v
python -m pytest tests/test_inhibition.py -v
python -m pytest tests/test_ablation.py -v
```

## Expected output format

Most scripts emit readable logs plus JSON payloads with this schema:

```json
{
  "experiment": "context_bloat",
  "arms": {
    "static": {"accuracy": 0.98, "avg_tokens": 6066},
    "rag": {"accuracy": 0.90, "avg_tokens": 1000},
    "crabpath_corrected": {"accuracy": 0.96, "avg_tokens": 297}
  }
}
```

For single-run utilities, validate summary metrics against the claim table above.
