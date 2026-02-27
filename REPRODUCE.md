# Reproduce CrabPath Results

All paper claims are verified by deterministic simulations. No API keys, no network calls, no randomness.

## Quick Start

```bash
git clone https://github.com/jonathangu/crabpath.git
cd crabpath
pip install -e .
python sims/run_all.py
```

## Expected Output

```
simulation                 claim        summary
-------------------------------------------------------------------------------
deploy_pipeline            PASS         final=1.000
negation                   PASS         bad=-0.940
context_reduction          PASS         30.0->2.6
forgetting                 PASS         dormant=0.933
edge_damping               PASS         dr=True ur=False
domain_separation          PASS         cross=5
brain_death                PASS         dormant=0.975
individuation              PASS         diff_edges=27
```

## Simulations → Paper Claims

| Sim | Paper Claim | What It Tests |
|-----|-------------|---------------|
| `deploy_pipeline` | Deploy edges compile to reflex (weight → 1.0) | 50 queries, 4 pipeline edges, all reach reflex tier |
| `negation` | Inhibitory weight reaches -0.94 | Bad path suppressed via negative outcomes |
| `context_reduction` | 30 → 2.6 nodes (91% reduction) | Active context shrinks as graph learns |
| `forgetting` | 93.3% dormant after 100 queries | Unused edges decay, graph stays sparse |
| `edge_damping` | Damped edges discover branches, undamped loops | w' = w × 0.3^k prevents traversal cycles |
| `domain_separation` | 5 cross-file edges emerge | Separate domains connected by co-firing |
| `brain_death` | Autotune detects 97.5% dormant | Health metrics catch over-decay |
| `individuation` | 27 edges differ between twin agents | Same files, different usage → different graphs |

## Run Tests

```bash
pip install -e .
python -m pytest -q
```

Expected: 138 passed in <1s.

## Details

Each sim writes `*_results.json` with full data. The `run_all.py` runner checks each result's `claim` field and reports PASS/FAIL. All sims are deterministic — same input produces same output on any machine with Python 3.10+.

## Benchmarks

CrabPath provides a local benchmark for the repository docs in `benchmarks/queries.json` and `benchmarks/run_benchmark.py`.

### Run

```bash
python3 benchmarks/run_benchmark.py
python3 -m pytest tests/ -x -q
```

The benchmark builds a graph from the repo using `split_workspace`, builds a hash vector index, runs:

1. keyword overlap
2. hash embed similarity
3. CrabPath traverse
4. CrabPath with replay

and writes comparison output to `benchmarks/results.json`.
