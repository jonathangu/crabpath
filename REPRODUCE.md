# Reproduce CrabPath Results

All paper claims are verified by deterministic simulations and one benchmark harness. No API keys are needed for simulation.

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

All eight simulations are deterministic and all PASS.

## Table: Simulations → Paper Claims

| Sim | Paper Claim | What It Tests |
|-----|-------------|---------------|
| `deploy_pipeline` | Repeated edges compile to reflex | 50 queries, 4 pipeline edges, reflex-tier convergence |
| `negation` | Corrective negatives suppress bad guidance | Inhibitory learning (`-0.940`) on stale edge |
| `context_reduction` | Context becomes focused over time | 30→2.6 fired nodes |
| `forgetting` | Selective dormancy is learned | Dormant shares reach 93.3% |
| `edge_damping` | Damped traversal avoids loops | `λ=0.3` reaches target; `λ=1.0` loops |
| `domain_separation` | Sparse bridges while preserving clusters | 5 cross-file edges emerge |
| `brain_death` | Health checks surface recovery pressure | Dormancy and recovery control behavior |
| `individuation` | Same topology with workload-specific weights | 27 edges differ by >0.05 after divergent workloads |

## Run Tests

```bash
pip install -e .
python3 -m pytest tests/ -x -q
```

Expected: 150 passed.

## Benchmarks

CrabPath includes a deterministic benchmark harness at `benchmarks/run_benchmark.py`.

### Run

```bash
python3 benchmarks/run_benchmark.py
python3 -m pytest tests/ -x -q
```

Benchmark output compares:

1. keyword overlap
2. hash embedding
3. CrabPath traversal
4. CrabPath traversal + session replay

Current reference results (CR@3): `0.942`, `0.933`, `0.892`, `0.908`.
