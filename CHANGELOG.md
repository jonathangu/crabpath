# Changelog

All notable changes to CrabPath are documented here.

## [1.0.0] — 2026-02-26

### Architecture
- **MemoryController** — single authoritative policy layer for query → retrieve → learn cycles
- **LearningPhaseManager** — explicit two-phase learning control (Hebbian → RL transition)
- **Inhibition module** (`inhibition.py`) — dedicated negative-edge routing with `apply_correction()`, `score_with_inhibition()`, `is_inhibited()`, and `get_inhibitory_edges()`
- **Enriched Node/Edge primitives** — nodes carry `summary`, `type` (fact/procedure/action/tool_call); edges carry `kind`, `decay_step`, `metadata`
- Legacy activation engine preserved in `crabpath/legacy/` for backward compatibility

### Learning & RL
- **Corrected policy gradient** (Gu 2016) — trajectory-summed credit assignment across full traversal path
- **Ablation study** — 7-arm study (Full, BM25, No-RL, Myopic, No-inhibition, No-synaptogenesis, No-autotune) with bootstrap 95% CIs
- **BM25 external baseline** — CrabPath matches BM25 on accuracy (0.742 vs 0.737), dominates on negation (1.000 vs 0.000)
- **Phase transition diagnostics** — weight entropy + gradient magnitude tracking confirms two-phase learning
- **RL signal fixes** — softmax temperature parameter, episode-family baselines, sibling jitter for symmetry breaking
- **Negative RL flow** — harmful nodes punish the entire traversal path

### Graph Evolution
- **Synaptogenesis** (`synaptogenesis.py`) — proto-edge formation → promotion → Hebbian reinforcement → skip penalty → edge competition
- **Mitosis** (`mitosis.py`) — recursive cell division: LLM-driven splitting, merging, neurogenesis. No magic numbers — LLM decides everything
- **Autotuner** (`autotune.py`) — 5-knob self-regulation with meta-learning (`TuneMemory`), safety guardrails, emergency brake
- **Decay** (`decay.py`) — exponential weight decay with configurable half-life

### Embeddings
- **Multi-provider support** — OpenAI (`text-embedding-3-small`), Gemini (`text-embedding-004`), Cohere (`embed-v4`), Ollama (`nomic-embed-text`)
- **`auto_embed()`** — tries providers in order: OpenAI → Gemini → Ollama
- Keyword-based fallback when no provider is configured

### CLI & Integration
- **13 CLI commands** — `query`, `learn`, `stats`, `health`, `evolve`, `sim`, `split`, `add`, `remove`, `consolidate`, `snapshot`, `feedback`, `migrate`
- **MCP server** (`mcp_server.py`) — stdio JSON-RPC for agent integration
- **OpenAI tools spec** (`tools/openai-tools.json`) + **OpenAPI spec** (`tools/openapi.yaml`)
- **pip installable** — `pip install crabpath` with optional `[embeddings]` extra
- **`OpenClawCrabPathAdapter`** — drop-in adapter for OpenClaw sessions

### Experiments & Benchmarks
- Context Bloat (95% reduction), Gate Bloat (99%), Stale Context (90%), Negation (84%), Procedure (63%)
- Deploy Pipeline — safe path reaches reflex (>0.9) by episode 10, dangerous shortcut becomes dormant
- Giraffe Test — LLM handles negation at episode 3, weights cross by episode 8
- Shadow mode on production workspace: 235 queries, avg reward 0.99, 90% context reduction

### Code Quality
- 278 tests, 0 ruff lint errors, 0 audit items
- Paper↔code consistency audit: 18 discrepancies found and resolved
- Zero external dependencies (stdlib only)

---

## [0.6.0] — 2026-02-25

### Added
- **Auto-neurogenesis** — cosine novelty detection, deterministic IDs, quality gates
- 84 tests

## [0.5.2] — 2026-02-25

### Added
- `add` + `remove` CLI commands for live node creation/destruction

## [0.5.1] — 2026-02-25

### Added
- **CLI** — JSON-in/JSON-out interface: `query`, `learn`, `snapshot`, `feedback`, `stats`, `consolidate`
- 76 tests

## [0.5.0] — 2026-02-25

### Added
- **OpenClaw integration** — `OpenClawCrabPathAdapter`, feedback loop, delayed outcome attribution
- 69 tests

## [0.4.0] — 2026-02-24

### Added
- **Neurogenesis** — node creation from novel concepts
- **Consolidation** — splitting, merging, pruning, protected nodes
- Embedding-based quality gates

## [0.3.0] — 2026-02-24

### Added
- **Embedding-based semantic seeding** — cosine similarity entry point selection

## [0.2.0] — 2026-02-24

### Added
- **STDP timing** — causal edge strengthening based on firing order
- **Traces** — persistent node warmth across activations

## [0.1.0] — 2026-02-23

### Added
- **Neuron model** — nodes with potential/threshold, edges with signed weights
- **Spreading activation** — multi-step propagation with decay and inhibition
- **JSON persistence** — save/load graphs, zero dependencies
- Initial test suite (61 tests)
