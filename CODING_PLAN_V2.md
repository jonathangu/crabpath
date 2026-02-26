> **Note:** This design doc is historical. The implementation lives in crabpath/*.py. See ARCHITECTURE_REVIEW.md for current architecture.

# CrabPath v2 Rewrite Coding Plan (`v0.6.0` Mechanical → LLM-guided RL v2)

Date: 2026-02-25

This plan is a full implementation blueprint to migrate CrabPath from mechanical activation (v0.6.0) to v2 traversal+RL, while preserving reusable modules and interfaces. It follows a phase-by-phase build order so each phase is independently testable.

## 1) Scope and non-negotiable constraints

- Keep v0.6.0 stable codepaths available until v2 passes all tests.
- Preserve public CLI behavior (argparse entrypoints and JSON output contract).
- Use mockable deterministic router for unit tests; never require external LLM in tests.
- Do not commit private graph data or API keys in public repo.
- Target no behavior regression for unchanged modules (Graph persistence, embeddings, CLI wiring).
- Final target: `80+` tests, including new simulator regression suite.

## 2) What to keep from v0.6.0 (preserve)

### `crabpath/graph.py`
- Keep existing graph core abstractions and persistence flow.
- Keep behavior of `Graph.add_node`, `Graph.add_edge`, `save_graph`, `load_graph`, `snapshot`, and in-memory store.
- Keep `Node`, `Edge`, `Graph` object identity and iteration semantics.
- Retain backward-compatibility fields/methods (or wrappers) used by legacy tests/CLI until migrated.

### `crabpath/embeddings.py`
- Keep `EmbeddingIndex` class and vector index behavior.
- Keep `EmbeddingIndex.build`, `seed`, `search` semantics.
- Keep `openai_embed` adapter behavior unless provider strategy changes elsewhere.
- Add v2 methods where missing: upsert, remove, raw_scores (if not already matching exact needed signature).

### `crabpath/cli.py`
- Keep command structure (`query`, `learn`, `snapshot`, `feedback`, `stats`, `add`, `remove`, `consolidate`).
- Keep JSON envelope shape, stderr logging conventions, and human-readable fallback.

### `tests/*.py`
- Keep pytest setup, fixtures, and graph/persistence tests.
- Refactor tests in modules tied to rewritten internals, but preserve naming and collection strategy.

### `pyproject.toml`
- Keep build/test metadata, dependencies, and tool config.
- Only add new deps required by v2 (prompt schema validator, optional policy/test utilities).

## 3) What to replace and why

### `crabpath/activation.py` → new `crabpath/traversal.py`
- Replace mechanical LIF/STDP traversal logic.
- New traversal layer becomes LLM-guided with three tiers: reflex/habitual/dormant.
- `learn()` moves to `crabpath/learning.py`.

### `crabpath/neurogenesis.py`
- Remove cosine-band heuristic from activation flow.
- Move LLM-driven novelty creation into traversal process in v2.

### `crabpath/adapter.py`
- Replace v0.6 adapter orchestration with v2 integration path.
- Preserve adapter public API (`load_graph`, `seed`, `query`, `learn`, persistence helpers) but delegate to v2 modules.

### `crabpath/feedback.py`
- Fold correction signal handling into `crabpath/learning.py` trajectory reward/advantage update pipeline.
- Keep compatibility shim only if some command still expects previous function names.

## 4) What to add (new files + exact APIs)

### 4.1 `crabpath/traversal.py`

Core responsibility: produce trajectory and context from query with policy + tiers.

- `from __future__ import annotations`
- `from dataclasses import dataclass, field`
- `from typing import Any, Iterable, Mapping, Sequence`

Classes / functions:

- `@dataclass
  class TraversalConfig:`
  - `max_hops: int = 8`
  - `reflex_limit: int = 2`
  - `habitual_limit: int = 4`
  - `dormant_limit: int = 2`
  - `novelty_threshold: float = 0.42`
  - `min_context_chars: int = 320`
  - `temperature: float = 0.2`
  - `branch_beam: int = 3`
  - Description: policy/tier traversal runtime settings.

- `@dataclass
  class TierDecision:`
  - `tier: str`
  - `selected_edges: list[tuple[str, str]]`
  - `scores: dict[str, float]`
  - `rationale: str | None = None`
  - Description: result from one tier stage.

- `@dataclass
  class TraversalStep:`
  - `from_node: str`
  - `to_node: str`
  - `edge_weight: float`
  - `tier: str`
  - `action_type: str`
  - `raw_router_output: dict[str, Any] | None`
  - `log_prob: float | None`
  - Description: one directed decision in episode path.

- `@dataclass
  class TraversalTrajectory:`
  - `steps: list[TraversalStep]`
  - `visit_order: list[str]`
  - `context_nodes: list[str]`
  - `fallback_used: bool`
  - `decision_log: list[TierDecision]`
  - `raw_context: str`
  - Description: complete path representation used by simulator and learner.

- `def build_seed_candidates(graph: Graph, query: str, topk: int = 8) -> list[str]`
  - Description: deterministic lexical/embedding seed ranking for initial frontier.

- `def classify_tier(node: Node, *, fired_count: int, is_reflex_enabled: bool = True, habituation_gate: float = 0.3, is_dormant: bool = False) -> str`
  - Description: returns `"reflex" | "habitual" | "dormant"`.

- `def select_reflex_actions(graph: Graph, frontier: Sequence[str], limit: int) -> list[tuple[str, str]]`
  - Description: deterministic fast-path edges for obvious relevance / high prior support.

- `def select_habitual_actions(graph: Graph, current_node: str, config: TraversalConfig) -> list[tuple[str, str]]`
  - Description: policy-driven or top-weight edges based on pointer weights.

- `def select_dormant_actions(graph: Graph, used_nodes: set[str], limit: int) -> list[tuple[str, str]]`
  - Description: exploratory fallback edges for discovery.

- `def maybe_spawn_knowledge_nodes(graph: Graph, router_context: Mapping[str, Any], config: TraversalConfig) -> list[str]`
  - Description: invokes LLM-driven neurogenesis from traversal context when novelty/uncertainty warrants.

- `def traverse(
    query: str,
    graph: Graph,
    router: "Router",
    config: TraversalConfig,
    embedding_index: EmbeddingIndex | None = None,
    seed_nodes: Sequence[str] | None = None,
    max_steps: int | None = None,
) -> TraversalTrajectory:`
  - Description: executes full v2 path-finding loop.

- `def render_context(trajectory: TraversalTrajectory, graph: Graph, max_tokens: int = 4096) -> str`
  - Description: composes cheap summary + full node content for answer generation.

- `def choose_next_with_tiers(
    current_node: str,
    graph: Graph,
    router: "Router",
    state: Mapping[str, Any],
    config: TraversalConfig
) -> TraversalStep | None`
  - Description: chooses next edge with deterministic tier precedence and router policy fallback.

### 4.2 `crabpath/learning.py`

Core responsibility: policy-gradient learning + credit assignment + update application.

- `from __future__ import annotations`
- `from dataclasses import dataclass`

Classes / functions:

- `@dataclass
  class RewardSignal:
    episode_id: str
    final_reward: float
    step_rewards: list[float] | None = None
    outcome: str | None = None
    feedback: str | None = None
    metadata: dict[str, Any] | None = None`
  - Description: container for scalar and optional dense reward annotations.

- `@dataclass
  class LearningConfig:
    learning_rate: float = 0.05
    discount: float = 0.99
    baseline_decay: float = 0.95
    entropy_beta: float = 0.01
    clip_min: float = -5.0
    clip_max: float = 5.0
    min_weight: float = -5.0
    max_weight: float = 5.0
  - Description: all knobs for Gu-corrected update behavior.

- `@dataclass
  class EdgeUpdate:
    source: str
    target: str
    delta: float
    new_weight: float
    rationale: str
  - Description: one normalized edge delta to apply.

- `@dataclass
  class LearningResult:
    updates: list[EdgeUpdate]
    baseline: float
    avg_reward: float
    entropy: float | None = None
    skipped: list[str] | None = None
  - Description: returned by update call for telemetry + tests.

- `def compute_baseline(state_key: str, reward: float, alpha: float, default: float = 0.0) -> float`
  - Description: running EMA baseline state update for credit normalization.

- `def gu_corrected_advantage(trajectory: TraversalTrajectory, reward: RewardSignal, baseline: float, discount: float) -> list[float]`
  - Description: returns per-step advantages with full-trajectory credit assignment.

- `def policy_gradient_update(
    trajectory: TraversalTrajectory,
    reward: RewardSignal,
    config: LearningConfig,
    baseline: float = 0.0,
) -> tuple[float, list[float]]`
  - Description: core PG math, returns scalar loss and step-level advantage list.

- `def weight_delta(
    trajectory: TraversalTrajectory,
    advantages: Sequence[float],
    config: LearningConfig,
    edge_entropy: float | None = None,
) -> list[tuple[str, str, float]]`
  - Description: per-edge signed delta candidate.

- `def apply_weight_updates(graph: Graph, deltas: Sequence[tuple[str, str, float]], config: LearningConfig) -> list[EdgeUpdate]`
  - Description: apply clamped updates; updates edge metadata (`follow_count`, `skip_count`, timestamps).

- `def apply_feedback_reward(graph: Graph, trajectory: TraversalTrajectory, feedback_json: Mapping[str, Any], config: LearningConfig) -> LearningResult`
  - Description: turns CLI feedback payload (correct/incorrect + explanations) into explicit reward signal and updates.

- `def make_learning_step(graph: Graph, trajectory: TraversalTrajectory, reward: RewardSignal, config: LearningConfig) -> LearningResult`
  - Description: one-step orchestrator used by simulator and adapter learn flow.

- `def edge_entropy_from_trajectory(trajectory: TraversalTrajectory) -> float`
  - Description: optional entropy regularizer term.

- `def should_prune_edge(weight: float, min_weight: float) -> bool`
  - Description: optional pruning heuristic used by consolidation.

### 4.3 `crabpath/router.py`

Core responsibility: LLM interaction layer and structured command output.

- `class RouterError(RuntimeError): ...`
  - Description: typed errors for parsing/unavailable model cases.

- `@dataclass
  class RouterConfig:
    model: str = "gpt-5-mini"
    temperature: float = 0.2
    timeout_s: float = 8.0
    max_retries: int = 2
    fallback_behavior: str = "heuristic"
  - Description: injection/config for deterministic and testable routing.

- `@dataclass
  class RouterDecision:
    chosen_target: str
    rationale: str
    confidence: float
    tier: str
    alternatives: list[tuple[str, float]]
    raw: dict[str, Any]
  - Description: normalized structured decision output.

- `class Router:`
  - `def __init__(self, config: RouterConfig | None = None, client: Any | None = None) -> None`
    - Description: construct with optional mockable LLM client.
  - `def build_prompt(self, query: str, candidates: list[tuple[str, float]], context: Mapping[str, Any], budget: int) -> str`
    - Description: strict system prompt for edge selection.
  - `def decide_next(
      self,
      query: str,
      current_node_id: str,
      candidate_nodes: list[tuple[str, float]],
      context: Mapping[str, Any],
      tier: str,
      previous_reasoning: str | None = None,
    ) -> RouterDecision`
    - Description: returns structured decision JSON; validates schema.
  - `def parse_json(self, raw: str) -> dict[str, Any]`
    - Description: validates parser contract and raises `RouterError`.
  - `def fallback(self, candidates: list[tuple[str, float]], tier: str) -> RouterDecision`
    - Description: deterministic/heuristic fallback when LLM output unavailable.

- `def normalize_router_payload(payload: Mapping[str, Any]) -> RouterDecision`
  - Description: mapping validation + coercion.

### 4.4 `crabpath/decay.py`

Core responsibility: time-/turn-based edge weight decay.

- `@dataclass
  class DecayConfig:
    half_life_turns: int = 120
    half_life_seconds: float | None = None
    min_weight: float = -5.0
    max_weight: float = 5.0`
  - Description: controls exponential decay rate.

- `def decay_factor(half_life_turns: int, elapsed_turns: int) -> float`
  - Description: computes multiplicative decay weight based on elapsed turns.

- `def decay_weight(weight: float, elapsed_turns: int, config: DecayConfig) -> float`
  - Description: per-edge decay function with clipping.

- `def apply_decay(graph: Graph, turns_elapsed: int, now_ts: float | None = None, config: DecayConfig | None = None) -> dict[str, float]`
  - Description: applies decay to all edges and returns map of changed weights.

- `def should_decay_last_followed(last_followed_ts: float | None, now_ts: float | None, cfg: DecayConfig) -> bool`
  - Description: decides whether decayed application should skip inactive nodes.

### 4.5 `crabpath/simulator.py` (primary file), plus compatibility `crabpath/simulate.py`

Core responsibility: scriptable + interactive loop running full v2 cycle.

- `@dataclass
  class ScenarioStep:`
    - `query: str`
    - `expected_answer_fragments: list[str]`
    - `expected_path_contains: list[str] | None = None`
    - `expected_terminal_types: list[str] | None = None`
    - `feedback: dict[str, Any] | None = None`
    - `max_turns: int = 1`
  - Description: one scripted scenario test step.

- `@dataclass
  class SimulatorConfig:
    query_budget: int = 12
    turns: int = 1
    decay_interval: int = 1
    consolidation_interval: int = 10
    enable_interactive: bool = False
    output_json: bool = True
    output_dashboard: bool = True
  - Description: simulator runtime controls.

- `@dataclass
  class EpisodeMetrics:
    query: str
    trajectory: TraversalTrajectory
    answer: str
    reward: float
    updates: list[EdgeUpdate]
    created_nodes: int
    pruned_edges: int
    pruned_nodes: int
  - Description: one-episode run telemetry.

- `def load_scenarios(path: str) -> list[ScenarioStep]`
  - Description: parse JSONL scripted scenarios.

- `def run_episode(query: str, graph: Graph, router: Router, cfg: SimulatorConfig, learning_cfg: LearningConfig, answer_fn: Any | None = None) -> EpisodeMetrics`
  - Description: end-to-end episode: traverse, answer, collect feedback, learn, decay/consolidate hooks.

- `def run_batch(
    scenarios: list[ScenarioStep],
    graph: Graph,
    router: Router,
    cfg: SimulatorConfig,
    learning_cfg: LearningConfig,
  ) -> list[EpisodeMetrics]`
  - Description: executes many scripted scenarios and aggregates metrics.

- `def run_interactive(graph: Graph, router: Router, cfg: SimulatorConfig, learning_cfg: LearningConfig) -> None`
  - Description: stdin loop for manual queries + optional feedback.

- `def render_dashboard(metrics: Sequence[EpisodeMetrics], graph: Graph) -> dict[str, Any]`
  - Description: outputs graph state deltas, created/pruned counts, and top weight updates.

- `def run_simulator(argv: Sequence[str] | None = None) -> int`
  - Description: CLI-style entrypoint for simulator command.

- `# crabpath/simulate.py`
  - `def main(argv: Sequence[str] | None = None) -> int`
    - Description: compatibility script that imports `run_simulator` from `simulator.py`.

### 4.6 `crabpath/schemas.py` (new lightweight validation module, optional but recommended)

- `def normalize_node_type(value: str) -> str`
- `def validate_node_schema(node: Mapping[str, Any]) -> None`
- `def validate_edge_schema(edge: Mapping[str, Any]) -> None`
- `def ensure_backward_compat(record: Mapping[str, Any]) -> dict[str, Any]`
- Description: centralize guardrails for migration and load/save validation.

## 5) New v2 schemas

### 5.1 Node schema (canonical)
- `id` (str)
- `content` (str)
- `summary` (str)
- `type` (`fact` | `procedure` | `action` | `tool_call` | `guardrail`)
- `threshold` (float)
- `metadata` (dict): `source`, `created_ts`, `fired_count`, `probationary`, `last_visited_ts`, `notes`, plus any extension keys
- Backward-compatibility policy:
  - legacy `potential` may remain in metadata alias.
  - missing `summary` defaults to short stable preview of content.
  - missing `type` defaults to `fact`.

### 5.2 Edge schema (canonical)
- `source` (str)
- `target` (str)
- `weight` (float, signed)
- `decay_rate` (float)
- `last_followed_ts` (float | None)
- `metadata` (dict): `created_by` (`auto`|`manual`|`llm`), `follow_count`, `skip_count`
- Optional derived fields:
  - `follow_count` starts at `0` and increments when edge traversed.
  - `skip_count` increments on candidate rejection/avoidance.
  - `decay_rate` default from graph-level policy.

## 6) Private repo migration scope and required edits

### Private script: `/Users/guclaw/crabpath-private/scripts/sync_learning_to_graph.py`
- Update parser/writer to accept new node/edge schema.
- Emit `summary`, `type`, `threshold`, `metadata`, and `decay_rate` in serialized output.
- Preserve migration mode that maps legacy `weight->weight`, `potential->metadata.potential` safely.
- Add `--dry-run` to inspect converted edges count and unknown-type warnings.

### Private script: `/Users/guclaw/crabpath-private/scripts/crabpath_query_log.py`
- Replace call path to import/execute new traversal (`traversal.py`) and router decisions.
- Preserve query-log schema for compatibility; add `v2_decision_log` optional field.
- Ensure feedback replay reuses `learning.make_learning_step` instead of legacy `learn`.

### Private scripts `crabpath-bootstrap.py`, `crabpath-query.py`
- Ensure bootstrap emits required v2 keys.
- Keep public output contract; no schema changes visible to external users unless feature-flagged.

### Graph migration (391 nodes, 35K edges)
- Add migration utility in repo: `scripts/migrate_graph_v2.py` (or equivalent):
  - Read v0.6 JSON snapshots.
  - Backfill missing node summary/type/metadata keys.
  - Backfill edge signed weight/deacy metadata.
  - Produce immutable baseline snapshot path argument unchanged (timestamped copy).
- Keep baseline snapshot untouched by default; write converted artifact under new path.

## 7) Execution order (phase-by-phase)

### Phase 1: schema migration foundation
- Files: `crabpath/graph.py`, `crabpath/embeddings.py`, migration helpers + tests.
- Deliverables:
  - Graph accepts both v0.6 and v2 shape.
  - Save/load round-trip for v2 schemas.
  - Backward-compatible node/edge readers.
  - Graph-only tests for schema validation and migration helpers.
- Independent acceptance:
  - Existing persistence tests pass unchanged.
  - New migration test fixture validates 391/35K migration counts.

### Phase 2: router + traversal
- Files: `crabpath/router.py`, `crabpath/traversal.py`
- Deliverables:
  - deterministic fallback behavior + structured JSON parser.
  - three-tier traversal policy with complete trajectory output.
  - neurogenesis callsite in traversal when novelty criteria met.
- Independent acceptance:
  - Unit tests for tier ordering and router fallback behavior without LLM.

### Phase 3: learning + decay
- Files: `crabpath/learning.py`, `crabpath/decay.py`
- Deliverables:
  - Gu-corrected PG with full-trajectory credit and baseline.
  - weight update application + edge metadata update.
  - periodic decay function and edge decay metadata semantics.
- Independent acceptance:
  - Tests for sign/magnitude of updates in myopic-corrected toy path.
  - reward shaping edge cases and clipping.

### Phase 4: simulator
- Files: `crabpath/simulator.py`, `crabpath/simulate.py`, `scenarios/*.jsonl`
- Deliverables:
  - scripted JSONL runner and interactive stdin mode.
  - JSON dashboard output metrics.
  - scenario corpus for Giraffe, Forbidden Door, Procedure.
- Independent acceptance:
  - scenario run outputs deterministic with mock router.

### Phase 5: CLI + adapter v2
- Files: `crabpath/adapter.py`, `crabpath/cli.py`
- Deliverables:
  - Adapter uses traversal + learning + decay.
  - Feedback routed into learning module.
  - Command behavior unchanged; outputs unchanged JSON schema keys.
- Independent acceptance:
  - CLI golden tests and `--help` output unaffected.

### Phase 6: private repo integration and migration
- Files: private scripts + data migration execution.
- Deliverables:
  - migration outputs schema-correct graph copy.
  - query log replay uses v2 feedback loop.
  - baseline snapshot preserved.
- Independent acceptance:
  - private migration script dry run and applied run produce expected counts.

## 8) Modified files and exact changes

### `crabpath/graph.py` (target 320-360 LOC)
- Add schema constants for valid node types and required keys.
- Extend `Node` dataclass:
  - new fields: `summary`, `type`.
  - keep existing fields, including legacy compatibility aliases.
- Extend `Edge` dataclass:
  - add `decay_rate`, `last_followed_ts`, metadata with `created_by`, `follow_count`, `skip_count`.
- Add migration helpers:
  - `Graph.normalize_node(record)`
  - `Graph.normalize_edge(record)`
  - `Graph.to_v2_dict()`
- Ensure `from_dict` handles absent v2 keys.
- Keep old `add_node`, `add_edge`, merge and load/save semantics.
- Add lightweight utility `prune_zero_weight_edges` and `active_edge_count` (non-breaking).

### `crabpath/embeddings.py` (target 220-260 LOC)
- Verify/add methods:
  - `upsert(node_id, vector, metadata)`
  - `remove(node_id)`
  - `raw_scores(query)` returning `list[tuple[node_id, float]]`
- Optional normalization: if `summary` index exists, use `summary` first for quick previews.
- Keep all old APIs.

### `crabpath/activation.py` (new file status)
- Legacy file kept for compatibility only, import shim:
  - `from .traversal import traverse as activate`
  - `from .learning import make_learning_step as learn`
  - Marked deprecated; logs warning when imported directly.

### `crabpath/traversal.py` (new 280-360 LOC)
- Replace all mechanical traversal internals with tiered selection + LLM policy.
- Emit `TraversalTrajectory` with rich decision metadata.

### `crabpath/learning.py` (new 260-340 LOC)
- Replace `learn()` and STDP-style internals.
- Implement Gu-corrected PG + full trajectory credit.
- Integrate feedback mapping and baseline tracking.

### `crabpath/neurogenesis.py` (rewired)
- Keep only shared heuristics/constants used by scripts.
- Remove usage as active pipeline driver.
- Expose utility stubs for optional manual/diagnostic use.

### `crabpath/adapter.py` (target 320-380 LOC)
- Query pipeline:
  - tokenized query, embedding seed, traversal, answer synthesis.
  - call learning only for explicit feedback or implicit success signals.
- Replace ad-hoc novelty checks with traversal-invoked `maybe_spawn_knowledge_nodes`.
- Replace legacy feedback handler with `learning.apply_feedback_reward`.

### `crabpath/cli.py` (target 360-420 LOC)
- Keep top-level commands untouched.
- Update command handlers internally to call v2 modules.
- Ensure JSON output keys still include: `status`, `query`, `answer`, `nodes`, `edges`, `elapsed_ms` (or existing contract keys).

### `crabpath/feedback.py` (legacy compatibility)
- Keep module with deprecation warning.
- Export compatibility wrappers:
  - `apply_feedback(...) -> apply_feedback_reward(...)`
  - remove direct internal learning state.

### `tests/*.py` (target total `80+`)
- Existing graph/persistence tests kept and expanded if schema changes.
- activation tests converted to traversal tests.
- neurogenesis tests converted to neurogenesis routing tests + creation policy.
- feedback tests converted to learning integration tests.

## 9) New scenario files

Create `scenarios/` with JSONL files:
- `scenarios/giraffe_test.jsonl`
- `scenarios/forbidden_door.jsonl`
- `scenarios/procedure_test.jsonl`

Each line format:

```json
{"name":"giraffe baseline","query":"Who is the strongest in the room?","expected_answer_fragments":["Giraffe"],"expected_path_contains":["giraffe"],"feedback":{"reward":1,"outcome":"correct","explanation":"Correctly avoided giraffe trap"},"max_turns":1}
```

Include expected_path_contains for regression that checks path-level learning behavior.

## 10) Test plan (with explicit mapping)

### Keep (minimal change)
- Graph tests: Node/Edge/Graph persistence invariants.
- Embeddings tests: build/upsert/remove/search.
- CLI argument/output formatting tests.

### Replace
- `tests/test_activation.py` → `tests/test_traversal.py`
  - reflex/habitual/dormant ordering
  - deterministic fallback on router failure
  - trajectory completeness
  - context assembly ordering and truncation
- `tests/test_neurogenesis.py` → `tests/test_traversal_neurogenesis.py`
  - novelty triggers
  - summary/type assignment expectations

- `tests/test_feedback.py` and activation learning sections → `tests/test_learning.py`
  - Gu-corrected per-step deltas
  - baseline behavior
  - full-trajectory credit assignment
  - feedback mapping to edge updates

### Add new tests
- `tests/test_simulator_integration.py`
  - run scenario file end-to-end with mocked router
  - verify JSON dashboard fields and episode summary
- `tests/test_giraffe_corrected.py`
  - run myopic wrong policy vs corrected policy and show improved cumulative reward over repeated episodes
- `tests/test_forbidden_door.py`
  - verify guardrail node blocks hazardous branch while alternative exists
- `tests/test_decay.py`
  - time/turn based decay and clipping behavior
- `tests/test_adapter_v2_contract.py`
  - public adapter methods still exist and return expected schema.

### Suggested count target
- Existing tests: ~84
- New/rewrites: +20 to +35
- Target total: `100-120` (>=80 guaranteed).

## 11) Line count and dependency order estimate

### Estimated LOC by file
- `crabpath/graph.py`: 320-360
- `crabpath/embeddings.py`: 220-260
- `crabpath/router.py`: 180-240
- `crabpath/traversal.py`: 280-360
- `crabpath/learning.py`: 260-340
- `crabpath/decay.py`: 70-110
- `crabpath/simulator.py`: 260-340
- `crabpath/adapter.py`: 320-380
- `crabpath/cli.py`: 360-420
- `crabpath/feedback.py`: 40-80 (compat shims)
- `tests/test_traversal.py`: 120-200
- `tests/test_learning.py`: 160-240
- `tests/test_simulator_integration.py`: 120-180
- `tests/test_giraffe_corrected.py`: 80-140
- `tests/test_forbidden_door.py`: 60-120
- `tests/test_decay.py`: 40-90
- `scenarios/*.jsonl`: 150-400 lines

### Dependency order (topological)
1. `graph.py` and `embeddings.py`
2. `decay.py` (leaf utility)
3. `router.py`
4. `traversal.py` (depends on graph+embeddings+router)
5. `learning.py` (depends on graph+traversal trajectory structs)
6. `adapter.py` + `cli.py` (depends on router/traversal/learning/decay)
7. `simulator.py` (depends on all above)
8. tests.

## 12) Risk register
- Structured LLM output drift: mitigate with strict schema + retries + heuristic fallback.
- Reward sparsity: mitigate with baseline + path bonuses + per-step eligibility decay.
- Legacy snapshot compatibility: maintain reader fallback and keep dual-format migration.
- Test determinism: enforce fixed random seed and deterministic fallback path.
- Private/public boundary leakage: ensure scripts in public repo only reference local public paths and env var keys never hardcoded.

## 13) Acceptance criteria (v2 “ready” definition)
- All old graph/persistence tests pass.
- v2 traversal produces deterministic trajectory with mocked router.
- Learning updates change expected edges for corrected feedback in the giraffe scenario.
- Forbidden-door path is avoided with guardrail preference.
- Simulator produces dashboard and scenario pass/fail metrics.
- Private migration scripts run on a graph copy while keeping baseline snapshot untouched.
