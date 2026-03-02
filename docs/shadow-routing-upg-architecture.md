# Shadow Routing + Ultimate Policy Gradient (PR 2)

This PR introduces explicit protocol and policy boundaries without changing CLI/API behavior.

## Two timescales

- Runtime fast path: synchronous daemon query handling (`query`) with deterministic retrieval and routing decisions.
- Async teacher updates: background/ops flows (`async_route_pg` and related maintenance/learning tasks) that improve edge policy over time.

The runtime path must remain low-latency and deterministic. Teacher updates can be slower and can aggregate broader evidence.

## Route function contracts

- `runtime_route_fn`: deterministic function used during traversal to rank habitual edge candidates for a single query.
  - Inputs: source id, candidate edges, query text.
  - Behavior: deterministic sort by score, then target id tie-break.
  - Ownership: `openclawbrain.policy.make_runtime_route_fn`.

- `async_route_fn`: teacher-side policy update behavior (outside this PR), used to adjust route behavior based on delayed outcomes.
  - Inputs/outputs may evolve with teacher signals and reward decomposition.
  - Must not break runtime determinism contract.

## Why protocol + policy modules

- `openclawbrain.protocol`:
  - Adds versioned request/response helpers and typed query parameter parsing (`QueryRequest`, `QueryParams`, `QueryResponse`).
  - Centralizes validation and conversion from raw dicts.
  - Keeps error semantics deterministic and consistent.

- `openclawbrain.policy`:
  - Isolates routing policy shape (`RoutingPolicy`) from daemon request handling.
  - Encapsulates deterministic runtime routing function construction.
  - Makes policy behavior testable without daemon process setup.

This split keeps daemon focused on orchestration, while protocol and policy remain independently testable modules.

## Reward source weighting

Async updates now support explicit reward channels:

- `human` (default weight `1.0`)
- `self` (default weight `0.6`)
- `harvester` (default weight `0.3`)
- `teacher` (default weight `0.1`)

`async-route-pg` applies policy-gradient outcomes as:

`scaled_outcome = scale_reward(score_scale * teacher_score, reward_source, reward_weights)`

Current behavior remains backward compatible because the default source is `teacher`, and teacher scores still drive updates exactly as before except for the source multiplier.

## Trace schema (state/action/candidate set)

PR2 introduces first-class replayable routing traces in `openclawbrain.trace`:

- `RouteCandidate`
  - `target_id`, `edge_weight`, `edge_relevance`, optional `similarity`
  - `target_preview`, `target_file`, `target_authority`
- `RouteDecisionPoint`
  - `query_text`, `source_id`, `source_preview`
  - `candidates[]`
  - `teacher_choose[]`, `teacher_scores{}`
  - `ts`, `reward_source`
- `RouteTrace`
  - `query_id`, `ts`, optional `chat_id`
  - `query_text`, `seeds`, `fired_nodes`
  - `traversal_config`, `route_policy`
  - `decision_points[]`

Determinism contract:

- JSON serialization uses sorted keys.
- Candidate ordering is deterministic: edge weight descending, then `target_id` ascending.
- Trace JSONL can be replayed exactly for reproducible teacher-labeling runs.

## Replay and teacher labeling flow

`async-route-pg` supports two execution paths:

1. Build traces from recent journal query events.
2. Or load prebuilt traces via `--traces-in`.
3. Optionally persist traces before labeling via `--traces-out`.
4. Label decision points via teacher (`openai` or custom labeler).
5. Apply PG updates using weighted reward scaling.

This split decouples trajectory sampling from supervision so label policies can be rerun deterministically against fixed decision-point sets.
