# Shadow Routing + Ultimate Policy Gradient (PR 1)

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

## Next PRs

- Reward-source weighting:
  - Separate reward channels (user correction, task success, explicit directives) with configurable weighting.
- Traces and decision-point schema:
  - Add structured trace artifacts for route decisions and teacher supervision.
- Storage boundary:
  - Introduce clearer persistence interfaces so runtime state and async training state can evolve independently.
