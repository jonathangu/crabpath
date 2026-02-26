> **Note:** This design doc is historical. The implementation lives in crabpath/*.py. See ARCHITECTURE_REVIEW.md for current architecture.

# Activation v2 Design: LLM-Guided Graph Activation for CrabPath

## 0) Scope and baseline

`activate()` currently routes energy mechanically with `weight × signal` and STDP updates weights after fact in `learn()`. This v2 design introduces an LLM in the propagation loop while preserving the existing graph semantics and API where possible.

Current code touchpoints:

- `crabpath/graph.py` stores neurons (`Node`) and directed weighted edges (`Edge`).
- `crabpath/activation.py` runs the synchronous leaky integrate-and-fire pass.
- `crabpath/adapter.py` provides query/seed/activate flow and existing auto-node creation hooks.
- `crabpath/neurogenesis.py` provides novelty gating + deterministic node IDs.
- `crabpath/feedback.py` stores delayed feedback snapshots used for learning.
- `NEUROGENESIS_DESIGN.md` and `IMPLEMENTATION_PLAN.md` already establish auto-node creation and an habitual-tier control path.

Paper context confirms current claim: CrabPath learns what gets loaded, not just what is semantically similar, and encodes sequencing via STDP timing.

## 1) ARCHITECTURE: LLM-in-the-loop activation loop

### 1.1 Core state and contracts

Introduce two optional data structures without changing node/edge dataclasses:

- `RouteEvent`: `{source, target, multiplier, confidence, reason, edge_exists, created, flagged_for_removal}`
- `ActivationContext`: holds user message, fired node states, pending LLM decisions, and per-step route logs.

Keep all existing fields in `Firing` and add optional companion object:

- `Firing.routes: dict[(source_id, target_id), RouteEvent]`
- `Firing.route_metadata: dict[str, object]`
- `Firing.new_nodes: list[dict]`
- `Firing.prune_candidates: list[dict]`

This keeps backward compatibility for current CLI/tests while enabling learning to use router outputs.

### 1.2 Activation step-by-step

For each `activate_v2(graph, seeds, user_message, max_steps, ...)`:

1. Decay traces and apply seeds exactly as today.
2. For each step:
3. Compute `to_fire` from mechanical threshold + not-fired check.
4. For each fired node, collect outgoing edges and classify each candidate with tier policy.
5. Build a single batched LLM request for the whole step containing all non-reflex candidates.
6. Apply routing multipliers:
7. Apply inhibitory and excitatory transfers for each outgoing target.
8. Seed any newly created nodes into current turn state.
9. Apply refractory reset and leak.
10. Optionally create/remove nodes mid-step from LLM decisions.
11. Continue until `max_steps`.

### 1.3 Input format to LLM (per step)

- `user_message`: raw user text
- `global_context`: session tags, open context IDs, turn index, novelty signals
- `fired_nodes`: up to `max_fired_context` entries with
  - `node_id`, `content`, `energy_at_fire`, `trace`, `step_fired`, `metadata`
- `outgoing_candidates`: per fired node list of edges with
  - `source_id`, `target_id` (nullable for absent edge), `edge_weight`, `edge_tier`, `seed_overlap`, `edge_context`
- `missing_concepts`: candidate concept tokens from user not mapped to known nodes
- `policy`: thresholds and safety constraints

Output schema is strict JSON (one-shot parse).

### 1.4 Number of LLM calls per activation

- `0` calls: all candidates are reflex tier or early bypass confidence gate.
- `1` call/step: one batched call for all habitual/novel candidates in that step.
- `max_steps` = 3 by default, so up to `3` routing calls.
- Optional pre-step novelty call before propagation if no seeds and high novelty risk.
- Typical upper bound = 4 calls/turn, controlled by budget-aware short-circuit.

## 2) COST MODEL

### 2.1 Token estimate

Let:

- `B`: base prompt tokens (~180-320)
- `N`: fired nodes in request
- `E`: candidate edges sent to model
- `C_n`: tokens per node (`~36`)
- `C_e`: tokens per edge (`~24`)

Approx:

`input_tokens = B + N * C_n + E * C_e`

`output_tokens`: target 120-220 (strict JSON, concise).

Example:

- `N=8`, `E=24`, `B=240` ⇒ ~840 input tokens.
- `output=180`.

### 2.2 Cost with current public rates (as of 2026-02-25)

- OpenAI GPT-4o-mini: $0.15 in / $0.60 out per 1M tokens.
  - 840 in + 180 out ≈ `$0.000228`/call.
- Gemini 2.5 Flash: $0.30 in / $2.50 out per 1M tokens (text) on the standard tier.
  - 840 in + 180 out ≈ `$0.000774`/call.

At 1.0 call/turn:

- GPT-4o-mini ≈ `$0.00023`/turn.
- Gemini Flash ≈ `$0.00077`/turn.

At 3 calls/turn:

- GPT-4o-mini ≈ `$0.00068`/turn.
- Gemini Flash ≈ `$0.00232`/turn.

### 2.3 Budget ceilings

Recommended guardrails:

- Hard token ceiling per call: 1200 in, 240 out.
- Hard spend ceiling per turn: `$0.003` for Gemini, `$0.001` for GPT-mini.
- Monthly spend kill-switch default: 20% of budget allocated for retrieval/context operations.
- Budget-aware bypass: if expected tokens × expected calls exceeds ceiling, skip LLM and use mechanical routing.

## 3) HYBRID THREE-TIER SYSTEM

### 3.1 Tier definitions

- Reflex: `edge.weight > 0.8` or `edge.weight < -0.5`.
  - Mechanical routing, no LLM.
- Habitual: `0.2 <= edge.weight <= 0.8` or `-0.5 <= edge.weight <= -0.2`.
  - LLM routing only.
- Novel: no edge exists, or concept absent in graph.
  - LLM may create node, choose target, return positive/negative multiplier, and emit routing decision.

### 3.2 Decision precedence

- If any novel concept is asserted in user message, route a `create_node_here` action at highest priority for that step.
- Novel node insertion is speculative until step-local validation passes:
  - not duplicate (`deterministic_auto_id`-style check)
  - content quality gate (min chars, no blocked chatter phrases)
  - top-edge relevance threshold met

### 3.3 Candidate selection

For each fired node, compute outgoing candidates in this order:

1. existing reflex candidates
2. existing habitual candidates
3. conceptual candidates extracted from user message
4. recently co-firing symbolic neighbors (for consolidation continuity)

## 4) PROMPT ENGINEERING AND FAST INFERENCE CONTRACT

### 4.1 Required behavior

- `temperature=0`
- `max_output_tokens=220`
- `response_format` = JSON (or structured outputs where supported)
- `stop` on schema end
- reject hidden chain-of-thought

### 4.2 Actual system+user template

Use this template exactly with provider-level templating:

```
SYSTEM:
You are CrabPath Router, a deterministic graph activation policy controller.
Return strict JSON only, no markdown, no prose.
- multipliers are in [-1.000, 1.000]
- do not fabricate target nodes
- do not return duplicate route entries
- never create a node whose id is missing or empty
- if uncertain, use neutral multiplier 0.0 with confidence <= 0.35
- if user directly negates a concept, prefer negative multiplier for that path

USER:
TURN_ID={{turn_id}}
MODEL_VERSION={{model_version}}
USER_MESSAGE={{user_message}}
SEED_NODES={{top_seed_nodes JSON}}
FIRED_NODES={{fired_nodes JSON}}
CANDIDATE_EDGES={{candidate_edges JSON}}
NOVELTY_CONTEXT={{novelty_signals JSON}}
POLICY={{policy JSON}}

Return JSON with keys:
{
  "step": int,
  "routes": [
    {
      "source_id": "string",
      "target_id": "string or null",
      "multiplier": float,
      "confidence": 0.0,
      "edge_exists": bool,
      "create_node_here": false,
      "prune_target": false,
      "reason": "short rationale"
    }
  ],
  "new_nodes": [
    {
      "node_id": "string optional",
      "content": "string",
      "from_source_ids": ["string"],
      "proposed_threshold": 0.8,
      "initial_in_edges_from": ["string"],
      "proposed_out_edges_to": ["string"],
      "confidence": 0.0
    }
  ],
  "prune_candidates": [
    {
      "node_id": "string",
      "reason": "superseded|contradicted|stale",
      "confidence": 0.0
    }
  ],
  "route_summary": {
    "total_edges_considered": int,
    "mutations": int,
    "skipped": int
  }
}
```

### 4.3 Output parser (strict)

- Parse with UTF-8 JSON only.
- Validate all entries against schema.
- Drop rows with out-of-range multiplier.
- Clamp values to bounds only when value is numeric and parse succeeded.
- If parse fails, fall back to deterministic baseline.

### 4.4 Parsing speed

One-shot parser path should be schema-driven and allocation-light:

- no regex extraction
- no markdown stripping beyond basic `.strip()`
- one `json.loads`
- fast-fail on schema mismatch by route count cap and numeric checks

## 5) LEARNING FROM ROUTING DECISIONS

### 5.1 Separation of concerns

Introduce a router signal `m_uv` and timing signal `f(dt)` and keep STDP semantics.

- `m_uv` comes from LLM multiplier in `[-1,1]` for every candidate edge used in this activation.
- `f(dt)` is unchanged from current timing factor in `learn()`.
- For edges without LLM input, fallback `m_uv = tanh(edge.weight / w_ref)`.

### 5.2 Edge update rule

Let:

- `w` = current edge weight.
- `r` = route multiplier from LLM (or fallback fallback).
- `g` = confidence in decision in `[0,1]`.
- `y` = outcome in `[-1,1]`.
- `f` = timing factor.
- `η_route = 0.25` (new default for v2).

```
z = max(-1.0, min(1.0, w / w_ref))
target = (1.0 - λ_safety) * z + λ_safety * r
Δw_llm = η_route * y * f * g * (target - z)
Δw_stdp = η_stdp * y * f
w_new = clamp(w + Δw_llm + Δw_stdp, -10, 10)
```

Take `w_ref = 1.0` for routing-sensitive learning.

Interpretation: repeated `r=0.9` pushes effective policy toward high positive routing if outcomes are positive; `r=-0.8` pushes away. If `η_route=0.25`, one positive outcome moves strongly and repeatedly, while negative outcomes reverse it. Existing STDP remains present but becomes a smoothed second-order correction with the same `y * f` asymmetry.

### 5.3 Edge creation from router

If `create_node_here` is requested and novelty is strong:

- Add node immediately in same turn.
- Create provisional weak edges with signed weights from `m_uv * 0.15`.
- Mark edge metadata:
  - `origin="llm_router"`
  - `created_at_turn`
  - `first_route_mult`
  - `route_count`

### 5.4 Bridging to existing `learn()`

Keep `learn()` signature but add optional `router_feedback` argument:

- `learn(graph, result, outcome, route_events=None, route_rate=0.25)`.
- If route events missing, fallback behavior identical to current v0.6 mechanics.
- If present, apply route-smoothed update first, then classical STDP and edge creation gates.

## 6) FAILURE MODES AND CIRCUIT BREAKERS

### 6.1 Hallucinated edges/nodes

- Validate `target_id` must exist unless `create_node_here=true`.
- For new nodes, apply strict content filters:
  - length 8..220
  - no blocked tokens list (`hello`, `thanks`, etc.)
  - no all-punctuation
  - dedupe with hash + embedding similarity guard
- Require at least 2/3 agreement between router calls if same edge appears contradictory across two adjacent steps.

If node quality is low:

- `state=probationary`
- not added to durable context ranking
- only persistent after 2 successful firings or one explicit positive outcome

### 6.2 Slow or unstable latency

- Timeout per call: 350ms p95 target, 800ms hard cap.
- On timeout: return mechanical result for that step and mark degraded mode for next step.
- If 3 consecutive hard-timeouts in a session: disable routing for N=25 turns.

### 6.3 Parse/contract failures

- Malformed JSON or missing required keys: skip LLM for the step.
- Invalid numeric values: clamp and continue.
- Route for non-existing target: ignore and log warning event.

### 6.4 Unsafe inhibition and oscillation

- Hard cap on negative energy injection per step.
- If inhibition makes a node oscillate above threshold-rebound pattern for >3 turns, clamp target multiplier to 0.0 for that node/step.
- Track `inhibited_count_by_node` and auto-quarantine if above anomaly limit.

### 6.5 Garbage growth and churn

- per-turn new nodes budget: max 2.
- per-session new nodes budget: configurable (default 1/turn max, 60/session/day).
- per-edge new route mutation cap per turn: 48.
- periodic consolidation continues to prune weak edges and orphans.

## 7) COMPARISON TO EXISTING WORK

### 7.1 GraphRAG (Microsoft)

- GraphRAG builds/queries a knowledge graph (entities + relationships) before retrieval; edge traversal is mostly symbolic and static per indexing cycle.
- CrabPath v2 routes in a running activation dynamics loop with continuous firing/inhibition and outcome feedback.
- New contribution: edge strengths are policy parameters at inference time, not retrieval indices only.

### 7.2 Generative Agents

- Generative Agents stores episodic memory and uses prompt logic to pick context from recency and relevance.
- CrabPath v2 introduces per-edge signed multipliers in a recurrent graph dynamic.
- Novelty: LLM performs low-latency synaptic-style decisions repeatedly inside the propagation step, not once per response.

### 7.3 MemGPT

- MemGPT has explicit memory hierarchy and controller actions (write/read/summarize) managed by an LLM.
- CrabPath v2 keeps graph operations local and differentiable-like (weights/threshold dynamics) while still using compact router signals.
- Novelty: router is not a full agent controller, only a routing micro-controller.

### 7.4 Voyager

- Voyager-style agents use long-lived memory plus LLM-evolved skill modules and exploration strategy in environments.
- CrabPath v2 differs by learning retrieval-policy structure from node co-firing and by having explicit inhibitory edges in the same graph.
- Novelty: graph edges are both inhibitory and excitatory propagation channels updated by outcomes.

### 7.5 Think-on-Graph

- Think-on-Graph systems reason over graph nodes/paths externally before answering.
- CrabPath v2 differs by integrating routing inside a neuron-like activation loop, with local state (`trace/potential`) and timing-dependent learning.
- Novelty: no external planner is required for every turn; routing is an embedded control law.

## 8) IMPLEMENTATION PLAN

### 8.1 Phase A: minimal viable v2

- Add `RouteEvent` and `ActivationContext` dataclasses in `activation.py`.
- Extend `activate()` to optionally accept `router` callback and `llm_client`.
- Add `--v2` path in `adapter.query()` to collect and forward router feedback but still write through current `learn()` compatibility.
- Add `openai` and `gemini` adapters in a new `router.py` with strict schema parser.
- Add token/cost counters in per-session state.
- Keep no behavior change when `router=None`.

### 8.2 Phase B: LLM-guided routing + real-time neurogenesis

- Add `create_node_here` handling and `new_nodes` application during step.
- Add prune candidates as low-priority metadata flags (deferred deletion).
- Add `learn(..., route_events=...)` path.
- Add probationary lifecycle for newly created nodes.

### 8.3 Phase C: production hardening

- Add circuit-breaker and fallback telemetry.
- Add strict per-session budget controls and short-circuit confidence gate.
- Add regression tests for malformed JSON, malformed node IDs, and prune actions.

### 8.4 A/B testing design

Compare three controllers over same logs:

1. Mechanical baseline (current v0.6 behavior).
2. Mechanical + novelty gating only (no per-edge LLM routing).
3. Full v2 routing (hybrid tiers + real-time neurogenesis + LLM prune candidates).

Use a time-split replay.

Primary metrics:

- task success correction-adjusted
- number of user corrections per 100 turns
- mean activated-node precision/recall against known-good context
- context token footprint
- mean latency
- cost/turn and cost/session

Secondary metrics:

- edge churn rate
- node churn rate
- prune false-positive rate
- inhibitory recall (does harmful paths get blocked when contradicted)

## 9) PAPER FRAMING: WHAT THIS MEANS

This becomes a new claim, not just an optimization patch:

1. CrabPath becomes a learned activation policy network where a compact LLM controls a dynamic routing kernel.
2. The loop is no longer purely mechanical; it is a hybrid controller: mechanical core with learned micro-gating.
3. The novelty is in treating LLM routing as a low-cost, low-latency control signal and treating STDP as the slow, outcome-conditioned integrator of those micro-decisions.
4. This gives a stronger novelty over prior work:
   - not just retrieval (GraphRAG),
   - not just memory controller (MemGPT),
   - not just planning graph (Think-on-Graph),
   - but a policy graph with explicit inhibition, timing asymmetry, and online concept growth.

Potential paper title update:

- "LLM-Guided Synaptic Routing in Agent Memory Graphs"
- Subtitle: "A Hybrid Mechanical-LLM Activation Loop with Timing-Aware Outcome Integration"

Suggested reframe in abstract:

- "CrabPath v2 shifts from fixed weighted propagation to adaptive graph routing where routing preferences are inferred from context per step, then integrated into long-term edge dynamics via outcome-conditioned, timing-aware updates."
