> **Note:** This design doc is historical. The implementation lives in crabpath/*.py. See ARCHITECTURE_REVIEW.md for current architecture.

# CrabPath v2.0 Specification and Paper Rewrite Plan

Date: 2026-02-25

## 1) CRITICAL ANALYSIS OF THE v2 VISION

### 1.1 What is strong

CrabPath v2 has a clear and important idea: replace purely numeric neuron dynamics with an LLM-routing policy while retaining a persistent graph scaffold. That gives the system semantic competence (negation, intent, counterfactual reasoning) without discarding the long-run memory effects already built into the graph.

The current v0.x implementation already has a coherent base for this migration:

- LIF-style activation with thresholds, inhibitory/excitatory edges, trace bookkeeping, and STDP-like learning is implemented in `crabpath/activation.py`.
- A minimal feedback pipeline and delayed attribution helpers already exist in `crabpath/feedback.py` and `crabpath/adapter.py`.
- Incremental neurogenesis already exists in `crabpath/neurogenesis.py` and adapter query path, so there is a concrete place to hang v2 behavior.

Three elements are genuinely compelling and deserve to stay:

- Three-level routing policy (reflex/habitual/novel) gives a direct latency/cost control surface.
- Use-it-or-lose-it decay plus hard pruning provides a natural self-pruning loop.
- Deterministic auto-node IDs and edge caps already reduce duplication and churn.

### 1.2 What is weak

The architecture is currently incomplete as a learning algorithm, not just as an implementation detail.

- Credit assignment remains under-specified. `learn()` in the current code uses STDP timing with a crude `outcome` scalar but does not model whether selected edges were causally useful for the user-visible failure/success.
- The vision implicitly collapses two roles into one set of weights: attention routing and semantic truth value of content. That can create brittle behavior when good answers are stored in poor locations.
- The proposal assumes “LLM is activation function” but does not define an interface contract (JSON schema guarantees, grounding constraints, function-calling mode, stop conditions, deterministic fallback). Without this, one bad JSON parse or policy drift can silently poison traversal.
- “Two- to three-hop” depth is fixed but no explicit coverage guarantee is given for multi-step procedures. A shallow depth with no fallback search can miss required tool chains.
- No robust adversarial path controls are specified. With only threshold heuristics, a bad auto-created or high-weight branch can route unsafe outputs repeatedly until decay catches it.
- The v2 summary suggests all seeds may be overridden by LLM judgment. If not explicitly constrained, this can collapse to recency/cognitive bias of the LLM and erase learned structure.
- Version drift exists in artifacts: `__version__` is `0.4.0` while the v2 vision and prompt text refer to `v0.6.0` and larger graph sizes. A paper submission based on these claims must lock artifacts and reproducibility assumptions.

### 1.3 What is missing

- No explicit objective and reward model for routing. Is feedback optimizing task success, safety, token efficiency, or sequence quality? Multiple objectives need explicit scalarization.
- No explicit anti-hallucination contract for LLM-supplied context selection.
- No explicit handling of absent/nearly-duplicate content during neurogenesis. The current `deterministic_auto_id` avoids exact hash collisions but not semantic duplication.
- No policy for protected vs killable nodes beyond `metadata.protected` in consolidation.
- No explicit replay/attribution protocol for delayed feedback in the traversal layer beyond coarse snapshot windowing in existing docs.
- No ablation protocol that directly tests the new routing paradigm against current STDP-only routing.

### 1.4 Where it will break (high-probability failure modes)

1. Infinite loops and cycle explosions are likely if cycle handling is underspecified. Current graph allows arbitrary cycles.
2. The graph can overfit to early LLM preferences because reflex paths become sticky unless explicit anti-entropy is active.
3. False positives in auto-node creation for short/chatter text will still occur even with basic token gates.
4. Sparse-branching sessions will route to near-empty context unless fallback to static/static-with-rules is enforced.
5. Delayed corrections can bias only recent path and wash out long-horizon credit signals, especially if feedback arrives beyond the attribution window.
6. Cost blow-up if the LLM is called per hop (`2-3` calls/turn) instead of per-turn routing pass.
7. Stale embedding index risk: vector index rebuild lag can produce bad novelty decisions and bad auto-node creation.

---

## 2) FORMAL ARCHITECTURE FOR CRABPATH v2

### 2.1 Core representation

CrabPath v2 treats each node as a document and each edge as a typed, decaying pointer. The activation function is an LLM policy that chooses which pointers to traverse, constrained by edge caches.

### 2.2 Node schema

Use this schema in JSON and persistence.

`Node`:

- `id`: `str`, immutable, unique.
- `content`: `str`, canonical body used as LLM context.
- `type`: `enum["fact","procedure","action","tool_call","guardrail","policy_hint","diagnostic","meta"]`.
- `threshold`: `float`, default `0.8`.
- `potential`: `float`, transient during traversal (optional in storage).
- `trace`: `float`, warm-state signal.
- `metadata`:
  - `created_ts`: `float`
  - `updated_ts`: `float`
  - `source`: `str` (`"manual"|"auto"`)
  - `error_class`: `str|None`
  - `namespace`: `str|None`
  - `fired_count`: `int`
  - `success_count`: `int`
  - `failure_count`: `int`
  - `tool_call_count`: `int`
  - `session_last_seen`: `str`
  - `content_signature`: `str` (hash for dedupe)
  - `auto_probationary`: `bool` (if created via auto-neurogenesis)
  - `version`: `int`
- `lifecycle`:
  - `state`: `enum["normal","probation","protected","retiring","dead"]`
  - `ttl_turns`: `int`
  - `last_seen_turn`: `int`
  - `death_score`: `float`

Recommended serialized JSON node object:

```json
{
  "id": "giraffe-codeword",
  "content": "Giraffe is Jon's codeword for testing CrabPath.",
  "type": "fact",
  "threshold": 0.8,
  "potential": 0.0,
  "trace": 0.12,
  "metadata": {
    "created_ts": 1739750000.0,
    "updated_ts": 1739754000.0,
    "source": "auto",
    "source": "manual",
    "fired_count": 4,
    "success_count": 3,
    "failure_count": 1,
    "tool_call_count": 2,
    "auto_probationary": false,
    "content_signature": "sha1:..."
  },
  "lifecycle": {
    "state": "normal",
    "ttl_turns": 200,
    "last_seen_turn": 1142,
    "death_score": 0.24
  }
}
```

### 2.3 Pointer/edge schema

`Edge`:

- `source`: `str`
- `target`: `str`
- `weight`: `float`, range `[-10.0, 10.0]`
- `kind`: `enum["support","follow_up","inhibit","tool_link","anti_goal"]`
- `created_ts`: `float`
- `updated_ts`: `float`
- `decay_lambda`: `float` in `[0, 1]` (default `0.02` per turn)
- `last_seen_turn`: `int`
- `use_count`: `int`
- `skip_count`: `int`
- `evidence`: `float` cumulative support score
- `created_by_turn`: `int`
- `is_reflex_override`: `bool`
- `lock_version`: `int`
- `decay_floor`: `float` (default `0.0`, optional nonzero for protected links)

Recommended serialized JSON edge object:

```json
{
  "source": "crabpath-testing",
  "target": "codewords",
  "weight": 0.71,
  "kind": "support",
  "created_ts": 1739750000.0,
  "updated_ts": 1739754012.0,
  "decay_lambda": 0.02,
  "last_seen_turn": 1142,
  "use_count": 31,
  "skip_count": 4,
  "evidence": 0.84,
  "created_by_turn": 1097,
  "is_reflex_override": false,
  "lock_version": 3,
  "decay_floor": 0.0
}
```

### 2.4 Three-tier routing policy

- Reflex tier: `weight >= 0.80` follow automatically without LLM call.
- Habitual tier: `0.30 <= weight < 0.80` candidates are submitted to LLM policy.
- Dormant tier: `weight < 0.30` normally skipped.

Global context budget rule:
- `max_nodes_per_turn = 24`
- `max_new_nodes_followed_per_turn = 6`
- `max_hops = 3`

### 2.5 Traversal algorithm (pseudocode)

```text
function v2_traverse(query, session_state, turn_id, max_hops=3, k_out=8):
    seeds = semantic_seed(query, top_k=20) ∪ symbolic_seeds(session_state)
    candidates0 = normalize_and_rank(seeds)

    # novelty and optional node creation happens before traversal
    novelty = assess_novelty(query, raw_scores(candidates0))
    if novelty.should_create:
        auto_node = create_or_refresh_auto_node(query, candidates0, session_state)
        candidates0[auto_node.id] = max(candidates0.get(auto_node.id, 0.0), session_state.auto_seed_boost)

    frontier = priority_queue(by=init_score_desc)
    for n in top_k(candidates0, 20):
        frontier.push((n.id, init_score(n), 0))

    visited = map(node_id -> 0)
    fired = []  # ordered list of (node_id, depth, evidence)
    follow_log = []

    for step in 0 .. max_hops-1:
        if frontier.empty(): break

        current_layer = pop_frontier(frontier)
        for (node_id, path_score, depth) in current_layer:
            if visited[node_id] >= 1: continue
            if depth >= max_hops: continue
            if node_dormant(node_id): continue

            fire_node(node_id, path_score)
            fired.append((node_id, depth, path_score))
            visited[node_id] += 1

            outgoing = outgoing_edges(node_id)
            reflex_edges = filter(outgoing, weight >= 0.80)
            dormant_edges = filter(outgoing, weight < 0.30)
            llm_edges = filter(outgoing, 0.30 <= weight < 0.80)

            selected = []
            for e in reflex_edges:
                selected.append((e.target, e, "reflex", depth+1))

            if should_call_llm(node_id, llm_edges, step, session_state):
                llm_decision = policy_llm_select(
                    node_id=node_id,
                    node_content=node.content,
                    outgoing=decorate(outgoing),
                    context=query,
                    budget=k_out,
                    guardrails=session_state.guardrails,
                    visited=visited,
                )
                for choice in llm_decision.follow:
                    selected.append((choice.target, choice.edge, "llm", depth+1))

            for e in selected:
                if e.weight >= 0.30 and not creates_cycle(visited, e.target, remaining_depth=max_hops-depth-1):
                    frontier.push((e.target, path_score * sigmoid(e.weight), depth+1))
                    follow_log.append((node_id, e.target, e.weight, decision_channel(e)))

            apply_cycle_fallback(visited, fired, max_hops)

    return fired, follow_log, context_bundle(fired, top_k=24)
```

Cycle handling:
- `visited[node_id]` limit is 1 for strict safety, or 2 if `session_state.exploration_mode`.
- If a candidate target appears in the last `L=2` steps, skip.
- If frontier stagnates before `max_hops` due to cycle suppression, terminate early.
- A hard guardrail: if no node added in two consecutive steps, terminate.

### 2.6 Learning rules

Let edge `e=(i,j)`.

Signals:

- `d_{ij} = 1` if edge was followed in this turn, else `0`.
- `o` outcome scalar in `[-1, 1]`.
- `A_{ij}` attention at edge-time as LLM score (`0` for auto-skipped reflex edges; defaults to `0.5` for auto-follow).
- `u_{ij}` utility decay by traversal depth from source node firing: `u_{ij}=exp(-α * depth_gap)`.
- `g_{ij}` anti-oscillation gate from recent sign stability: `g_{ij} = sigmoid(0.5*(use_count-skip_count))`.

Immediate routing cache update (always applied after traversal):


a_f = +η_f
b_f = -η_f

w_{ij} ← clip(w_{ij} + η_{
follow}*(2d_{ij}-1), -W_{max}, W_{max})
```

with `η_follow = 0.02`, `W_max = 10.0` by default.

Delayed feedback update:

```
Δw_{ij}^{feedback} = η_{fb} * o * (2d_{ij}-1) * A_{ij} * g_{ij} * u_{ij}
```

Update equation:

`w_{ij} ← clip(w_{ij} + Δw_{ij}^{feedback}, -10.0, 10.0)`

This gives the four events in the vision:

- followed + positive: `d=1, o>0` => large positive update
- followed + correction: `d=1, o<0` => large negative update
- followed but no feedback: only tiny `η_follow` push
- skipped: `d=0` gives small negative drift

Optional STDP-aware temporal modifier (only for followed edges):

```
f(Δt) =
  1/(1+Δt)            if Δt > 0
  0.5                  if Δt = 0
  -0.5/(1+|Δt|)       if Δt < 0
```

Then `η_fb` is multiplied by `f(Δt)` to keep causal ordering effects.

`Δt` is defined as target fire depth minus source fire depth; zero if unknown.

### 2.7 Neurogenesis triggers and node creation logic

A node is proposed when all gates pass:

- novelty score indicates unknown concept
- token/quality gate passes
- blocklist gate passes
- anti-churn gates allow creation

Novelty test uses raw embedding top hits:

- `top1 >= 0.60` ⇒ known, no creation
- `top1 <= 0.28` and blocked flags false ⇒ noise gate fail, no creation
- `0.28 < top1 < 0.58` and relative prominence gate pass ⇒ candidate
- `top1=0` with no seeds and explicit symbolic hit ⇒ fallback candidate for safety if configured

Quality gate:
- `token_count >= 3`
- normalized alnum ratio above threshold
- no blocked phrases list
- no near-duplicate signature match within hash bucket

Creation and bootstrapping:

1. `node_id = sha1(normalized_query)[:12]` prefixed with `auto:`
2. if node exists, refresh `auto_seed_count`, `last_seen`, and probation counter
3. else create with `threshold=0.8`, `auto_probationary=true`, `source="auto"`
4. connect from:
   - top seeded nodes in current turn (incoming) with initial `w∈[0.10,0.20]`
   - prior-turn fired nodes (incoming/outgoing) with `w∈[0.06,0.12]`
5. upsert embedding immediately and mark index dirty; rebuild asynchronously
6. force-seed into current frontier with low seed boost.

### 2.8 Consolidation and death

Edge decay per turn:

- if `followed` then `w ← w` (or slight decay only for safety)
- if not followed and not in top-context for `M` turns:
  - `w ← w * (1 - decay_lambda)`
  - default `decay_lambda = 0.03`

Consolidation pass:

- prune edge if `abs(w) < 0.05`
- prune edge also if `evidence < 0.02` and `age_turns > 14`
- node death score:
  - `DS = α*|degree| + β*activity - γ*success_rate - δ*age_penalty`
  - if `DS < 0` and node is not protected and no inbound/outbound strong weight, node dies
- orphan removal: node with no incoming/outbound edges and `auto_probationary=true` and no recent fire dies immediately after `probation_turns >= 14`
- duplicate merge: if `jaccard(content_embedding_neighborhood)` high and signature close, merge keeper with larger persistence stats

### 2.9 Exact cost model (LLM calls, tokens, USD)

Assume per-turn routing calls only when not fully reflexive.

Let:
- `p_amb` be the probability that a turn needs LLM-based routing.
- `C_in` input tokens to routing model.
- `C_out` output tokens.
- `r_in`, `r_out` token prices in USD per token.
- `C_embed` embedding call tokens.
- `p_call = 1` in ambiguous turns.
- `p_reflex` turns where all decisions are cached/reflex.

Formula:

`LLM_calls_per_turn = 1 - p_reflex`

`Cost_turn = (1-p_reflex) * (C_in*r_in + C_out*r_out) + C_embed`

Base assumptions (as of 2026-02-25 benchmark snapshot used for planning):
- routing model: `r_in = $0.15 / 1,000,000`, `r_out = $0.60 / 1,000,000`
- embedding model: `r_emb = $0.02 / 1,000,000`
- routing call token budget: `C_in = 900`, `C_out = 180`
- embedding call token budget: `C_embed_tokens = 180`

Then:
- ambiguous-turn cost ≈ `0.00000015*900 + 0.0000006*180 = $0.000225`
- embedding cost ≈ `$0.0000036`
- total ambiguous turn cost ≈ `$0.0002286`
- if `p_reflex=0.70`, expected cost ≈ `$0.00006858`

Per-1k turns/day at 70% reflex:
- `70` costly calls/`1000` turns = `$0.0158`
- `1000` embedding calls = `$0.0036`
- total ≈ `$0.0194`/day

This is acceptable only if routing calls are bounded to ≤1 call/turn. If per-hop calls are used (max 3 calls):
- multiply ambiguous-turn routing cost by up to `3` and cost increases to `$0.0006858` ambiguous turn.
- this must be treated as hard-stop fallback only.

### 2.10 Pseudocode API contract

`traverse(request) -> TraversalResult`

Input:
- `query_text: str`
- `session_state: dict`
- `memory_search_ids: list[str]`
- `workspace_context: dict`
- `heartbeat: dict`
- `top_k_seed: int = 20`
- `max_hops: int = 3`

Output:
- `fired_ids: list[str]`
- `contents: list[str]`
- `follow_log: list[dict{source,target,weight,decision_channel}]`
- `guardrails: list[str]`
- `auto_node: dict|None`
- `novelty: dict`
- `llm_calls: int`
- `cost_estimate_usd: float`
- `trace: list[dict{node_id, depth, energy}]`

---

## 3) CRABPATH PAPER REWRITE OUTLINE (WORKSHOP VERSION)

### 3.1 New title options

1. `CrabPath v2: Learning Routing Policies for Agent Memory with LLM-Guided Graph Traversal`
2. `From Neural Firing to LLM Routing: A Practical Memory Graph for Agent Context Selection`
3. `Cache-Based Neurogenesis for Agent Memory: A Hybrid Graph–LLM Retrieval Architecture`
4. `When Edges Learn: A Three-Tier Policy Routing Graph for Persistent Agent Memory`

### 3.2 200-word abstract draft

Agent memory systems still behave like static retrieval pipelines: they return relevant documents by lexical or embedding similarity, then rely on prompting to decide usage. We present CrabPath v2, a hybrid graph memory architecture that replaces fixed retrieval choice with an LLM-guided pointer-routing policy over a learned directed graph. Nodes are typed documents containing facts, procedures, and tool-call snippets; directed edges encode learned routing priors with signed weights for support and inhibition. A three-tier mechanism controls latency and cost: reflex edges are always followed, dormant edges are skipped, and habitual edges are selected by a low-cost LLM policy model. The model receives node content and candidate outgoing pointers, returns a constrained follow set, and learns from delayed task feedback by updating both edge confidence and routing priors. We retain biological-inspired mechanisms—neurogenesis, decay, and consolidation—while grounding them in software constraints: bounded depth, bounded branching, anti-cycle rules, and explicit edge decay. Automatic node creation is triggered by semantic novelty and correction events, with deterministic IDs, probationary lifecycle, and merge/consolidation safeguards to control graph growth. We evaluate the design against an RAG-only baseline on agent task logs, measuring success-corrected retrieval precision, correction rate over delay, context tokens, and routing latency. CrabPath v2 reframes memory selection as a long-horizon routing optimization problem and provides a practical path to systems that improve over time while keeping per-turn costs bounded.

### 3.3 Section outline with key claims

1. Introduction
- Define the problem: context loading is costly and non-adaptive in current RAG pipelines.
- Position the core claim: memory routing can be learned.

2. Related Work and Positioning
- Contrast with RAG, MemGPT, tool-memory, GraphRAG, and STDP-inspired associative memory.
- Explicitly define what is new in v2 versus prior work.

3. Baseline and Failure Modes
- Summarize existing CrabPath v0.4/v0.5 mechanics.
- Provide concrete cases where pure embedding retrieval fails on procedure/task intent.

4. v2 Architecture
- Describe Node/Edge schema.
- Three-tier routing and depth-constrained traversal.
- LLM policy contract and anti-cycle safeguards.

5. Learning and Adaptation
- Exact equations for routing-cache updates and delayed feedback.
- Decay, consolidation, and auto-node lifecycle.
- Neurogenesis and anti-churn controls.

6. Complexity and Cost
- Asymptotics in node/edge/turn dimensions.
- LLM call budget and token-cost model.

7. Experimental Method
- Offline replay protocol (train/test temporal split).
- Protocol for delayed feedback attribution.
- Baselines and ablations.

8. Results
- Correction rate, first-correction latency, context token load, p50/p95 latency, safety-guardrail adherence.

9. Discussion and Failure Cases
- Credit assignment, exploration collapse, embedding staleness, adversarial loops.
- Limits and mitigation.

10. Future Work
- Better attribution, policy distillation, graph sparsification and cache compression.

### 3.4 Keep / Drop / New relative to v1 paper

Keep:
- conceptual mapping of nodes/edges and inhibitory edges
- core STDP/temporal asymmetry story
- consolidation/contraction story
- explicit acknowledgment of credit-assignment weakness

Drop:
- extensive neuroscience metaphor as primary proof language
- broad mathematical detours (spectral notes, Krylov speculation) unless tied to experiments
- long static exposition style; replace with implementation-first framing

New:
- LLM-guided routing policy contract with bounded JSON I/O
- formal three-tier decision policy as systems control
- explicit cost model and latency budget
- explicit anti-cycle and safety logic
- neurogenesis lifecycle and probationary policy
- ablation and rollout protocol for workshop-level evaluation credibility

### 3.5 Novelty claim (one sentence)

CrabPath v2 is the first architecture we know of that treats agent memory as a bounded-depth, three-tier graph policy where an LLM performs constrained pointer routing while a learned signed edge cache preserves long-run behavior and keeps most turns on deterministic, low-cost execution.

### 3.6 Minimum viable experiment

A feasible and publication-safe minimum includes:

- 20-day or longer historical agent log corpus with tool traces and delayed corrections.
- Temporal split: first 70% turns for initialization, final 30% for evaluation.
- Systems compared:
  - Baseline A: embedding retrieval + fixed prompt selection.
  - Baseline B: current CrabPath v0.4-style firing selection.
  - Treatment C: CrabPath v2 with 2-hop and 3-hop settings.
  - Ablation C1: no three-tier; C2: no delayed feedback; C3: no neurogenesis.
- Metrics:
  - correction rate and correction lag at 1/3/5 turns
  - task success rate, known-bad-action count
  - median/95th percentile latency
  - average context tokens and LLM routing calls/turn
  - edge churn and stability (`pruned/created` ratio)
- Success criteria:
  - statistically significant reduction in correction rate and context tokens
  - no worse than baseline in latency budget violations.

### 3.7 Exact references to include

- Collins & Loftus, 1975. A spreading-activation theory of semantic processing.
- Daw, Niv & Dayan, 2005. Uncertainty-based competition between brain systems.
- Fields, 2008. White matter in learning and cognition.
- Graybiel, 2008. Habits and evaluative systems.
- Gu, J. 2020. Correcting update rules in reinforcement learning.
- Lewis et al., 2020. Retrieval-augmented generation.
- Packer et al., 2023. MemGPT.
- Page et al., 1999. PageRank.
- Sutton, Precup & Singh, 1999. Between MDPs and SMDPs.
- Microsoft Research GraphRAG (2024).
- Any source of LLM token pricing used in the final cost table.

---

## 4) IMPLEMENTATION-LEVEL RISKS TO DECIDE NOW

- Decide whether a single routing LLM call per turn is mandatory; if yes, enforce compact JSON schema and strict truncation fallback.
- Decide whether v2 uses only in-memory node-level potentials or no potentials at all (pure policy graph).
- Decide whether graph persistence keeps dynamic fields (`potential`, `trace`) or reconstructs them from event logs only.
- Decide a single attribution policy now (winner-takes-most or fractional across turns) to prevent silent divergence later.
- Decide whether neurogenesis is on by default or explicit rollout-gated with feature flags.
