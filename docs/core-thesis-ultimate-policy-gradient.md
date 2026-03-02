# The Core Thesis: Shadow Routing + Ultimate Policy Gradient (OpenClawBrain)

Date: 2026-03-02

## TL;DR
OpenClawBrain should feel *fast* at runtime and *smarter* every day.

We get both by splitting the system into two loops:

1) **Runtime (fast, deterministic, no LLM calls):**
   - Embed the query.
   - Retrieve seed nodes.
   - Traverse a learned memory graph.
   - Emit a compact `prompt_context`.

2) **Learning (async, parallel, high leverage):**
   - Replay recent traversal decision points.
   - Ask a teacher model what it *would have routed to*.
   - Convert dense labels into policy-gradient updates over edges.

This yields **fewer turns, fewer tokens, better context**, while preserving stable runtime latency.

---

## Why this matters
The failure mode of most "memory" systems is *interactive pointer chasing*:

- Retrieve something vague → see a pointer → retrieve again → follow another pointer → repeat

That creates:

- More tool calls
- More context bloat
- More turns
- Worse tail latency

OpenClawBrain’s goal is the opposite: **one fast context load** that gets you into the right neighborhood, plus a background learning loop that makes the next context load even better.

---

## Definitions

### `runtime_route_fn` (money-maker)
A **lightning-fast local routing policy** used during traversal to choose which candidate edges to expand.

**Inputs** (conceptually):
- `query_embedding` (already computed)
- `source_node` (current node)
- `candidate_edges` (habitual edges) with:
  - learned `edge.weight`
  - optional dense weak-supervision `edge.metadata["relevance"]`
  - target node vector (cached in index)

**Output:**
- deterministic list of next target node IDs (top-K)

**Key requirement:** no network calls; CPU-only math; deterministic.

### `async_route_fn` (teacher labeler)
An **offline** labeling function that answers:

> Given the query + the current node + candidate edges, what should we expand next?

It can be an LLM (default: `gpt-5-mini`) and is allowed to be slow — because it runs in the background and is heavily parallelized.

**Output (robust JSON):**
- `choose`: a short list of target IDs, and/or
- `scores`: per-target scores in [-1, 1]

---

## The ultimate policy gradient
We treat traversal as a policy over actions (edges):

- **State**: (query, current node, candidate edges)
- **Action**: choose the next edge(s) to expand
- **Policy parameters**: edge weights (+ optional relevance metadata)

The **ultimate policy gradient** idea:

- Human feedback provides *high-authority reward*.
- Self-learning (agent outcomes) provides strong reward.
- Harvester signals provide medium reward.
- Async teacher labels provide *massive volume* of weak supervision.

All of these are different *reward sources* that can be unified into the same policy-gradient update rule.

In practice this looks like:

- Use the teacher to generate dense edge-level labels.
- Apply bounded REINFORCE updates (`apply_outcome_pg`) to edge weights.
- Optionally store teacher signal as `edge.metadata["relevance"]` to influence logits.

The key is **asymmetry**:
- Teacher is lower weight per sample.
- Teacher is higher volume.
- Humans are highest weight.

---

## Runtime routing (deterministic, context-conditioned)
The simplest fast policy that already conditions on context is:

```
score(edge) = edge.weight
          + (edge.relevance if enabled)
          + α * cosine(query_vec, target_vec)
```

Then choose `top_k` edges by score (deterministic tie-break by target_id).

This is "transformer → number" without paying transformer cost twice:
- the transformer already produced the query embedding
- routing is just dot products and adds

---

## Structural plasticity remains core
The background policy-gradient loop does **not** replace structural graph evolution.

We still want (and keep):
- **split**: break oversized nodes
- **merge**: consolidate duplicates
- **connect**: propose new edges
- **prune**: remove dead wood
- **inject**: create new nodes from explicit corrections/teachings/directives

The new async teacher + PG loop makes these mechanisms smarter by producing:
- cleaner trajectories
- denser, less noisy edge-strength signals
- better edge relevance metadata

---

## Implementation milestones

### Milestone 0 (shipped)
- `async-route-pg`: teacher shadow loop + PG updates (dry-run by default; `--apply` gated)

### Milestone 1 (shipped)
- `route_mode=edge+sim`: runtime query-conditioned routing based on embeddings + learned weights + relevance

### Milestone 2 (shipped)
- Explicit reward-source weighting (`openclawbrain.reward.RewardWeights`) across human/self/harvester/teacher.
- Replayable route traces in `openclawbrain.trace` + unified labels in `openclawbrain.labels`.
- Learned runtime route policy in `openclawbrain.route_model` + `route_mode=learned`.
- Storage boundary interfaces in `openclawbrain.storage` for state/event persistence.

### Next milestones
- Add an eval harness that measures: token count, turn count, and retrieval correctness.
- Add shortcut/pointer edge creation offline (teacher-assisted connect during maintain).

---

## How to run (operator sketch)
1) Run `async-route-pg` in dry-run and inspect JSON deltas.
2) Apply to a copied state first.
3) Train `route_model.npz` from traces/labels and enable runtime `route_mode=learned`.

---

## The promise
If we do this right, OpenClawBrain becomes a system where:

- Runtime stays fast and predictable.
- Every conversation generates training data.
- The brain improves continuously.
- Humans steer direction; teachers provide volume.

That is the core thesis.

## Math appendix
- [Ultimate Policy Gradient Routing Math](ultimate-policy-gradient-routing-math.md)
