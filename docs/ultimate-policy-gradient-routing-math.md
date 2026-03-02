# Ultimate Policy Gradient Routing Math (OpenClawBrain)

Date: 2026-03-02

This appendix formalizes how OpenClawBrain can translate supervision labels into a learned runtime routing policy over graph edges.

## Status in code today

- `async-route-pg` currently uses teacher `scores` as reward shaping fed into `apply_outcome_pg` (REINFORCE-style updates), not full distribution distillation/KL optimization yet.
- The distillation objective below is a proposed/optional extension, not the default behavior in current v0 code.

## 1) Decision Point and Notation

A single routing decision point is:

\[
\mathcal{D} = (x, u, \{v_i\}_{i=1}^{m})
\]

where:

- \(x\): user query text
- \(u\): current source node (the node being expanded)
- \(v_i\): candidate target nodes reachable from \(u\)
- \(m\): number of candidates at that decision point

Let:

- \(\mathbf{q} \in \mathbb{R}^{d_q}\): query-context embedding (from query and optional source context)
- \(\mathbf{t}_i \in \mathbb{R}^{d_t}\): target-node embedding for candidate \(v_i\)
- \(\mathbf{f}_i \in \mathbb{R}^{d_f}\): scalar edge/candidate features (existing edge weight, authority flags, recency, etc.)

## 2) Router Parameterization (Low-Rank Bilinear / Two-Tower)

A practical learned logit for candidate \(i\):

\[
z_i = (\mathbf{q}A) \cdot (\mathbf{t}_iB) + \mathbf{w}^{\top}\mathbf{f}_i + b
\]

with parameters:

- \(A \in \mathbb{R}^{d_q \times r}\)
- \(B \in \mathbb{R}^{d_t \times r}\)
- \(\mathbf{w} \in \mathbb{R}^{d_f}\)
- \(b \in \mathbb{R}\)
- rank \(r \ll \min(d_q,d_t)\)

Equivalent form:

\[
z_i = \mathbf{q}^{\top} M \mathbf{t}_i + \mathbf{w}^{\top}\mathbf{f}_i + b, \quad M = AB^{\top}
\]

### Precompute target projections

Because target embeddings are relatively stable, precompute:

\[
\tilde{\mathbf{t}}_i = \mathbf{t}_i B
\]

Then online scoring is:

\[
z_i = (\mathbf{q}A) \cdot \tilde{\mathbf{t}}_i + \mathbf{w}^{\top}\mathbf{f}_i + b
\]

This keeps runtime CPU-local and cheap (dot products + adds).

## 3) Policy Over Edges

Define a temperature-scaled softmax policy:

\[
p_i = \pi_\theta(a=i\mid x,u,\{v_j\}) = \frac{\exp(z_i/T)}{\sum_{j=1}^{m}\exp(z_j/T)}
\]

- \(T>0\): temperature
- lower \(T\): sharper policy
- higher \(T\): smoother policy

**STOP note:** training includes STOP in the softmax action set; the STOP logit is fixed today. Learning a STOP logit is future work.

## 4) Teacher Labels to Target Distribution \(\mathbf{y}\)

For each decision point, teacher supervision arrives in two common forms.

### A) `choose` set

If teacher returns a subset \(C \subseteq \{1,\dots,m\}\), construct:

\[
y_i =
\begin{cases}
\frac{1}{|C|}, & i \in C \\
0, & i \notin C
\end{cases}
\]

### B) `scores` map

If teacher returns raw candidate scores \(s_i\), convert to a probability target with label temperature \(\tau\):

\[
y_i = \frac{\exp(s_i/\tau)}{\sum_{j=1}^{m}\exp(s_j/\tau)}
\]

- smaller \(\tau\): emphasizes top teacher choices
- larger \(\tau\): preserves softer preference structure

## 5) Distillation Objective with Reward-Source Weighting

Let each labeled example carry a source tag
\(r \in \{\text{human},\text{self},\text{harvester},\text{teacher}\}\)
with weight \(\lambda_r\).

Per-example cross-entropy:

\[
\ell_{\text{CE}} = -\sum_{i=1}^{m} y_i \log p_i
\]

Weighted training loss over examples \(n=1\dots N\):

\[
\mathcal{L}_{\text{distill}} = \frac{1}{N}\sum_{n=1}^{N} \lambda_{r_n} \, \ell_{\text{CE}}^{(n)}
\]

So higher-authority sources directly contribute larger effective gradient magnitude.

## 6) Policy-Gradient Fine-Tuning (REINFORCE)

After deployment or replay, sample/observe action \(a\) and scalar outcome reward \(R\). With baseline \(b\):

\[
\nabla_\theta J(\theta) = \mathbb{E}\left[(R-b)\nabla_\theta \log \pi_\theta(a\mid s)\right]
\]

where state \(s\equiv(x,u,\{v_i\})\).

Stochastic update form:

\[
\Delta\theta \propto \lambda_r (R-b)\nabla_\theta \log p_a
\]

- \(A = R-b\) is the advantage
- baseline \(b\) reduces variance without changing expected gradient
- \(\lambda_r\) applies the same reward-source hierarchy to RL updates

## 7) Reward-Source Hierarchy

OpenClawBrain policy learning follows:

\[
\lambda_{\text{human}} > \lambda_{\text{self}} > \lambda_{\text{harvester}} > \lambda_{\text{teacher}}
\]

Interpretation:

- Human feedback: highest authority, lower volume
- Self outcomes: strong, medium volume
- Harvester signals: medium trust, high volume
- Teacher labels: weakest per-sample trust, very high volume

This makes the model data-efficient without letting weak supervision dominate trusted supervision.

## 8) Runtime Determinism vs Training Stochasticity

Training can be stochastic (sampling, exploration, replay shuffling). Runtime routing should remain deterministic for stable latency and reproducibility:

1. Compute logits \(z_i\) for candidate edges.
2. Rank by \(z_i\) (deterministic tie-break by target ID).
3. Return top-\(k\) actions.

So randomness is a training-time tool, not a serving-time behavior.

Current mismatch to track: runtime `route_mode=edge+sim` ranks by
\(w_i^{\text{edge}} + r_i^{\text{relevance}} + \alpha \cos(\mathbf{q},\mathbf{t}_i)\),
but current PG updates do not backprop through the cosine term; that is future two-tower router-objective work.

## 9) Mapping to Current OCB Implementation (v0)

This math maps directly onto current architecture primitives:

- **Trace source:** query traces are logged in journal events, then replayed into decision points.
- **Async teacher loop:** `openclawbrain async-route-pg` collects decision points, asks teacher for `choose`/`scores`, and applies policy-gradient style updates.
- **Runtime policy v0:** `route_mode=edge+sim` uses deterministic edge ranking from edge priors + optional relevance metadata + query-target similarity.

Current v0 runtime scorer is:

\[
z_i^{(v0)} = w_i^{\text{edge}} + r_i^{\text{relevance}} + \alpha\,\cos(\mathbf{q},\mathbf{t}_i)
\]

which is a simple special case of the broader learned router family above.

## 10) Practical Reading

- Distillation aligns router logits to teacher/human preference distributions.
- REINFORCE aligns router decisions to delayed outcomes.
- Reward-source weights control trust and prevent weak-signal overfitting.
- Deterministic top-\(k\) serving keeps OpenClawBrain operationally predictable.
