# CrabPath Roadmap

## Phase 1: Foundation (Weeks 1-2)
*Get the graph working and prove warm-start adds value*

### Core Data Structures
- [ ] `MemoryGraph` class (NetworkX DiGraph + SQLite metadata store)
- [ ] Node types: Fact, Rule, Tool, Action, Sequence, Episode, ErrorClass, Hub
- [ ] Edge types: Association, Sequence, Causation, Contingency, Inhibition, Preference, Abstraction
- [ ] Node schema: content, summary, tags, priors (recency × frequency), timestamps
- [ ] Edge schema: type, weight (signed for inhibition), condition, success_stats

### Warm-Start Pipeline
- [ ] Log parser for OpenClaw session transcripts
- [ ] Entity normalizer (canonicalize files, tools, commands, topics)
- [ ] Episode boundary detector
- [ ] Node extractor (corrections → Rules, tool docs → Tools, traces → Actions/Sequences)
- [ ] Edge builder (co-occurrence + sequential transition + PMI)
- [ ] Immune system seed (quarantine from known failures)
- [ ] Initial weight calibration (IDF-weighted, degree-capped)

### Baseline Evaluation
- [ ] Golden task suite (16+ tasks from existing eval)
- [ ] Naive RAG baseline (vector search, top-k)
- [ ] Static loading baseline (current file-based approach)
- [ ] Metrics: retrieval precision/recall, task success, token cost, latency

## Phase 2: Activation Engine (Weeks 3-4)
*Make the graph actually route queries*

### Activation Propagation
- [ ] Seed node selection (intent classifier + embedding similarity + recency)
- [ ] Spreading activation with damping (α < 1)
- [ ] Inhibitory channel (separate W+ and W- matrices)
- [ ] Refractory periods (recently activated nodes get cooldown)
- [ ] Hop limit + convergence check (‖x_{t+1} - x_t‖ < ε)
- [ ] Top-K node selection for context assembly

### LLM-as-Activation-Function
- [ ] Edge gating: lightweight model scores candidate neighbors
- [ ] Candidate generation pipeline: tag filter → weight filter → LLM gate
- [ ] Model selection by query risk score (cheap for routine, expensive for novel)
- [ ] Spike output format: fire flag, activation strength, gist, positive/negative pointers

### Three-Tier Routing
- [ ] Reflex check (is there a myelinated path for this query pattern?)
- [ ] Habitual routing (cheap model activation)
- [ ] Deliberative fallback (full model when confidence low)
- [ ] Cost tracking per tier

## Phase 3: Learning Loop (Weeks 5-6)
*Make the graph learn from outcomes*

### Hebbian Learning
- [ ] Co-activation + outcome → edge weight update
- [ ] Temporal credit assignment (eligibility traces along traversal path)
- [ ] Negative learning (failure → strengthen inhibitory edges)
- [ ] Counterfactual evaluation (what SHOULD have been retrieved?)
- [ ] Learning rate scheduling (critical periods: high early, stable later)

### Graph Maintenance
- [ ] Decay: unused edges lose weight over time
- [ ] Pruning: remove low-weight edges, archive isolated nodes
- [ ] Neurogenesis: new nodes enter probation (must fire 3+ times to persist)
- [ ] Consolidation (nightly): replay successful traces, merge duplicates, promote patterns

### Immune System
- [ ] Quarantine nodes correlated with repeated failures
- [ ] Rehabilitation protocol (successful re-use → gradual reinstatement)
- [ ] Context-conditioned quarantine rules

## Phase 4: Myelination (Weeks 7-8)
*Compile hot paths into near-zero-cost reflexes*

### Option Extraction
- [ ] Frequent sequence mining (PrefixSpan on traversal traces)
- [ ] Option formalization: initiation set, internal policy, termination condition
- [ ] Precondition/postcondition verification contracts
- [ ] Confidence threshold for myelination (frequency × success rate × stability)

### Reflex Cache
- [ ] Compiled macro storage format (signature → nodes + justification + health stats)
- [ ] Fast matching (embedding similarity on query → reflex signature)
- [ ] Fallback mechanism (if reflex fails → degrade to habitual)
- [ ] Degradation detection (exponential moving average of success; drop below threshold → unmyelinate)

### Economics Validation
- [ ] Cost per query tracking across tiers
- [ ] Myelination rate over time
- [ ] Amortized cost model (including maintenance overhead)
- [ ] Learning curve visualization

## Phase 5: Evaluation & Paper (Ongoing)

### Offline Replay Experiment (THE key experiment)
- [ ] Time-split holdout: train on first 70% of days, test on last 30%
- [ ] Build CrabPath graph from train only
- [ ] Simulate: no memory vs RAG vs CrabPath (full) vs CrabPath ablations
- [ ] Metrics: tool-call correctness, error-class incidence, token cost, steps-to-resolution
- [ ] Learning curves and ablation results

### Ablation Matrix
- [ ] E0: No graph (RAG baseline)
- [ ] E1: Graph + activation, no learning
- [ ] E2: + Hebbian learning
- [ ] E3: + Pruning
- [ ] E4: + Myelination
- [ ] E5: + Inhibition
- [ ] E6: + Immune system
- [ ] E7: Full CrabPath (all mechanisms)

### Model Quality Sweep
- [ ] Activation model: Haiku vs Sonnet vs Opus vs distilled 7B
- [ ] Success vs cost over time for each
- [ ] Identify minimum capability threshold

### Graph Health Monitoring
- [ ] Entropy of outgoing transitions
- [ ] Hub dominance (Gini coefficient)
- [ ] Component structure over time
- [ ] Spectral radius / mixing time estimates

---

## Technical Decisions (TBD)

- **Storage**: SQLite + NetworkX (start simple) vs Neo4j (if scale demands)
- **Embedding model**: for seed node selection and reflex matching
- **Activation model**: start with Haiku/GPT-mini, upgrade as needed
- **Integration**: OpenClaw plugin vs standalone library
- **Language**: Python (prototyping) → Rust core (if performance matters)
