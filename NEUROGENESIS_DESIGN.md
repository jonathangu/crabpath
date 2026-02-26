> **Note:** This design doc is historical. The implementation lives in crabpath/*.py. See ARCHITECTURE_REVIEW.md for current architecture.

# Auto-Neurogenesis Design for CrabPath v0.5.2

## 1) Objective

Enable automatic node growth from novel user input while avoiding graph explosion, noisy nodes (e.g. greetings), and index inconsistency.  
Keep existing behavior of `activate()` and `learn()` unchanged.

## 2) Current behavior that constrains the design

1. `learn()` currently only creates edges between existing co-fired nodes.
2. Nodes are created today only through sync/bootstrap or `crabpath add`.
3. `EmbeddingIndex` is currently rebuilt as a full recompute and has no add/remove API.
4. `activate()` currently accepts only a seed map and does not consult embeddings or node creation.
5. CLI and adapter query path already support:
   1. semantic seeding via `index.seed()`
   2. optional `memory_search`-style symbolic seeds
   3. activation + snapshot persistence

Therefore auto-neurogenesis belongs in an orchestration layer, not in core activation.

## 3) Proposed architecture

1. Add `crabpath/neurogenesis.py` (new module).
2. Add a small `AutoNeurogenesisConfig` and `AutoNeurogenesisPolicy` there.
3. Extend `OpenClawCrabPathAdapter` with `query()` and `query_and_learn_step()` helpers that:
   1. run normal seeding
   2. run novelty detection
   3. optionally create/update auto-node
   4. run activation
   5. optionally attach auto-node outcomes for future learning
4. Keep `activate()` as pure inference primitive; do not add side effects inside it.
5. Add minimal optional CLI support via flags on `query`:
   1. `--auto-neurogenesis` to enable
   2. `--auto-max-per-session`, `--auto-window`, `--auto-known-threshold`, etc.
6. Extend `EmbeddingIndex` with incremental helpers:
   1. `upsert(node_id, text_or_vector, embed_fn)`
   2. `remove(node_id)`
   3. `dirty` marker for deferred rebuild

## 4) Component responsibilities

1. `graph.py`
   1. no functional change required for MVP.
   2. optional: add prune helper that can preserve flagged nodes during probation.
2. `embeddings.py`
   1. keep `seed()` API stable.
   2. add raw similarity helper for novelty scoring:
      1. return `(node_id, score)` list for `min_score=0.0`
3. `adapter.py`
   1. owns session-scoped auto-neurogenesis state (`previous_fired_ids`, `last_query_ts`, counters).
4. `neurogenesis.py` (new)
   1. all novelty heuristics, node creation, dedupe, and probation logic.
5. `cli.py`
   1. expose auto-neurogenesis as explicit opt-in mode.
   2. default behavior remains unchanged.

## 5) Novel concept detection

Use both raw and relative signals to avoid over-creating nodes.

### 5.1 Signals

1. `top1`: highest cosine from raw neighbor search.
2. `top2`: second best cosine, if available.
3. `tail`: mean of ranks 2..k for `k=5` (fallback 0).
4. `rel`: `(top1 - tail) / (top1 + 1e-6)`; low when there is no strong peak.
5. `len_ok`: normalized token-length gate.
6. `novelty_gate`: text-level gate for greetings/chitchat.

### 5.2 Heuristic (configurable)

1. Known if `top1 >= 0.60`.
2. Raw-novel if `top1 <= 0.58`.
3. Low-confidence/noise gate if `top1 < 0.28`.
4. Relative novelty if `rel < 0.25`.
5. Goodness gate:
   1. token_count >= 3
   2. alpha_ratio >= 0.45
   3. not in blocked utterance set (`hello`, `thanks`, `yes`, `no`, `ok`, `got it`, etc.)
   4. not all punctuation/noise.
6. Final decision: create node when `(raw-novel and relative-novel and goodness)` OR `(no seeds and raw-novel and goodness)`, and not blocked by session caps.

### 5.3 Rationale from provided cosine points

1. `0.76` for `"worktree drift codex reset"` maps to known region (`>=0.60`), so no new node.
2. `0.32` for `"giraffe codeword"` is below raw-novel boundary and above noise floor, so eligible.
3. `0.24` for `"xyzzy plugh nonsense"` is below noise floor and should not create a node unless explicit override.

## 6) Auto-node creation

### 6.1 Content and metadata

1. Content: start with normalized query text only.
2. Metadata schema:
   1. `source="auto"`
   2. `created_ts`
   3. `last_seen_ts`
   4. `session_id` (current session on creation)
   5. `created_turn_id`
   6. `last_fired_turn_id`
   7. `auto_probationary=true`
   8. `auto_seed_count` initialized to 1 and incremented on repeated hits
   9. `content_signature`: stable hash input used for deterministic dedupe
3. Keep raw context in metadata (`query_context_ids`, `prev_turn_ids`) and do not bloat `content`.

### 6.2 Deterministic ID format

1. `nid = f"auto:{sha1(normalized_query)[:12]}"`.
2. If ID already exists, update metadata (`last_seen_ts`, `auto_seed_count`) and refresh threshold if needed.
3. If different node content produces same query hash collision extremely unlikely; if collision with different content, include session and time bucket in hash namespace.

### 6.3 Initial node parameters

1. Threshold: `0.8` (lower than default 1.0 to allow future recall).
2. Seed injection when created: `1.0..1.3` at creation turn.
3. Mark `auto_probationary=true`.

## 7) Edge creation for new nodes

### 7.1 Sources of initial connections

1. Current turn seeded nodes with highest seed score (excluding pure noise hits).
2. Previous `N` turns’ fired node IDs (`N=3` by default) from adapter memory.
3. Last known `memory_search` hits in that turn, if any.

### 7.2 Direction and weights

1. Add incoming from prior context into the new node:
   1. `w_in = 0.10` to `0.20` scaled by prior node seed/score.
2. Add outgoing from new node to current-turn semantic anchors:
   1. `w_out = 0.12` to `0.25` scaled by anchor score.
3. Add low-strength reverse edges only when explicit evidence exists, never as default.
4. Initial absolute weights must be > 0.06 so they survive immediate pruning at min-weight 0.05.

### 7.3 Learning behavior

1. New-node edges are weak by design.
2. STDP from `learn()` strengthens or prunes them naturally as outcomes arrive.
3. Auto edges are capped as regular edges (`[-10, 10]`).

## 8) Growth control and stability

### 8.1 Hard controls

1. `max_auto_nodes_per_session`: 200.
2. `max_auto_nodes_per_turn`: 1.
3. `auto_window_turns_cooldown`: 8 turns before creating another auto-node for same hash family.
4. `auto_min_query_tokens`: 3.
5. `auto_noise_max_per_hour` as a fallback circuit breaker.

### 8.2 Probationary phase

1. Auto nodes receive `auto_probationary=true` for `M=6` turns.
2. If an auto node has not fired at least once in probation and gets low engagement, collapse confidence:
   1. lower threshold by `+0.2`? (hard to fire) and/or
   2. reduce outgoing weights by 50% (decay)
   3. drop if `age > 14` turns and `fire_count == 0`.
3. Auto nodes with at least one successful outcome leave probation and become normal nodes.

### 8.3 Consolidation integration

1. Keep existing `consolidate(min_weight=0.05)` semantics.
2. During consolidation runs, preserve:
   1. high-confidence non-probation nodes
   2. probation nodes with firing evidence or successful outcomes
3. Run consolidation every `T=50` turns or once per hour for heavy sessions.

## 9) Embedding index update strategy

### 9.1 Current limitation

`EmbeddingIndex` currently supports only full `build()` and load/save.

### 9.2 Incremental plan

1. Add `upsert(node_id, text_or_vector, embed_fn=None)`:
   1. if vector provided, insert directly
   2. else compute vector through `embed_fn` for single text
   3. set `dirty=True` and update in-memory vectors dict
2. Add `remove(node_id)` for deleted nodes.
3. Add `rebuild_if_needed()`:
   1. immediate upsert for single nodes is enough for same-turn queryability
   2. schedule full rebuild every `R=500` mutations or once per hour
4. Add simple versioning:
   1. `index_version` increment on write
   2. optional compatibility marker with graph save timestamp.

## 10) Pseudocode

```python
def query_with_auto_neurogenesis(adapter, query_text, session_id, turn_id, opts):
    # 1) Build seeds as today
    semantic_neighbors_raw = adapter.index.top_neighbors_raw(
        query_text, top_k=8, embed_fn=adapter.embed_fn, min_score=0.0
    )
    seeds = adapter.seed(query_text, top_k=opts.top_k)

    # 2) Novelty detection
    novelty = assess_novelty(
        query_text=query_text,
        raw=semantic_neighbors_raw,
        top_k=opts.top_k,
        turn_state=adapter.turn_state,
        config=opts.neuro_cfg,
    )

    auto_node_id = None
    if novelty.should_create:
        auto_node_id = deterministic_auto_id(query_text)
        node = adapter.graph.get_node(auto_node_id)
        if node is None:
            node = Node(
                id=auto_node_id,
                content=normalize_query_text(query_text),
                threshold=0.8,
                metadata=novelty.metadata,
            )
            adapter.graph.add_node(node)
            adapter.index.upsert(auto_node_id, node.content, adapter.embed_fn)
        else:
            bump_auto_metadata(node, session_id, turn_id)

        connect_new_node(
            graph=adapter.graph,
            new_node_id=auto_node_id,
            turn_fired_ids=adapter.last_fired_ids,
            current_seed_ids=seeds.keys(),
            weights=opts.edge_init_weights,
        )

        adapter.turn_state.auto_created_ids.append(auto_node_id)
        seeds[auto_node_id] = max(1.0, seeds.get(auto_node_id, 0.0))

    # 3) Run inference with warm context
    firing = adapter.activate(seeds, max_steps=opts.max_steps, decay=opts.decay, top_k=opts.top_k)

    # 4) Track for future neurogenesis and optional pruning
    adapter.turn_state.record(turn_id, query_text, firing, auto_node_id)
    adapter.mark_auto_node_fired(firing)

    # 5) Optional persistence
    if opts.autosave:
        adapter.save()
    return firing, {"auto_node_id": auto_node_id, "created": auto_node_id is not None}
```

## 11) Failure modes and mitigations

1. False positives from short utterances (e.g., "hi", "thanks")
   1. mitigation: quality gate + blocked phrase list + token count minimum.
2. Duplicate auto-nodes for same concept
   1. mitigation: deterministic IDs + periodic dedupe hash migration.
3. Over-edgeing from weak contexts
   1. mitigation: hard per-turn edge budget + weak initial weight caps.
4. Index staleness
   5. mitigation: upsert + dirty marker + periodic rebuild.
5. Graph drift / churn
   1. mitigation: probationary pruning + periodic `consolidate`.
6. Memory growth under high-noise workloads
   1. mitigation: session/hour quotas + cooldown + optional global kill switch.

## 12) Test scenarios (what to implement after design)

1. `novel_detection_known` with seed `0.76` does not create node.
2. `novel_detection_new_phrase` with seed `0.32` creates deterministic auto-node and ID is stable across runs.
3. `novel_detection_nonsense` with seed `0.24` does not create node.
4. `novelty_greeting` for `"hello"` / `"thanks"` never creates node even when no semantic match.
5. `deterministic_id_reuse` updates existing auto-node instead of duplicate.
6. `auto_node_connects_current_turn` ensures edges are added to top firing anchors.
7. `auto_node_connects_temporal` adds weak edges from prior-turn nodes.
8. `probationary_prune` removes never-fired probation nodes after TTL.
9. `probationary_success` retains and strengthens node after positive outcomes.
10. `index_upsert_query_in_same_turn` newly created node is returned by similarity search without full rebuild.
11. `cli_auto_flag_off` keeps v0.5.2 behavior unchanged.
12. `cli_auto_flag_on` enables auto-neurogenesis and writes updated graph/index atomically.

## 13) Build order (MVP vs later)

### MVP (1–2 days)
1. Add `neurogenesis.py` with novelty classifier + deterministic ID generation + edge bootstrap.
2. Add `Adapter.query()` path invoking novelty + creation.
3. Add `EmbeddingIndex.upsert()` with per-node add and dirty flag.
4. Add CLI `--auto-neurogenesis` gate for explicit opt-in rollout.
5. Add gating tests for known/novel/nonsense behavior.

### Later
1. Probation metadata and auto-node lifecycle pruning.
2. Temporal-context edge formation and decay policy.
3. Full delayed-feedback integration and automatic consolidation schedule hooks.
4. Offline replay experiments for threshold calibration and false-positive rates.
