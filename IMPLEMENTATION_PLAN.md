> **Note:** This design doc is historical. The implementation lives in crabpath/*.py. See ARCHITECTURE_REVIEW.md for current architecture.

# CrabPath v0.5 Integration Plan (Post-0.4.0)

Current status: v0.4.0, 61 passing tests, pure-Python core, zero dependencies, graph currently 153 nodes / 143 edges.
This plan is for integrating CrabPath into an OpenClaw agent loop with existing `memory_search`, workspace files, and heartbeat signals.

## 1) AGENT LOOP INTEGRATION

### 1.1 Component layout

1. Add a session-scoped adapter in the agent runtime: `OpenClawCrabPathAdapter`.
2. Keep these persistent artifacts:
   1. `crabpath_graph.json` (Graph)
   2. `crabpath_embeddings.json` (EmbeddingIndex)
   3. `crabpath_events.db` (session snapshots + correction bridge)
3. Keep these in-memory per session:
   1. warm fire history window
   2. last `Firing` result
   3. last selected node IDs and scores

### 1.2 Startup hook

1. On OpenClaw session start:
   1. load graph and index if present, else fallback bootstrap.
   2. initialize `CrabPathState(session_id, user_id, workspace_id)`.
   3. start heartbeat cursor: `last_heartbeat_ts`, `last_workspace_version`.

### 1.3 Pre-response flow

1. Build seeding context before every assistant turn:
   1. `task_text = user_message + intent hints + heartbeat delta + workspace summary`.
   2. generate base seeds with `EmbeddingIndex.seed()`.
   3. augment with symbolic recall from OpenClaw `memory_search` IDs.
2. Candidate node construction:
   1. combine seeds and explicit IDs from `memory_search`.
   2. dedupe and keep top-N by energy (default N=20).
3. Run graph activation:
   1. `firing = activate(graph, seeds=seed_map, max_steps=3, decay=0.1, top_k=12, reset=False, trace_decay=0.05)`.
4. Optional habitual tier (see section 2) filters/expands fired nodes.
5. Final context bundle generation:
   1. include selected node `content` lines.
   2. include top inhibitory nodes as structured guardrails.
   3. attach provenance metadata for learning (`fired_ids`, `fired_at`, `energies`, `inhibited`).
6. Persist turn snapshot before sending LLM context so delayed feedback can be attributed.

### 1.4 OpenClaw hook map

1. `memory_search` hook:
   1. pass `memory_search` output as seed candidates with weak energy (`0.25` each).
2. workspace hook:
   1. include file list + diff signals in task context and optionally as anti-bias features to the habitual tier.
3. heartbeat hook:
   1. add active heartbeat signal to prevent stale paths (e.g., prefer node metadata containing same task area as recent heartbeat tags).

### 1.5 Suggested adapter skeleton

```python
# pseudocode
class OpenClawCrabPathAdapter:
    def __init__(self, graph, index, db, model_client, habit_cfg):
        self.graph = graph
        self.index = index
        self.db = db
        self.model_client = model_client
        self.habit_cfg = habit_cfg
        self.turn_buffer = deque(maxlen=20)

    def on_session_start(self, session):
        state = load_or_init_state(session.id)
        return state

    def select_context(self, session, user_msg, memory_hits, workspace, heartbeat):
        seed_map = make_seeds(user_msg, memory_hits, workspace, heartbeat)
        firing = activate(self.graph, seed_map, max_steps=3, top_k=12, reset=False)
        selected = maybe_habitual_filter(firing, user_msg, heartbeat, workspace)
        self._persist_turn(session.id, selected)
        return selected

    def on_feedback(self, session, correction_event):
        apply_delayed_feedback(self.db, self.graph, session.id, correction_event)
```

## 2) THE HABITUAL TIER

### 2.1 Purpose and policy

1. Keep graph selection cheap and deterministic when confidence is high.
2. Use a cheap LLM only when graph output is noisy, stale, or too large.
3. Let the habit tier decide:
   1. which fired nodes to load,
   2. whether to follow outgoing pointers,
   3. whether to keep inhibitory constraints out-of-band.

### 2.2 Invocation policy

1. Skip habitual tier when all are true:
   1. `len(firing.fired) <= 6`
   2. top score margin > 0.35 and minimum score > 0.9
   3. no negative edges with high magnitude (`edge.weight <= -0.8` and connected to fired nodes)
   4. heartbeat freshness < 60s and workspace unchanged.
2. Invoke habitual tier when any is true:
   1. `len(firing.fired) > 12`
   2. seed energy entropy too flat
   3. context budget remaining < expected static context load
   4. recently observed failure on similar intent class.

### 2.3 Input format (JSON)

1. `intent`: short user task/goal
2. `candidate_nodes`: array of up to 20 objects
   1. `node_id`, `energy`, `fired_step`, `trace`, `content`, `metadata`.
3. `top_outgoing_pointers`: for each fired node, up to K outgoing edges with source/target and weight.
4. `workspace_digest`: changed files, open files, touched paths, heartbeat tags.
5. `recent_failures`: last 3 correction classes and timestamps.
6. `policy`:
   1. `context_token_budget`
   2. `max_nodes`.

### 2.4 Prompt template

1. System instruction: return strict JSON, no markdown.
2. Objective:
   1. maximize task success probability,
   2. minimize context tokens,
   3. keep known-safe constraints.
3. Candidate selection scoring heuristic to mimic in prompt:
   1. high `energy`
   2. high `trace`
   3. inhibitory edge awareness.
4. Required output schema:
   1. `load_ids`: ordered list of node ids to include
   2. `follow_outgoing`: boolean
   3. `follow_ids`: ordered list of target ids if following outgoing edges
   4. `drop_ids`: node ids explicitly rejected
   5. `explanation`: one short reason string.

### 2.5 Cost budget per query

1. Default budget:
   1. input <= 700 tokens,
   2. output <= 180 tokens,
   3. latency SLO <= 120ms p95 in fast path, <= 300ms p95 in expensive path.
2. Hard cap:
   1. max 2 retries
   2. disable habit tier if budget exceeded for 3 consecutive turns.
3. Approximate spend target:
   1. <= 5% of per-turn context work budget,
   2. <= 0.05 equivalent LLM units per active day per agent.

### 2.6 Behavioral output handling

1. If `follow_outgoing=true`, append pointer targets to candidate list with dampened energy `0.4 * edge.weight_sign`.
2. For `drop_ids`, remove from final context regardless of high energy.
3. If model output JSON parse fails, fallback to deterministic top-N fired nodes.

## 3) OUTCOME FEEDBACK LOOP

### 3.1 Positive vs negative signals

1. Positive outcome (`+1.0`):
   1. no correction within grace window,
   2. explicit success flag in session metadata,
   3. user marks task completed and no follow-up corrective message.
2. Negative outcome (`-1.0`):
   1. explicit correction event,
   2. rollback marker in tool traces,
   3. tool-failure sequence tied to this response turn,
   4. user feedback containing clear invalidation.
3. Neutral / partial (`0.0` to `+0.3`):
   1. non-blocking edits,
   2. clarification requests,
   3. ambiguous feedback with no explicit fault.

### 3.2 Snapshot capture at pre-response

1. Persist one row per assistant turn:
   1. `turn_id`
   2. `session_id`
   3. `timestamp`
   4. `fired_ids` ordered
   5. `fired_scores`
   6. `fired_at`
   7. `inhibited_ids`
   8. `context_snippet_hash`
   9. `assistant_message_id`
   10. `workspace_snapshot_id`
2. Store serialized `Firing` and selected final context IDs, not just raw graph nodes.
3. Keep rolling retention to at least last 20 turns for delayed feedback.

### 3.3 Mapping correction to session-fired nodes

1. On correction event, compute candidate turns:
   1. same `session_id`
   2. `assistant_turn_id <= correction_turn_id`
   3. message distance `<= 10` by default
   4. no correction already consumed for those turns.
2. Pick by highest relevance score:
   1. closeness of correction text to workspace/action context,
   2. temporal distance decay.
3. Build a single aggregate feedback event per chosen turn:
   1. `base_outcome = -1.0` for hard correction,
   2. `weight = base_outcome * exp(-delta_turns/5)`.
4. Reconstruct or fetch `Firing` and call:
   1. `learn(graph, firing, outcome=weight, rate=adaptive_rate)`.

### 3.4 Delayed feedback 5-message design

1. Use `window_turns=10`, `window_seconds=900`.
2. If correction arrives 5 messages later, attribution uses the turn with smallest `turn_distance` and largest overlap.
3. If multiple possible turns within window, apply fractional feedback:
   1. `w_i = exp(-λ * delta_turn_i)` normalized.
   2. apply `learn(... outcome=negative_sum(w_i))` per turn.
4. Mark all consumed turns as “attributed” to prevent double-learning.

### 3.5 Drift-safe learning schedule

1. Never call `learn()` inline in user-facing path.
2. Append outcomes to queue, apply in background worker every 5-30s.
3. Batch contiguous outcomes per session and persist updated graph once per batch.

## 4) INCREMENTAL GRAPH UPDATES

### 4.1 New node vs update policy

1. New node when:
   1. no existing node has semantic match > 0.92,
   2. error class or tool namespace is new,
   3. new entity/action signature not seen in node history.
2. Update existing node when:
   1. semantic match > 0.92,
   2. same action class and tool type,
   3. only metadata/content counters update (`frequency`, `last_seen`, `source_refs`).
3. Node metadata schema additions:
   1. `last_seen_ts`, `first_seen_ts`, `fired_count`, `success_count`, `failure_count`, `correction_count`, `error_classes`, `source_turn_ids`.

### 4.2 Edge creation from session co-occurrence

1. In addition to `learn()` STDP edges, create explicit co-occurrence edges nightly:
   1. between consecutive nodes in final selected context,
   2. between fired nodes and executed tool call classes in same turn.
2. Use weighted evidence counts:
   1. `cooccurrence_weight += 0.2` for temporal adjacency,
   2. `context_to_action_weight += 0.3` for successful tool transitions.
3. Keep these edges below 1.0 initially and let `learn()` dominate long-term strength.
4. Run `graph.consolidate(min_weight=0.05)` after ingestion window.

### 4.3 Embedding index refresh strategy

1. For now, treat EmbeddingIndex as full-rebuildable with cheap graphs:
   1. detect graph changed nodes nightly,
   2. rebuild every 24h during low-traffic window,
   3. atomically write `embeddings.tmp` then replace.
2. For larger graphs, add incremental path:
   1. `EmbeddingIndex.vectors` in-memory add/remove patch before rebuild,
   2. schedule full rebuild weekly.
3. Version indices with `embedding_index_version`.
4. Reject stale index usage when graph `version` changed and last build lag > 24h.

### 4.4 Consolidation and growth controls

1. Keep `consolidate(min_weight=0.05)` daily if graph change > 20 edges.
2. If node churn > 15% weekly, reduce consolidation threshold temporarily to 0.03 to preserve learning opportunities.
3. Never delete nodes with `metadata.protected=True` or `fired_count > 10` and no recent negative trend.

## 5) A/B EVALUATION

### 5.1 Treatments

1. Control arm: static context path only (current workflow with memory_search + workspace + heartbeats).
2. Treatment arm: CrabPath-selected context with optional habitual tier.
3. Optional secondary arms:
   1. CrabPath without habitual tier,
   2. CrabPath without inhibitory edges,
   3. naive STDP timing removed (symmetric learning).

### 5.2 Parallel execution without contamination

1. Deterministic split per conversation/session hash by stable flag:
   1. hash(`session_id`) % 100 < N.
2. Each arm uses independent graph and index artifacts.
3. Disable shared persistence during experiment.
4. Log the same raw traces for both arms including ground-truth labels and correction events.
5. At rollout checkpoints, promote only one artifact branch after statistical checks.

### 5.3 Metrics

1. Core metrics:
   1. correction rate (per 100 turns),
   2. correction rate lag by 1, 5, and 10 turns,
   3. task success rate,
   4. known-bad-action avoidance,
   5. context tokens loaded before response,
   6. first-correction turn index,
   7. tool sequence precision against baseline sequence labels.
2. Runtime metrics:
   1. end-to-end latency p50/p95,
   2. habit-tier invocation count and fallback rate,
   3. `activate` step count and fired count.
3. Scientific metrics:
   1. spectral gap trend,
   2. edge-sign stability,
   3. average inhibition applied per response,
   4. learning effect half-life after failures.

### 5.4 Evaluation protocol

1. Use time-split by date where possible to expose drift.
2. Minimum sample size target:
   1. 500 turns per arm for directional metrics,
   2. 2000 turns for task-success and correction rate significance.
3. Compare with paired bootstrapped confidence intervals at session level.
4. Define explicit fail condition:
   1. if treatment correction rate > control by 20% for 2 consecutive checkpoints, rollback artifact.

## 6) RISKS AND FAILURE MODES

### 6.1 Graph ossification

1. Risk: dominant weights force narrow trajectories and suppress exploration.
2. Mitigation:
   1. spectral gap monitor and alert when it exceeds threshold,
   2. weight clipping is already bounded [-10,10], but add periodic noise:
      1. `w += Uniform(-eps, eps)` for exploration with `eps` decayed by age,
   3. enforce minimum entropy in selected nodes (`min_unique_intent_ratio`).

### 6.2 Cold start

1. Risk: sparse graph and poor seeding causes low coverage.
2. Mitigation:
   1. bootstrap with canonical procedural graph,
   2. lower thresholds for new nodes,
   3. route to static context when fired count < 2,
   4. disable habituation until node_count and hit rate stabilize.

### 6.3 Embedding drift

1. Risk: stale vectors on text revisions and stale topic shifts.
2. Mitigation:
   1. node versioning,
   2. periodic rebuild,
   3. detect fallback from low seed overlap (seed_count == 0) and trigger rebuild + refresh.

### 6.4 Latency budget overrun

1. Risk: habit tier + activation + DB + context packing exceeds latency target.
2. Mitigation:
   1. cache seed results by normalized task fingerprint for 30s TTL,
   2. precompute top-k from latest workspace change,
   3. strict habit-tier timeout (120ms), fallback to deterministic path on timeout.

### 6.5 Credit misattribution

1. Risk: STDP strengthens nodes that co-fired but were not causally useful.
2. Mitigation:
   1. delay- weighted feedback,
   2. temporal alignment with correction source,
   3. explicit anti-causal dampening already in learn()
   4. experiment track of attribution precision.

## 7) Implementation sequence (phased)

1. Phase 0 (1-2 days): adapter + persistence wiring, no behavior change to core memory selection.
2. Phase 1 (3-5 days): add pre-response capture and correction bridge with delayed mapping.
3. Phase 2 (1-2 weeks): habit tier integration with strict fallback.
4. Phase 3 (1 week): daily graph/index build + co-occurrence enrichment + A/B harness.
5. Phase 4 (1-2 weeks): full parallel run, evaluate and harden.

## 8) Code contracts to codify before implementation

1. Define `TurnContextSnapshot` and `FeedbackEvent` structs (or DB tables) that include exactly:
   1. `session_id`, `turn_id`, `message_id`, `assistant_message_id`,
   2. `fired_ids`, `fired_scores`, `fired_at`, `inhibited_ids`,
   3. `outcome`, `severity`, `source`.
2. Define adapter boundary for OpenClaw to avoid hard dependency on specific agent internals:
   1. `get_memory_search_hits(session, text)`
   2. `get_workspace_state(session)`
   3. `get_heartbeat_tags(session)`
3. Define a single `CrabPathConfig` object for tunables:
   1. `activation_top_k`, `habituation_token_budget`, `feedback_window_turns`,
   2. `learn_rate`, `enable_create_edges`, `max_new_edges`, `embedding_rebuild_interval_hours`.
