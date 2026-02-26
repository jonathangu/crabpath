from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._structural_utils import classify_edge_tier
from .graph import Graph
from .router import Router


@dataclass
class TraversalConfig:
    max_hops: int = 30
    episode_edge_damping: float = 0.3
    episode_visit_penalty: float = 0.0
    temperature: float | None = None  # Use model default
    branch_beam: int = 3


@dataclass
class TraversalStep:
    from_node: str
    to_node: str
    edge_weight: float
    tier: str
    candidates: list[tuple[str, float]]
    effective_weight: float = 0.0


@dataclass
class TraversalTrajectory:
    steps: list[TraversalStep]
    visit_order: list[str]
    context_nodes: list[str]
    raw_context: str


def _normalize_seed_nodes(seed_nodes: Any) -> list[tuple[str, float]]:
    if seed_nodes is None:
        return []

    if isinstance(seed_nodes, Mapping):
        normalized: list[tuple[str, float]] = []
        for node_id, score in seed_nodes.items():
            try:
                normalized.append((str(node_id), float(score)))
            except (TypeError, ValueError):
                normalized.append((str(node_id), 1.0))
        return normalized

    if not isinstance(seed_nodes, Sequence) or isinstance(seed_nodes, (str, bytes)):
        return [(str(seed_nodes), 1.0)]

    normalized: list[tuple[str, float]] = []
    for item in seed_nodes:
        if isinstance(item, tuple | list) and len(item) >= 2:
            node_id, score = item[0], item[1]
            try:
                normalized.append((str(node_id), float(score)))
            except (TypeError, ValueError):
                normalized.append((str(node_id), 1.0))
        else:
            normalized.append((str(item), 1.0))
    return normalized


def _seed_nodes_from_index(
    query: str,
    embedding_index: Any,
    top_k: int,
) -> list[tuple[str, float]]:
    raw_scores = getattr(embedding_index, "raw_scores", None)
    if raw_scores is None or not callable(raw_scores):
        return []

    try:
        scores = raw_scores(query, top_k=top_k)
    except TypeError:
        scores = raw_scores(query, None, top_k=top_k)
    except Exception as exc:
        import warnings

        warnings.warn(
            "CrabPath: embedding index raw_scores failed: "
            f"{exc}. Falling back to safe seed extraction.",
            stacklevel=2,
        )
        return []

    normalized: list[tuple[str, float]] = []
    for item in scores or []:
        if not isinstance(item, tuple | list) or len(item) < 2:
            continue
        normalized.append((str(item[0]), float(item[1])))
    return normalized


def _build_router_context(
    query: str,
    graph: Graph,
    current_node_id: str,
    visit_order: list[str],
    candidates: list[tuple[str, float]],
) -> dict[str, Any]:
    current_node = graph.get_node(current_node_id)
    node_summary = current_node.summary if current_node else ""

    return {
        "query": query,
        "visit_order": list(visit_order),
        "current_node": current_node_id,
        "current_node_summary": node_summary,
        "candidate_count": len(candidates),
    }


def _select_by_node(candidates: list[tuple[str, float]], node_id: str) -> tuple[str, float] | None:
    for candidate in candidates:
        if candidate[0] == node_id:
            return candidate
    return None


def traverse(
    query: str,
    graph: Graph,
    router: Router,
    config: TraversalConfig | None = None,
    embedding_index: Any | None = None,
    seed_nodes: Sequence[tuple[str, float]] | list[str] | None = None,
) -> TraversalTrajectory:
    cfg = config or TraversalConfig()
    all_steps: list[TraversalStep] = []
    visit_order: list[str] = []
    context_nodes: list[str] = []

    normalized: list[tuple[str, float]] = []
    if seed_nodes is not None:
        normalized = _normalize_seed_nodes(seed_nodes)
    elif embedding_index is not None:
        normalized = _seed_nodes_from_index(
            query=query,
            embedding_index=embedding_index,
            top_k=max(cfg.branch_beam * cfg.max_hops, 1),
        )

    normalized = [item for item in normalized if graph.get_node(item[0]) is not None]
    if not normalized:
        return TraversalTrajectory(
            steps=all_steps,
            visit_order=visit_order,
            context_nodes=context_nodes,
            raw_context="",
        )

    normalized.sort(key=lambda item: item[1], reverse=True)
    start_node = normalized[0][0]
    visit_order.append(start_node)
    context_nodes.append(start_node)

    edge_traversal_count: dict[tuple[str, str], int] = {}
    node_visit_count: dict[str, int] = {start_node: 1}
    edge_damping = max(0.0, float(cfg.episode_edge_damping))
    visit_penalty = max(0.0, float(cfg.episode_visit_penalty))

    current_node = start_node
    # max_hops indicates edge hops to traverse; visit order grows by one per hop
    # so the resulting node count is at most max_hops + 1.
    for _ in range(cfg.max_hops):
        outgoing = graph.outgoing(current_node)
        if not outgoing:
            break

        # Candidate list includes all outgoing options for learning.
        all_candidates: list[tuple[str, float]] = [
            (target.id, float(edge.weight)) for target, edge in outgoing
        ]
        all_candidates.sort(key=lambda item: item[1], reverse=True)

        effective_candidates: list[tuple[str, float]] = []
        for target_id, base_weight in all_candidates:
            effective_weight = base_weight
            if base_weight > 0.0 and edge_damping != 1.0:
                traversals = edge_traversal_count.get((current_node, target_id), 0)
                effective_weight *= edge_damping ** traversals

            if visit_penalty > 0.0:
                effective_weight -= visit_penalty * node_visit_count.get(target_id, 0)

            effective_candidates.append((target_id, effective_weight))

        effective_candidates.sort(key=lambda item: item[1], reverse=True)
        if not any(weight > 0.0 for _, weight in effective_candidates):
            break

        chosen: tuple[str, float] | None = None
        tier = "dormant"

        if any(
            classify_edge_tier(weight) == "reflex"
            for _, weight in effective_candidates
        ):
            reflex_candidates = [
                c
                for c in effective_candidates
                if classify_edge_tier(c[1]) == "reflex"
            ]
            tier = "reflex"
            chosen = sorted(
                reflex_candidates, key=lambda item: (item[1], item[0]), reverse=True
            )[0]
        elif any(
            classify_edge_tier(weight) == "habitual"
            for _, weight in effective_candidates
        ):
            habitual_candidates = [
                c
                for c in effective_candidates
                if classify_edge_tier(c[1]) == "habitual"
            ]
            tier = "habitual"
            context = _build_router_context(
                query=query,
                graph=graph,
                current_node_id=current_node,
                visit_order=visit_order,
                candidates=habitual_candidates,
            )
            decision = router.decide_next(
                query=query,
                current_node_id=current_node,
                candidate_nodes=habitual_candidates[: cfg.branch_beam],
                context=context,
                tier="habitual",
            )
            chosen = _select_by_node(habitual_candidates, decision.chosen_target)
            if chosen is None:
                # Chosen edge was invalid or leads to a visited node; fallback to
                # the highest-weight habitual candidate.
                chosen = sorted(
                    habitual_candidates,
                    key=lambda item: (item[1], item[0]),
                    reverse=True,
                )[0]
        else:
            # All candidates are dormant for this node.
            break

        if chosen is None:
            break

        chosen_target, chosen_weight = chosen
        all_candidates_list = list(effective_candidates)
        base_weight_lookup = {node_id: weight for node_id, weight in all_candidates}
        effective_weight = chosen_weight
        step = TraversalStep(
            from_node=current_node,
            to_node=chosen_target,
            edge_weight=base_weight_lookup.get(chosen_target, chosen_weight),
            effective_weight=effective_weight,
            tier=tier,
            candidates=all_candidates_list,
        )
        all_steps.append(step)

        edge_traversal_count[(current_node, chosen_target)] = (
            edge_traversal_count.get((current_node, chosen_target), 0) + 1
        )
        node_visit_count[chosen_target] = node_visit_count.get(chosen_target, 0) + 1

        visit_order.append(chosen_target)
        context_nodes.append(chosen_target)
        current_node = chosen_target

        if not graph.outgoing(current_node):
            break

    raw_context = render_context(
        TraversalTrajectory(
            steps=all_steps,
            visit_order=visit_order,
            context_nodes=context_nodes,
            raw_context="",
        ),
        graph=graph,
        max_chars=1_000_000,
    )
    return TraversalTrajectory(
        steps=all_steps,
        visit_order=visit_order,
        context_nodes=context_nodes,
        raw_context=raw_context,
    )


def render_context(trajectory: TraversalTrajectory, graph: Graph, max_chars: int = 4096) -> str:
    ordered_nodes = []
    for node_id in trajectory.visit_order:
        node = graph.get_node(node_id)
        if node is None:
            continue
        ordered_nodes.append(node.content)

    context = "\n\n".join(ordered_nodes)
    if len(context) <= max_chars:
        return context
    return context[:max_chars]
