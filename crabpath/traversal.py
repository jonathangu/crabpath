from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .graph import Graph
from .router import Router


@dataclass
class TraversalConfig:
    max_hops: int = 3
    temperature: float = 0.2
    branch_beam: int = 3


@dataclass
class TraversalStep:
    from_node: str
    to_node: str
    edge_weight: float
    tier: str
    candidates: list[tuple[str, float]]


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
        return [(str(node_id), float(score)) for node_id, score in seed_nodes.items()]

    if not isinstance(seed_nodes, Sequence) or isinstance(seed_nodes, (str, bytes)):
        return [(str(seed_nodes), 1.0)]

    normalized: list[tuple[str, float]] = []
    for item in seed_nodes:
        if isinstance(item, tuple | list) and len(item) >= 2:
            node_id, score = item[0], item[1]
            normalized.append((str(node_id), float(score)))
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
    except Exception:
        return []

    normalized: list[tuple[str, float]] = []
    for item in scores or []:
        if not isinstance(item, tuple | list) or len(item) < 2:
            continue
        normalized.append((str(item[0]), float(item[1])))
    return normalized


def _classify_tier(weight: float) -> str:
    if weight > 0.8:
        return "reflex"
    if weight >= 0.3:
        return "habitual"
    return "dormant"


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
    visited = set(visit_order)

    current_node = start_node
    for _ in range(cfg.max_hops):
        outgoing = graph.outgoing(current_node)
        if not outgoing:
            break

        # Candidate list includes all outgoing options for learning.
        all_candidates: list[tuple[str, float]] = [
            (target.id, float(edge.weight)) for target, edge in outgoing
        ]
        all_candidates.sort(key=lambda item: item[1], reverse=True)
        unvisited_candidates = [c for c in all_candidates if c[0] not in visited]
        if not unvisited_candidates:
            break

        chosen: tuple[str, float] | None = None
        tier = "dormant"

        if any(
            _classify_tier(weight) == "reflex" and node_id not in visited
            for node_id, weight in all_candidates
        ):
            reflex_candidates = [
                c
                for c in all_candidates
                if _classify_tier(c[1]) == "reflex" and c[0] not in visited
            ]
            tier = "reflex"
            chosen = sorted(reflex_candidates, key=lambda item: (item[1], item[0]), reverse=True)[0]
        elif any(
            _classify_tier(weight) == "habitual" and node_id not in visited
            for node_id, weight in all_candidates
        ):
            habitual_candidates = [
                c
                for c in all_candidates
                if _classify_tier(c[1]) == "habitual" and c[0] not in visited
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
            chosen = _select_by_node(unvisited_candidates, decision.chosen_target)
            if chosen is None:
                # Chosen edge was invalid or leads to a visited node; fallback to
                # the highest-weight habitual candidate.
                chosen = sorted(
                    habitual_candidates, key=lambda item: (item[1], item[0]), reverse=True
                )[0]
        else:
            # All candidates are dormant for this node.
            break

        if chosen is None:
            break

        chosen_target, chosen_weight = chosen
        all_candidates_list = list(all_candidates)
        step = TraversalStep(
            from_node=current_node,
            to_node=chosen_target,
            edge_weight=chosen_weight,
            tier=tier,
            candidates=all_candidates_list,
        )
        all_steps.append(step)

        if chosen_target in visited:
            break
        visit_order.append(chosen_target)
        context_nodes.append(chosen_target)
        visited.add(chosen_target)
        current_node = chosen_target

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
