"""Memory controller for orchestrated query, retrieval, and learning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from math import exp, log
from typing import Any, Callable

from .decay import DecayConfig, apply_decay
from .graph import Graph, Node
from .inhibition import InhibitionConfig, score_with_inhibition
from .learning import LearningConfig, RewardSignal, make_learning_step
from .synaptogenesis import (
    SynaptogenesisConfig,
    SynaptogenesisState,
    record_cofiring,
    record_skips,
)


@dataclass
class ControllerConfig:
    learning: LearningConfig  # from learning.py
    synaptogenesis: SynaptogenesisConfig  # from synaptogenesis.py
    inhibition: InhibitionConfig  # from inhibition.py
    decay: DecayConfig  # from decay.py
    traversal_max_hops: int = 3
    enable_learning: bool = True
    enable_synaptogenesis: bool = True
    enable_inhibition: bool = True
    enable_decay: bool = True
    decay_interval: int = 5  # apply decay every N queries

    @classmethod
    def default(cls) -> "ControllerConfig":
        return cls(
            learning=LearningConfig(),
            synaptogenesis=SynaptogenesisConfig(),
            inhibition=InhibitionConfig(),
            decay=DecayConfig(),
        )


@dataclass
class QueryResult:
    query: str
    selected_nodes: list[str]
    context: str
    context_chars: int
    trajectory: list[dict[str, Any]]
    candidates_considered: int


@dataclass
class PhaseState:
    current_phase: int = 1
    weight_entropy: float = 0.0
    avg_gradient_magnitude: float = 0.0
    queries_in_phase: int = 0
    phase_history: list[dict] = field(default_factory=list)


class LearningPhaseManager:
    def __init__(
        self,
        entropy_threshold: float = 1.2,
        gradient_threshold: float = 0.02,
        min_queries_before_transition: int = 20,
        hysteresis: int = 5,
    ):
        self.state = PhaseState()
        self.entropy_threshold = entropy_threshold
        self.gradient_threshold = gradient_threshold
        self.min_queries_before_transition = min_queries_before_transition
        self.hysteresis = hysteresis
        self._consecutive_phase2_signals = 0
        self._consecutive_phase1_signals = 0
        self._gradient_observations = 0

    def compute_weight_entropy(self, graph: Graph) -> float:
        weights = [float(edge.weight) for edge in graph.edges() if edge.weight > 0]
        if not weights:
            return 0.0

        max_weight = max(weights)
        weights = [exp(weight - max_weight) for weight in weights]
        total = sum(weights)
        if total <= 0.0:
            return 0.0

        entropy = 0.0
        for probability in (weight / total for weight in weights):
            if probability <= 0.0:
                continue
            entropy -= probability * log(probability)
        return entropy

    def _extract_gradient_magnitude(self, learning_summary: dict) -> float:
        updates = learning_summary.get("updates", [])
        if not isinstance(updates, list) or not updates:
            return 0.0

        deltas = []
        for update in updates:
            raw_delta = None
            if hasattr(update, "delta"):
                raw_delta = getattr(update, "delta")
            elif isinstance(update, dict):
                raw_delta = update.get("delta")
            if raw_delta is None:
                continue
            try:
                deltas.append(abs(float(raw_delta)))
            except (TypeError, ValueError):
                continue

        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def update(self, graph: Graph, learning_summary: dict) -> int:
        self.state.weight_entropy = self.compute_weight_entropy(graph)
        instant_gradient = self._extract_gradient_magnitude(learning_summary)
        self._gradient_observations += 1
        self.state.avg_gradient_magnitude = (
            (self.state.avg_gradient_magnitude * (self._gradient_observations - 1))
            + instant_gradient
        ) / self._gradient_observations
        self.state.queries_in_phase += 1

        meets_entropy = self.state.weight_entropy < self.entropy_threshold
        meets_gradient = self.state.avg_gradient_magnitude > self.gradient_threshold
        phase2_signal = (
            self.state.current_phase == 1
            and meets_entropy
            and meets_gradient
            and self.state.queries_in_phase >= self.min_queries_before_transition
        )
        phase1_signal = (
            self.state.current_phase == 2
            and not meets_entropy
            and not meets_gradient
            and self.state.queries_in_phase >= self.min_queries_before_transition
        )

        if phase2_signal:
            self._consecutive_phase2_signals += 1
            self._consecutive_phase1_signals = 0
        elif phase1_signal:
            self._consecutive_phase1_signals += 1
            self._consecutive_phase2_signals = 0
        else:
            self._consecutive_phase2_signals = 0
            self._consecutive_phase1_signals = 0

        if self.state.current_phase == 1 and self._consecutive_phase2_signals >= self.hysteresis:
            previous = self.state.current_phase
            self.state.current_phase = 2
            self.state.queries_in_phase = 0
            self.state.phase_history.append(
                {
                    "from": previous,
                    "to": self.state.current_phase,
                    "weight_entropy": self.state.weight_entropy,
                    "avg_gradient_magnitude": self.state.avg_gradient_magnitude,
                }
            )
            self._consecutive_phase2_signals = 0

        if self.state.current_phase == 2 and self._consecutive_phase1_signals >= self.hysteresis:
            previous = self.state.current_phase
            self.state.current_phase = 1
            self.state.queries_in_phase = 0
            self.state.phase_history.append(
                {
                    "from": previous,
                    "to": self.state.current_phase,
                    "weight_entropy": self.state.weight_entropy,
                    "avg_gradient_magnitude": self.state.avg_gradient_magnitude,
                }
            )
            self._consecutive_phase1_signals = 0

        return self.state.current_phase

    @property
    def phase(self) -> int:
        return self.state.current_phase

    def should_skip_pg(self) -> bool:
        return self.state.current_phase == 1

    def should_dampen_hebbian(self) -> bool:
        return self.state.current_phase == 2 and self.state.queries_in_phase > 100


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _normalize_trajectory_step(
    step: dict[str, Any],
) -> dict[str, Any]:
    candidates = step.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    normalized = []
    for candidate in candidates:
        if not isinstance(candidate, tuple | list) or len(candidate) < 2:
            continue
        try:
            normalized.append((str(candidate[0]), float(candidate[1])))
        except (TypeError, ValueError):
            continue
    return {
        "from_node": str(step.get("from_node", "")),
        "to_node": str(step.get("to_node", "")),
        "edge_weight": float(step.get("edge_weight", 0.0)),
        "candidates": normalized,
    }


class MemoryController:
    def __init__(self, graph: Graph, config: ControllerConfig | None = None):
        self.graph = graph
        self.config = config or ControllerConfig.default()
        self._syn_state = SynaptogenesisState()
        self._query_count = 0
        self.phase_manager = LearningPhaseManager()

    @property
    def query_count(self) -> int:
        return self._query_count

    def query(self, query_text: str, llm_call: Callable | None = None) -> QueryResult:
        """Single entry point for all queries."""
        self._query_count += 1

        query = query_text.strip()
        seed_nodes = self._seed_nodes(query)
        if not seed_nodes:
            return QueryResult(
                query=query,
                selected_nodes=[],
                context="",
                context_chars=0,
                trajectory=[],
                candidates_considered=0,
            )

        visited = set(seed_nodes[:1])
        selected_nodes = [seed_nodes[0]]
        trajectory: list[dict[str, Any]] = []
        candidates_considered = 0

        current_node_id = seed_nodes[0]
        current_node = self.graph.get_node(current_node_id)
        if current_node is None:
            return QueryResult(
                query=query,
                selected_nodes=[],
                context="",
                context_chars=0,
                trajectory=[],
                candidates_considered=0,
            )

        # Increase access count immediately for starting node.
        current_node.access_count += 1

        for _ in range(max(0, self.config.traversal_max_hops - 1)):
            outgoing = self.graph.outgoing(current_node_id)
            if not outgoing:
                break

            candidates = [(target.id, edge.weight) for target, edge in outgoing]
            candidates_considered += len(candidates)

            # Always use edge-weight candidates and apply inhibition before selection.
            scored = score_with_inhibition(
                candidates=candidates,
                graph=self.graph,
                source_node=current_node_id,
                config=self.config.inhibition,
            )
            if not scored:
                break

            selected_target = self._select_next(query, current_node_id, scored, llm_call)
            if selected_target is None or selected_target not in {nid for nid, _ in scored}:
                selected_target = scored[0][0]

            edge = self.graph.get_edge(current_node_id, selected_target)
            step = {
                "from_node": current_node_id,
                "to_node": selected_target,
                "edge_weight": edge.weight if edge is not None else 0.0,
                "candidates": scored,
            }
            trajectory.append(step)

            if selected_target in visited:
                break

            next_node = self.graph.get_node(selected_target)
            if next_node is None:
                break

            next_node.access_count += 1
            selected_nodes.append(selected_target)
            visited.add(selected_target)
            current_node_id = selected_target

            # If we can’t move, stop early.
            if self.graph.outgoing(current_node_id):
                continue
            break

        context = self._render_context(selected_nodes)
        return QueryResult(
            query=query,
            selected_nodes=selected_nodes,
            context=context,
            context_chars=len(context),
            trajectory=trajectory,
            candidates_considered=candidates_considered,
        )

    def learn(self, result: QueryResult, reward: float) -> dict:
        """Apply all learning from a completed query."""
        reward_value = float(reward)
        reward_signal = RewardSignal(episode_id=result.query, final_reward=reward_value)

        selected_nodes = list(dict.fromkeys(result.selected_nodes))
        normalized_trajectory = [
            _normalize_trajectory_step(step) for step in result.trajectory
        ]
        candidates_considered = int(result.candidates_considered)
        learning_step_updates = {}
        skip_penalties = 0

        if self.config.enable_synaptogenesis and selected_nodes:
            synaptogenesis_config = self.config.synaptogenesis
            if self.phase_manager.should_dampen_hebbian():
                synaptogenesis_config = replace(
                    self.config.synaptogenesis,
                    hebbian_increment=self.config.synaptogenesis.hebbian_increment / 2.0,
                )
            synaptogenesis = record_cofiring(
                self.graph,
                selected_nodes,
                self._syn_state,
                synaptogenesis_config,
            )
        else:
            synaptogenesis = {"reinforced": 0, "proto_created": 0, "promoted": 0}

        if selected_nodes:
            self._increment_evidence_counts(selected_nodes)

        for step in normalized_trajectory:
            current = str(step.get("from_node", ""))
            candidates = step.get("candidates") or []
            selected = [str(step.get("to_node", ""))]
            if current:
                skip_penalties += record_skips(
                    self.graph,
                    current,
                    candidates=[str(item[0]) for item in candidates],
                    selected=selected,
                    config=self.config.synaptogenesis,
                )

        if self.config.enable_learning and normalized_trajectory:
            if self.phase_manager.should_skip_pg():
                learning_step_updates = {
                    "updates": [],
                    "baseline": 0.0,
                    "avg_reward": reward_value,
                }
            else:
                learning_result = make_learning_step(
                    self.graph,
                    normalized_trajectory,
                    reward_signal,
                    self.config.learning,
                )
                learning_updates = []
                for update in learning_result.updates:
                    learning_updates.append(
                        update._asdict() if hasattr(update, "_asdict") else update
                    )
                learning_step_updates = {
                    "updates": learning_updates,
                    "baseline": learning_result.baseline,
                    "avg_reward": learning_result.avg_reward,
                }
        else:
            learning_step_updates = {
                "updates": [],
                "baseline": 0.0,
                "avg_reward": reward_value,
            }

        if reward_value < 0 and self.config.enable_inhibition:
            corrections = self._apply_inhibitory_correction(selected_nodes, reward_value)
        else:
            corrections = []

        if reward_value < 0:
            for node_id in selected_nodes:
                node = self.graph.get_node(node_id)
                if node is not None:
                    node.failure_count = int(node.failure_count) + 1

        decay_changes = {}
        if (
            self.config.enable_decay
            and self.config.decay_interval > 0
            and self._query_count % self.config.decay_interval == 0
        ):
            decay_changes = apply_decay(
                self.graph,
                turns_elapsed=self.config.decay_interval,
                config=self.config.decay,
            )

        self.phase_manager.update(self.graph, learning_step_updates)

        return {
            "query": result.query,
            "reward": reward_value,
            "candidates_considered": candidates_considered,
            "synaptogenesis": synaptogenesis,
            "skip_penalties": skip_penalties,
            "learning": learning_step_updates,
            "corrections": corrections,
            "decayed": decay_changes,
            "phase": {
                "current": self.phase_manager.phase,
                "entropy": self.phase_manager.state.weight_entropy,
                "gradient": self.phase_manager.state.avg_gradient_magnitude,
            },
        }

    def stats(self) -> dict[str, Any]:
        return {
            "query_count": self._query_count,
            "graph": {
                "node_count": self.graph.node_count,
                "edge_count": self.graph.edge_count,
                "active_edges": self.graph.active_edge_count(),
            },
            "controller": {
                "enable_learning": self.config.enable_learning,
                "enable_synaptogenesis": self.config.enable_synaptogenesis,
                "enable_inhibition": self.config.enable_inhibition,
                "enable_decay": self.config.enable_decay,
                "decay_interval": self.config.decay_interval,
                "traversal_max_hops": self.config.traversal_max_hops,
                "proto_edges": len(self._syn_state.proto_edges),
            },
        }

    def _seed_nodes(self, query_text: str) -> list[str]:
        query_tokens = set(_tokenize(query_text))
        if not query_tokens:
            return []

        scored: list[tuple[str, float]] = []
        for node in self.graph.nodes():
            if not isinstance(node, Node):
                continue
            node_tokens = set(_tokenize(f"{node.id} {node.content} {node.summary}"))
            overlap = len(query_tokens.intersection(node_tokens))
            if overlap == 0:
                continue
            score = overlap / max(len(query_tokens), 1)
            if score > 0.0:
                scored.append((node.id, score))

        if not scored:
            return []
        scored.sort(key=lambda item: item[1], reverse=True)
        return [node_id for node_id, _ in scored]

    def _select_next(
        self,
        query_text: str,
        current_node_id: str,
        candidates: list[tuple[str, float]],
        llm_call: Callable | None,
    ) -> str | None:
        if llm_call is None:
            return candidates[0][0]
        if not candidates:
            return None

        chosen_id: str | None = None
        try:
            chosen_id = llm_call(query_text, current_node_id, candidates)
        except TypeError:
            try:
                chosen_id = llm_call(query_text, candidates)
            except TypeError:
                chosen_id = llm_call(candidates)
        except Exception:
            chosen_id = None

        if chosen_id is None:
            return candidates[0][0]
        if isinstance(chosen_id, (list, tuple)) and chosen_id:
            if isinstance(chosen_id[0], str):
                return str(chosen_id[0])
            return None
        if isinstance(chosen_id, str):
            return chosen_id
        return None

    def _apply_inhibitory_correction(self, selected_nodes: list[str], reward: float) -> list[dict]:
        if reward >= 0 or len(selected_nodes) < 2:
            return []

        from .inhibition import apply_correction

        return apply_correction(self.graph, selected_nodes, reward, self.config.inhibition)

    def _increment_evidence_counts(self, selected_nodes: list[str]) -> None:
        for source, target in zip(selected_nodes, selected_nodes[1:]):
            edge = self.graph.get_edge(source, target)
            if edge is None:
                continue
            edge.evidence_count = int(edge.evidence_count) + 1

    def _render_context(self, node_ids: list[str]) -> str:
        parts = []
        for node_id in node_ids:
            node = self.graph.get_node(node_id)
            if node is None:
                continue
            if node.content:
                parts.append(node.content)
        return "\n\n".join(parts)
