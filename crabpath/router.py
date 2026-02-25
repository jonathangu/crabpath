from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .graph import Graph
from ._structural_utils import parse_markdown_json


_SYSTEM_PROMPT = (
    "You are a memory router. Given a query and candidate document pointers, "
    'choose which to follow. Output JSON: '
    '{"target": "node_id", "confidence": 0.0-1.0, "rationale": "brief"}'
)

_SELECT_SYSTEM_PROMPT = (
    "You are a memory router. Given a query and candidate nodes with summaries, "
    "select which nodes are needed to answer this query. You may select 0, 1, or multiple. "
    "Select 0 if the query is trivial (greeting, thanks, yes/no) or none are relevant. "
    "Select multiple if the query needs context from several nodes together. "
    'Output JSON: {"selected": ["node_id1", "node_id2"], "rationale": "brief"}'
)


class RouterError(RuntimeError):
    """Raised when routing fails due to parsing issues or unavailable model output."""


@dataclass
class RouterConfig:
    model: str = "gpt-5-mini"
    temperature: float | None = None  # Use model default
    timeout_s: float = 8.0
    max_retries: int = 2
    fallback_behavior: str = "heuristic"
    max_select: int = 5


@dataclass
class RouterDecision:
    chosen_target: str
    rationale: str
    confidence: float
    tier: str
    alternatives: list[tuple[str, float]]
    raw: dict[str, Any]


@dataclass
class _NormalizedCandidate:
    node_id: str
    weight: float
    summary: str | None = None


def normalize_router_payload(payload: Mapping[str, Any]) -> RouterDecision:
    """Validate and normalize a raw routing payload into a `RouterDecision`."""

    if not isinstance(payload, Mapping):
        raise RouterError("Router payload must be a mapping")

    if "target" not in payload:
        raise RouterError("Router payload missing required key: target")

    target = payload["target"]
    if not isinstance(target, str) or not target:
        raise RouterError("Router payload target must be a non-empty string")

    if "confidence" not in payload:
        raise RouterError("Router payload missing required key: confidence")
    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError):
        raise RouterError("Router payload confidence must be numeric")

    if not (0.0 <= confidence <= 1.0):
        raise RouterError("Router payload confidence must be between 0.0 and 1.0")

    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise RouterError("Router payload rationale must be a string")

    raw_alternatives = payload.get("alternatives", [])
    alternatives: list[tuple[str, float]] = []
    if raw_alternatives is None:
        raw_alternatives = []
    if not isinstance(raw_alternatives, list):
        raise RouterError("Router payload alternatives must be a list")

    for alt in raw_alternatives:
        if not isinstance(alt, (tuple, list)) or len(alt) < 2:
            raise RouterError("Each alternative must be a [node_id, score] pair")
        node_id, score = alt[0], alt[1]
        if not isinstance(node_id, str):
            raise RouterError("Alternative node_id must be a string")
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            raise RouterError("Alternative score must be numeric")
        alternatives.append((node_id, score_f))

    tier = str(payload.get("tier", "heuristic"))
    raw = dict(payload)

    return RouterDecision(
        chosen_target=target,
        rationale=rationale,
        confidence=confidence,
        tier=tier,
        alternatives=alternatives,
        raw=raw,
    )


class Router:
    def __init__(self, config: RouterConfig | None = None, client: Any | None = None) -> None:
        self.config = config or RouterConfig()
        self.client = client

    def _coerce_candidates(
        self, candidates: Sequence[tuple[str, float]]
    ) -> list[_NormalizedCandidate]:
        normalized: list[_NormalizedCandidate] = []
        for item in candidates:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                raise RouterError("Candidates must be iterable of (node_id, weight) tuples")

            node_id, score = item[0], item[1]
            if not isinstance(node_id, str):
                raise RouterError("Candidate node_id must be a string")

            try:
                weight = float(score)
            except (TypeError, ValueError) as exc:
                raise RouterError("Candidate weight must be numeric") from exc

            summary = None
            if len(item) >= 3:
                maybe_summary = item[2]
                if maybe_summary is not None:
                    summary = str(maybe_summary)

            normalized.append(_NormalizedCandidate(node_id=node_id, weight=weight, summary=summary))

        return normalized

    @staticmethod
    def _token_count(text: str) -> int:
        return len(text.split())

    def build_prompt(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        context: Mapping[str, Any],
        budget: int,
    ) -> str:
        # Keep prompt under both the caller budget and the Router hard ceiling.
        max_tokens = min(int(budget) if budget else 0, 199)
        if max_tokens <= 0:
            max_tokens = 199

        node_summary = context.get("node_summary")
        if node_summary is None:
            node_summary = context.get("current_node_summary", "")
        if node_summary is None:
            node_summary = ""

        normalized = self._coerce_candidates(candidates)

        header = (
            "System: {_system}\\n"
            "User: Query: {query}\\n"
            "Current: {summary}\\n"
            "These pointers have weights from your past routing decisions.\\n"
            "Higher weight = you followed this more often and it worked.\\n"
            "Candidates:\\n"
        ).format(_system=_SYSTEM_PROMPT, query=query, summary=node_summary)

        candidate_lines: list[str] = []
        for candidate in normalized:
            line = f"→ {candidate.node_id} ({candidate.weight:.2f})"
            if candidate.summary:
                line += f": {candidate.summary}"
            candidate_lines.append(line)

        # Greedily include as many candidate lines as possible without
        # breaking the token cap.
        included: list[str] = []
        for line in candidate_lines:
            if not included:
                next_prompt = f"{header}{line}"
            else:
                included_block = "\n".join(included)
                next_prompt = f"{header}{included_block}\n{line}"
            if (
                self._token_count(next_prompt) < 200
                and self._token_count(next_prompt) <= max_tokens
            ):
                included.append(line)
            else:
                break

        body = "\n".join(included)
        prompt = f"{header}{body}"

        # In case context or system/user lines are already too long, trim
        # with a very conservative fallback that keeps template validity.
        if self._token_count(prompt) >= 200 or self._token_count(prompt) > max_tokens:
            # leave room for a short suffix showing truncation and include only the top line.
            fallback_query = (
                query
                if len(query.split()) < max_tokens
                else " ".join(query.split()[: max_tokens // 3])
            )
            prompt = (
                f"System: {_SYSTEM_PROMPT}\nUser: Query: {fallback_query}\n"
                f"Current: {str(node_summary)[:40]}\nCandidates:\n"
                f"(no candidates shown: token budget reached)"
            )

        return prompt

    def parse_json(self, raw: str) -> dict[str, Any]:
        try:
            payload = parse_markdown_json(raw, require_object=True)
        except (TypeError, ValueError) as exc:
            raise RouterError("Router output is not valid JSON") from exc

        if not isinstance(payload, dict):
            raise RouterError("Router payload must be a JSON object")

        for required_key in ("target", "confidence", "rationale"):
            if required_key not in payload:
                raise RouterError(f"Router payload missing required key: {required_key}")

        if not isinstance(payload["target"], str):
            raise RouterError("Router payload target must be a string")

        if not isinstance(payload["rationale"], str):
            raise RouterError("Router payload rationale must be a string")

        try:
            confidence = float(payload["confidence"])
        except (TypeError, ValueError):
            raise RouterError("Router payload confidence must be numeric")

        if not (0.0 <= confidence <= 1.0):
            raise RouterError("Router payload confidence must be between 0.0 and 1.0")

        if "alternatives" in payload:
            alternatives = payload["alternatives"]
            if not isinstance(alternatives, list):
                raise RouterError("Router payload alternatives must be a list")

            for alt in alternatives:
                if not isinstance(alt, (tuple, list)) or len(alt) < 2:
                    raise RouterError("Alternative entries must be [node_id, score]")
                if not isinstance(alt[0], str):
                    raise RouterError("Alternative node_id must be a string")
                try:
                    float(alt[1])
                except (TypeError, ValueError):
                    raise RouterError("Alternative score must be numeric")

        return payload

    def _extract_model_output(self, messages: Sequence[dict[str, str]]) -> str:
        if not self.client:
            raise RouterError("No client configured")

        # Primary path for OpenAI-style client.
        if hasattr(self.client, "chat"):
            kwargs: dict = {
                "model": self.config.model,
                "messages": list(messages),
                "timeout": self.config.timeout_s,
            }
            if self.config.temperature is not None:
                kwargs["temperature"] = self.config.temperature
            response = self.client.chat.completions.create(**kwargs)

            if not response.choices:
                raise RouterError("LLM response missing choices")

            choice = response.choices[0]
            content = choice.message.content if hasattr(choice, "message") else None
            if not content:
                raise RouterError("LLM response content missing")
            return str(content)

        # Generic callable path for tests/mocks.
        if callable(self.client):
            return str(self.client(messages))

        complete = getattr(self.client, "complete", None)
        if callable(complete):
            return str(complete(messages))

        raise RouterError("Unknown client interface")

    def decide_next(
        self,
        query: str,
        current_node_id: str,
        candidate_nodes: list[tuple[str, float]],
        context: Mapping[str, Any],
        tier: str,
        previous_reasoning: str | None = None,
    ) -> RouterDecision:
        del previous_reasoning  # reserved for future reasoning carry-over

        candidates = self._coerce_candidates(candidate_nodes)
        if not candidates:
            raise RouterError("No candidates provided")

        if not self.client or self.config.fallback_behavior == "heuristic":
            return self.fallback([(c.node_id, c.weight) for c in candidates], tier)

        prompt = self.build_prompt(query, [(c.node_id, c.weight) for c in candidates], context, 199)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        last_error: Exception | None = None
        for _attempt in range(max(self.config.max_retries, 0) + 1):
            try:
                raw_output = self._extract_model_output(messages)
                payload = self.parse_json(raw_output)
                decision = normalize_router_payload(payload)
                if "tier" not in payload:
                    decision.tier = tier
                decision.raw = payload
                return decision
            except (RouterError, Exception) as exc:
                import warnings

                warnings.warn(
                    f"CrabPath: Router.decide_next failed: {exc}. Falling back to heuristic routing.",
                    stacklevel=2,
                )
                last_error = exc

        if self.config.fallback_behavior == "heuristic":
            return self.fallback([(c.node_id, c.weight) for c in candidates], tier)

        if last_error is None:
            raise RouterError("Routing failed")
        if isinstance(last_error, RouterError):
            raise
        raise RouterError("Routing failed") from last_error

    def select_nodes(
        self,
        query: str,
        candidates: list[tuple[str, float, str]],
        current_node_id: str | None = None,
        graph: Graph | None = None,
    ) -> list[str]:
        """Select 0, 1, or N nodes relevant to a query. LLM decides.

        Args:
            query: The user query.
            candidates: List of (node_id, score, summary) tuples.
            current_node_id: Optional current context node (used to avoid candidates
                with inhibitory edges from it.
            graph: Optional graph for current-context edge checks.

        Returns:
            List of selected node_ids (may be empty).
        """
        if not candidates:
            return []

        if not self.client or self.config.fallback_behavior == "heuristic":
            return self._select_fallback(query, candidates, current_node_id, graph)

        candidate_lines = "\n".join(
            f"→ {'AVOID: ' if score < 0 else ''}{nid} ({score:.2f}): {summary[:100]}"
            for nid, score, summary in candidates
        )
        user_msg = f"Query: {query}\n\nCandidates:\n{candidate_lines}"

        messages = [
            {"role": "system", "content": _SELECT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        for _attempt in range(max(self.config.max_retries, 0) + 1):
            try:
                raw_output = self._extract_model_output(messages)
                payload = self.parse_select_json(raw_output)
                selected = payload.get("selected", [])
                if isinstance(selected, list):
                    # Filter to only valid candidate IDs
                    valid_ids = {nid for nid, _, _ in candidates}
                    selected_nodes = [str(s) for s in selected if str(s) in valid_ids]
                    if self.config.max_select > 0:
                        selected_nodes = selected_nodes[: self.config.max_select]
                    return selected_nodes
                return []
            except Exception as exc:
                import warnings

                warnings.warn(
                    f"CrabPath: select_nodes failed: {exc}. Falling back to heuristic selection.",
                    stacklevel=2,
                )
                continue

        return self._select_fallback(query, candidates, current_node_id, graph)

    def _select_fallback(
        self,
        query: str,
        candidates: list[tuple[str, float, str]],
        current_node_id: str | None = None,
        graph: Graph | None = None,
    ) -> list[str]:
        """Heuristic fallback: select top candidates by score."""
        if not candidates:
            return []

        if current_node_id and graph is not None:
            filtered_candidates: list[tuple[str, float, str]] = []
            for node_id, score, summary in candidates:
                edge = graph.get_edge(current_node_id, node_id)
                if edge is not None and edge.weight < 0:
                    continue
                filtered_candidates.append((node_id, score, summary))
            candidates = filtered_candidates

        # Trivial query detection
        trivial = {"hello", "hi", "thanks", "yes", "no", "ok", "sure", "yeah", "bye"}
        words = set(query.lower().split())
        if words and words.issubset(trivial | {""}):
            return []

        # Select all candidates above a reasonable threshold
        sorted_candidates = sorted(candidates, key=lambda c: c[1], reverse=True)
        if not sorted_candidates:
            return []

        best_score = sorted_candidates[0][1]
        if best_score <= 0:
            selected: list[str] = []
        else:
            threshold = best_score * 0.6
            selected = [
                nid for nid, score, _ in sorted_candidates if score >= threshold
            ]

        # At minimum return the top one if anything scored decently
        if not selected and sorted_candidates and sorted_candidates[0][1] > 0.15:
            selected = [sorted_candidates[0][0]]

        if self.config.max_select > 0:
            selected = selected[: self.config.max_select]

        return selected

    def parse_select_json(self, raw: str) -> dict[str, Any]:
        """Parse the selection response JSON."""
        try:
            payload = parse_markdown_json(raw, require_object=True)
        except (TypeError, ValueError) as exc:
            raise RouterError("Select output is not valid JSON") from exc

        if not isinstance(payload, dict):
            raise RouterError("Select payload must be a JSON object")

        return payload

    def fallback(self, candidates: list[tuple[str, float]], tier: str) -> RouterDecision:
        normalized = self._coerce_candidates(candidates)
        if not normalized:
            raise RouterError("Cannot fallback with empty candidate list")

        ranked = sorted(normalized, key=lambda c: (c.weight, c.node_id), reverse=True)
        chosen = ranked[0]

        alternatives = [(candidate.node_id, candidate.weight) for candidate in ranked[1:]]

        confidence = (chosen.weight + 1.0) / 2.0
        if confidence < 0:
            confidence = 0.0
        elif confidence > 1:
            confidence = 1.0

        return RouterDecision(
            chosen_target=chosen.node_id,
            rationale=(
                f"Fallback selected highest-weight candidate '{chosen.node_id}' "
                f"in tier '{tier}'."
            ),
            confidence=confidence,
            tier=tier,
            alternatives=alternatives,
            raw={
                "method": "fallback",
                "tier": tier,
                "target": chosen.node_id,
                "confidence": confidence,
                "rationale": (
                    f"Fallback selected highest-weight candidate '{chosen.node_id}' "
                    f"in tier '{tier}'."
                ),
                "alternatives": alternatives,
            },
        )
