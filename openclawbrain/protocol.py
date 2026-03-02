"""Versioned protocol helpers for daemon NDJSON requests/responses."""

from __future__ import annotations

from dataclasses import dataclass

ROUTE_MODES = {"off", "edge", "edge+sim", "learned"}
PROTOCOL_VERSION = "v1"



def parse_int(value: object, label: str, default: int | None = None, min_value: int = 1) -> int:
    """Validate integer input."""
    if value is None:
        if default is None:
            raise ValueError(f"{label} is required")
        return default
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < min_value:
        raise ValueError(f"{label} must be >= {min_value}")
    return value



def parse_float(value: object, label: str, required: bool = False, default: float | None = None) -> float:
    """Validate float input."""
    if value is None:
        if not required:
            if default is None:
                raise ValueError(f"{label} is required")
            return default
        raise ValueError(f"{label} is required")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)



def parse_bool(value: object, label: str, *, default: bool) -> bool:
    """Validate boolean input with a default."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value



def parse_route_mode(value: object) -> str:
    """Validate route mode."""
    if value is None:
        return "off"
    if not isinstance(value, str):
        raise ValueError("route_mode must be one of: off, edge, edge+sim, learned")
    mode = value.strip().lower()
    if mode not in ROUTE_MODES:
        raise ValueError("route_mode must be one of: off, edge, edge+sim, learned")
    return mode



def parse_str_list(value: object, label: str, required: bool = True) -> list[str]:
    """Parse a JSON list of non-empty strings."""
    if value is None:
        if required:
            raise ValueError(f"{label} is required")
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    entries: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} must only contain non-empty strings")
        entries.append(item)
    if not entries and required:
        raise ValueError(f"{label} must not be empty")
    return entries



def parse_chat_id(value: object, label: str, required: bool = True) -> str | None:
    """Parse optional chat_id with non-empty normalization."""
    if value is None:
        if required:
            raise ValueError(f"{label} is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = value.strip()
    if required and not normalized:
        raise ValueError(f"{label} is required")
    return normalized if normalized else None


@dataclass(frozen=True)
class QueryParams:
    """Typed query params used by daemon query routing/execution."""

    query: str
    top_k: int = 4
    max_prompt_context_chars: int = 30000
    max_context_chars: int = 30000
    max_fired_nodes: int = 30
    route_mode: str = "off"
    route_top_k: int = 5
    route_alpha_sim: float = 0.5
    route_use_relevance: bool = True
    prompt_context_include_node_ids: bool = True
    exclude_files: tuple[str, ...] = ()
    exclude_file_prefixes: tuple[str, ...] = ()
    chat_id: str | None = None

    @classmethod
    def from_dict(cls, params: dict[str, object]) -> "QueryParams":
        """Build validated params from raw JSON dict."""
        query_text = params.get("query")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query must be a non-empty string")

        top_k = parse_int(params.get("top_k"), "top_k", default=4)

        max_prompt_chars = params.get("max_prompt_context_chars")
        if not isinstance(max_prompt_chars, int):
            max_prompt_chars = 30000

        max_context_chars = params.get("max_context_chars")
        if not isinstance(max_context_chars, int):
            max_context_chars = max_prompt_chars

        return cls(
            query=query_text,
            top_k=top_k,
            max_prompt_context_chars=max_prompt_chars,
            max_context_chars=max_context_chars,
            max_fired_nodes=parse_int(params.get("max_fired_nodes"), "max_fired_nodes", default=30),
            route_mode=parse_route_mode(params.get("route_mode")),
            route_top_k=parse_int(params.get("route_top_k"), "route_top_k", default=5),
            route_alpha_sim=parse_float(params.get("route_alpha_sim"), "route_alpha_sim", default=0.5),
            route_use_relevance=parse_bool(params.get("route_use_relevance"), "route_use_relevance", default=True),
            prompt_context_include_node_ids=parse_bool(
                params.get("prompt_context_include_node_ids"),
                "prompt_context_include_node_ids",
                default=True,
            ),
            exclude_files=tuple(parse_str_list(params.get("exclude_files"), "exclude_files", required=False)),
            exclude_file_prefixes=tuple(
                parse_str_list(params.get("exclude_file_prefixes"), "exclude_file_prefixes", required=False)
            ),
            chat_id=parse_chat_id(params.get("chat_id"), "chat_id", required=False),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to JSON-serializable dict."""
        return {
            "query": self.query,
            "top_k": self.top_k,
            "max_prompt_context_chars": self.max_prompt_context_chars,
            "max_context_chars": self.max_context_chars,
            "max_fired_nodes": self.max_fired_nodes,
            "route_mode": self.route_mode,
            "route_top_k": self.route_top_k,
            "route_alpha_sim": self.route_alpha_sim,
            "route_use_relevance": self.route_use_relevance,
            "prompt_context_include_node_ids": self.prompt_context_include_node_ids,
            "exclude_files": list(self.exclude_files),
            "exclude_file_prefixes": list(self.exclude_file_prefixes),
            "chat_id": self.chat_id,
        }


@dataclass(frozen=True)
class QueryRequest:
    """Versioned daemon request wrapper for query calls."""

    id: object
    method: str
    params: dict[str, object]
    protocol_version: str = PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, request: object) -> "QueryRequest":
        """Parse and validate one daemon request envelope."""
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be a JSON object")
        if not isinstance(method, str):
            raise ValueError("method must be a string")
        return cls(id=req_id, method=method, params=params)

    def query_params(self) -> QueryParams:
        """Parse query params using protocol v1 schema."""
        return QueryParams.from_dict(self.params)


@dataclass(frozen=True)
class QueryResponse:
    """Versioned daemon response envelope for query and other methods."""

    id: object
    result: object | None = None
    error: dict[str, object] | None = None
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        has_result = self.result is not None
        has_error = self.error is not None
        if has_result and has_error:
            raise ValueError("response cannot include both result and error")
        if not has_result and not has_error:
            raise ValueError("response must include result or error")

    def to_dict(self) -> dict[str, object]:
        """Convert envelope to NDJSON-compatible dict."""
        if self.error is not None:
            return {"id": self.id, "error": self.error}
        return {"id": self.id, "result": self.result}
