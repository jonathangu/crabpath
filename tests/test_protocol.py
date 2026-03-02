from __future__ import annotations

import pytest

from openclawbrain.protocol import QueryParams, QueryRequest, QueryResponse


def test_query_request_from_dict_parses_envelope() -> None:
    request = QueryRequest.from_dict({"id": "1", "method": "query", "params": {"query": "alpha"}})
    assert request.id == "1"
    assert request.method == "query"
    assert request.params == {"query": "alpha"}


def test_query_request_from_dict_validates_shape() -> None:
    with pytest.raises(ValueError, match="request must be a JSON object"):
        QueryRequest.from_dict("bad")

    with pytest.raises(ValueError, match="params must be a JSON object"):
        QueryRequest.from_dict({"id": 1, "method": "query", "params": []})

    with pytest.raises(ValueError, match="method must be a string"):
        QueryRequest.from_dict({"id": 1, "method": None, "params": {}})


def test_query_params_defaults_follow_daemon_behavior() -> None:
    params = QueryParams.from_dict({"query": "alpha"})
    assert params.query == "alpha"
    assert params.top_k == 4
    assert params.max_prompt_context_chars == 30000
    assert params.max_context_chars == 30000
    assert params.max_fired_nodes == 30
    assert params.route_mode == "off"
    assert params.route_top_k == 5
    assert params.route_alpha_sim == 0.5
    assert params.route_use_relevance is True
    assert params.prompt_context_include_node_ids is True
    assert params.exclude_files == ()
    assert params.exclude_file_prefixes == ()
    assert params.chat_id is None


def test_query_params_max_context_defaults_to_prompt_chars() -> None:
    params = QueryParams.from_dict({"query": "alpha", "max_prompt_context_chars": 777})
    assert params.max_prompt_context_chars == 777
    assert params.max_context_chars == 777


def test_query_params_accepts_learned_route_mode() -> None:
    params = QueryParams.from_dict({"query": "alpha", "route_mode": "learned"})
    assert params.route_mode == "learned"


def test_query_params_validation_errors() -> None:
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        QueryParams.from_dict({"query": ""})

    with pytest.raises(ValueError, match="top_k must be an integer"):
        QueryParams.from_dict({"query": "x", "top_k": "2"})

    with pytest.raises(ValueError, match=r"route_mode must be one of: off, edge, edge\+sim, learned"):
        QueryParams.from_dict({"query": "x", "route_mode": "bad"})

    with pytest.raises(ValueError, match="route_use_relevance must be a boolean"):
        QueryParams.from_dict({"query": "x", "route_use_relevance": "yes"})


def test_query_response_to_dict_result_and_error() -> None:
    ok = QueryResponse(id="1", result={"x": 1}).to_dict()
    assert ok == {"id": "1", "result": {"x": 1}}

    err = QueryResponse(id="1", error={"code": -1, "message": "bad"}).to_dict()
    assert err == {"id": "1", "error": {"code": -1, "message": "bad"}}

    with pytest.raises(ValueError, match="response must include result or error"):
        QueryResponse(id="1")
