#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crabpath.graph import Edge, Graph, Node


STABLE_NODES = [
    {
        "id": "sc_coordinator_old",
        "type": "fact",
        "content": "Fact: The incident coordinator is Jordan and must receive all priority alerts.",
        "summary": "Old coordinator",
    },
    {
        "id": "sc_coordinator_new",
        "type": "fact",
        "content": "Fact: The incident coordinator is Morgan and must receive all priority alerts.",
        "summary": "New coordinator",
    },
    {
        "id": "sc_region_old",
        "type": "fact",
        "content": "Fact: Primary production traffic should use us-east-1.",
        "summary": "Old region",
    },
    {
        "id": "sc_region_new",
        "type": "fact",
        "content": "Fact: Primary production traffic should use eu-west-1.",
        "summary": "New region",
    },
    {
        "id": "sc_retry_old",
        "type": "fact",
        "content": "Fact: Retry policy is 2 attempts with exponential backoff for transient errors.",
        "summary": "Old retry",
    },
    {
        "id": "sc_retry_new",
        "type": "fact",
        "content": "Fact: Retry policy is 4 jittered attempts with exponential backoff for transient errors.",
        "summary": "New retry",
    },
    {
        "id": "sc_coordinator_guard",
        "type": "guardrail",
        "content": "Guardrail: do not keep using a coordinator field if policy has moved to a replacement name.",
        "summary": "Coordinator guardrail",
    },
    {
        "id": "sc_region_guard",
        "type": "guardrail",
        "content": "Guardrail: do not keep routing traffic to the old region once migration is active.",
        "summary": "Region guardrail",
    },
    {
        "id": "sc_retry_guard",
        "type": "guardrail",
        "content": "Guardrail: do not lock retry policy to stale fixed 2-attempt defaults.",
        "summary": "Retry guardrail",
    },
    {
        "id": "sc_tool_runbook",
        "type": "tool_call",
        "content": "Tool call: open on-call runbook and on-call schedule before answering coordination or routing questions.",
        "summary": "Runbook lookup",
    },
]

TOPIC_ENTRIES = [
    {
        "id": "sc_entry_coordinator",
        "type": "fact",
        "content": "Coordinator policy retrieval node",
        "summary": "coordinator lookup",
        "relevant": ["sc_coordinator_old", "sc_coordinator_new", "sc_coordinator_guard", "sc_tool_runbook"],
    },
    {
        "id": "sc_entry_region",
        "type": "fact",
        "content": "Region policy retrieval node",
        "summary": "region lookup",
        "relevant": ["sc_region_old", "sc_region_new", "sc_region_guard", "sc_tool_runbook"],
    },
    {
        "id": "sc_entry_retry",
        "type": "fact",
        "content": "Retry policy retrieval node",
        "summary": "retry lookup",
        "relevant": ["sc_retry_old", "sc_retry_new", "sc_retry_guard", "sc_tool_runbook"],
    },
]

SCENARIO_TEMPLATE = [
    {"topic": "coordinator", "query": "Who is the current incident coordinator?", "old": "Jordan", "new": "Morgan"},
    {"topic": "region", "query": "Which primary region should traffic use?", "old": "us-east-1", "new": "eu-west-1"},
    {"topic": "retry", "query": "What retry policy applies right now?", "old": "2 attempts", "new": "4 attempts"},
]


CORRECTION_TURNS = [
    {"topic": "coordinator", "query": "Correction: The incident coordinator is Morgan, not Jordan."},
    {"topic": "region", "query": "Correction: Production traffic now uses eu-west-1, not us-east-1."},
    {"topic": "retry", "query": "Correction: Use 4 jittered retry attempts, not 2."},
    {"topic": "coordinator", "query": "The coordinator is now Morgan and not Jordan."},
    {"topic": "region", "query": "Remember, production routing is now eu-west-1."},
]

RECALL_TEMPLATES = {
    "coordinator": {
        "queries": [
            "Who should own priority incidents?",
            "Need the active coordinator name.",
            "Who is currently on-call for major incidents?",
        ],
        "new": "Morgan",
    },
    "region": {
        "queries": [
            "Which region should production use now?",
            "Which region currently serves production traffic?",
            "Primary region after migration?",
        ],
        "new": "eu-west-1",
    },
    "retry": {
        "queries": [
            "What retry policy applies now?",
            "How many retry attempts should clients use?",
            "What are the transient retry limits?",
        ],
        "new": "4 attempts",
    },
}



def _build_topics(graph: Graph) -> None:
    for entry in TOPIC_ENTRIES:
        graph.add_node(Node(entry["id"], entry["content"], entry["summary"], entry["type"]))
        graph.add_edge(Edge(source="stale_root", target=entry["id"], weight=1.0))
        for idx, node_id in enumerate(entry["relevant"], start=1):
            graph.add_edge(Edge(source=entry["id"], target=node_id, weight=0.82 - 0.08 * (idx - 1)))


def build_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node("stale_root", "Stale-context experiment root", "", "fact"))

    for record in STABLE_NODES:
        graph.add_node(Node(record["id"], record["content"], record["summary"], record["type"]))

    _build_topics(graph)

    # Inhibition supports faster correction and decay-like suppression of stale facts.
    graph.add_edge(Edge(source="sc_region_new", target="sc_region_old", weight=-0.55))
    graph.add_edge(Edge(source="sc_coordinator_new", target="sc_coordinator_old", weight=-0.50))
    graph.add_edge(Edge(source="sc_retry_new", target="sc_retry_old", weight=-0.50))
    return graph


def _build_scenario_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for turn in range(10):
        template = SCENARIO_TEMPLATE[turn % len(SCENARIO_TEMPLATE)]
        topic_key = template["topic"]
        rows.append(
            {
                "query": template["query"],
                "feedback": {"reward": 1.0},
                "expected_answer_fragments": [template["old"]],
                "topic": topic_key,
            }
        )

    for item in CORRECTION_TURNS:
        topic_key = item["topic"]
        old_new = next(t for t in SCENARIO_TEMPLATE if t["topic"] == topic_key)
        rows.append(
            {
                "query": item["query"],
                "feedback": {"reward": 1.0},
                "expected_answer_fragments": [old_new["new"]],
                "topic": topic_key,
            }
        )

    for idx in range(15):
        topic_key = SCENARIO_TEMPLATE[idx % len(SCENARIO_TEMPLATE)]["topic"]
        template = RECALL_TEMPLATES[topic_key]
        query = template["queries"][idx % len(template["queries"]) ]
        expected = [template["new"]]
        reward = 1.0
        # Introduce one intentional stale slip at turn 20 to measure catch-up behavior.
        if idx == 10:
            reward = -1.0
            expected = ["Jordan", "us-east-1", "2 attempts"]
        rows.append(
            {
                "query": query,
                "feedback": {"reward": reward},
                "expected_answer_fragments": expected,
                "topic": topic_key,
            }
        )

    return rows


def build_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "query": row["query"],
            "feedback": row["feedback"],
            "expected_answer_fragments": row["expected_answer_fragments"],
        }
        for row in _build_scenario_rows()
    ]


def write_outputs(graph_path: Path, scenario_path: Path) -> None:
    graph = build_graph()
    graph.save(str(graph_path))
    with scenario_path.open("w", encoding="utf-8") as f:
        for row in build_scenarios():
            f.write(json.dumps(row))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build stale-context experiment graph and scenarios")
    parser.add_argument("--graph", default="experiments/stale_context_graph.json", help="Path to write stale-context graph JSON")
    parser.add_argument("--scenario", default="scenarios/stale_context.jsonl", help="Path to write stale-context scenario JSONL")
    args = parser.parse_args()

    write_outputs(Path(args.graph), Path(args.scenario))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
