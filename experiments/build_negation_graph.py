#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crabpath.graph import Edge, Graph, Node


NEGATION_NODES = [
    {
        "id": "ng_do_run_migrations_now",
        "type": "fact",
        "content": "Policy A: Run database migrations directly in production during immediate patch windows.",
        "summary": "Migrations in production",
    },
    {
        "id": "ng_do_not_run_migrations_in_prod",
        "type": "guardrail",
        "content": "Policy B: Do NOT run database migrations directly in production; use a staged deployment first.",
        "summary": "No prod migrations",
    },
    {
        "id": "ng_use_staging_migration",
        "type": "procedure",
        "content": "Procedure: migrate first in staging, then validate and promote to production.",
        "summary": "Staged migration",
    },
    {
        "id": "ng_use_prod_window",
        "type": "procedure",
        "content": "Procedure: schedule production migration during an approved maintenance window.",
        "summary": "Maintenance window",
    },
    {
        "id": "ng_feature_flag",
        "type": "procedure",
        "content": "Procedure: gate risky operations behind a feature flag before broad execution.",
        "summary": "Feature flag",
    },
    {
        "id": "ng_tool_validate",
        "type": "tool_call",
        "content": "Tool call: validate migration plan against latest schema before executing any change.",
        "summary": "Schema validate",
    },
]

SCENARIO = [
    {"query": "Can we run a migration now?", "expected": ["Directly in production", "run now"], "reward": 0.0},
    {"query": "What is the migration policy for this request?", "expected": ["Directly in production", "run now"], "reward": 0.0},
    {"query": "Could we skip staging and run immediately?", "expected": ["Directly in production", "skip staging"], "reward": 0.0},
    {"query": "Correction: Do NOT run migrations directly in production; stage first.", "expected": ["Do NOT", "staged"], "reward": 1.0},
    {"query": "What is the updated migration policy?", "expected": ["Do NOT", "staged"], "reward": 1.0},
    {"query": "Should I run it in production right now?", "expected": ["Do NOT", "staged"], "reward": 1.0},
    {"query": "Confirm the policy for migration execution.", "expected": ["maintenance window", "staged"], "reward": 1.0},
    {"query": "What if this migration is urgent?", "expected": ["maintenance window", "Do NOT"], "reward": 1.0},
    {"query": "Can you explain the right sequence?", "expected": ["staged", "feature flag"], "reward": 1.0},
    {"query": "What should we execute before migration in prod?", "expected": ["validate", "staged"], "reward": 1.0},
    {"query": "Is direct prod migration still allowed?", "expected": ["Do NOT", "staged"], "reward": 1.0},
]


def build_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node("negation_root", "Negation contrast graph", "", "fact"))

    for record in NEGATION_NODES:
        graph.add_node(Node(record["id"], record["content"], record["summary"], record["type"]))

    graph.add_node(Node("neg_query", "Migration query entry", "negation entry", "fact"))
    graph.add_edge(Edge(source="negation_root", target="neg_query", weight=1.0))

    graph.add_edge(Edge(source="neg_query", target="ng_do_run_migrations_now", weight=0.85))
    graph.add_edge(Edge(source="neg_query", target="ng_do_not_run_migrations_in_prod", weight=0.85))

    graph.add_edge(Edge(source="ng_do_run_migrations_now", target="ng_use_prod_window", weight=0.78))
    graph.add_edge(Edge(source="ng_do_run_migrations_now", target="ng_tool_validate", weight=0.60))
    graph.add_edge(Edge(source="ng_do_not_run_migrations_in_prod", target="ng_use_staging_migration", weight=0.90))
    graph.add_edge(Edge(source="ng_do_not_run_migrations_in_prod", target="ng_feature_flag", weight=0.64))

    graph.add_edge(Edge(source="ng_do_not_run_migrations_in_prod", target="ng_do_run_migrations_now", weight=-0.75))
    graph.add_edge(Edge(source="ng_do_run_migrations_now", target="ng_do_not_run_migrations_in_prod", weight=-0.20))

    for guard_node in ["ng_do_not_run_migrations_in_prod", "ng_do_run_migrations_now"]:
        graph.add_edge(Edge(source="neg_query", target=guard_node, weight=0.15))

    return graph


def build_scenarios() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in SCENARIO:
        rows.append(
            {
                "query": step["query"],
                "feedback": {"reward": float(step["reward"])},
                "expected_answer_fragments": step["expected"],
            }
        )
    return rows


def write_outputs(graph_path: Path, scenario_path: Path) -> None:
    graph = build_graph()
    graph.save(str(graph_path))
    with scenario_path.open("w", encoding="utf-8") as f:
        for row in build_scenarios():
            f.write(json.dumps(row))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build negation experiment graph and scenarios")
    parser.add_argument("--graph", default="experiments/negation_graph.json", help="Path to write negation graph JSON")
    parser.add_argument("--scenario", default="scenarios/negation.jsonl", help="Path to write negation scenario JSONL")
    args = parser.parse_args()
    write_outputs(Path(args.graph), Path(args.scenario))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
