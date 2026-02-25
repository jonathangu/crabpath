#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crabpath.graph import Edge, Graph, Node


PROCEDURE_NODES = [
    {
        "id": "pp_check_logs",
        "type": "procedure",
        "content": "Procedure step 1: check logs for recent errors and capture trace IDs.",
        "summary": "Check logs",
    },
    {
        "id": "pp_read_config",
        "type": "procedure",
        "content": "Procedure step 2: read relevant service config and verify environment overrides.",
        "summary": "Read config",
    },
    {
        "id": "pp_fix_code",
        "type": "procedure",
        "content": "Procedure step 3: fix the code path and rerun targeted checks.",
        "summary": "Fix code",
    },
    {
        "id": "pp_run_tests",
        "type": "procedure",
        "content": "Procedure step 4: run automated tests and confirm green status.",
        "summary": "Run tests",
    },
    {
        "id": "pp_investigate_alert",
        "type": "procedure",
        "content": "Decoy: investigate unrelated alert and skip core procedure.",
        "summary": "Investigate other alert",
    },
    {
        "id": "pp_fix_style",
        "type": "procedure",
        "content": "Decoy: fix formatting/style before runtime validation.",
        "summary": "Fix style",
    },
    {
        "id": "pp_open_ticket",
        "type": "procedure",
        "content": "Decoy: open support ticket before any operational action.",
        "summary": "Open ticket",
    },
    {
        "id": "pp_reboot_worker",
        "type": "procedure",
        "content": "Decoy: reboot worker nodes without checking config first.",
        "summary": "Reboot workers",
    },
]

SCENARIO_TEXTS = [
    "Resolve a production bug in the checkout path.",
    "Troubleshoot a failing deployment and patch quickly.",
    "Debug a flaky test in the service health check path.",
    "Fix a regression in a customer-facing endpoint.",
    "Address a production exception in background jobs.",
    "Recover from config drift in runtime.",
    "Handle a test-only failure before release.",
    "Stabilize a request timeout regression.",
    "Repair a broken startup sequence.",
    "Patch a service path after monitoring alert.",
]


def build_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node("pp_root", "Procedure-learning root", "", "fact"))

    for record in PROCEDURE_NODES:
        graph.add_node(Node(record["id"], record["content"], record["summary"], record["type"]))

    graph.add_node(Node("pp_query", "Procedure planning entry", "procedure planning", "fact"))
    graph.add_edge(Edge(source="pp_root", target="pp_query", weight=1.0))

    # Provide one weakly correct chain and several tempting alternatives.
    graph.add_edge(Edge(source="pp_query", target="pp_check_logs", weight=0.65))
    graph.add_edge(Edge(source="pp_query", target="pp_investigate_alert", weight=0.62))
    graph.add_edge(Edge(source="pp_query", target="pp_read_config", weight=0.56))
    graph.add_edge(Edge(source="pp_query", target="pp_fix_style", weight=0.54))

    graph.add_edge(Edge(source="pp_check_logs", target="pp_read_config", weight=0.72))
    graph.add_edge(Edge(source="pp_check_logs", target="pp_fix_code", weight=0.31))

    graph.add_edge(Edge(source="pp_read_config", target="pp_fix_code", weight=0.68))
    graph.add_edge(Edge(source="pp_read_config", target="pp_run_tests", weight=0.34))

    graph.add_edge(Edge(source="pp_fix_code", target="pp_run_tests", weight=0.66))
    graph.add_edge(Edge(source="pp_fix_code", target="pp_open_ticket", weight=0.33))

    graph.add_edge(Edge(source="pp_investigate_alert", target="pp_reboot_worker", weight=0.44))
    graph.add_edge(Edge(source="pp_investigate_alert", target="pp_check_logs", weight=0.58))

    graph.add_edge(Edge(source="pp_reboot_worker", target="pp_run_tests", weight=0.30))
    graph.add_edge(Edge(source="pp_run_tests", target="pp_check_logs", weight=0.20))

    # Inhibitory signal once new sequence is validated should reduce wrong jump points.
    graph.add_edge(Edge(source="pp_run_tests", target="pp_reboot_worker", weight=-0.45))
    graph.add_edge(Edge(source="pp_run_tests", target="pp_open_ticket", weight=-0.40))

    return graph


def build_scenarios() -> list[dict[str, object]]:
    scenario: list[dict[str, object]] = []
    for i, query in enumerate(SCENARIO_TEXTS, start=1):
        reward = 1.0
        if i in (2, 3):
            reward = -1.0
        elif i in (4, 5):
            reward = 0.4
        scenario.append(
            {
                "query": query,
                "feedback": {"reward": reward},
                "expected_answer_fragments": [
                    "check logs",
                    "read config",
                    "fix code",
                    "run tests",
                ],
            }
        )
    return scenario


def write_outputs(graph_path: Path, scenario_path: Path) -> None:
    graph = build_graph()
    graph.save(str(graph_path))
    with scenario_path.open("w", encoding="utf-8") as f:
        for row in build_scenarios():
            f.write(json.dumps(row))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build procedure-learning experiment graph and scenarios")
    parser.add_argument("--graph", default="experiments/procedure_graph.json", help="Path to write procedure graph JSON")
    parser.add_argument("--scenario", default="scenarios/procedure.jsonl", help="Path to write procedure scenario JSONL")
    args = parser.parse_args()

    write_outputs(Path(args.graph), Path(args.scenario))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
