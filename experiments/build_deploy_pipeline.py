#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crabpath.graph import Edge, Graph, Node


DEPLOY_NODES = [
    {
        "id": "deploy_root",
        "type": "fact",
        "content": "Deployment pipeline entry point",
    },
    {
        "id": "check_tests",
        "type": "action",
        "content": "Run: pytest tests/ -v. Verifies all tests pass before deployment.",
    },
    {
        "id": "check_ci",
        "type": "action",
        "content": "Run: gh run list --limit 1. Verify CI pipeline is green.",
    },
    {
        "id": "deploy_staging",
        "type": "tool_call",
        "content": "Run: ssh staging cd /app && git pull && pm2 restart. Deploy to staging first.",
    },
    {
        "id": "deploy_prod",
        "type": "tool_call",
        "content": "Run: ssh prod cd /app && git pull && pm2 restart. Deploy to production.",
    },
    {
        "id": "rollback",
        "type": "procedure",
        "content": "If deploy fails: ssh prod git checkout HEAD~1 && pm2 restart. Rollback to previous version.",
    },
    {
        "id": "skip_tests",
        "type": "action",
        "content": "Deploy without running tests. DANGEROUS - can break production.",
    },
    {
        "id": "hotfix",
        "type": "procedure",
        "content": "For urgent fixes: run specific test, deploy staging, verify, then prod.",
    },
]


DEPLOY_EDGES = [
    ("deploy_root", "check_tests", 0.5),
    ("deploy_root", "skip_tests", 0.5),
    ("deploy_root", "rollback", 0.3),
    ("check_tests", "check_ci", 0.6),
    ("check_ci", "deploy_staging", 0.6),
    ("deploy_staging", "deploy_prod", 0.7),
    ("skip_tests", "deploy_prod", 0.7),
    ("deploy_prod", "rollback", 0.3),
    ("deploy_root", "hotfix", 0.3),
]


def build_graph() -> Graph:
    graph = Graph()
    for record in DEPLOY_NODES:
        graph.add_node(Node(record["id"], record["content"], "", record["type"]))

    for source, target, weight in DEPLOY_EDGES:
        graph.add_edge(Edge(source=source, target=target, weight=weight))

    return graph


def write_outputs(graph_path: Path) -> None:
    graph = build_graph()
    graph.save(str(graph_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deploy pipeline experiment graph and save as JSON."
    )
    parser.add_argument(
        "--graph",
        default="experiments/deploy_pipeline_graph.json",
        help="Path to write deploy pipeline graph JSON",
    )
    args = parser.parse_args()

    write_outputs(Path(args.graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
