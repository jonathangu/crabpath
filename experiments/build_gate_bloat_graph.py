#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crabpath.graph import Edge, Graph, Node


def _gate_records() -> list[dict[str, str]]:
    categories = {
        "coding": "Coding",
        "deployment": "Deployment",
        "memory": "Memory",
        "security": "Security",
        "review": "Review",
    }

    templates = {
        "coding": [
            "Prefer explicit types on public interfaces",
            "Run linters before committing",
            "Avoid broad exception catches",
            "Keep functions focused and small",
            "Write regression tests for bugfixes",
            "Document API behavior changes",
            "Avoid changing dependencies without review",
            "Use feature flags for risky changes",
            "Prefer idempotent handlers",
            "Keep variable names descriptive",
            "Refactor before adding special cases",
            "Fail fast on invalid input",
            "Log context for user-visible actions",
        ],
        "deployment": [
            "Use staged rollout for production",
            "Validate schema before migration",
            "Use blue-green for critical services",
            "Verify health checks before expansion",
            "Set conservative timeout budgets",
            "Throttle background jobs under load",
            "Pause rollout when error budget burns quickly",
            "Capture snapshots before risky changes",
            "Test rollback in dry-run mode",
            "Rotate credentials during off-hours",
            "Coordinate deployment windows with maintainers",
            "Drain workers before patching",
            "Pin infrastructure versions in config",
        ],
        "memory": [
            "Load only relevant project-specific rules",
            "Bound context by task intent",
            "Prefer summaries over raw logs",
            "Expire stale task-state pointers",
            "Favor high-precision retrieval",
            "Keep memory budget under control",
            "Avoid loading duplicate instructions",
            "Do not load obsolete migration guidance",
            "Record negative feedback explicitly",
            "Log user intent before selection",
            "Use guardrail-only path when confidence is low",
            "Keep trace data for recent turns",
            "Prune dead-end context paths",
        ],
        "security": [
            "Block secrets in plain text",
            "Escalate when privilege changes are requested",
            "Require explicit approval for prod write",
            "Disable unsafe shell commands by default",
            "Redact PII before storage",
            "Check auth scopes on every tool call",
            "Reject prompt-injection command strings",
            "Audit outputs for credential leaks",
            "Rate limit failed auth attempts",
            "Use signed artifact checks",
            "Never execute unknown binaries",
            "Log every access to secret store",
            "Validate webhook signatures",
        ],
        "review": [
            "Require owner sign-off for schema edits",
            "Request reviewer on production-impacting files",
            "Mark uncertain suggestions as tentative",
            "Track dependency provenance",
            "Prefer minimal diffs",
            "Require issue references in major changes",
            "Resolve comments before merge",
            "Avoid one-off scripts without docs",
            "Enforce commit message quality",
            "Update runbook for SOP changes",
            "Summarize risk in PR description",
            "Close stale findings after 30 days",
            "Keep generated artifacts out of source",
        ],
    }

    records: list[dict[str, str]] = []
    category_keys = list(categories)
    for index in range(130):
        key = category_keys[index % len(category_keys)]
        phrase = templates[key][index % len(templates[key])]
        records.append(
            {
                "id": f"gb_gate_{index + 1:03d}",
                "type": "guardrail",
                "content": f"Guardrail ({categories[key]}): {phrase}",
                "summary": f"{categories[key]} gate {index + 1}",
                "category": key,
            }
        )
    return records


QUERY_RECORDS: list[dict[str, Any]] = [
    {"id": "gb_query_01", "query": "Preparing a hotfix for a critical API bug", "category": "coding", "expected": ["Regression tests", "idempotent", "Fail fast"]},
    {"id": "gb_query_02", "query": "Deploying a database migration under production pressure", "category": "deployment", "expected": ["staged rollout", "schema", "snapshot"]},
    {"id": "gb_query_03", "query": "Need to cleanly summarize context for the current incident task", "category": "memory", "expected": ["task intent", "context", "budget"]},
    {"id": "gb_query_04", "query": "Running auth changes and preparing for production write", "category": "security", "expected": ["explicit approval", "prod write", "scopes"]},
    {"id": "gb_query_05", "query": "Submitting PR for a risky infrastructure change", "category": "review", "expected": ["issue references", "risk", "review"]},
    {"id": "gb_query_06", "query": "Rolling out canary traffic to a shared service", "category": "deployment", "expected": ["health checks", "expand", "blue-green"]},
    {"id": "gb_query_07", "query": "Investigating a bug in user-facing serialization", "category": "coding", "expected": ["invalid input", "regression tests", "explicit types"]},
    {"id": "gb_query_08", "query": "Building a concise memory plan for this coding task", "category": "memory", "expected": ["relevant", "summaries", "duplicate"]},
    {"id": "gb_query_09", "query": "Responding to elevated token usage and possible abuse", "category": "security", "expected": ["rate limit", "credential leaks", "audit"]},
    {"id": "gb_query_10", "query": "Preparing branch-level review for production impact", "category": "review", "expected": ["owner sign-off", "reviewer", "minimal diffs"]},
    {"id": "gb_query_11", "query": "Scheduling maintenance and avoiding risky commands", "category": "deployment", "expected": ["off-hours", "safely", "rollout"]},
    {"id": "gb_query_12", "query": "Fixing flaky behavior in a new feature", "category": "coding", "expected": ["regression tests", "focused", "small diffs"]},
    {"id": "gb_query_13", "query": "Handling user request context and not overloading memory", "category": "memory", "expected": ["context", "budget", "relevant"]},
    {"id": "gb_query_14", "query": "Preparing access to secret store for a script", "category": "security", "expected": ["plain text", "scopes", "approval"]},
    {"id": "gb_query_15", "query": "Checking for stale instructions before final merge", "category": "review", "expected": ["stale", "close findings", "PR description"]},
    {"id": "gb_query_16", "query": "Release window decision for a major service", "category": "deployment", "expected": ["deployment window", "health checks", "error budget"]},
    {"id": "gb_query_17", "query": "Refactoring for maintainability while staying within policy", "category": "coding", "expected": ["feature flags", "explicit", "tests"]},
    {"id": "gb_query_18", "query": "Summarizing session memory for handoff", "category": "memory", "expected": ["high-precision retrieval", "prune", "summaries"]},
    {"id": "gb_query_19", "query": "Tool policy for external integrations and webhooks", "category": "security", "expected": ["webhook signatures", "validation", "audit"]},
    {"id": "gb_query_20", "query": "Preparing PR docs and changelog before merge", "category": "review", "expected": ["issue references", "risk", "changelog"]},
]


def build_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node("gb_root", "Gate bloat root", "", "fact"))

    gates = _gate_records()
    for item in gates:
        graph.add_node(Node(item["id"], item["content"], item["summary"], item["type"]))

    for query_record in QUERY_RECORDS:
        query_node_id = query_record["id"]
        query_text = query_record["query"]
        category = query_record["category"]

        graph.add_node(Node(query_node_id, query_text, "query entry", "fact"))
        graph.add_edge(Edge(source="gb_root", target=query_node_id, weight=1.0))

        relevant = [gate for gate in gates if gate["category"] == category][:6]
        for index, gate in enumerate(relevant[:3], start=1):
            graph.add_edge(Edge(source=query_node_id, target=gate["id"], weight=0.95 - 0.07 * (index - 1)))

        noisy_gates = [gate for gate in gates if gate["category"] != category][:2]
        for index, gate in enumerate(noisy_gates[:2], start=1):
            graph.add_edge(Edge(source=query_node_id, target=gate["id"], weight=0.20 + 0.01 * index))

    return graph


def build_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "query": record["query"],
            "feedback": {"reward": 1.0},
            "expected_answer_fragments": record["expected"],
        }
        for record in QUERY_RECORDS
    ]


def write_outputs(graph_path: Path, scenario_path: Path) -> None:
    graph = build_graph()
    graph.save(str(graph_path))

    with scenario_path.open("w", encoding="utf-8") as f:
        for row in build_scenarios():
            f.write(json.dumps(row))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build gate-bloat experiment graph and scenarios")
    parser.add_argument("--graph", default="experiments/gate_bloat_graph.json", help="Path to write gate-bloat graph JSON")
    parser.add_argument("--scenario", default="scenarios/gate_bloat.jsonl", help="Path to write gate-bloat scenario JSONL")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    scenario_path = Path(args.scenario)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    write_outputs(graph_path, scenario_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
