#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crabpath.graph import Edge, Graph, Node


CONTEXT_BLOAT_NODES: list[dict[str, str]] = [
    {
        "id": "fact_merge_conflict_triage",
        "type": "fact",
        "content": "Fact: Merge conflicts are usually solved by inspecting git diff output and resolving both file versions before committing.",
        "summary": "Merge conflict triage",
    },
    {
        "id": "fact_git_checkout",
        "type": "fact",
        "content": "Fact: Create a dedicated branch before making repair commits so the main branch history stays clean.",
        "summary": "Use branches",
    },
    {
        "id": "fact_ci_signal",
        "type": "fact",
        "content": "Fact: CI failures in tests should be interpreted by first checking the most recent failing test name.",
        "summary": "CI signal",
    },
    {
        "id": "fact_log_buckets",
        "type": "fact",
        "content": "Fact: Application logs should be segmented by request_id, user_id, and trace_id for fast debugging.",
        "summary": "Log segmentation",
    },
    {
        "id": "fact_db_migration_window",
        "type": "fact",
        "content": "Fact: Run schema migrations only inside a short planned deployment window to avoid data race windows.",
        "summary": "Migration window",
    },
    {
        "id": "fact_cache_layering",
        "type": "fact",
        "content": "Fact: Layered cache keys must include request context and version to prevent stale read contamination.",
        "summary": "Cache layering",
    },
    {
        "id": "fact_error_budget",
        "type": "fact",
        "content": "Fact: For reliability engineering, treat error budget burn above 10 percent as an incident-level signal.",
        "summary": "Error budget threshold",
    },
    {
        "id": "fact_api_idempotency",
        "type": "fact",
        "content": "Fact: POST retry handlers should only be idempotent-safe when an idempotency key is supplied.",
        "summary": "Idempotency",
    },
    {
        "id": "fact_container_restart",
        "type": "fact",
        "content": "Fact: Restart only one container at a time when draining traffic to avoid cascade failures.",
        "summary": "Safe restart sequence",
    },
    {
        "id": "fact_secret_handling",
        "type": "fact",
        "content": "Fact: Never commit secrets; load secrets from environment variables or a secret manager at runtime.",
        "summary": "No secret commits",
    },
    {
        "id": "fact_dependency_drift",
        "type": "fact",
        "content": "Fact: Dependency drift should be documented in a changelog before merging release branches.",
        "summary": "Dependency drift",
    },
    {
        "id": "fact_memory_retention",
        "type": "fact",
        "content": "Fact: Tool context windows should keep only relevant behavioral rules for current task.",
        "summary": "Context retention",
    },
    {
        "id": "fact_ratelimit",
        "type": "fact",
        "content": "Fact: Enforce per-endpoint rate limit buckets and return 429 when throttling.",
        "summary": "Rate limiting",
    },
    {
        "id": "fact_roll_forward",
        "type": "fact",
        "content": "Fact: Roll-forward migrations are preferred over roll-back when data transforms are not reversible.",
        "summary": "Migration rollback policy",
    },
    {
        "id": "fact_readme_update",
        "type": "fact",
        "content": "Fact: Update README links and examples when interfaces or command flags change.",
        "summary": "Docs update",
    },
    {
        "id": "fact_test_isolation",
        "type": "fact",
        "content": "Fact: Tests in shared runners should be isolated with temporary schema or containerized state.",
        "summary": "Test isolation",
    },
    {
        "id": "fact_observability",
        "type": "fact",
        "content": "Fact: Correlation IDs should cross logs, traces, and metrics when debugging incidents.",
        "summary": "Observability",
    },
    {
        "id": "procedure_reproduce_bug",
        "type": "procedure",
        "content": "Procedure: reproduce bug with the smallest dataset and exact repro command before attempting patching.",
        "summary": "Repro bug",
    },
    {
        "id": "procedure_fix_merge",
        "type": "procedure",
        "content": "Procedure: fix merge conflict by keeping both branch changes, then manually reconciling line-level semantics.",
        "summary": "Resolve merge",
    },
    {
        "id": "procedure_review_diff",
        "type": "procedure",
        "content": "Procedure: review diff hunks in small chunks, run targeted tests after each grouped change.",
        "summary": "Review diff",
    },
    {
        "id": "procedure_collect_logs",
        "type": "procedure",
        "content": "Procedure: collect logs from last deployment window and check for new warning signatures first.",
        "summary": "Collect logs",
    },
    {
        "id": "procedure_rotate_secrets",
        "type": "procedure",
        "content": "Procedure: rotate credentials, invalidate old tokens, and re-run authentication smoke tests.",
        "summary": "Rotate secrets",
    },
    {
        "id": "procedure_run_smoke",
        "type": "procedure",
        "content": "Procedure: run smoke tests before broad test suites to validate health endpoints and startup path.",
        "summary": "Smoke tests",
    },
    {
        "id": "procedure_scale_down",
        "type": "procedure",
        "content": "Procedure: scale down non-critical workers, drain queues, then patch the faulty worker type.",
        "summary": "Scale down",
    },
    {
        "id": "procedure_rollback_plan",
        "type": "procedure",
        "content": "Procedure: rollback only after confirming database, cache, and queue states are captured in a snapshot.",
        "summary": "Rollback plan",
    },
    {
        "id": "procedure_fix_security_bug",
        "type": "procedure",
        "content": "Procedure: reproduce exploit path, add validation check, and add regression test before deploy.",
        "summary": "Fix security bug",
    },
    {
        "id": "procedure_update_instrumentation",
        "type": "procedure",
        "content": "Procedure: add request-scoped telemetry around the risky codepath, then observe before tuning.",
        "summary": "Update instrumentation",
    },
    {
        "id": "procedure_validate_backup",
        "type": "procedure",
        "content": "Procedure: validate backups by restoring a random shard into staging and checking consistency checks.",
        "summary": "Validate backup",
    },
    {
        "id": "procedure_prepare_release",
        "type": "procedure",
        "content": "Procedure: prepare release notes, changelog, and deployment checklist before pushing the tag.",
        "summary": "Prepare release",
    },
    {
        "id": "procedure_threat_check",
        "type": "procedure",
        "content": "Procedure: run static scan, dependency audit, and secret pattern scan before merging security changes.",
        "summary": "Threat check",
    },
    {
        "id": "procedure_config_audit",
        "type": "procedure",
        "content": "Procedure: audit configuration drift and pin required environment overrides before restart.",
        "summary": "Config audit",
    },
    {
        "id": "procedure_trace_follow",
        "type": "procedure",
        "content": "Procedure: follow one trace from ingress to DB write path and verify each hop for latency outliers.",
        "summary": "Trace follow",
    },
    {
        "id": "guard_no_force_push",
        "type": "guardrail",
        "content": "Guardrail: never force-push to shared or protected branches.",
        "summary": "No force push",
    },
    {
        "id": "guard_no_plaintext_secrets",
        "type": "guardrail",
        "content": "Guardrail: do not include plaintext API keys or credentials in logs or prompts.",
        "summary": "No plaintext secrets",
    },
    {
        "id": "guard_no_unapproved_prod",
        "type": "guardrail",
        "content": "Guardrail: do not run destructive commands on production without explicit approval tag.",
        "summary": "No destructive prod",
    },
    {
        "id": "guard_no_direct_db_reset",
        "type": "guardrail",
        "content": "Guardrail: do not reset shared databases during business hours.",
        "summary": "No direct db reset",
    },
    {
        "id": "guard_validate_schema",
        "type": "guardrail",
        "content": "Guardrail: validate schema migrations against staging first before production rollout.",
        "summary": "Validate schema first",
    },
    {
        "id": "guard_small_batch_rollout",
        "type": "guardrail",
        "content": "Guardrail: rollout risky changes in small batches, monitor for two error windows before expansion.",
        "summary": "Small batch rollout",
    },
    {
        "id": "guard_idempotent_apis",
        "type": "guardrail",
        "content": "Guardrail: endpoints mutating state must not be treated as safe for automatic retry loops.",
        "summary": "Retry safety",
    },
    {
        "id": "guard_tainting_flags",
        "type": "guardrail",
        "content": "Guardrail: never ignore validation errors from taint checking output.",
        "summary": "Taint check",
    },
    {
        "id": "guard_lock_budget",
        "type": "guardrail",
        "content": "Guardrail: enforce budget and lock checks before allowing high-cost training jobs.",
        "summary": "Budget guardrail",
    },
    {
        "id": "guard_test_before_merge",
        "type": "guardrail",
        "content": "Guardrail: do not merge until smoke tests and static checks have passed.",
        "summary": "Test before merge",
    },
    {
        "id": "guard_reviewers",
        "type": "guardrail",
        "content": "Guardrail: require at least one reviewer for changes touching authentication code.",
        "summary": "Review policy",
    },
    {
        "id": "guard_rollback_drift",
        "type": "guardrail",
        "content": "Guardrail: verify dependency versions before rollback to avoid package drift issues.",
        "summary": "Rollback version check",
    },
    {
        "id": "tool_read_diff",
        "type": "tool_call",
        "content": "Tool call: `git diff --merge` to inspect conflict markers and local vs remote changes.",
        "summary": "git diff --merge",
    },
    {
        "id": "tool_log_query",
        "type": "tool_call",
        "content": "Tool call: `kubectl logs --since=30m --selector app=api` for tailing service logs.",
        "summary": "kubectl logs",
    },
    {
        "id": "tool_rollback_cmd",
        "type": "tool_call",
        "content": "Tool call: `helm rollback` to revert a bad release once checks indicate failure conditions.",
        "summary": "helm rollback",
    },
    {
        "id": "tool_config_lint",
        "type": "tool_call",
        "content": "Tool call: `python -m yaml` and schema validation for configuration file correctness.",
        "summary": "config lint",
    },
    {
        "id": "tool_db_migrate",
        "type": "tool_call",
        "content": "Tool call: `alembic upgrade head` only inside a planned migration window.",
        "summary": "alembic upgrade",
    },
    {
        "id": "tool_secret_scan",
        "type": "tool_call",
        "content": "Tool call: `trivy` secret scan to catch credentials before merge.",
        "summary": "trivy secret scan",
    },
    {
        "id": "tool_unit_tests",
        "type": "tool_call",
        "content": "Tool call: `pytest tests/unit -q` after applying a targeted patch.",
        "summary": "pytest unit",
    },
    {
        "id": "tool_infra_plan",
        "type": "tool_call",
        "content": "Tool call: `terraform plan` before modifying infrastructure or autoscaling rules.",
        "summary": "terraform plan",
    },
    {
        "id": "tool_trace_explore",
        "type": "tool_call",
        "content": "Tool call: run trace search in distributed tracing UI for a failing request id.",
        "summary": "trace search",
    },
]

SCENARIOS: list[dict[str, Any]] = [
    {
        "query": "How do I resolve a git merge conflict fast?",
        "expected_answer_fragments": ["Resolve merge", "git diff --merge", "conflict markers"],
        "path": ["procedure_fix_merge", "procedure_review_diff", "tool_read_diff", "guard_no_force_push"],
    },
    {
        "query": "What is the safe way to start debugging a failing CI run?",
        "expected_answer_fragments": ["failed test name", "collect logs", "targeted tests"],
        "path": ["fact_ci_signal", "procedure_collect_logs", "procedure_review_diff", "procedure_run_smoke"],
    },
    {
        "query": "Which rule prevents accidentally pushing secrets?",
        "expected_answer_fragments": ["No plaintext secrets", "never commit secrets", "secret manager"],
        "path": ["guard_no_plaintext_secrets", "fact_secret_handling", "procedure_rotate_secrets", "tool_secret_scan"],
    },
    {
        "query": "How should I handle a schema migration in production?",
        "expected_answer_fragments": ["staging", "migration window", "roll-forward"],
        "path": ["fact_db_migration_window", "guard_validate_schema", "procedure_rollback_plan", "tool_db_migrate"],
    },
    {
        "query": "How do I isolate flaky tests and rerun?",
        "expected_answer_fragments": ["smallest dataset", "isolated", "pytest unit"],
        "path": ["procedure_reproduce_bug", "fact_test_isolation", "procedure_review_diff", "tool_unit_tests"],
    },
    {
        "query": "How do I prevent cache contamination in multi-tenant systems?",
        "expected_answer_fragments": ["cache keys", "context and version", "configuration drift"],
        "path": ["fact_cache_layering", "procedure_config_audit", "procedure_update_instrumentation", "tool_config_lint"],
    },
    {
        "query": "What is the rollout safety rule for risky deploys?",
        "expected_answer_fragments": ["small batches", "error windows", "monitor"],
        "path": ["guard_small_batch_rollout", "procedure_scale_down", "fact_error_budget", "procedure_collect_logs"],
    },
    {
        "query": "How do I triage a high traffic incident?",
        "expected_answer_fragments": ["request_id", "trace", "correlation IDs"],
        "path": ["fact_log_buckets", "procedure_collect_logs", "procedure_trace_follow", "fact_observability"],
    },
    {
        "query": "When can I restart containers with active traffic?",
        "expected_answer_fragments": ["one container at a time", "drain traffic", "tool log query"],
        "path": ["fact_container_restart", "procedure_scale_down", "guard_no_direct_db_reset", "tool_log_query"],
    },
    {
        "query": "What should I do before publishing a release?",
        "expected_answer_fragments": ["release notes", "changelog", "test before merge"],
        "path": ["procedure_prepare_release", "procedure_review_diff", "guard_test_before_merge", "tool_infra_plan"],
    },
    {
        "query": "How can I fix an API idempotency bug?",
        "expected_answer_fragments": ["idempotency key", "safe retry", "guardrail"],
        "path": ["fact_api_idempotency", "guard_idempotent_apis", "procedure_fix_security_bug", "tool_unit_tests"],
    },
    {
        "query": "How should rate limits be implemented at scale?",
        "expected_answer_fragments": ["per-endpoint", "429", "ratelimit"],
        "path": ["fact_ratelimit", "procedure_update_instrumentation", "procedure_run_smoke", "fact_observability"],
    },
    {
        "query": "What is the right flow for security-related fixes?",
        "expected_answer_fragments": ["reproduce exploit", "validation check", "secret scan"],
        "path": ["procedure_fix_security_bug", "procedure_threat_check", "tool_secret_scan", "procedure_run_smoke"],
    },
    {
        "query": "How do I prepare for a planned rollback?",
        "expected_answer_fragments": ["snapshot", "rollback", "helm rollback"],
        "path": ["procedure_rollback_plan", "guard_rollback_drift", "tool_rollback_cmd", "fact_roll_forward"],
    },
    {
        "query": "How do I validate deployment readiness?",
        "expected_answer_fragments": ["smoke tests", "ready check", "health endpoints"],
        "path": ["procedure_run_smoke", "fact_test_isolation", "tool_unit_tests", "fact_ci_signal"],
    },
    {
        "query": "How to keep authentication changes safe?",
        "expected_answer_fragments": ["reviewer", "risk", "authentication code"],
        "path": ["guard_reviewers", "procedure_threat_check", "fact_secret_handling", "tool_secret_scan"],
    },
    {
        "query": "What to do when dependencies change before merge?",
        "expected_answer_fragments": ["dependency drift", "changelog", "readme update"],
        "path": ["fact_dependency_drift", "procedure_prepare_release", "fact_readme_update", "tool_unit_tests"],
    },
    {
        "query": "How do I check production readiness for database changes?",
        "expected_answer_fragments": ["backup validation", "restore", "staging"],
        "path": ["procedure_validate_backup", "guard_validate_schema", "tool_db_migrate", "fact_db_migration_window"],
    },
    {
        "query": "Which guardrail blocks dangerous workspace commands?",
        "expected_answer_fragments": ["destructive commands", "production", "approval tag"],
        "path": ["guard_no_unapproved_prod", "procedure_collect_logs", "procedure_scale_down", "tool_log_query"],
    },
    {
        "query": "Where should I start when telemetry looks broken?",
        "expected_answer_fragments": ["trace", "request to db write", "latency"],
        "path": ["procedure_trace_follow", "tool_trace_explore", "fact_observability", "procedure_collect_logs"],
    },
]


def _add_nodes(graph: Graph) -> None:
    graph.add_node(Node("cb_root", "Context bloat root hub for simulation", "", "fact"))
    for record in CONTEXT_BLOAT_NODES:
        graph.add_node(
            Node(record["id"], record["content"], record["summary"], record["type"])
        )


def _add_edges(graph: Graph) -> None:
    node_ids = [record["id"] for record in CONTEXT_BLOAT_NODES]

    for index, scenario in enumerate(SCENARIOS, start=1):
        entry_id = f"cb_query_{index:02d}_entry"
        query_text = scenario["query"]
        relevant = scenario["path"]

        graph.add_node(
            Node(entry_id, f"Query entry for: {query_text}", "Query seed", "fact")
        )
        graph.add_edge(Edge(source="cb_root", target=entry_id, weight=1.0))

        distractor_pool = [nid for nid in node_ids if nid not in relevant][:4]

        for offset, target_id in enumerate(relevant, start=1):
            graph.add_edge(Edge(source=entry_id, target=target_id, weight=0.95 - 0.03 * (offset - 1)))

            if offset > 1:
                previous = relevant[offset - 2]
                graph.add_edge(
                    Edge(
                        source=previous,
                        target=target_id,
                        weight=0.88 - 0.05 * (offset - 2),
                    )
                )

        # Add low-weight distractors so myopic and top-k baselines can pick some noise.
        for offset, target_id in enumerate(distractor_pool[:2], start=1):
            graph.add_edge(Edge(source=entry_id, target=target_id, weight=0.23 - 0.02 * offset))
            if offset == 1 and relevant:
                graph.add_edge(Edge(source=relevant[0], target=target_id, weight=0.21))



def build_graph() -> Graph:
    graph = Graph()
    _add_nodes(graph)
    _add_edges(graph)
    return graph


def build_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "query": record["query"],
            "feedback": {"reward": 1.0},
            "expected_answer_fragments": record["expected_answer_fragments"],
        }
        for record in SCENARIOS
    ]


def write_outputs(graph_path: Path, scenario_path: Path) -> None:
    graph = build_graph()
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.parent.mkdir(parents=True, exist_ok=True)

    graph.save(str(graph_path))

    payload = build_scenarios()
    with scenario_path.open("w", encoding="utf-8") as f:
        for step in payload:
            f.write(json.dumps(step))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build context-bloat experiment graph and scenarios")
    parser.add_argument(
        "--graph",
        default="experiments/context_bloat_graph.json",
        help="Path to write experiment graph JSON.",
    )
    parser.add_argument(
        "--scenario",
        default="scenarios/context_bloat.jsonl",
        help="Path to write context-bloat scenario JSONL.",
    )
    args = parser.parse_args()

    write_outputs(Path(args.graph), Path(args.scenario))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
