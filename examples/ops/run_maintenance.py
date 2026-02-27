#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crabpath import load_state
from crabpath.maintain import run_maintenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run slow-loop maintenance")
    parser.add_argument("--state", required=True, help="Path to state.json")
    parser.add_argument("--tasks", default="health,decay,merge,prune")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-merges", type=int, default=5)
    parser.add_argument("--prune-below", type=float, default=0.01)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    state_path = Path(args.state)
    if not state_path.exists():
        raise SystemExit(f"state not found: {state_path}")

    _ = load_state(str(state_path))
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    report = run_maintenance(
        state_path=str(state_path),
        tasks=tasks,
        dry_run=args.dry_run,
        max_merges=args.max_merges,
        prune_below=args.prune_below,
        journal_path=str(state_path.parent / "journal.jsonl"),
    )

    if args.json:
        print(json.dumps(report.__dict__))
        return

    print(f"Maintenance report for {state_path}")
    print(f"  tasks: {', '.join(report.tasks_run) if report.tasks_run else '(none)'}")
    print(f"  nodes: {report.health_before['nodes']} -> {report.health_after['nodes']}")
    print(f"  edges: {report.edges_before} -> {report.edges_after}")
    print(f"  merges: {report.merges_applied}/{report.merges_proposed}")
    print(f"  pruned: edges={report.pruned_edges} nodes={report.pruned_nodes}")
    print(f"  dry_run: {args.dry_run}")
    if report.notes:
        print(f"  notes: {', '.join(report.notes)}")


if __name__ == "__main__":
    main()
