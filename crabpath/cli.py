"""
CrabPath CLI — Bootstrap, query, and manage memory graphs.

Usage:
    crabpath init [--db PATH]              Initialize a new graph
    crabpath import-openclaw PATH          Bootstrap from OpenClaw workspace
    crabpath activate QUERY                Activate the graph for a query
    crabpath stats                         Show graph statistics
    crabpath learn --outcome success|fail  Record outcome for last activation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph import MemoryGraph


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="crabpath",
        description="🦀 CrabPath — Activation-driven memory graphs for AI agents",
    )
    parser.add_argument("--db", default="crabpath.db", help="Path to graph database")
    
    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Initialize a new graph database")

    # stats
    sub.add_parser("stats", help="Show graph statistics")

    # activate
    p_act = sub.add_parser("activate", help="Activate the graph for a query")
    p_act.add_argument("query", help="Query string")
    p_act.add_argument("--top-k", type=int, default=10, help="Number of nodes to return")
    p_act.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # import-openclaw
    p_import = sub.add_parser("import-openclaw", help="Bootstrap from OpenClaw workspace")
    p_import.add_argument("workspace", help="Path to OpenClaw workspace directory")
    p_import.add_argument("--learning-db", help="Path to learning harness SQLite DB")

    # learn
    p_learn = sub.add_parser("learn", help="Record outcome for learning")
    p_learn.add_argument("--outcome", choices=["success", "failure"], required=True)
    p_learn.add_argument("--trace", help="Path to activation trace JSON")

    args = parser.parse_args(argv)

    if args.command == "init":
        cmd_init(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "activate":
        cmd_activate(args)
    elif args.command == "import-openclaw":
        cmd_import_openclaw(args)
    elif args.command == "learn":
        cmd_learn(args)
    else:
        parser.print_help()


def cmd_init(args):
    db_path = Path(args.db)
    if db_path.exists():
        print(f"Database already exists: {db_path}")
        sys.exit(1)
    g = MemoryGraph(db_path=db_path)
    print(f"🦀 Initialized CrabPath graph at {db_path}")


def cmd_stats(args):
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No database found at {db_path}. Run: crabpath init")
        sys.exit(1)
    g = MemoryGraph(db_path=db_path)
    # TODO: load from DB
    stats = g.stats()
    print(f"🦀 CrabPath Graph Stats")
    print(f"   Nodes: {stats['nodes']}")
    print(f"   Edges: {stats['edges']}")
    print(f"   Quarantined: {stats['quarantined']}")
    if stats['node_types']:
        print(f"   Node types: {json.dumps(stats['node_types'], indent=6)}")
    if stats['edge_types']:
        print(f"   Edge types: {json.dumps(stats['edge_types'], indent=6)}")


def cmd_activate(args):
    from .activation import ActivationConfig, ActivationEngine

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No database found at {db_path}. Run: crabpath init")
        sys.exit(1)

    g = MemoryGraph(db_path=db_path)
    # TODO: load graph from DB
    config = ActivationConfig(top_k=args.top_k)
    engine = ActivationEngine(g, config=config)
    result = engine.activate(args.query)

    if args.as_json:
        output = {
            "query": result.query,
            "tier": result.tier,
            "hops": result.hops,
            "nodes": [
                {"id": n.id, "type": n.node_type.value, "score": s, "summary": n.summary}
                for n, s in result.activated_nodes
            ],
            "inhibited": result.inhibited_nodes,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"🦀 CrabPath activation for: {result.query}")
        print(f"   Tier: {result.tier} | Hops: {result.hops}")
        print(f"   Activated {len(result.activated_nodes)} nodes:")
        for node, score in result.activated_nodes:
            print(f"     [{score:.3f}] {node.node_type.value}: {node.summary or node.content[:60]}")
        if result.inhibited_nodes:
            print(f"   Inhibited: {', '.join(result.inhibited_nodes)}")


def cmd_import_openclaw(args):
    from .integrations.openclaw import import_workspace

    db_path = Path(args.db)
    workspace = Path(args.workspace)

    if not workspace.exists():
        print(f"Workspace not found: {workspace}")
        sys.exit(1)

    g = MemoryGraph(db_path=db_path)
    stats = import_workspace(g, workspace, learning_db=args.learning_db)
    
    print(f"🦀 Imported OpenClaw workspace into CrabPath")
    print(f"   Nodes created: {stats['nodes_created']}")
    print(f"   Edges created: {stats['edges_created']}")
    print(f"   Sources: {', '.join(stats['sources'])}")


def cmd_learn(args):
    print(f"🦀 Recorded outcome: {args.outcome}")
    # TODO: load last activation trace and update weights


if __name__ == "__main__":
    main()
