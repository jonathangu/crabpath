"""Minimal OpenAI function-calling loop with CrabPath.

This keeps the model loop tiny: ask, call `query`, then update via `learn`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from crabpath import Edge, Graph, MemoryController, Node

if TYPE_CHECKING:
    from openai import OpenAI  # pragma: no cover


WORKSPACE = Path(__file__).with_name("openai_example_graph.json")
TOOLS_SPEC = Path(__file__).resolve().parents[1] / "tools" / "openai-tools.json"


def _build_graph() -> Graph:
    if WORKSPACE.exists():
        return Graph.load(str(WORKSPACE))

    graph = Graph()
    graph.add_node(Node(id="check", content="Check deployment logs"))
    graph.add_node(Node(id="mitigate", content="Mitigate with rollback when needed"))
    graph.add_node(Node(id="verify", content="Verify service recovery"))
    graph.add_edge(Edge(source="check", target="mitigate", weight=0.8))
    graph.add_edge(Edge(source="mitigate", target="verify", weight=0.6))
    graph.save(str(WORKSPACE))
    return graph


def _load_tools() -> list[dict[str, Any]]:
    with TOOLS_SPEC.open("r", encoding="utf-8") as stream:
        return json.load(stream).get("tools", [])


def _run_tool(
    call_name: str,
    args: dict[str, Any],
    controller: MemoryController,
    state: dict[str, Any],
) -> dict[str, Any]:
    if call_name == "query":
        result = controller.query(str(args.get("query", "")))
        state["last_result"] = result
        return {
            "query": args.get("query", ""),
            "fired_ids": result.selected_nodes,
            "context": result.context,
        }

    if call_name == "learn":
        last = state.get("last_result")
        if last is None:
            return {"error": "learn called before any query result"}

        outcome = float(args.get("outcome", 0.0))
        update = controller.learn(last, outcome)
        controller.graph.save(str(WORKSPACE))
        return {"ok": True, "learning": update}

    if call_name == "stats":
        return {"nodes": controller.graph.node_count, "edges": controller.graph.edge_count}

    return {"error": f"Unsupported tool: {call_name}"}


def _build_openai_client() -> "OpenAI":
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Install openai (pip install openai) to run this example.") from exc

    return OpenAI()


def main() -> None:
    # Tool loop is intentionally simple and deterministic for demo readability.
    turns = [
        ("What should I check next for a failing deploy?", 1.0),
        ("Did mitigation succeed?", 1.0),
    ]

    graph = _build_graph()
    controller = MemoryController(graph)
    state = {"last_result": None}
    client = _build_openai_client()
    tools = _load_tools()

    messages = [
        {
            "role": "system",
            "content": (
                "Use the query tool for retrieval. Then learn from outcome in supervised "
                "signals when available."
            ),
        }
    ]

    for user_text, outcome in turns:
        messages.append({"role": "user", "content": user_text})
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
        }
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
        messages.append(assistant_msg)

        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            tool_result = _run_tool(tool_call.function.name, args, controller, state)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(tool_result),
                }
            )

            # In this tiny loop we demonstrate learn immediately after each query.
            if tool_call.function.name == "query":
                last = state.get("last_result")
                if last is None:
                    raise RuntimeError("Expected a query result before learn.")
                learn_outcome = {
                    "fired_ids": ",".join(last.selected_nodes),
                    "outcome": outcome,
                }
                _run_tool("learn", learn_outcome, controller, state)

        followup = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        print(followup.choices[0].message.content or "")


if __name__ == "__main__":
    main()
