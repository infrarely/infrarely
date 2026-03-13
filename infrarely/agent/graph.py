"""
agent/graph.py — LangGraph graph construction for the Student Agent.

Graph topology:
                        ┌─────────────────────────────────┐
                        │                                 │
  START → plan → reason → (router) → tools → observe ──→ reason
                       ↓
                    verify → respond → END

The router decides:
  • "tools"  → last AI message has pending tool calls → execute them
  • "verify" → last AI message is a final answer     → check it
"""

import functools
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from infrarely.agent.state import AgentState
from infrarely.agent.nodes import (
    MAX_TURNS,
    _has_tool_calls,
    observe_node,
    plan_node,
    reason_node,
    respond_node,
    verify_node,
)
from config import build_llm
from infrarely.memory.memory_manager import MemoryManager
from infrarely.observability.logger import get_logger
from infrarely.tools.registry import ToolRegistry

log = get_logger("agent.graph")


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────────────────────────────


def _router(state: AgentState) -> Literal["tools", "verify", "__end__"]:
    """
    Conditional edge: examine last AI message to decide the next node.

    Routes:
      "tools"  → tool calls are pending
      "verify" → final answer produced; needs verification
      END      → hard turn-count cap reached (safety valve)
    """
    messages = state["messages"]
    turn = state.get("turn_count", 0)

    # Safety valve
    if turn >= MAX_TURNS:
        log.info("agent.graph", f"Router → END (MAX_TURNS {MAX_TURNS} reached)")
        return END

    last_ai = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    if last_ai is None:
        return END

    if _has_tool_calls(last_ai):
        log.info(
            "agent.graph",
            f"Router → tools (pending: "
            f"{[tc['name'] for tc in last_ai.tool_calls]})",
        )
        return "tools"

    log.info("agent.graph", "Router → verify")
    return "verify"


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH FACTORY
# ─────────────────────────────────────────────────────────────────────────────


def build_graph(
    memory_manager: MemoryManager, student_context: dict, checkpointer=None
):
    """
    Assemble, compile, and return the Student Agent LangGraph.

    Args:
        memory_manager:  Initialised MemoryManager for the session thread.
        student_context: Dict with student profile and permissions.
        checkpointer:    Optional LangGraph checkpointer (SqliteSaver).

    Returns:
        Compiled StateGraph ready for `graph.invoke(...)`.
    """
    log.info("agent.graph", "Building Student Agent graph...")

    # ── LLMs ─────────────────────────────────────────────────────────────────
    tool_registry = build_tool_registry()
    lc_tools = get_langchain_tools(tool_registry, student_context)

    base_llm = build_llm()
    llm_with_tools = build_llm(bind_tools=lc_tools)

    # Wire memory_manager's LLM for summarisation
    memory_manager.set_llm(base_llm)

    # ── Partial-apply dependencies into node functions ────────────────────────
    plan_fn = functools.partial(plan_node, llm=base_llm, memory_manager=memory_manager)
    reason_fn = functools.partial(reason_node, llm_with_tools=llm_with_tools)
    verify_fn = functools.partial(verify_node, llm=base_llm)
    respond_fn = functools.partial(respond_node, memory_manager=memory_manager)

    # ── ToolNode: handles all LangChain tool calls ────────────────────────────
    # handle_tool_errors=True → converts exceptions into ToolMessage error strings
    tool_node = ToolNode(tools=lc_tools, handle_tool_errors=True)

    # ── Build StateGraph ──────────────────────────────────────────────────────
    builder = StateGraph(AgentState)

    builder.add_node("plan", plan_fn)
    builder.add_node("reason", reason_fn)
    builder.add_node("tools", tool_node)
    builder.add_node("observe", observe_node)
    builder.add_node("verify", verify_fn)
    builder.add_node("respond", respond_fn)

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "reason")
    builder.add_conditional_edges(
        "reason",
        _router,
        {"tools": "tools", "verify": "verify", END: END},
    )
    builder.add_edge("tools", "observe")
    builder.add_edge("observe", "reason")  # observation feeds back into reasoning
    builder.add_edge("verify", "respond")
    builder.add_edge("respond", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    graph = builder.compile(**compile_kwargs)
    log.info("agent.graph", "✅ Graph compiled successfully.")
    return graph
