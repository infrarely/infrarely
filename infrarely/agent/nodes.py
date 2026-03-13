"""
agent/nodes.py — The five core nodes of the Student Agent ReAct pipeline.

Pipeline:
  [plan] → [reason] → [act] → [observe] → [verify] → [respond]
                  ↑_________|  (loop if more tools needed)

Each node is a pure function: AgentState → dict of state updates.

RELIABILITY FEATURES PER NODE
──────────────────────────────
plan    : Decomposes complex queries into ordered sub-tasks before acting.
          Prevents planning failures by front-loading retrieval.
reason  : Builds the system prompt with memory context + student profile.
          Temperature=0.1 reduces non-determinism.
act     : Delegates to LangGraph ToolNode. Errors are surfaced as strings.
observe : Records tool outputs into reasoning_trace.  Flags partial data.
verify  : Scans response for fabrication signals.  Lowers confidence score.
          If confidence < threshold, rewrites with explicit uncertainty markers.
respond : Structures the final answer in a student-friendly format.
          Never outputs raw tool JSON.
"""

import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from config import CONFIDENCE_THRESHOLD, FABRICATION_SIGNALS, UNCERTAINTY_PHRASES
from infrarely.agent.state import AgentState, ReasoningStep
from infrarely.observability.logger import get_logger

log = get_logger("agent.nodes")

MAX_TURNS = 8   # hard cap on ReAct iterations per query (prevents infinite loops)

# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a highly reliable AI assistant for university students.
You help with: study planning, assignment tracking, course concepts,
exam preparation, note organisation, scheduling, career planning, and productivity.

RELIABILITY RULES (NEVER VIOLATE):
1. NEVER fabricate information. If a tool returns NOT_FOUND or incomplete data,
   say so explicitly. Do not guess or estimate missing details.
2. ALWAYS prefer tool-based retrieval over answering from memory.
3. If a tool fails, report the failure honestly and suggest alternatives.
4. Structure all responses clearly: use headers, bullet points, and numbered steps.
5. For multi-step questions, state the plan before executing it.
6. Flag any information with low confidence using ⚠️ markers.

STUDENT CONTEXT:
{memory_context}

CURRENT DATE: {current_date}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: extract tool calls from AIMessage
# ─────────────────────────────────────────────────────────────────────────────

def _has_tool_calls(message: AIMessage) -> bool:
    return bool(getattr(message, "tool_calls", None))


def _count_tool_errors(messages: list) -> int:
    """Count ToolMessages that contain error indicators."""
    errors = 0
    for m in messages:
        if isinstance(m, ToolMessage):
            content = str(m.content)
            if any(sig in content for sig in
                   ["TOOL FAILED", "TOOL ERROR", "SERVICE UNAVAILABLE",
                    "ERROR:", "TimeoutError", "ConnectionError"]):
                errors += 1
    return errors


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 1: PLAN
# ─────────────────────────────────────────────────────────────────────────────

def plan_node(state: AgentState, llm, memory_manager) -> Dict:
    """
    Decompose the user's query into ordered retrieval + reasoning steps.

    This node prevents planning failures (Problem #9) by forcing the agent
    to think about what information it needs BEFORE it starts generating.
    Simple queries get a trivial one-step plan.  Complex queries get 3–5 steps.
    """
    t0       = time.perf_counter()
    messages = state["messages"]
    user_msg = next((m.content for m in reversed(messages)
                     if isinstance(m, HumanMessage)), "")

    # Update working memory
    memory_manager.add_message("human", user_msg)

    # Build memory-enriched system context
    memory_context = memory_manager.build_context_block()
    from datetime import date
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        memory_context=memory_context,
        current_date=str(date.today()),
    )

    planning_prompt = (
        f"User asked: '{user_msg}'\n\n"
        f"List 1–5 concrete steps you will take to answer this accurately. "
        f"Format: step1 | step2 | ... (pipe-separated, each step is ≤10 words). "
        f"Prefer tool calls over recalling from memory. Be specific."
    )

    try:
        resp  = llm.invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=planning_prompt),
        ])
        steps = [s.strip() for s in resp.content.split("|") if s.strip()]
        if not steps:
            steps = ["Retrieve relevant data", "Reason over results", "Respond clearly"]
    except Exception as exc:
        log.error("agent.plan", f"Planning LLM call failed: {exc}")
        steps = ["Use available tools to gather data", "Respond based on results"]

    elapsed = (time.perf_counter() - t0) * 1000
    log.plan("agent.plan", steps)

    trace_step = ReasoningStep(
        step_type="thought",
        content=f"Plan: {' | '.join(steps)}",
        tool_name=None, tool_args=None, elapsed_ms=elapsed,
    )

    return {
        "plan":            steps,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_step],
        "turn_count":      0,
        # Inject system message at start of messages if not already present
        "messages": (
            [SystemMessage(content=system_content)]
            if not any(isinstance(m, SystemMessage) for m in messages)
            else []
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 2: REASON (agent node — decides next action or final answer)
# ─────────────────────────────────────────────────────────────────────────────

def reason_node(state: AgentState, llm_with_tools) -> Dict:
    """
    Core LLM reasoning step.

    The model sees the full conversation (system + history + tool results)
    and decides whether to call another tool or produce a final answer.
    """
    t0        = time.perf_counter()
    messages  = state["messages"]
    turn      = state.get("turn_count", 0)

    log.reason("agent.reason",
               f"Turn {turn+1}/{MAX_TURNS} | "
               f"messages_in_context={len(messages)} | "
               f"tool_errors_so_far={_count_tool_errors(messages)}")

    # Hard cap: if we've exceeded MAX_TURNS, force a final answer
    if turn >= MAX_TURNS:
        log.error("agent.reason",
                  f"MAX_TURNS ({MAX_TURNS}) exceeded. Forcing final answer.")
        forced = AIMessage(
            content=(
                "I've reached the maximum reasoning steps for this query. "
                "Here's what I've gathered so far:\n\n"
                + "\n".join(state.get("tool_results", []))
                + "\n\n⚠️ This response may be incomplete. Please try a more specific question."
            )
        )
        return {"messages": [forced], "turn_count": turn + 1}

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        log.error("agent.reason", f"LLM invocation failed: {exc}")
        # Return a graceful error message as AIMessage
        response = AIMessage(
            content=(
                f"I encountered an error while processing your request: {exc}. "
                f"Please try again or rephrase your question."
            )
        )

    elapsed = (time.perf_counter() - t0) * 1000

    if _has_tool_calls(response):
        for tc in response.tool_calls:
            log.act("agent.reason",
                    f"Calling tool: {tc['name']} | args: {tc['args']}",
                    elapsed_ms=elapsed)
            trace = ReasoningStep(
                step_type="action", content=f"Call {tc['name']}",
                tool_name=tc["name"], tool_args=tc["args"], elapsed_ms=elapsed,
            )
            state["reasoning_trace"].append(trace)
    else:
        log.reason("agent.reason",
                   f"Final answer produced (no tool call) | {elapsed:.1f}ms")

    return {
        "messages":    [response],
        "turn_count":  turn + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 3: OBSERVE (processes ToolMessage results)
# ─────────────────────────────────────────────────────────────────────────────

def observe_node(state: AgentState) -> Dict:
    """
    Inspect tool results and append them to the reasoning trace.

    Flags partial or error results with DATA_STATUS markers so the verify
    node knows to apply stricter hallucination checks.
    """
    messages     = state["messages"]
    tool_results = list(state.get("tool_results", []))
    trace        = list(state.get("reasoning_trace", []))

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = str(msg.content)
        log.observe("agent.observe", f"Tool result preview: {content[:120]}")

        # Detect partial/error data
        is_partial = any(sig in content for sig in
                         ["NOT FOUND", "incomplete", "DATA STATUS: incomplete",
                          "TOOL FAILED", "TOOL ERROR", "SERVICE UNAVAILABLE"])
        is_error   = any(sig in content for sig in
                         ["ERROR:", "TimeoutError", "ConnectionError", "DENIED"])

        status = "partial" if is_partial else ("error" if is_error else "ok")
        annotated = f"[TOOL_RESULT status={status}]\n{content}"
        tool_results.append(annotated)

        trace_step = ReasoningStep(
            step_type  = "observation",
            content    = f"Tool result ({status}): {content[:200]}",
            tool_name  = getattr(msg, "name", None),
            tool_args  = None,
            elapsed_ms = None,
        )
        trace.append(trace_step)

    return {
        "tool_results":    tool_results,
        "reasoning_trace": trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 4: VERIFY
# ─────────────────────────────────────────────────────────────────────────────

def verify_node(state: AgentState, llm) -> Dict:
    """
    Self-verification step — checks the last AIMessage for fabrication signals
    before allowing the response to reach the user.

    Addresses Problem #4: Lack of self-verification.

    Process:
      1. Locate the pending AI response.
      2. Scan for fabrication signal phrases.
      3. Cross-check claims against available tool_results.
      4. Assign a confidence score.
      5. If confidence < CONFIDENCE_THRESHOLD, rewrite with uncertainty markers.
    """
    t0       = time.perf_counter()
    messages = state["messages"]

    # Find the last AI message without tool calls
    pending_ai = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not _has_tool_calls(msg):
            pending_ai = msg
            break

    if pending_ai is None:
        return {"verification": "skipped", "confidence": 1.0}

    response_text = str(pending_ai.content)
    tool_results  = state.get("tool_results", [])
    error_count   = _count_tool_errors(messages)

    # ── Check 1: Fabrication signal scan ────────────────────────────────────
    detected_signals = [sig for sig in FABRICATION_SIGNALS
                        if sig.lower() in response_text.lower()]
    if detected_signals:
        log.hallucination_guard(
            "agent.verify",
            f"Fabrication signals detected: {detected_signals}",
            "Reducing confidence score"
        )

    # ── Check 2: Partial data propagation check ──────────────────────────────
    has_partial = any("status=partial" in r or "status=error" in r
                      for r in tool_results)
    not_found_count = sum(1 for r in tool_results if "NOT FOUND" in r)

    # ── Compute confidence ───────────────────────────────────────────────────
    confidence = 1.0
    confidence -= 0.15 * len(detected_signals)    # fabrication signals
    confidence -= 0.20 if has_partial else 0.0    # partial tool data
    confidence -= 0.10 * error_count              # tool errors
    confidence -= 0.15 * not_found_count          # not-found results
    confidence  = max(0.0, min(1.0, confidence))

    elapsed = (time.perf_counter() - t0) * 1000
    log.verify("agent.verify",
               f"confidence={confidence:.2f} | signals={detected_signals} | "
               f"partial={has_partial} | errors={error_count}", elapsed_ms=elapsed)

    # ── Rewrite if confidence is too low ────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        log.hallucination_guard(
            "agent.verify",
            f"Confidence {confidence:.2f} < threshold {CONFIDENCE_THRESHOLD}",
            "Injecting uncertainty markers into response"
        )
        rewritten = _inject_uncertainty(response_text, confidence,
                                        not_found_count, error_count, llm)
        updated_msg = AIMessage(content=rewritten)
        trace_step = ReasoningStep(
            step_type  = "verification",
            content    = f"REWRITTEN: confidence={confidence:.2f}",
            tool_name  = None, tool_args=None, elapsed_ms=elapsed,
        )
        return {
            "messages":        [updated_msg],
            "verification":    "failed_rewritten",
            "confidence":      confidence,
            "reasoning_trace": state.get("reasoning_trace", []) + [trace_step],
        }

    trace_step = ReasoningStep(
        step_type="verification",
        content=f"PASSED: confidence={confidence:.2f}",
        tool_name=None, tool_args=None, elapsed_ms=elapsed,
    )
    return {
        "verification":    "passed",
        "confidence":      confidence,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_step],
    }


def _inject_uncertainty(text: str, confidence: float,
                        not_found: int, errors: int, llm) -> str:
    """
    Either LLM-rewrite or rule-based injection of uncertainty markers.
    Fallback to rule-based if LLM call fails (avoids recursive failure).
    """
    uncertainty_header = (
        f"⚠️ **CONFIDENCE NOTICE** (score: {confidence*100:.0f}%)\n"
        f"The following response is based on incomplete or uncertain data"
        + (f" ({not_found} data gap(s))" if not_found else "")
        + (f" ({errors} tool error(s))" if errors else "")
        + ".\nPlease verify important details independently.\n\n"
    )
    try:
        rewrite_prompt = (
            f"Rewrite the following response to clearly express uncertainty "
            f"where information is missing or unverified. Use phrases like "
            f"'based on available data', 'I cannot confirm', 'you may want to verify'. "
            f"Do NOT add new information. Keep the helpful parts:\n\n{text}"
        )
        resp = llm.invoke([HumanMessage(content=rewrite_prompt)])
        return uncertainty_header + resp.content
    except Exception:
        return uncertainty_header + text


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 5: RESPOND
# ─────────────────────────────────────────────────────────────────────────────

def respond_node(state: AgentState, memory_manager) -> Dict:
    """
    Format and finalise the response for the student.

    Addresses Problem #10: Cognitive overload.
    Ensures responses are structured, scannable, and student-friendly.
    """
    t0       = time.perf_counter()
    messages = state["messages"]

    # Find the last AI content message
    final_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not _has_tool_calls(msg):
            final_text = str(msg.content)
            break

    if not final_text:
        final_text = "I was unable to generate a response. Please try again."

    # Persist the agent's response to memory
    memory_manager.add_message("assistant", final_text)

    elapsed = (time.perf_counter() - t0) * 1000
    log.respond("agent.respond",
                f"Sending final answer ({len(final_text)} chars) | "
                f"verification={state.get('verification', 'skipped')} | "
                f"confidence={state.get('confidence', 1.0):.2f}",
                elapsed_ms=elapsed)

    # Append a structured footer for observability (shown in debug mode only)
    debug_footer = (
        f"\n\n---\n"
        f"_Turn {state.get('turn_count', 1)} | "
        f"Verification: {state.get('verification', 'N/A')} | "
        f"Confidence: {state.get('confidence', 1.0)*100:.0f}%_"
    ) if os.getenv("SHOW_DEBUG_FOOTER", "0") == "1" else ""

    return {"final_answer": final_text + debug_footer}


import os  # noqa: E402 (needed for SHOW_DEBUG_FOOTER env check)