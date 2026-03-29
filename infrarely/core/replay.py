from __future__ import annotations

from typing import Any, Dict, List

from infrarely.core.run_store import RunStore

store = RunStore()


def replay(run_id: str) -> Dict[str, Any]:
    """Replay a saved run deterministically from persisted tool-call outputs."""
    original = store.load(run_id)
    saved_calls = original.get("tool_calls", [])
    if not isinstance(saved_calls, list):
        saved_calls = []

    replay_calls: List[Dict[str, Any]] = []
    divergences: List[Dict[str, Any]] = []

    for index, saved in enumerate(saved_calls):
        if not isinstance(saved, dict):
            divergences.append(
                {
                    "index": index,
                    "reason": "malformed_tool_call",
                    "details": str(saved),
                }
            )
            continue

        replay_calls.append(
            {
                "index": index,
                "tool": saved.get("tool", ""),
                "input": saved.get("input", {}),
                "output": saved.get("output"),
                "depth": int(saved.get("depth", 0) or 0),
                "replayed": True,
            }
        )

    return {
        "run_id": original.get("run_id", run_id),
        "original_input": original.get("input"),
        "original_output": original.get("output"),
        "tool_calls_replayed": len(replay_calls),
        "replay_calls": replay_calls,
        "divergences": divergences,
        "saved_at": original.get("saved_at"),
        "execution_trace": original.get("execution_trace", {}),
    }


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    return store.list_runs(limit=limit)
