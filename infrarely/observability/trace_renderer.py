from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List

COLORS = {
    "AGENT_START": "\033[94m",
    "AGENT_COMPLETE": "\033[94m",
    "ROUTING_PASS": "\033[92m",
    "VALIDATION_PASS": "\033[92m",
    "OUTPUT_PASS": "\033[92m",
    "DEPTH_CHECK_PASS": "\033[92m",
    "TOOL_EXECUTED": "\033[96m",
    "LLM_RESPONSE": "\033[97m",
    "ROUTING_VIOLATION": "\033[91m",
    "VALIDATION_VIOLATION": "\033[91m",
    "OUTPUT_VIOLATION": "\033[91m",
    "DEPTH_VIOLATION": "\033[91m",
    "INFRARELY_BLOCK": "\033[91m",
}
RESET = "\033[0m"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _step_duration_s(step: Dict[str, Any]) -> float:
    return _safe_float(step.get("duration_ms", 0.0)) / 1000.0


def _normalize_events(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = trace.get("events", [])
    if isinstance(events, list) and events:
        normalized: List[Dict[str, Any]] = []
        for event in events:
            if isinstance(event, dict):
                normalized.append(event)
        return normalized

    # Convert InfraRely's current execution-trace shape into event-like rows.
    out: List[Dict[str, Any]] = []
    elapsed = 0.0

    out.append({"timestamp": elapsed, "type": "AGENT_START", "data": {}})

    for step in trace.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        elapsed += _step_duration_s(step)
        event_type = "TOOL_EXECUTED"
        if step.get("skipped"):
            event_type = "STEP_SKIPPED"
        elif step.get("success") is False:
            event_type = "STEP_FAILED"
        out.append(
            {
                "timestamp": elapsed,
                "type": event_type,
                "data": {
                    "tool": step.get("tool") or step.get("name"),
                    "reason": (
                        step.get("error", "") if step.get("success") is False else ""
                    ),
                },
            }
        )

    for tc in trace.get("tool_calls", []) or []:
        if not isinstance(tc, dict):
            continue
        elapsed += _safe_float(tc.get("duration_ms", 0.0)) / 1000.0
        event_type = "TOOL_EXECUTED" if tc.get("success", True) else "OUTPUT_VIOLATION"
        out.append(
            {
                "timestamp": elapsed,
                "type": event_type,
                "data": {
                    "tool": tc.get("tool_name", ""),
                    "output_length": len(str(tc.get("output_preview", ""))),
                },
            }
        )

    for llm_call in trace.get("llm_calls", []) or []:
        if not isinstance(llm_call, dict):
            continue
        elapsed += _safe_float(llm_call.get("duration_ms", 0.0)) / 1000.0
        out.append(
            {
                "timestamp": elapsed,
                "type": "LLM_RESPONSE",
                "data": {
                    "finish_reason": llm_call.get("reason", "goal_synthesis"),
                    "model": llm_call.get("model", ""),
                },
            }
        )

    for err in trace.get("errors", []) or []:
        out.append(
            {
                "timestamp": elapsed,
                "type": "INFRARELY_BLOCK",
                "data": {"reason": str(err)},
            }
        )

    out.append(
        {
            "timestamp": elapsed,
            "type": "AGENT_COMPLETE",
            "data": {"success": bool(trace.get("success", True))},
        }
    )
    return out


def render_terminal(trace: Dict[str, Any]) -> None:
    run_id = str(trace.get("run_id") or trace.get("trace_id") or "unknown")
    duration = trace.get("duration_ms", 0)
    events = _normalize_events(trace)

    print(f"\n{'-' * 60}")
    print(f"  INFRARELY TRACE  |  {run_id[:16]}  |  {duration}ms")
    print(f"{'-' * 60}")

    for event in events:
        ts = f"{_safe_float(event.get('timestamp', 0.0)):.3f}s"
        etype = str(event.get("type", "UNKNOWN"))
        data = event.get("data", {})
        if not isinstance(data, dict):
            data = {"value": str(data)}
        color = COLORS.get(etype, "\033[97m")

        inline = ""
        if "tool" in data and data.get("tool"):
            inline = f"-> {data['tool']}"
        elif "attempted_tool" in data and data.get("attempted_tool"):
            inline = f"-> {data['attempted_tool']} BLOCKED"
        elif "finish_reason" in data and data.get("finish_reason"):
            inline = f"-> {data['finish_reason']}"
        elif "output_length" in data:
            inline = f"-> {data['output_length']} chars"
        elif "reason" in data and data.get("reason"):
            inline = f"-> {data['reason']}"
        elif "depth" in data:
            inline = f"-> depth {data['depth']}/{data.get('max', '?')}"

        print(f"  {color}[{ts}] {etype:<30} {inline}{RESET}")

    violations = [
        e
        for e in events
        if "VIOLATION" in str(e.get("type", "")) or "BLOCK" in str(e.get("type", ""))
    ]
    print(f"{'-' * 60}")
    if violations:
        print(f"  \033[91m{len(violations)} enforcement violation(s){RESET}")
    else:
        print(f"  \033[92mClean execution - zero violations{RESET}")
    print(f"{'-' * 60}\n")


def render_html(trace: Dict[str, Any], output_path: str | None = None) -> str:
    run_id = str(trace.get("run_id") or trace.get("trace_id") or "unknown")
    duration = trace.get("duration_ms", 0)
    events = _normalize_events(trace)

    rows = ""
    for event in events:
        etype = str(event.get("type", ""))
        is_bad = "VIOLATION" in etype or "BLOCK" in etype
        is_good = "PASS" in etype or "COMPLETE" in etype
        color = "#ef4444" if is_bad else "#10b981" if is_good else "#94a3b8"
        data_obj = event.get("data", {})
        if not isinstance(data_obj, dict):
            data_obj = {"value": str(data_obj)}
        data_str = html.escape(json.dumps(data_obj, indent=2, ensure_ascii=False))
        rows += f"""<tr>
            <td style=\"color:#64748b;font-family:monospace;white-space:nowrap\">{_safe_float(event.get('timestamp', 0.0)):.3f}s</td>
            <td style=\"color:{color};font-weight:600;white-space:nowrap\">{html.escape(etype)}</td>
            <td><pre style=\"margin:0;font-size:11px;color:#94a3b8;white-space:pre-wrap\">{data_str}</pre></td>
        </tr>"""

    violations = [
        e
        for e in events
        if "VIOLATION" in str(e.get("type", "")) or "BLOCK" in str(e.get("type", ""))
    ]
    status_color = "#ef4444" if violations else "#10b981"
    status_text = f"{len(violations)} violation(s)" if violations else "Clean execution"

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"UTF-8\">
<title>InfraRely Trace - {html.escape(run_id[:8])}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#060b18;color:#f1f5f9;margin:0;padding:32px}}
h1{{font-size:18px;font-weight:800;margin-bottom:4px}}
.meta{{font-size:12px;color:#64748b;margin-bottom:24px}}
.status{{display:inline-block;padding:6px 14px;border-radius:6px;font-weight:700;font-size:13px;
         background:rgba(0,0,0,.3);border:1px solid {status_color};color:{status_color};margin-bottom:20px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;padding:8px 12px;color:#475569;font-size:10px;
    text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #1e293b}}
td{{padding:8px 12px;border-bottom:1px solid #0f172a;vertical-align:top}}
tr:hover td{{background:rgba(255,255,255,.02)}}
</style>
</head>
<body>
<h1>InfraRely Execution Trace</h1>
<div class=\"meta\">Run: {html.escape(run_id)} &nbsp;|&nbsp; {duration}ms &nbsp;|&nbsp; {len(events)} events</div>
<div class=\"status\">{status_text}</div>
<table>
<thead><tr><th>Time</th><th>Event</th><th>Data</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body>
</html>"""

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_doc, encoding="utf-8")
    return html_doc
