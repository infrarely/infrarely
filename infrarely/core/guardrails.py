from __future__ import annotations

import inspect
import json
import threading
import uuid
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Union,
    get_args,
    get_origin,
)

BLOCKS_LOG = Path(".infrarely/blocks.log")
BLOCKS_LOG.parent.mkdir(parents=True, exist_ok=True)

_LOG_LOCK = threading.Lock()

REGISTRY: Dict[str, Dict[str, str]] = {
    "IR-001": {
        "name": "Routing Contract Violation",
        "fix": "Add the tool to allowed_tools or update your routing policy.",
    },
    "IR-002": {
        "name": "Tool Input Validation Failure",
        "fix": "Check required fields in your tool schema.",
    },
    "IR-003": {
        "name": "Enum Constraint Violation",
        "fix": "Update enum values in tool schema or clarify in prompt.",
    },
    "IR-004": {
        "name": "Execution Depth Exceeded",
        "fix": "Increase max_depth or check for agent loops.",
    },
    "IR-005": {
        "name": "Output Verification Failure",
        "fix": "Check if LLM is returning a proper response after tool use.",
    },
}


class GuardrailViolation(Exception):
    def __init__(
        self,
        guardrail_id: str,
        layer: str,
        severity: str,
        reason: str,
        fix: str,
        blocked_value: Any = None,
        run_id: Optional[str] = None,
    ):
        self.guardrail_id = guardrail_id
        self.layer = layer
        self.severity = severity
        self.reason = reason
        self.fix = fix
        self.blocked_value = blocked_value
        self.run_id = run_id
        self.violation_id = uuid.uuid4().hex[:8]
        self.timestamp = datetime.now(timezone.utc).isoformat()
        super().__init__(self._format())

    def _format(self) -> str:
        sep = "=" * 60
        return (
            f"\n{sep}\n"
            "  INFRARELY GUARDRAIL TRIGGERED\n"
            f"  ID       : {self.guardrail_id} [{self.violation_id}]\n"
            f"  Layer    : {self.layer}\n"
            f"  Severity : {self.severity.upper()}\n"
            f"  Reason   : {self.reason}\n"
            f"  Fix      : {self.fix}\n"
            f"  Blocked  : {self.blocked_value}\n"
            f"{sep}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "guardrail_id": self.guardrail_id,
            "layer": self.layer,
            "severity": self.severity,
            "reason": self.reason,
            "fix": self.fix,
            "blocked_value": _safe_json_value(self.blocked_value),
            "run_id": self.run_id,
            "timestamp": self.timestamp,
        }

    def log(self) -> None:
        line = json.dumps(self.to_dict(), ensure_ascii=True)
        with _LOG_LOCK:
            with BLOCKS_LOG.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _append_trace_error(trace: Any, violation: GuardrailViolation) -> None:
    try:
        if (
            trace is not None
            and hasattr(trace, "errors")
            and isinstance(trace.errors, list)
        ):
            trace.errors.append(str(violation))
    except Exception:
        # Guardrails should never fail the caller while trying to record metadata.
        pass


def _raise_violation(
    guardrail_id: str,
    layer: str,
    severity: str,
    reason: str,
    blocked_value: Any,
    trace: Any,
    run_id: Optional[str],
) -> None:
    violation = GuardrailViolation(
        guardrail_id=guardrail_id,
        layer=layer,
        severity=severity,
        reason=reason,
        fix=REGISTRY[guardrail_id]["fix"],
        blocked_value=blocked_value,
        run_id=run_id,
    )
    violation.log()
    _append_trace_error(trace, violation)
    raise violation


def _as_tool_set(allowed_tools: Iterable[str] | Mapping[str, Any] | None) -> set[str]:
    if allowed_tools is None:
        return set()
    if isinstance(allowed_tools, Mapping):
        return {str(k) for k in allowed_tools.keys()}
    return {str(v) for v in allowed_tools}


def enforce_routing(
    tool_name: str,
    allowed_tools: Iterable[str] | Mapping[str, Any] | None,
    trace: Any,
    run_id: Optional[str] = None,
) -> None:
    allowed = _as_tool_set(allowed_tools)
    if tool_name not in allowed:
        _raise_violation(
            "IR-001",
            "ROUTING",
            "critical",
            f"Tool '{tool_name}' not in allowed set.",
            {"attempted": tool_name, "allowed": sorted(allowed)},
            trace,
            run_id,
        )


def validate_tool_call(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
    schema: Mapping[str, Any] | None,
    trace: Any,
    run_id: Optional[str] = None,
) -> None:
    if tool_input is None:
        tool_input = {}
    if not isinstance(tool_input, Mapping):
        _raise_violation(
            "IR-002",
            "TOOL_VALIDATION",
            "high",
            f"Tool '{tool_name}' input must be a mapping.",
            {"received_type": type(tool_input).__name__},
            trace,
            run_id,
        )

    schema = schema or {}
    required = list(schema.get("required", []))
    props = schema.get("properties", {}) or {}

    missing = [field for field in required if field not in tool_input]
    if missing:
        _raise_violation(
            "IR-002",
            "TOOL_VALIDATION",
            "high",
            f"Tool '{tool_name}' missing required fields: {missing}.",
            {"missing": missing, "received": dict(tool_input)},
            trace,
            run_id,
        )

    for field, value in tool_input.items():
        prop = props.get(field, {}) if isinstance(props, Mapping) else {}
        enum_values = prop.get("enum") if isinstance(prop, Mapping) else None
        if enum_values and value not in enum_values:
            _raise_violation(
                "IR-003",
                "TOOL_VALIDATION",
                "high",
                f"Field '{field}' value '{value}' not in allowed enum.",
                {
                    "field": field,
                    "value": value,
                    "allowed": list(enum_values),
                },
                trace,
                run_id,
            )


def enforce_depth(
    depth: int, max_depth: int, trace: Any, run_id: Optional[str] = None
) -> None:
    if depth >= max_depth:
        _raise_violation(
            "IR-004",
            "DEPTH_LIMIT",
            "critical",
            f"Depth {depth} exceeded max {max_depth}.",
            {"depth": depth, "max": max_depth},
            trace,
            run_id,
        )


def verify_output(
    output: Any, tool_name: str, trace: Any, run_id: Optional[str] = None
) -> None:
    if output is None:
        _raise_violation(
            "IR-005",
            "OUTPUT_VERIFICATION",
            "medium",
            f"Tool '{tool_name}' output is None.",
            {"output": None},
            trace,
            run_id,
        )

    if isinstance(output, str):
        if len(output.strip()) < 10:
            _raise_violation(
                "IR-005",
                "OUTPUT_VERIFICATION",
                "medium",
                "Agent output is empty or too short.",
                {"output_length": len(output)},
                trace,
                run_id,
            )
        return

    if isinstance(output, Mapping) and (
        output.get("__infrarely_error") or output.get("__aos_error")
    ):
        _raise_violation(
            "IR-005",
            "OUTPUT_VERIFICATION",
            "high",
            f"Tool '{tool_name}' returned an explicit error payload.",
            dict(output),
            trace,
            run_id,
        )


def build_tool_schema(tool_fn: Callable[..., Any]) -> Dict[str, Any]:
    """Build a lightweight schema from a callable signature.

    Captures required parameters and enum constraints based on Literal annotations.
    This is intentionally conservative and never blocks unknown fields.
    """
    schema: Dict[str, Any] = {"required": [], "properties": {}}
    sig = inspect.signature(tool_fn)

    def _enum_values(annotation: Any) -> list[Any] | None:
        origin = get_origin(annotation)
        if origin is Literal:
            values = list(get_args(annotation))
            return values or None

        # Optional[Literal[...]] may be represented as typing.Union or |.
        if origin in (Union, types.UnionType):
            for arg in get_args(annotation):
                if arg is type(None):
                    continue
                if get_origin(arg) is Literal:
                    values = list(get_args(arg))
                    if values:
                        return values
        return None

    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if param.default is inspect.Parameter.empty:
            schema["required"].append(name)

        prop: Dict[str, Any] = {}
        enum_values = _enum_values(param.annotation)
        if enum_values:
            prop["enum"] = enum_values

        if prop:
            schema["properties"][name] = prop

    return schema
