"""
infrarely/_internal/bridge.py — Main SDK ↔ InfraRely Infrastructure Bridge
═══════════════════════════════════════════════════════════════════════════════
Connects SDK's clean API surface to the InfraRely 7-layer infrastructure.
Developers never import this. SDK does it internally.

This bridge resolves goals using:
  1. Knowledge Layer (bypass LLM if confidence >= threshold)
  2. Tool matching (deterministic routing)
  3. Capability matching (multi-step workflows)
  4. LLM synthesis (last resort)
"""

from __future__ import annotations

import hashlib
import re
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from infrarely.core.config import get_config
from infrarely.core.result import Result, Error, ErrorType, _ok, _fail
from infrarely.memory.knowledge import (
    KnowledgeManager,
    KnowledgeResult,
    get_knowledge_manager,
)
from infrarely.core.decorators import (
    ToolRegistry,
    get_tool_registry,
    CapabilityRegistry,
    get_capability_registry,
)
from infrarely.runtime.workflow import Workflow, StepResult
from infrarely.observability.observability import (
    ExecutionTrace,
    TraceStep,
    TraceLLMCall,
    TraceKnowledgeQuery,
    TraceStateTransition,
    TraceToolCall,
    TraceStore,
    get_metrics,
    get_logger,
)


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFIER — deterministic first, LLM never for routing
# ═══════════════════════════════════════════════════════════════════════════════


class _ParameterExtractor:
    """
    Extracts function parameters from natural language goals.
    Maps extracted values to actual tool function signatures.

    Pipeline:
      1. Inspect function signature → get param names + types + defaults
      2. Extract candidate values from goal (numbers, strings, identifiers)
      3. Match candidates to params by type + position
      4. Fill defaults for unmatched params
    """

    # Word → number mapping for NL number words
    _WORD_NUMS: Dict[str, int] = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "thousand": 1000,
        "million": 1_000_000,
    }

    @classmethod
    def _words_to_numbers(cls, text: str) -> str:
        """Convert English number words in *text* to digit strings.

        Handles compound forms like 'twenty five' → '25'.
        """
        words = text.lower().split()
        result: list[str] = []
        i = 0
        while i < len(words):
            w = words[i]
            if w in cls._WORD_NUMS:
                num = cls._WORD_NUMS[w]
                # Absorb following number words (e.g. "twenty five")
                while i + 1 < len(words) and words[i + 1] in cls._WORD_NUMS:
                    i += 1
                    nxt = cls._WORD_NUMS[words[i]]
                    if nxt < num:
                        num += nxt
                    else:
                        num *= nxt
                result.append(str(num))
            else:
                result.append(w)
            i += 1
        return " ".join(result)

    @classmethod
    def extract(cls, goal: str, fn: Callable, meta: Any = None) -> Dict[str, Any]:
        """
        Extract parameters for `fn` from natural language `goal`.
        Returns kwargs dict ready to call fn(**result).
        """
        import inspect

        sig = inspect.signature(fn)
        params_info = []
        for pname, param in sig.parameters.items():
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                ann = None
            default = (
                param.default if param.default is not inspect.Parameter.empty else None
            )
            has_default = param.default is not inspect.Parameter.empty
            params_info.append(
                {
                    "name": pname,
                    "type": ann,
                    "default": default,
                    "has_default": has_default,
                }
            )

        if not params_info:
            return {}

        # ── Convert word-numbers before extracting digits ─────────────────
        normalised_goal = cls._words_to_numbers(goal)

        # Extract candidate values from normalised goal
        numbers = re.findall(r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)", normalised_goal)
        numbers = [float(n) if "." in n else int(n) for n in numbers]

        quoted = re.findall(r'"([^"]+)"', goal)
        identifiers = re.findall(r"\b([A-Z]{1,5}\d{2,5})\b", goal)

        # Remaining text (strip numbers, tool name, filler words)
        filler = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "for",
            "in",
            "of",
            "with",
            "on",
            "at",
            "from",
            "by",
            "that",
            "this",
            "it",
            "do",
            "does",
            "please",
            "can",
            "you",
            "tell",
            "me",
            "get",
            "find",
            "show",
            "give",
            "how",
            "about",
            "much",
            "many",
            # Apostrophe fragments — "what's" → ["what", "s"]
            "s",
            "t",
            "re",
            "ve",
            "ll",
            "d",
            "m",
            "nt",
        }
        # Also remove tool name words
        tool_name = getattr(fn, "__name__", "")
        tool_words = set(tool_name.replace("_", " ").lower().split())
        goal_words = [
            w
            for w in re.findall(r"[a-zA-Z]+", goal.lower())
            if w not in filler and w not in tool_words
        ]

        result = {}
        num_idx = 0
        str_idx = 0

        for pinfo in params_info:
            pname = pinfo["name"]
            ptype = pinfo["type"]

            # Try to match by type
            if ptype in (int, float) or (
                ptype is None and numbers and num_idx < len(numbers)
            ):
                if num_idx < len(numbers):
                    val = numbers[num_idx]
                    num_idx += 1
                    if ptype is int:
                        val = int(val)
                    elif ptype is float:
                        val = float(val)
                    result[pname] = val
                    continue

            if ptype is str:
                # Prefer quoted strings, then identifiers, then remaining words
                if quoted and str_idx < len(quoted):
                    result[pname] = quoted[str_idx]
                    str_idx += 1
                    continue
                if identifiers:
                    result[pname] = identifiers.pop(0)
                    continue
                # Use remaining non-filler words joined
                if goal_words:
                    result[pname] = " ".join(goal_words)
                    goal_words = []
                    continue

            if ptype is dict:
                # Try to pass context
                continue

            if ptype is None:
                # Unknown type — try numbers first, then strings
                if num_idx < len(numbers):
                    result[pname] = numbers[num_idx]
                    num_idx += 1
                    continue
                if quoted and str_idx < len(quoted):
                    result[pname] = quoted[str_idx]
                    str_idx += 1
                    continue
                if goal_words:
                    result[pname] = " ".join(goal_words)
                    goal_words = []
                    continue

            # Fall back to default
            if pinfo["has_default"]:
                continue  # Will use function's default

        return result


class _IntentClassifier:
    """
    Classifies user intent to determine tool/capability routing.
    Fully deterministic — pattern matching + keyword scoring.
    """

    # Common math patterns — the "what is" variant requires the ENTIRE
    # remainder to be pure math tokens (with optional trailing '?') so
    # "what is 10 added to 25?" is NOT captured.
    _MATH_PATTERN = re.compile(
        r"^[\d\s\+\-\*/\(\)\.\^%]+\??\s*$|"
        r"^what\s+is\s+[\d\s\+\-\*/\(\)\.\^%]+\??\s*$|"
        r"^(?:calculate|compute|solve|evaluate)\b",
        re.IGNORECASE,
    )

    # Verb / synonym map  →  canonical tool-name words
    _SYNONYMS: Dict[str, str] = {
        "sum": "add",
        "plus": "add",
        "total": "add",
        "combine": "add",
        "minus": "subtract",
        "difference": "subtract",
        "take": "subtract",
        "remove": "subtract",
        "times": "multiply",
        "product": "multiply",
        "multiplied": "multiply",
        "divided": "divide",
        "split": "divide",
        "over": "divide",
        "ratio": "divide",
        "search": "find",
        "lookup": "find",
        "locate": "find",
        "fetch": "get",
        "retrieve": "get",
        "obtain": "get",
        "list": "get",
        "display": "show",
        "print": "show",
        "create": "make",
        "generate": "make",
        "build": "make",
        "delete": "remove",
        "erase": "remove",
        "drop": "remove",
        "update": "edit",
        "modify": "edit",
        "change": "edit",
        "check": "verify",
        "validate": "verify",
        "test": "verify",
        "send": "send",
        "mail": "send",
        "notify": "send",
        "count": "count",
        "tally": "count",
    }

    _SYNONYM_TARGETS = set(_SYNONYMS.values())

    @staticmethod
    def _stem(word: str) -> str:
        """Minimal English suffix stripping (Porter-lite)."""
        w = word.lower()
        for suffix in (
            "tion",
            "ment",
            "ness",
            "ing",
            "ied",
            "ies",
            "ed",
            "er",
            "ly",
            "es",
            "s",
        ):
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                return w[: -len(suffix)]
        return w

    @classmethod
    def _normalise_word(cls, word: str) -> str:
        """Return canonical form: synonym lookup → stem → lower."""
        w = word.lower()
        if w in cls._SYNONYMS:
            return cls._SYNONYMS[w]
        # Don't stem words that are canonical synonym targets
        if w in cls._SYNONYM_TARGETS:
            return w
        return cls._stem(w)

    @staticmethod
    def classify(
        goal: str,
        tool_names: List[str],
        capability_names: List[str],
        tools: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Classify goal into: math, tool, capability, or general.
        Returns {"type": ..., "name": ..., "confidence": ..., "params": dict}

        When a tool is matched, params are extracted using _ParameterExtractor
        which inspects the actual function signature and maps NL values to it.
        """
        goal_lower = goal.lower().strip()
        tools = tools or {}

        # 1. Math/calculation — deterministic (highest priority)
        if _IntentClassifier._MATH_PATTERN.match(goal_lower):
            return {
                "type": "math",
                "name": "math_eval",
                "confidence": 1.0,
                "params": {"expr": goal},
            }

        # Pre-compute normalised goal words (once)
        goal_raw_words = set(goal_lower.split())
        goal_norm_words = {_IntentClassifier._normalise_word(w) for w in goal_raw_words}

        # 2. Direct tool name match (tool name words appear in goal)
        #    Now uses normalised words so "added"→"add" matches tool "add"
        for tname in tool_names:
            name_words = set(tname.replace("_", " ").lower().split())
            name_norm = {_IntentClassifier._normalise_word(w) for w in name_words}
            overlap = name_norm & goal_norm_words
            if len(overlap) >= min(2, len(name_norm)):
                fn = tools.get(tname)
                params = (
                    _ParameterExtractor.extract(goal, fn)
                    if fn
                    else _IntentClassifier._extract_raw_params(goal)
                )
                return {
                    "type": "tool",
                    "name": tname,
                    "confidence": 0.90,
                    "params": params,
                }

        # 3. Keyword-based tool suggestion (description overlap)
        best_tool = None
        best_score = 0.0
        registry = get_tool_registry()
        for tname in tool_names:
            meta = registry.get_meta(tname)
            if not meta:
                continue
            # Score by description match + parameter name match
            desc_words = set(meta.description.lower().split())
            desc_norm = {_IntentClassifier._normalise_word(w) for w in desc_words}
            desc_overlap = desc_norm & goal_norm_words
            score = len(desc_overlap) / max(len(goal_norm_words), 1)
            # Bonus if tool name partially matches (normalised)
            name_words = set(tname.replace("_", " ").lower().split())
            name_norm = {_IntentClassifier._normalise_word(w) for w in name_words}
            if name_norm & goal_norm_words:
                score += 0.3
            if score > best_score:
                best_score = score
                best_tool = tname

        if best_tool and best_score > 0.25:
            fn = tools.get(best_tool)
            params = (
                _ParameterExtractor.extract(goal, fn)
                if fn
                else _IntentClassifier._extract_raw_params(goal)
            )
            return {
                "type": "tool",
                "name": best_tool,
                "confidence": min(0.95, best_score),
                "params": params,
            }

        # 4. Capability match
        for cname in capability_names:
            name_words = set(cname.replace("_", " ").lower().split())
            goal_words = set(goal_lower.split())
            if name_words & goal_words:
                return {
                    "type": "capability",
                    "name": cname,
                    "confidence": 0.80,
                    "params": _IntentClassifier._extract_raw_params(goal),
                }

        # 5. No specific tool/capability match → general/knowledge
        return {
            "type": "general",
            "name": "",
            "confidence": 0.5,
            "params": _IntentClassifier._extract_raw_params(goal),
        }

    @staticmethod
    def _extract_raw_params(goal: str) -> Dict[str, Any]:
        """Extract common parameters from natural language (raw, untyped)."""
        params = {}
        quoted = re.findall(r'"([^"]+)"', goal)
        if quoted:
            params["subjects"] = quoted
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", goal)
        if numbers:
            params["numbers"] = [float(n) if "." in n else int(n) for n in numbers]
        ids = re.findall(r"\b([A-Z]{1,5}\d{2,5})\b", goal)
        if ids:
            params["identifiers"] = ids
        return params


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE MATH EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_math_eval(expr: str) -> Any:
    """Evaluate simple math expressions safely (no exec/eval)."""
    # Clean the expression
    expr = re.sub(r"[^0-9\+\-\*/\.\(\)\s\^%]", "", expr)
    expr = expr.replace("^", "**")
    # Only allow safe characters
    if not re.match(r"^[\d\s\+\-\*/\.\(\)\*%]+$", expr.strip()):
        return None
    try:
        # Use compile + eval with empty namespace for safety
        code = compile(expr.strip(), "<math>", "eval")
        # Verify no names are used
        if code.co_names:
            return None
        return eval(code, {"__builtins__": {}}, {})
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT — wraps provider-specific calls
# ═══════════════════════════════════════════════════════════════════════════════


def _call_llm(prompt: str, system: str = "", max_tokens: int = 512) -> Dict[str, Any]:
    """
    Call the configured LLM provider. Returns structured response.
    Never raises — always returns a dict with 'content' or 'error'.
    """
    cfg = get_config()
    provider = cfg.get("llm_provider", "openai")
    model = cfg.get("llm_model", "gpt-4o-mini")
    api_key = cfg.get("api_key", "")
    temperature = cfg.get("llm_temperature", 0.3)

    logger = get_logger()
    start = time.monotonic()

    messages = [{"role": "user", "content": prompt}]

    try:
        if provider in ("ollama", "local"):
            return _call_ollama(
                model,
                ([{"role": "system", "content": system}] if system else []) + messages,
                max_tokens,
                temperature,
                cfg.get("llm_base_url"),
            )

        llm = cfg.get("llm")
        if llm is None:
            from infrarely.llm.registry import load_provider

            llm = load_provider(
                provider,
                api_key,
                model,
                base_url=cfg.get("llm_base_url"),
            )
            cfg.set("llm", llm)

        text, tokens = llm.chat(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = (time.monotonic() - start) * 1000
        return {
            "content": text,
            "error": None,
            "tokens": tokens,
            "duration_ms": elapsed,
        }
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        logger.error(f"LLM call failed: {e}", provider=provider)
        return {"content": None, "error": str(e), "tokens": 0, "duration_ms": elapsed}


def _call_openai(api_key, model, messages, max_tokens, temperature):
    import urllib.request
    import json as _json

    start = time.monotonic()
    data = _json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = _json.loads(resp.read())
    elapsed = (time.monotonic() - start) * 1000
    content = result["choices"][0]["message"]["content"]
    tokens = result.get("usage", {}).get("total_tokens", 0)
    return {"content": content, "error": None, "tokens": tokens, "duration_ms": elapsed}


def _call_anthropic(api_key, model, system, prompt, max_tokens, temperature):
    import urllib.request
    import json as _json

    start = time.monotonic()
    data = _json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system or "You are a helpful assistant.",
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = _json.loads(resp.read())
    elapsed = (time.monotonic() - start) * 1000
    content = result["content"][0]["text"]
    tokens = result.get("usage", {}).get("input_tokens", 0) + result.get(
        "usage", {}
    ).get("output_tokens", 0)
    return {"content": content, "error": None, "tokens": tokens, "duration_ms": elapsed}


def _call_groq(api_key, model, messages, max_tokens, temperature):
    import urllib.request
    import json as _json

    start = time.monotonic()
    data = _json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = _json.loads(resp.read())
    elapsed = (time.monotonic() - start) * 1000
    content = result["choices"][0]["message"]["content"]
    tokens = result.get("usage", {}).get("total_tokens", 0)
    return {"content": content, "error": None, "tokens": tokens, "duration_ms": elapsed}


def _call_gemini(api_key, model, prompt, max_tokens, temperature):
    import urllib.request
    import json as _json

    start = time.monotonic()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    data = _json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
    ).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = _json.loads(resp.read())
    elapsed = (time.monotonic() - start) * 1000
    content = result["candidates"][0]["content"]["parts"][0]["text"]
    tokens = result.get("usageMetadata", {}).get("totalTokenCount", 0)
    return {"content": content, "error": None, "tokens": tokens, "duration_ms": elapsed}


def _call_ollama(model, messages, max_tokens, temperature, base_url=None):
    import urllib.request
    import json as _json

    start = time.monotonic()
    url = f"{base_url or 'http://localhost:11434'}/api/chat"
    data = _json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
    ).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = _json.loads(resp.read())
    elapsed = (time.monotonic() - start) * 1000
    content = result.get("message", {}).get("content", "")
    tokens = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
    return {"content": content, "error": None, "tokens": tokens, "duration_ms": elapsed}


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION ENGINE — the heart of agent.run()
# ═══════════════════════════════════════════════════════════════════════════════


class ExecutionEngine:
    """
    Executes a goal through the InfraRely pipeline:
      1. Knowledge check (bypass LLM if high confidence)
      2. Intent classification (deterministic routing)
      3. Tool execution (if matched)
      4. Capability/workflow execution (if matched)
      5. LLM synthesis (last resort)
      6. Trace recording
    """

    def __init__(
        self,
        agent_name: str,
        tools: Dict[str, Callable],
        capabilities: Dict[str, Any],
        knowledge: KnowledgeManager,
        trace_store: TraceStore,
        hitl_gate: Any = None,
    ):
        self._agent_name = agent_name
        self._tools = tools
        self._capabilities = capabilities
        self._knowledge = knowledge
        self._trace_store = trace_store
        self._hitl_gate = hitl_gate
        self._logger = get_logger()
        self._metrics = get_metrics()

    def execute(self, goal: str, context: Optional[Any] = None) -> Result:
        """
        Execute a goal through the full InfraRely pipeline.

        Pipeline order (critical — do NOT reorder):
          1. Intent classification (math, tool match, capability)
          2. Math evaluation (purely deterministic)
          3. Tool execution (if intent matched a tool)
          4. Knowledge check (bypass LLM if high confidence)
          5. LLM synthesis (last resort)

        Knowledge runs AFTER math/tool so that deterministic answers
        are never overridden by fuzzy TF-IDF matches.
        """
        cfg = get_config()
        start = time.monotonic()
        trace = ExecutionTrace(agent_name=self._agent_name, goal=goal)
        state_transitions = ["IDLE", "PLANNING"]
        used_llm = False
        llm_calls_count = 0
        sources = []
        plan_source = "deterministic"

        try:
            # ── 1. Intent classification (FIRST — before knowledge) ───────────
            tool_names = list(self._tools.keys())
            cap_names = list(self._capabilities.keys())
            intent = _IntentClassifier.classify(
                goal, tool_names, cap_names, tools=self._tools
            )
            state_transitions.append("EXECUTING")

            # ── 2. Math evaluation (fully deterministic, highest priority) ────
            if intent["type"] == "math":
                result = _safe_math_eval(goal)
                if result is not None:
                    elapsed = (time.monotonic() - start) * 1000
                    state_transitions.extend(["VERIFYING", "COMPLETED"])
                    trace.steps.append(
                        TraceStep(
                            name="math_eval",
                            tool="math_eval",
                            success=True,
                            duration_ms=elapsed,
                        )
                    )
                    trace.completed_at = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )
                    trace.duration_ms = elapsed
                    trace.output_preview = str(result)
                    self._trace_store.save(trace)
                    self._metrics.record_task(True, False, elapsed)
                    return _ok(
                        output=result,
                        confidence=1.0,
                        used_llm=False,
                        sources=["math_eval"],
                        duration_ms=elapsed,
                        goal=goal,
                        agent_name=self._agent_name,
                        _state_transitions=state_transitions,
                        _plan_source="deterministic",
                        trace_id=trace.trace_id,
                    )

            # ── 3. Tool execution (deterministic routing) ─────────────────────
            if intent["type"] == "tool" and intent["name"] in self._tools:
                tool_fn = self._tools[intent["name"]]
                tool_start = time.monotonic()
                try:
                    params = intent.get("params", {})

                    # ── HITL gate check ──────────────────────────────────
                    if self._hitl_gate is not None:
                        approved, approval_req = self._hitl_gate.wait_and_check(
                            intent["name"], params
                        )
                        if not approved:
                            from infrarely.core.result import ErrorType as _ET

                            elapsed = (time.monotonic() - start) * 1000
                            status = getattr(approval_req, "status", None)
                            if status and status.value == "timed_out":
                                return _fail(
                                    _ET.APPROVAL_TIMEOUT,
                                    f"Approval timed out for tool '{intent['name']}'",
                                    step=intent["name"],
                                    goal=goal,
                                    agent_name=self._agent_name,
                                    duration_ms=elapsed,
                                )
                            else:
                                return _fail(
                                    _ET.APPROVAL_REJECTED,
                                    f"Execution of '{intent['name']}' was rejected by human reviewer",
                                    step=intent["name"],
                                    goal=goal,
                                    agent_name=self._agent_name,
                                    duration_ms=elapsed,
                                )

                    # ── Tool input validation ────────────────────────────
                    try:
                        from infrarely.platform.validation import SchemaValidator as _SV

                        _validator = _SV()
                        val_cfg = get_config()
                        if val_cfg.get("tool_validation_enabled", True) and params:
                            val_result = _validator.validate_call(tool_fn, params)
                            if not val_result.valid:
                                from infrarely.core.result import ErrorType as _ET

                                elapsed = (time.monotonic() - start) * 1000
                                err_msgs = "; ".join(
                                    e.message for e in val_result.errors
                                )
                                return _fail(
                                    _ET.VALIDATION,
                                    f"Tool input validation failed: {err_msgs}",
                                    step=intent["name"],
                                    goal=goal,
                                    agent_name=self._agent_name,
                                    duration_ms=elapsed,
                                )
                            if val_result.coerced_args:
                                params.update(val_result.coerced_args)
                    except ImportError:
                        pass

                    tool_result = tool_fn(**params) if params else tool_fn()
                    tool_elapsed = (time.monotonic() - tool_start) * 1000

                    trace.tool_calls.append(
                        TraceToolCall(
                            tool_name=intent["name"],
                            inputs=params,
                            output_preview=str(tool_result)[:200],
                            duration_ms=tool_elapsed,
                            success=True,
                        )
                    )
                    trace.steps.append(
                        TraceStep(
                            name=intent["name"],
                            tool=intent["name"],
                            success=True,
                            duration_ms=tool_elapsed,
                        )
                    )
                    sources.append(intent["name"])
                    self._metrics.record_tool_call(intent["name"], tool_elapsed, True)

                    # Check if tool returned an error
                    if isinstance(tool_result, dict) and (
                        tool_result.get("__infrarely_error")
                        or tool_result.get("__aos_error")
                    ):
                        raise Exception(tool_result.get("message", "Tool failed"))

                    # If tool result is a complete answer, return directly
                    if self._is_complete_answer(tool_result, goal):
                        elapsed = (time.monotonic() - start) * 1000
                        state_transitions.extend(["VERIFYING", "COMPLETED"])
                        trace.completed_at = time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        )
                        trace.duration_ms = elapsed
                        trace.output_preview = str(tool_result)[:200]
                        self._trace_store.save(trace)
                        self._metrics.record_task(True, False, elapsed)
                        return _ok(
                            output=tool_result,
                            confidence=intent["confidence"],
                            used_llm=False,
                            sources=sources,
                            duration_ms=elapsed,
                            goal=goal,
                            agent_name=self._agent_name,
                            _steps_executed=1,
                            _state_transitions=state_transitions,
                            _plan_source="deterministic",
                            trace_id=trace.trace_id,
                        )

                    # Tool result needs LLM synthesis — falls through
                    context_for_llm = tool_result

                except Exception as e:
                    tool_elapsed = (time.monotonic() - tool_start) * 1000
                    self._metrics.record_tool_call(intent["name"], tool_elapsed, False)
                    trace.tool_calls.append(
                        TraceToolCall(
                            tool_name=intent["name"],
                            inputs=intent.get("params", {}),
                            success=False,
                            duration_ms=tool_elapsed,
                        )
                    )
                    self._logger.warning(f"Tool {intent['name']} failed: {e}")
                    # Fall through to knowledge/LLM

            # ── 4. Capability/workflow execution ──────────────────────────────
            if intent["type"] == "capability" and intent["name"] in self._capabilities:
                cap_fn = self._capabilities[intent["name"]]
                params = intent.get("params", {})
                try:
                    wf = cap_fn(**params) if params else cap_fn()
                    if isinstance(wf, Workflow):
                        wf_results = wf.execute()
                        steps_executed = sum(
                            1 for r in wf_results.values() if not r.skipped
                        )
                        steps_skipped = sum(1 for r in wf_results.values() if r.skipped)

                        for name, sr in wf_results.items():
                            trace.steps.append(
                                TraceStep(
                                    name=name,
                                    tool=name,
                                    success=sr.success,
                                    skipped=sr.skipped,
                                    error=sr.error,
                                    duration_ms=sr.duration_ms,
                                )
                            )
                            if sr.success and not sr.skipped:
                                sources.append(name)

                        if wf.all_succeeded:
                            elapsed = (time.monotonic() - start) * 1000
                            state_transitions.extend(["VERIFYING", "COMPLETED"])
                            output = wf.final_output
                            trace.completed_at = time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            )
                            trace.duration_ms = elapsed
                            trace.output_preview = str(output)[:200]
                            self._trace_store.save(trace)
                            self._metrics.record_task(True, False, elapsed)
                            plan_source = "capability"
                            return _ok(
                                output=output,
                                confidence=0.9,
                                used_llm=False,
                                sources=sources,
                                duration_ms=elapsed,
                                goal=goal,
                                agent_name=self._agent_name,
                                _steps_executed=steps_executed,
                                _steps_skipped=steps_skipped,
                                _state_transitions=state_transitions,
                                _plan_source=plan_source,
                                trace_id=trace.trace_id,
                            )
                        else:
                            context_for_llm = {
                                n: str(r.output)[:200]
                                for n, r in wf_results.items()
                                if r.success and r.output
                            }
                except Exception as e:
                    self._logger.warning(f"Capability {intent['name']} failed: {e}")
                    trace.errors.append(str(e))

            # ── 5. Knowledge check (Rule 3: Knowledge before generation) ──────
            # Only reach here if no deterministic path (math/tool/cap) handled it
            knowledge_result = self._knowledge.query(goal)
            trace.knowledge_queries.append(
                TraceKnowledgeQuery(
                    query=goal,
                    confidence=knowledge_result.confidence,
                    sources=knowledge_result.source_names,
                    decision=knowledge_result.decision,
                    duration_ms=knowledge_result.duration_ms,
                )
            )

            if knowledge_result.decision in ("bypass_llm", "ground_llm"):
                elapsed = (time.monotonic() - start) * 1000
                state_transitions.extend(["VERIFYING", "COMPLETED"])
                output = self._format_knowledge_response(knowledge_result)
                trace.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                trace.duration_ms = elapsed
                trace.output_preview = str(output)[:200]
                reason = (
                    "knowledge_bypass"
                    if knowledge_result.decision == "bypass_llm"
                    else "knowledge_ground"
                )
                for s in state_transitions:
                    trace.state_transitions.append(
                        TraceStateTransition(
                            from_state=state_transitions[
                                max(0, state_transitions.index(s) - 1)
                            ],
                            to_state=s,
                            reason=reason,
                        )
                    )
                self._trace_store.save(trace)
                self._metrics.record_task(True, False, elapsed)

                return _ok(
                    output=output,
                    confidence=knowledge_result.confidence,
                    used_llm=False,
                    sources=knowledge_result.source_names,
                    duration_ms=elapsed,
                    goal=goal,
                    agent_name=self._agent_name,
                    _knowledge_sources=knowledge_result.source_names,
                    _state_transitions=state_transitions,
                    _plan_source="knowledge",
                    trace_id=trace.trace_id,
                )

            # ── 6. LLM synthesis (last resort) ────────────────────────────────
            state_transitions.append("EXECUTING")
            used_llm = True
            llm_calls_count = 1
            plan_source = "llm"

            # Build prompt with grounding
            system_prompt = (
                "You are a helpful, concise assistant. "
                "Answer in 4 sentences or fewer. "
                "Use only the provided context. Never invent facts."
            )
            user_prompt = goal

            # Ground with knowledge if available
            if knowledge_result and knowledge_result.chunks:
                context_text = "\n".join(
                    f"- {c.content}" for c in knowledge_result.chunks[:5]
                )
                user_prompt = (
                    f"Context (use ONLY these facts):\n{context_text}\n\n"
                    f"Question: {goal}"
                )
                sources.extend(knowledge_result.source_names)

            # Ground with tool context if available
            if context:
                if isinstance(context, Result):
                    user_prompt = f"Previous context:\n{context.output}\n\nTask: {goal}"
                else:
                    user_prompt = f"Context:\n{context}\n\nTask: {goal}"

            llm_start = time.monotonic()
            llm_response = _call_llm(user_prompt, system=system_prompt)
            llm_elapsed = (time.monotonic() - llm_start) * 1000

            trace.llm_calls.append(
                TraceLLMCall(
                    model=cfg.get("llm_model", ""),
                    prompt_tokens=llm_response.get("tokens", 0) // 2,
                    completion_tokens=llm_response.get("tokens", 0) // 2,
                    duration_ms=llm_elapsed,
                    reason="goal_synthesis",
                )
            )

            if llm_response.get("error"):
                # ── Fallback: return knowledge chunks if LLM is unavailable ──
                if knowledge_result and knowledge_result.chunks:
                    elapsed = (time.monotonic() - start) * 1000
                    state_transitions.extend(["VERIFYING", "COMPLETED"])
                    output = self._format_knowledge_response(knowledge_result)
                    trace.completed_at = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )
                    trace.duration_ms = elapsed
                    trace.output_preview = str(output)[:200]
                    self._trace_store.save(trace)
                    self._metrics.record_task(True, False, elapsed)
                    return _ok(
                        output=output,
                        confidence=max(knowledge_result.confidence, 0.3),
                        used_llm=False,
                        sources=knowledge_result.source_names,
                        duration_ms=elapsed,
                        goal=goal,
                        agent_name=self._agent_name,
                        _knowledge_sources=knowledge_result.source_names,
                        _state_transitions=state_transitions,
                        _plan_source="knowledge_fallback",
                        trace_id=trace.trace_id,
                    )

                elapsed = (time.monotonic() - start) * 1000
                state_transitions.extend(["FAILED"])
                trace.success = False
                trace.errors.append(llm_response["error"])
                trace.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                trace.duration_ms = elapsed
                self._trace_store.save(trace)
                self._metrics.record_task(False, False, elapsed)
                fallback_msg = (
                    f'I couldn\'t find an answer for "{goal}". '
                    "No matching tools, knowledge, or LLM are available. "
                    "Add relevant knowledge with agent.knowledge.add_data(), "
                    "register a tool with @infrarely.tool, or configure an LLM."
                )
                return _ok(
                    output=fallback_msg,
                    confidence=0.0,
                    used_llm=False,
                    sources=[],
                    duration_ms=elapsed,
                    goal=goal,
                    agent_name=self._agent_name,
                    _state_transitions=state_transitions,
                    _plan_source="fallback",
                    trace_id=trace.trace_id,
                )

            output = llm_response["content"]
            elapsed = (time.monotonic() - start) * 1000
            state_transitions.extend(["VERIFYING", "COMPLETED"])

            # Determine confidence
            confidence = 0.75
            if knowledge_result and knowledge_result.confidence > 0:
                confidence = min(0.95, knowledge_result.confidence + 0.1)

            trace.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            trace.duration_ms = elapsed
            trace.output_preview = str(output)[:200]
            self._trace_store.save(trace)
            self._metrics.record_task(
                True, True, elapsed, low_confidence=confidence < 0.7
            )

            return _ok(
                output=output,
                confidence=confidence,
                used_llm=True,
                sources=sources if sources else ["llm"],
                duration_ms=elapsed,
                goal=goal,
                agent_name=self._agent_name,
                _steps_executed=len(trace.steps),
                _llm_calls=llm_calls_count,
                _knowledge_sources=(
                    knowledge_result.source_names if knowledge_result else []
                ),
                _state_transitions=state_transitions,
                _plan_source=plan_source,
                trace_id=trace.trace_id,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            self._logger.error(f"Execution failed: {e}")
            trace.success = False
            trace.errors.append(str(e))
            trace.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            trace.duration_ms = elapsed
            try:
                self._trace_store.save(trace)
            except Exception:
                pass
            self._metrics.record_task(False, used_llm, elapsed)

            return _fail(
                ErrorType.UNKNOWN,
                f"Unexpected error: {e}",
                goal=goal,
                agent_name=self._agent_name,
                duration_ms=elapsed,
                trace_id=trace.trace_id,
            )

    @staticmethod
    def _is_complete_answer(result: Any, goal: str) -> bool:
        """Heuristic: is the tool result a complete answer by itself?"""
        if result is None:
            return False
        if isinstance(result, (int, float, bool)):
            return True
        if isinstance(result, str) and len(result) > 10:
            return True
        if isinstance(result, (list, dict)):
            return True
        return False

    @staticmethod
    def _format_knowledge_response(kr: KnowledgeResult) -> str:
        """Format knowledge chunks into a readable response."""
        if not kr.chunks:
            return ""
        if len(kr.chunks) == 1:
            return kr.chunks[0].content
        # Combine top chunks
        return "\n".join(f"• {c.content}" for c in kr.chunks[:5])
