"""
aos/decorators.py — @infrarely.tool and @infrarely.capability decorators
═══════════════════════════════════════════════════════════════════════════════
Transform plain Python functions into AOS-managed tools and capabilities.

@infrarely.tool wraps a function with:
  - Automatic retry with exponential backoff
  - Timeout enforcement
  - Result caching
  - Fallback tool routing
  - Circuit breaker protection
  - Structured error handling (never bare exceptions)
  - Auto-registration in the global tool registry

@infrarely.capability wraps a workflow definition for reuse.

Philosophy 1: Progressive complexity.
  @infrarely.tool               → zero-config, all defaults
  @infrarely.tool(retries=5)    → one override
  @infrarely.tool(retries=5, timeout=3000, cache=True) → full control
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER (per-tool)
# ═══════════════════════════════════════════════════════════════════════════════


class _CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class _CircuitBreaker:
    """Per-tool circuit breaker to stop hammering broken functions."""

    def __init__(self, threshold: int = 3, recovery_s: float = 30.0):
        self.threshold = threshold
        self.recovery_s = recovery_s
        self.failures = 0
        self.state = _CircuitState.CLOSED
        self.last_failure_time = 0.0

    def allow(self) -> bool:
        if self.state == _CircuitState.CLOSED:
            return True
        if self.state == _CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_s:
                self.state = _CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: allow one probe

    def record_success(self):
        self.failures = 0
        self.state = _CircuitState.CLOSED

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.state = _CircuitState.OPEN


class _RateLimiter:
    """Token-bucket rate limiter for tools."""

    def __init__(self, max_calls: int = 100, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            cutoff = now - self.window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.max_calls:
                return False
            self._timestamps.append(now)
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            cutoff = time.time() - self.window
            active = sum(1 for t in self._timestamps if t > cutoff)
            return max(0, self.max_calls - active)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT CACHE (per-tool LRU)
# ═══════════════════════════════════════════════════════════════════════════════


class _ToolCache:
    """Simple LRU cache with TTL for deterministic tool results."""

    def __init__(self, max_size: int = 256, ttl: int = 3600):
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(self, args: tuple, kwargs: dict) -> str:
        """Create a stable cache key from function arguments."""
        key_data = json.dumps(
            {
                "a": [str(a) for a in args],
                "k": {k: str(v) for k, v in sorted(kwargs.items())},
            },
            sort_keys=True,
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Tuple[bool, Any]:
        with self._lock:
            if key in self._store:
                value, ts = self._store[key]
                if time.time() - ts < self._ttl:
                    self._store.move_to_end(key)
                    return True, value
                del self._store[key]
            return False, None

    def put(self, key: str, value: Any):
        with self._lock:
            if key in self._store:
                del self._store[key]
            self._store[key] = (value, time.time())
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self):
        with self._lock:
            self._store.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL METADATA & REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ToolMeta:
    """Metadata for a registered tool."""

    name: str
    description: str
    parameters: Dict[str, Any]  # {param_name: type_annotation}
    retries: int = 2
    timeout: int = 10_000  # ms
    fallback: Optional[str] = None
    cache_enabled: bool = True
    cache_ttl: int = 3600
    deterministic: bool = True
    tags: List[str] = field(default_factory=list)
    call_count: int = 0
    error_count: int = 0
    total_ms: float = 0.0
    last_error: str = ""


class ToolRegistry:
    """
    Global registry of all @infrarely.tool decorated functions.
    Thread-safe. Singleton per process.
    """

    _instance: Optional["ToolRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ToolRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._tools: Dict[str, Callable] = {}
                cls._instance._meta: Dict[str, ToolMeta] = {}
            return cls._instance

    def register(self, fn: Callable, meta: ToolMeta) -> None:
        self._tools[meta.name] = fn
        self._meta[meta.name] = meta

    def get(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def get_meta(self, name: str) -> Optional[ToolMeta]:
        return self._meta.get(name)

    def list_tools(self) -> List[ToolMeta]:
        return list(self._meta.values())

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def clear(self) -> None:
        self._tools.clear()
        self._meta.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    return _registry


# ═══════════════════════════════════════════════════════════════════════════════
# @infrarely.tool DECORATOR
# ═══════════════════════════════════════════════════════════════════════════════


def tool(
    fn: Optional[Callable] = None,
    *,
    retries: int = 2,
    timeout: int = 10_000,
    fallback: Optional[str] = None,
    cache: bool = True,
    cache_ttl: int = 3600,
    deterministic: bool = True,
    description: str = "",
    tags: Optional[List[str]] = None,
    rate_limit: int = 0,
    rate_window: float = 60.0,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Decorator that registers a function as an AOS tool.

    Usage:
        @infrarely.tool
        def my_func(x: int) -> str: ...

        @infrarely.tool(retries=5, timeout=3000)
        def my_func(x: int) -> str: ...
    """

    def _wrap(func: Callable) -> Callable:
        # ── Extract metadata from function signature ──────────────────────────
        sig = inspect.signature(func)
        params = {}
        for pname, param in sig.parameters.items():
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                params[pname] = "Any"
            else:
                params[pname] = getattr(ann, "__name__", str(ann))

        tool_name = func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
        tool_desc = tool_desc.strip().split("\n")[0]  # first line only

        meta = ToolMeta(
            name=tool_name,
            description=tool_desc,
            parameters=params,
            retries=retries,
            timeout=timeout,
            fallback=fallback,
            cache_enabled=cache,
            cache_ttl=cache_ttl,
            deterministic=deterministic,
            tags=tags or [],
        )

        # ── Per-tool infrastructure ───────────────────────────────────────────
        _cb = _CircuitBreaker(threshold=3, recovery_s=30.0)
        _cache_store = _ToolCache(max_size=256, ttl=cache_ttl) if cache else None
        _limiter = (
            _RateLimiter(max_calls=rate_limit, window_seconds=rate_window)
            if rate_limit > 0
            else None
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.monotonic()
            meta.call_count += 1

            # ── Rate limit check ──────────────────────────────────────────────
            if _limiter and not _limiter.allow():
                meta.error_count += 1
                meta.last_error = "Rate limit exceeded"
                return {
                    "__aos_error": True,
                    "type": "TOOL_FAILURE",
                    "message": f"Tool '{tool_name}' rate limit exceeded "
                    f"({rate_limit} calls per {rate_window}s). Try again later.",
                    "tool": tool_name,
                }

            # ── Circuit breaker check ─────────────────────────────────────────
            if not _cb.allow():
                meta.error_count += 1
                meta.last_error = "Circuit breaker OPEN"
                return {
                    "__aos_error": True,
                    "type": "TOOL_FAILURE",
                    "message": f"Tool '{tool_name}' circuit breaker is OPEN. "
                    f"Too many recent failures. Will retry after cooldown.",
                    "tool": tool_name,
                }

            # ── Cache check ───────────────────────────────────────────────────
            if _cache_store and deterministic:
                cache_key = _cache_store._make_key(args, kwargs)
                hit, cached = _cache_store.get(cache_key)
                if hit:
                    elapsed = (time.monotonic() - start) * 1000
                    meta.total_ms += elapsed
                    return cached

            # ── Execute with retry + timeout ──────────────────────────────────
            last_error = None
            for attempt in range(max(1, retries + 1)):
                try:
                    # Timeout enforcement via threading
                    result_container = [None]
                    error_container = [None]

                    def _run():
                        try:
                            result_container[0] = func(*args, **kwargs)
                        except Exception as e:
                            error_container[0] = e

                    t = threading.Thread(target=_run, daemon=True)
                    t.start()
                    t.join(timeout=timeout / 1000.0)

                    if t.is_alive():
                        # Thread still running → timeout
                        raise TimeoutError(
                            f"Tool '{tool_name}' timed out after {timeout}ms"
                        )

                    if error_container[0] is not None:
                        raise error_container[0]

                    result = result_container[0]

                    # ── Success ───────────────────────────────────────────────
                    _cb.record_success()
                    elapsed = (time.monotonic() - start) * 1000
                    meta.total_ms += elapsed

                    if _cache_store and deterministic:
                        _cache_store.put(cache_key, result)

                    return result

                except Exception as e:
                    last_error = e
                    _cb.record_failure()
                    if attempt < retries:
                        # Exponential backoff: 0.1s, 0.2s, 0.4s, ...
                        time.sleep(0.1 * (2**attempt))
                    continue

            # ── All retries exhausted ─────────────────────────────────────────
            meta.error_count += 1
            meta.last_error = str(last_error)
            elapsed = (time.monotonic() - start) * 1000
            meta.total_ms += elapsed

            # ── Try fallback if configured ────────────────────────────────────
            if fallback:
                fb_fn = _registry.get(fallback)
                if fb_fn:
                    try:
                        fb_result = fb_fn(*args, **kwargs)
                        return fb_result
                    except Exception:
                        pass

            # ── Return structured error (never bare exception) ────────────────
            return {
                "__aos_error": True,
                "type": "TOOL_FAILURE",
                "message": f"Tool '{tool_name}' failed after {retries + 1} attempts: "
                f"{last_error}",
                "tool": tool_name,
                "attempts": retries + 1,
                "last_error": str(last_error),
            }

        # ── Attach metadata to wrapper ────────────────────────────────────────
        wrapper._aos_tool = True
        wrapper._aos_meta = meta
        wrapper._aos_cache = _cache_store
        wrapper._aos_circuit_breaker = _cb
        wrapper._aos_rate_limiter = _limiter

        # ── Register in global tool registry ──────────────────────────────────
        _registry.register(wrapper, meta)

        return wrapper

    # ── Handle both @infrarely.tool and @infrarely.tool(...) ──────────────────────────────
    if fn is not None:
        return _wrap(fn)
    return _wrap


# ═══════════════════════════════════════════════════════════════════════════════
# CAPABILITY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CapabilityMeta:
    """Metadata for a registered capability."""

    name: str
    description: str
    parameters: Dict[str, Any]
    workflow_fn: Optional[Callable] = None


class CapabilityRegistry:
    """Global registry of @infrarely.capability defined workflows."""

    _instance: Optional["CapabilityRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CapabilityRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._capabilities: Dict[str, CapabilityMeta] = {}
            return cls._instance

    def register(self, meta: CapabilityMeta) -> None:
        self._capabilities[meta.name] = meta

    def get(self, name: str) -> Optional[CapabilityMeta]:
        return self._capabilities.get(name)

    def list_capabilities(self) -> List[CapabilityMeta]:
        return list(self._capabilities.values())

    def names(self) -> List[str]:
        return list(self._capabilities.keys())

    def clear(self) -> None:
        self._capabilities.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._capabilities

    def __len__(self) -> int:
        return len(self._capabilities)


_cap_registry = CapabilityRegistry()


def get_capability_registry() -> CapabilityRegistry:
    return _cap_registry


# ═══════════════════════════════════════════════════════════════════════════════
# @infrarely.capability DECORATOR
# ═══════════════════════════════════════════════════════════════════════════════


def capability(
    fn: Optional[Callable] = None,
    *,
    name: str = "",
    description: str = "",
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Decorator that registers a function as an AOS capability (workflow).

    Usage:
        @infrarely.capability(name="exam_prep", description="Prepares for exams")
        def exam_prep(topic: str):
            return infrarely.workflow([...])
    """

    def _wrap(func: Callable) -> Callable:
        cap_name = name or func.__name__
        cap_desc = description or func.__doc__ or f"Capability: {cap_name}"
        cap_desc = cap_desc.strip().split("\n")[0]

        sig = inspect.signature(func)
        params = {}
        for pname, param in sig.parameters.items():
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                params[pname] = "Any"
            else:
                params[pname] = getattr(ann, "__name__", str(ann))

        meta = CapabilityMeta(
            name=cap_name,
            description=cap_desc,
            parameters=params,
            workflow_fn=func,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._aos_capability = True
        wrapper._aos_meta = meta

        _cap_registry.register(meta)

        return wrapper

    if fn is not None:
        return _wrap(fn)
    return _wrap
