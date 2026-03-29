"""
agent/execution_depth.py  — Gap 2: Execution Depth Guard
Prevents recursive tool/capability loops from running forever.
"""
from __future__ import annotations
import threading
import infrarely.core.app_config as config
from infrarely.observability import logger

_MAX_DEPTH = getattr(config, "MAX_EXECUTION_DEPTH", 8)
_depth = threading.local()


class ExecutionDepthGuard:
    """Context manager — raises DepthLimitExceeded if nesting exceeds MAX_EXECUTION_DEPTH."""

    def __enter__(self):
        current = getattr(_depth, "value", 0)
        if current >= _MAX_DEPTH:
            logger.error(f"ExecutionDepthGuard: depth {current} >= limit {_MAX_DEPTH}")
            raise DepthLimitExceeded(
                f"Execution depth {current} exceeds maximum {_MAX_DEPTH}. "
                "Possible recursive capability loop detected."
            )
        _depth.value = current + 1
        logger.debug(f"ExecutionDepthGuard: depth now {_depth.value}")
        return self

    def __exit__(self, *args):
        _depth.value = max(0, getattr(_depth, "value", 1) - 1)

    @staticmethod
    def current_depth() -> int:
        return getattr(_depth, "value", 0)


class DepthLimitExceeded(RuntimeError):
    pass