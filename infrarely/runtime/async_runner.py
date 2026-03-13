"""
aos/async_runner.py — Async/Concurrent Agent Execution
═══════════════════════════════════════════════════════════════════════════════
Non-blocking agent execution for web servers and async applications.

Every modern web framework (FastAPI, Sanic, Django async) uses async.
This module provides first-class async support.

Usage::

    result = await agent.arun("some task")

    results = await asyncio.gather(
        agent.arun("task 1"),
        agent.arun("task 2"),
    )

    # FastAPI integration
    @app.post("/ask")
    async def ask(question: str):
        result = await agent.arun(question)
        return {"answer": result.output}
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from infrarely.core.agent import Agent
    from infrarely.core.result import Result


# ═══════════════════════════════════════════════════════════════════════════════
# THREAD POOL — Shared pool for running sync agent code in threads
# ═══════════════════════════════════════════════════════════════════════════════

_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _get_executor(max_workers: int = 10) -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _executor
    if _executor is None or _executor._shutdown:
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="aos-async",
        )
    return _executor


def shutdown_executor() -> None:
    """Shut down the shared thread pool."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC RUN — Convert sync agent.run() to async
# ═══════════════════════════════════════════════════════════════════════════════


async def async_run(
    agent: "Agent",
    goal: str,
    *,
    context: Optional[Any] = None,
) -> "Result":
    """
    Run an agent goal asynchronously.

    Uses a thread pool to avoid blocking the event loop,
    since the sync execution engine may do I/O (LLM calls, file reads).

    Parameters
    ----------
    agent : Agent
        The agent to run.
    goal : str
        The goal to execute.
    context : Any, optional
        Context from a previous result.

    Returns
    -------
    Result
    """
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    result = await loop.run_in_executor(
        executor,
        lambda: agent.run(goal, context=context),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC GATHER — Run multiple agent tasks concurrently
# ═══════════════════════════════════════════════════════════════════════════════


async def async_gather(
    *tasks,
    return_exceptions: bool = False,
) -> List["Result"]:
    """
    Run multiple agent tasks concurrently.

    Parameters
    ----------
    *tasks : coroutine or tuple of (Agent, goal_str)
        Each task is either a coroutine (from agent.arun()) or a (agent, goal) pair.
    return_exceptions : bool
        If True, exceptions are returned as Results instead of raised.

    Returns
    -------
    List[Result]

    Example::

        results = await async_gather(
            agent.arun("task 1"),
            agent.arun("task 2"),
        )
        # or
        results = await async_gather(
            (agent1, "task 1"),
            (agent2, "task 2"),
        )
    """
    coros = []
    for task in tasks:
        if asyncio.iscoroutine(task):
            coros.append(task)
        elif isinstance(task, tuple) and len(task) == 2:
            agent, goal = task
            coros.append(async_run(agent, goal))
        else:
            raise TypeError(
                f"Expected coroutine or (Agent, goal) tuple, got {type(task)}"
            )
    results = await asyncio.gather(*coros, return_exceptions=return_exceptions)

    # Wrap exceptions in Result objects if needed
    from infrarely.core.result import _fail, ErrorType

    processed = []
    for r in results:
        if isinstance(r, Exception):
            processed.append(
                _fail(
                    ErrorType.UNKNOWN,
                    f"Async execution failed: {r}",
                )
            )
        else:
            processed.append(r)

    return processed


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC DELEGATE — Async version of agent.delegate()
# ═══════════════════════════════════════════════════════════════════════════════


async def async_delegate(
    from_agent: "Agent",
    to_agent: "Agent",
    task: str,
    *,
    context: Optional[Any] = None,
) -> "Result":
    """Async version of agent delegation."""
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    result = await loop.run_in_executor(
        executor,
        lambda: from_agent.delegate(to_agent, task, context=context),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC PARALLEL — Async version of aos.parallel()
# ═══════════════════════════════════════════════════════════════════════════════


async def async_parallel(
    tasks: List[Tuple["Agent", str]],
) -> List["Result"]:
    """
    Run tasks in parallel asynchronously.

    Parameters
    ----------
    tasks : list of (Agent, str)
        List of (agent, goal) pairs.

    Returns
    -------
    List[Result]

    Example::

        results = await async_parallel([
            (researcher, "Find facts about Mars"),
            (writer, "Write intro paragraph"),
        ])
    """
    return await async_gather(*tasks)
