"""
aos/streaming.py — Streaming Response Support
═══════════════════════════════════════════════════════════════════════════════
Token-by-token output streaming for real-time UIs.

Every modern UI expects streaming. This module enables:
- Synchronous streaming: ``for chunk in agent.stream(goal)``
- Async streaming: ``async for chunk in agent.astream(goal)``
- Full metadata available after stream completes

Usage::

    for chunk in agent.stream("Explain machine learning"):
        print(chunk, end="", flush=True)

    # Async
    async for chunk in agent.astream("Explain machine learning"):
        await websocket.send(chunk)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM CHUNK
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StreamChunk:
    """A single chunk in a streaming response."""

    text: str = ""
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM RESULT — Wraps a completed stream with full metadata
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StreamResult:
    """
    Result available after a stream completes.
    Contains all accumulated text and metadata.
    """

    output: str = ""
    success: bool = True
    used_llm: bool = False
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    trace_id: str = ""
    chunks_count: int = 0
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING ITERATOR — Synchronous streaming
# ═══════════════════════════════════════════════════════════════════════════════


class StreamIterator:
    """
    Synchronous iterator that yields text chunks from agent execution.

    Usage::

        stream = agent.stream("Explain quantum physics")
        for chunk in stream:
            print(chunk, end="", flush=True)

        # After iteration, access full result
        print(stream.result.confidence)
    """

    _SENTINEL = object()

    def __init__(self, goal: str, agent: Any):
        self._goal = goal
        self._agent = agent
        self._queue: queue.Queue = queue.Queue()
        self._result: Optional[StreamResult] = None
        self._accumulated: List[str] = []
        self._started = False
        self._finished = False
        self._error: Optional[str] = None

    @property
    def result(self) -> Optional[StreamResult]:
        """Get the final result after streaming completes."""
        return self._result

    def __iter__(self) -> Iterator[str]:
        if not self._started:
            self._start()

        while True:
            try:
                item = self._queue.get(timeout=60)
                if item is self._SENTINEL:
                    break
                if isinstance(item, Exception):
                    self._error = str(item)
                    break
                chunk_text = str(item)
                self._accumulated.append(chunk_text)
                yield chunk_text
            except queue.Empty:
                if self._finished:
                    break

    def _start(self) -> None:
        """Start the background execution thread."""
        self._started = True
        thread = threading.Thread(target=self._execute, daemon=True)
        thread.start()

    def _execute(self) -> None:
        """Execute the agent goal and stream results."""
        start_time = time.time()
        try:
            result = self._agent.run(self._goal)
            output = str(result.output) if result.output is not None else ""

            # Simulate streaming by chunking the output
            chunk_size = max(1, len(output) // 20) if len(output) > 20 else 1
            i = 0
            while i < len(output):
                end = min(i + chunk_size, len(output))
                chunk_text = output[i:end]
                self._queue.put(chunk_text)
                i = end
                # Small delay for natural streaming feel
                if len(output) > 50:
                    time.sleep(0.01)

            duration = (time.time() - start_time) * 1000
            self._result = StreamResult(
                output=output,
                success=result.success,
                used_llm=result.used_llm,
                confidence=result.confidence,
                sources=result.sources,
                duration_ms=duration,
                trace_id=result.trace_id,
                chunks_count=len(self._accumulated)
                + (len(output) // max(1, chunk_size)),
            )

        except Exception as e:
            self._queue.put(e)
            self._result = StreamResult(
                output="".join(self._accumulated),
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
        finally:
            self._finished = True
            self._queue.put(self._SENTINEL)


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC STREAMING ITERATOR
# ═══════════════════════════════════════════════════════════════════════════════


class AsyncStreamIterator:
    """
    Async iterator that yields text chunks from agent execution.

    Usage::

        async for chunk in agent.astream("Explain quantum physics"):
            await websocket.send(chunk)
    """

    _SENTINEL = object()

    def __init__(self, goal: str, agent: Any):
        self._goal = goal
        self._agent = agent
        self._result: Optional[StreamResult] = None
        self._accumulated: List[str] = []
        self._started = False
        self._finished = False
        self._queue: queue.Queue = queue.Queue()

    @property
    def result(self) -> Optional[StreamResult]:
        return self._result

    def __aiter__(self) -> "AsyncStreamIterator":
        return self

    async def __anext__(self) -> str:
        import asyncio

        if not self._started:
            self._started = True
            loop = asyncio.get_event_loop()
            self._future = loop.run_in_executor(None, self._execute)

        while True:
            try:
                item = self._queue.get_nowait()
                if item is self._SENTINEL:
                    raise StopAsyncIteration
                if isinstance(item, Exception):
                    raise StopAsyncIteration
                chunk_text = str(item)
                self._accumulated.append(chunk_text)
                return chunk_text
            except queue.Empty:
                if self._finished:
                    raise StopAsyncIteration
                await asyncio.sleep(0.01)

    def _execute(self) -> None:
        """Execute in background thread."""
        start_time = time.time()
        try:
            result = self._agent.run(self._goal)
            output = str(result.output) if result.output is not None else ""

            chunk_size = max(1, len(output) // 20) if len(output) > 20 else 1
            i = 0
            while i < len(output):
                end = min(i + chunk_size, len(output))
                self._queue.put(output[i:end])
                i = end
                if len(output) > 50:
                    time.sleep(0.01)

            duration = (time.time() - start_time) * 1000
            self._result = StreamResult(
                output=output,
                success=result.success,
                used_llm=result.used_llm,
                confidence=result.confidence,
                sources=result.sources,
                duration_ms=duration,
                trace_id=result.trace_id,
                chunks_count=len(self._accumulated)
                + (len(output) // max(1, chunk_size)),
            )

        except Exception as e:
            self._queue.put(e)
            self._result = StreamResult(
                output="".join(self._accumulated),
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
        finally:
            self._finished = True
            self._queue.put(self._SENTINEL)
