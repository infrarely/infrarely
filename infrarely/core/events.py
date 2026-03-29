"""
infrarely/events.py — Webhook & Event-Driven Architecture
═══════════════════════════════════════════════════════════════════════════════
SCALE GAP 6: Event triggers, webhooks, and scheduled tasks.

Usage:
    import infrarely

    agent = infrarely.Agent("notifier")

    @agent.on_event("ticket_created")
    def handle_ticket(event):
        return agent.run(f"Triage ticket: {event['title']}")

    @agent.on_webhook("/github")
    def handle_pr(payload):
        return agent.run(f"Review PR: {payload['pull_request']['title']}")

    @agent.on_schedule("0 9 * * *")  # 9 AM daily
    def morning_report():
        return agent.run("Generate daily standup summary")

    # Fire an event programmatically
    aos.event_bus.emit("ticket_created", {"title": "Login broken"})

Architecture:
    EventBus         — Publish/subscribe event system
    WebhookRegistry  — Maps URL paths to agent handlers
    ScheduleEntry    — Cron-like schedule definitions
    ScheduleManager  — Runs scheduled tasks
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT BUS — Publish/Subscribe
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Event:
    """An event that flows through the event bus."""

    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            import uuid

            object.__setattr__(self, "event_id", uuid.uuid4().hex[:12])


class EventBus:
    """
    Publish/subscribe event bus for inter-agent communication.

    Example
    -------
        bus = EventBus()
        bus.on("ticket_created", lambda e: print(e.data))
        bus.emit("ticket_created", {"title": "Bug!"})
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._global_handlers: List[Callable] = []
        self._history: List[Event] = []
        self._max_history = 1000
        self._lock = threading.Lock()

    def on(self, event_name: str, handler: Callable) -> Callable:
        """
        Register a handler for an event.

        Can be used as a decorator:
            @bus.on("ticket_created")
            def handle(event):
                ...
        """
        with self._lock:
            self._handlers.setdefault(event_name, []).append(handler)
        return handler

    def on_any(self, handler: Callable) -> Callable:
        """Register a handler that fires on ALL events."""
        with self._lock:
            self._global_handlers.append(handler)
        return handler

    def off(self, event_name: str, handler: Optional[Callable] = None) -> None:
        """
        Unregister handler(s) for an event.

        If handler is None, removes ALL handlers for that event.
        """
        with self._lock:
            if handler is None:
                self._handlers.pop(event_name, None)
            elif event_name in self._handlers:
                self._handlers[event_name] = [
                    h for h in self._handlers[event_name] if h is not handler
                ]

    def emit(
        self,
        event_name: str,
        data: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> Event:
        """
        Emit an event. All registered handlers are called synchronously.

        Parameters
        ----------
        event_name : str
            The event to fire.
        data : dict, optional
            Event payload.
        source : str
            Which agent/component emitted the event.

        Returns
        -------
        Event
            The emitted event object.
        """
        event = Event(name=event_name, data=data or {}, source=source)

        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

            handlers = list(self._handlers.get(event_name, []))
            global_handlers = list(self._global_handlers)

        # Call handlers outside the lock
        for handler in handlers + global_handlers:
            try:
                handler(event)
            except Exception:
                pass  # Handlers should not crash the emitter

        return event

    def handlers(self, event_name: str) -> List[Callable]:
        """Get handlers for an event."""
        with self._lock:
            return list(self._handlers.get(event_name, []))

    def events(self) -> List[str]:
        """List all event names with registered handlers."""
        with self._lock:
            return list(self._handlers.keys())

    def history(self, event_name: Optional[str] = None, limit: int = 50) -> List[Event]:
        """Get recent event history, optionally filtered by name."""
        with self._lock:
            events = list(self._history)
        if event_name:
            events = [e for e in events if e.name == event_name]
        return events[-limit:]

    def reset(self) -> None:
        """Reset the event bus. USE ONLY IN TESTS."""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()
            self._history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# WEBHOOK REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WebhookRoute:
    """A registered webhook endpoint."""

    path: str
    handler: Callable
    agent_name: str = ""
    method: str = "POST"
    description: str = ""
    created_at: float = field(default_factory=time.time)


class WebhookRegistry:
    """
    Maps URL paths to agent webhook handlers.

    This provides the registration layer. The actual HTTP server
    is handled by the host framework (Flask, FastAPI, etc.).
    InfraRely provides the routing and dispatch.
    """

    def __init__(self):
        self._routes: Dict[str, WebhookRoute] = {}
        self._lock = threading.Lock()

    def register(
        self,
        path: str,
        handler: Callable,
        agent_name: str = "",
        method: str = "POST",
        description: str = "",
    ) -> WebhookRoute:
        """Register a webhook handler for a URL path."""
        route = WebhookRoute(
            path=path,
            handler=handler,
            agent_name=agent_name,
            method=method.upper(),
            description=description,
        )
        with self._lock:
            self._routes[path] = route
        return route

    def dispatch(self, path: str, payload: Dict[str, Any]) -> Any:
        """
        Dispatch an incoming webhook to the registered handler.

        Parameters
        ----------
        path : str
            The URL path (e.g., "/github").
        payload : dict
            The webhook payload.

        Returns
        -------
        Any
            The handler's return value.

        Raises
        ------
        KeyError
            If no handler is registered for the path.
        """
        with self._lock:
            route = self._routes.get(path)
        if route is None:
            raise KeyError(f"No webhook handler registered for path: {path!r}")
        return route.handler(payload)

    def routes(self) -> List[WebhookRoute]:
        """List all registered webhook routes."""
        with self._lock:
            return list(self._routes.values())

    def unregister(self, path: str) -> None:
        """Remove a webhook handler."""
        with self._lock:
            self._routes.pop(path, None)

    def reset(self) -> None:
        """Reset all routes. USE ONLY IN TESTS."""
        with self._lock:
            self._routes.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULE MANAGER — Cron-like scheduling
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScheduleEntry:
    """A scheduled task definition."""

    cron_expr: str
    handler: Callable
    agent_name: str = ""
    description: str = ""
    last_run: float = 0.0
    run_count: int = 0
    enabled: bool = True
    created_at: float = field(default_factory=time.time)


class CronParser:
    """
    Minimal cron expression parser.

    Supports: minute hour day_of_month month day_of_week
    Special values: * (any), */N (every N), N (exact), N-M (range)
    """

    @staticmethod
    def matches(cron_expr: str, t: Optional[time.struct_time] = None) -> bool:
        """Check if a cron expression matches the given time (or now)."""
        if t is None:
            t = time.localtime()

        parts = cron_expr.strip().split()
        if len(parts) != 5:
            return False

        fields = [t.tm_min, t.tm_hour, t.tm_mday, t.tm_mon, t.tm_wday]

        for expr, value in zip(parts, fields):
            if not CronParser._field_matches(expr, value):
                return False
        return True

    @staticmethod
    def _field_matches(expr: str, value: int) -> bool:
        """Check if a single cron field matches a value."""
        if expr == "*":
            return True
        if expr.startswith("*/"):
            try:
                step = int(expr[2:])
                return value % step == 0
            except ValueError:
                return False
        if "-" in expr:
            try:
                lo, hi = expr.split("-", 1)
                return int(lo) <= value <= int(hi)
            except ValueError:
                return False
        if "," in expr:
            try:
                return value in {int(v) for v in expr.split(",")}
            except ValueError:
                return False
        try:
            return value == int(expr)
        except ValueError:
            return False


class ScheduleManager:
    """
    Manages cron-like scheduled tasks.

    The manager does NOT run a background loop by default.
    Call ``tick()`` from your own loop, or call ``start()``
    to run a background thread that checks every 60 seconds.
    """

    def __init__(self):
        self._schedules: Dict[str, ScheduleEntry] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add(
        self,
        name: str,
        cron_expr: str,
        handler: Callable,
        agent_name: str = "",
        description: str = "",
    ) -> ScheduleEntry:
        """
        Add a scheduled task.

        Parameters
        ----------
        name : str
            Unique schedule name.
        cron_expr : str
            Cron expression: "minute hour day month weekday"
        handler : callable
            Function to call when the schedule triggers.
        """
        entry = ScheduleEntry(
            cron_expr=cron_expr,
            handler=handler,
            agent_name=agent_name,
            description=description,
        )
        with self._lock:
            self._schedules[name] = entry
        return entry

    def remove(self, name: str) -> None:
        """Remove a schedule."""
        with self._lock:
            self._schedules.pop(name, None)

    def tick(self) -> List[str]:
        """
        Check all schedules and run any that match the current time.

        Returns a list of schedule names that were triggered.
        """
        now = time.localtime()
        triggered = []

        with self._lock:
            entries = list(self._schedules.items())

        for name, entry in entries:
            if not entry.enabled:
                continue
            # Don't run more than once per minute
            if time.time() - entry.last_run < 59:
                continue
            if CronParser.matches(entry.cron_expr, now):
                try:
                    entry.handler()
                except Exception:
                    pass
                entry.last_run = time.time()
                entry.run_count += 1
                triggered.append(name)

        return triggered

    def schedules(self) -> List[Dict[str, Any]]:
        """List all schedules."""
        with self._lock:
            result = []
            for name, entry in self._schedules.items():
                result.append(
                    {
                        "name": name,
                        "cron": entry.cron_expr,
                        "agent": entry.agent_name,
                        "enabled": entry.enabled,
                        "run_count": entry.run_count,
                        "last_run": entry.last_run,
                    }
                )
            return result

    def start(self, interval: float = 60.0) -> None:
        """Start background schedule checking."""
        if self._running:
            return
        self._running = True

        def loop():
            while self._running:
                try:
                    self.tick()
                except Exception:
                    pass
                time.sleep(interval)

        self._thread = threading.Thread(
            target=loop, daemon=True, name="aos-schedule-manager"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background schedule checking."""
        self._running = False

    def reset(self) -> None:
        """Reset all schedules. USE ONLY IN TESTS."""
        self.stop()
        with self._lock:
            self._schedules.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

_event_bus: Optional[EventBus] = None
_webhook_registry: Optional[WebhookRegistry] = None
_schedule_manager: Optional[ScheduleManager] = None


def get_event_bus() -> EventBus:
    """Get module-level EventBus singleton."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def get_webhook_registry() -> WebhookRegistry:
    """Get module-level WebhookRegistry singleton."""
    global _webhook_registry
    if _webhook_registry is None:
        _webhook_registry = WebhookRegistry()
    return _webhook_registry


def get_schedule_manager() -> ScheduleManager:
    """Get module-level ScheduleManager singleton."""
    global _schedule_manager
    if _schedule_manager is None:
        _schedule_manager = ScheduleManager()
    return _schedule_manager
