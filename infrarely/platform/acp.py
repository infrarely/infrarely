"""
aos/acp.py — Agent Collaboration Protocol (ACP)
═══════════════════════════════════════════════════════════════════════════════
A standard protocol that allows AOS agents to collaborate with agents built
on other frameworks (LangChain, CrewAI, AutoGPT, custom REST agents).

AOS becomes infrastructure — like TCP/IP for agents. You don't fight it,
you run on top of it.

Usage (outbound — AOS agent calls external agent)::

    result = my_agent.delegate_external(
        endpoint="http://other-service/agent",
        protocol="ACP/1.0",
        task="Research this topic",
        context=my_context,
    )

Usage (inbound — expose AOS agent as ACP endpoint)::

    server = ACPServer(my_agent, host="0.0.0.0", port=9000)
    server.start()  # Other framework agents can now POST to /acp/v1/task

Protocol wire format (ACP/1.0)::

    POST /acp/v1/task
    Content-Type: application/json

    {
        "protocol": "ACP/1.0",
        "message_id": "...",
        "sender": {"name": "langchain-researcher", "framework": "langchain"},
        "task": "Research quantum computing",
        "context": {"previous_output": "..."},
        "timeout_ms": 30000,
        "capabilities_required": ["research", "summarize"],
        "metadata": {}
    }

    Response:
    {
        "protocol": "ACP/1.0",
        "message_id": "...",
        "in_reply_to": "...",
        "sender": {"name": "aos-tutor", "framework": "aos"},
        "status": "success",
        "output": "...",
        "confidence": 0.95,
        "sources": [],
        "duration_ms": 123.4,
        "metadata": {}
    }
"""

from __future__ import annotations

import enum
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urljoin


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

ACP_VERSION = "ACP/1.0"
ACP_CONTENT_TYPE = "application/json"
ACP_PATH = "/acp/v1/task"
ACP_HEALTH_PATH = "/acp/v1/health"
ACP_CAPABILITIES_PATH = "/acp/v1/capabilities"


class ACPStatus(enum.Enum):
    """Status of an ACP response."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    PARTIAL = "partial"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


class ACPFramework(enum.Enum):
    """Known agent frameworks for interoperability."""

    AOS = "aos"
    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    AUTOGPT = "autogpt"
    AUTOGEN = "autogen"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# ACP IDENTITY — who is sending/receiving
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ACPIdentity:
    """Identifies a participant in an ACP exchange."""

    name: str = ""
    framework: str = "unknown"
    version: str = ""
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "framework": self.framework}
        if self.version:
            d["version"] = self.version
        if self.capabilities:
            d["capabilities"] = self.capabilities
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACPIdentity":
        return cls(
            name=data.get("name", ""),
            framework=data.get("framework", "unknown"),
            version=data.get("version", ""),
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ACP MESSAGE — the wire-format request
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ACPMessage:
    """
    Standard ACP request message.

    This is what gets sent over the wire when one agent delegates to another.
    """

    protocol: str = ACP_VERSION
    message_id: str = field(default_factory=lambda: f"acp_{uuid.uuid4().hex[:16]}")
    sender: ACPIdentity = field(default_factory=ACPIdentity)
    task: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: float = 30_000
    capabilities_required: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "message_id": self.message_id,
            "sender": self.sender.to_dict(),
            "task": self.task,
            "context": self.context,
            "timeout_ms": self.timeout_ms,
            "capabilities_required": self.capabilities_required,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACPMessage":
        sender_data = data.get("sender", {})
        sender = (
            ACPIdentity.from_dict(sender_data)
            if isinstance(sender_data, dict)
            else ACPIdentity()
        )
        return cls(
            protocol=data.get("protocol", ACP_VERSION),
            message_id=data.get("message_id", f"acp_{uuid.uuid4().hex[:16]}"),
            sender=sender,
            task=data.get("task", ""),
            context=data.get("context", {}),
            timeout_ms=data.get("timeout_ms", 30_000),
            capabilities_required=data.get("capabilities_required", []),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ACPMessage":
        return cls.from_dict(json.loads(raw))

    def validate(self):
        """Validate the message. Returns empty list (truthy) if valid, list of errors otherwise."""
        errors: List[str] = []
        if not self.protocol:
            errors.append("Missing protocol version")
        elif not self.protocol.startswith("ACP/"):
            errors.append(f"Invalid protocol: {self.protocol!r} (expected ACP/x.y)")
        if not self.task:
            errors.append("Missing task")
        if not self.message_id:
            errors.append("Missing message_id")
        if errors:
            return errors

        # Return an empty list that evaluates as truthy
        class _ValidResult(list):
            def __bool__(self):
                return True

        return _ValidResult()

    @property
    def is_valid(self) -> bool:
        """Return True if the message validates OK."""
        result = self.validate()
        return isinstance(result, list) and len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ACP RESPONSE — the wire-format reply
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ACPResponse:
    """
    Standard ACP response message.
    """

    protocol: str = ACP_VERSION
    message_id: str = field(default_factory=lambda: f"acp_{uuid.uuid4().hex[:16]}")
    in_reply_to: str = ""
    sender: ACPIdentity = field(default_factory=ACPIdentity)
    status: str = "success"  # ACPStatus.value
    output: Any = None
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error_message: str = ""
    error_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Normalize status to ACPStatus enum
        if isinstance(self.status, str):
            for member in ACPStatus:
                if member.value == self.status:
                    self.status = member
                    break

    @property
    def success(self) -> bool:
        """Return True if status is success."""
        s = self.status
        if hasattr(s, "value"):
            s = s.value
        return str(s).lower() == "success"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "protocol": self.protocol,
            "message_id": self.message_id,
            "in_reply_to": self.in_reply_to,
            "sender": self.sender.to_dict(),
            "status": (
                self.status.value if isinstance(self.status, ACPStatus) else self.status
            ),
            "output": self.output,
            "confidence": self.confidence,
            "sources": self.sources,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
        if self.error_message:
            d["error_message"] = self.error_message
        if self.error_type:
            d["error_type"] = self.error_type
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACPResponse":
        sender_data = data.get("sender", {})
        sender = (
            ACPIdentity.from_dict(sender_data)
            if isinstance(sender_data, dict)
            else ACPIdentity()
        )
        return cls(
            protocol=data.get("protocol", ACP_VERSION),
            message_id=data.get("message_id", ""),
            in_reply_to=data.get("in_reply_to", ""),
            sender=sender,
            status=data.get("status", "success"),
            output=data.get("output"),
            confidence=data.get("confidence", 1.0),
            sources=data.get("sources", []),
            duration_ms=data.get("duration_ms", 0.0),
            error_message=data.get("error_message", ""),
            error_type=data.get("error_type", ""),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ACPResponse":
        return cls.from_dict(json.loads(raw))


# ═══════════════════════════════════════════════════════════════════════════════
# ACP ENDPOINT — represents a remote agent
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ACPEndpoint:
    """
    Represents an external agent that speaks ACP.

    Attributes
    ----------
    url : str
        Base URL of the remote agent (e.g. "http://other-service:8080").
    name : str
        Friendly name for the remote agent.
    framework : str
        What framework the remote agent runs on.
    protocol : str
        Protocol version (default "ACP/1.0").
    auth_token : str
        Optional bearer token for authentication.
    timeout_ms : float
        Default timeout for requests to this endpoint.
    capabilities : list[str]
        What this endpoint can do.
    healthy : bool
        Last known health status.
    last_seen : float
        Timestamp of last successful interaction.
    """

    url: str = ""
    name: str = ""
    framework: str = "unknown"
    protocol: str = ACP_VERSION
    auth_token: str = ""
    timeout_ms: float = 30_000
    capabilities: List[str] = field(default_factory=list)
    healthy: bool = True
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def task_url(self) -> str:
        """Full URL for the task endpoint."""
        base = self.url.rstrip("/")
        return f"{base}{ACP_PATH}"

    @property
    def health_url(self) -> str:
        """Full URL for the health endpoint."""
        base = self.url.rstrip("/")
        return f"{base}{ACP_HEALTH_PATH}"

    @property
    def capabilities_url(self) -> str:
        """Full URL for the capabilities endpoint."""
        base = self.url.rstrip("/")
        return f"{base}{ACP_CAPABILITIES_PATH}"


# ═══════════════════════════════════════════════════════════════════════════════
# ACP TRANSPORT — HTTP transport layer
# ═══════════════════════════════════════════════════════════════════════════════


class ACPTransport:
    """
    HTTP transport for ACP messages. Uses urllib (zero external dependencies).
    Handles serialization, auth, timeouts, and response parsing.
    """

    def __init__(self, default_timeout_ms: float = 30_000):
        self._default_timeout_ms = default_timeout_ms

    def send(
        self,
        endpoint_or_msg: Any,
        message_or_endpoint: Any = None,
        *,
        timeout_ms: Optional[float] = None,
    ) -> ACPResponse:
        """
        Send an ACP message to a remote endpoint and return the response.

        Accepts either (endpoint, message) or (message, endpoint) order.
        """
        # Detect argument order
        if isinstance(endpoint_or_msg, ACPMessage):
            message = endpoint_or_msg
            endpoint = message_or_endpoint
        else:
            endpoint = endpoint_or_msg
            message = message_or_endpoint
        t_ms = timeout_ms or message.timeout_ms or self._default_timeout_ms
        timeout_secs = t_ms / 1000.0

        url = endpoint.task_url
        body = message.to_json().encode("utf-8")

        headers: Dict[str, str] = {
            "Content-Type": ACP_CONTENT_TYPE,
            "Accept": ACP_CONTENT_TYPE,
            "X-ACP-Protocol": message.protocol,
            "X-ACP-Message-ID": message.message_id,
        }
        if endpoint.auth_token:
            headers["Authorization"] = f"Bearer {endpoint.auth_token}"

        req = Request(url, data=body, headers=headers, method="POST")

        start = time.monotonic()
        try:
            with urlopen(req, timeout=timeout_secs) as resp:
                raw_body = resp.read().decode("utf-8")
                elapsed_ms = (time.monotonic() - start) * 1000

                if resp.status != 200:
                    return ACPResponse(
                        in_reply_to=message.message_id,
                        status=ACPStatus.ERROR.value,
                        error_message=f"HTTP {resp.status}: {raw_body[:500]}",
                        duration_ms=elapsed_ms,
                    )

                try:
                    response = ACPResponse.from_json(raw_body)
                except (json.JSONDecodeError, Exception) as e:
                    return ACPResponse(
                        in_reply_to=message.message_id,
                        status=ACPStatus.ERROR.value,
                        error_message=f"Invalid JSON response: {e}",
                        duration_ms=elapsed_ms,
                    )

                response.duration_ms = elapsed_ms
                return response

        except HTTPError as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            body_text = ""
            try:
                body_text = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            return ACPResponse(
                in_reply_to=message.message_id,
                status=ACPStatus.ERROR.value,
                error_message=f"HTTP {e.code}: {body_text or e.reason}",
                error_type="HTTP_ERROR",
                duration_ms=elapsed_ms,
            )

        except URLError as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            return ACPResponse(
                in_reply_to=message.message_id,
                status=ACPStatus.ERROR.value,
                error_message=f"Connection failed: {e.reason}",
                error_type="CONNECTION_ERROR",
                duration_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                return ACPResponse(
                    in_reply_to=message.message_id,
                    status=ACPStatus.TIMEOUT.value,
                    error_message=f"Request timed out after {t_ms}ms",
                    error_type="TIMEOUT",
                    duration_ms=elapsed_ms,
                )
            return ACPResponse(
                in_reply_to=message.message_id,
                status=ACPStatus.ERROR.value,
                error_message=f"Transport error: {e}",
                error_type="TRANSPORT_ERROR",
                duration_ms=elapsed_ms,
            )

    def check_health(self, endpoint: ACPEndpoint) -> bool:
        """Ping the health endpoint. Returns True if healthy."""
        try:
            req = Request(endpoint.health_url, method="GET")
            headers: Dict[str, str] = {"Accept": ACP_CONTENT_TYPE}
            if endpoint.auth_token:
                headers["Authorization"] = f"Bearer {endpoint.auth_token}"
            for k, v in headers.items():
                req.add_header(k, v)

            with urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    endpoint.healthy = True
                    endpoint.last_seen = time.time()
                    return True
        except Exception:
            pass
        endpoint.healthy = False
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ACP EXCHANGE LOG — audit trail of all cross-framework interactions
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ACPExchange:
    """Record of one ACP request/response pair."""

    message: ACPMessage = field(default_factory=ACPMessage)
    response: Optional[ACPResponse] = None
    endpoint_url: str = ""
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    duration_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ACP REGISTRY — directory of known external agents
# ═══════════════════════════════════════════════════════════════════════════════


class ACPRegistry:
    """
    Registry of known ACP-compatible external agents.

    Thread-safe directory that agents can query to find collaborators.
    """

    def __init__(self) -> None:
        self._endpoints: Dict[str, ACPEndpoint] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        url: str,
        *,
        framework: str = "unknown",
        auth_token: str = "",
        capabilities: Optional[List[str]] = None,
        timeout_ms: float = 30_000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ACPEndpoint:
        """
        Register an external agent endpoint.

        Example::

            aos.acp_registry.register(
                "langchain-researcher",
                "http://researcher-service:8080",
                framework="langchain",
                capabilities=["research", "summarize"],
            )
        """
        endpoint = ACPEndpoint(
            url=url,
            name=name,
            framework=framework,
            auth_token=auth_token,
            timeout_ms=timeout_ms,
            capabilities=capabilities or [],
            metadata=metadata or {},
        )
        with self._lock:
            self._endpoints[name] = endpoint
        return endpoint

    def unregister(self, name: str) -> bool:
        """Remove an endpoint. Returns True if it existed."""
        with self._lock:
            return self._endpoints.pop(name, None) is not None

    def get(self, name: str) -> Optional[ACPEndpoint]:
        """Get an endpoint by name."""
        with self._lock:
            return self._endpoints.get(name)

    def find_by_capability(self, capability: str) -> List[ACPEndpoint]:
        """Find all endpoints that advertise a given capability."""
        with self._lock:
            return [
                ep for ep in self._endpoints.values() if capability in ep.capabilities
            ]

    def find_by_framework(self, framework: str) -> List[ACPEndpoint]:
        """Find all endpoints running a specific framework."""
        with self._lock:
            return [
                ep
                for ep in self._endpoints.values()
                if ep.framework.lower() == framework.lower()
            ]

    def list_all(self) -> List[ACPEndpoint]:
        """List all registered endpoints."""
        with self._lock:
            return list(self._endpoints.values())

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._endpoints)

    def clear(self) -> None:
        with self._lock:
            self._endpoints.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full registry."""
        with self._lock:
            return {
                name: {
                    "url": ep.url,
                    "framework": ep.framework,
                    "capabilities": ep.capabilities,
                    "healthy": ep.healthy,
                    "last_seen": ep.last_seen,
                }
                for name, ep in self._endpoints.items()
            }


# ═══════════════════════════════════════════════════════════════════════════════
# ACP ADAPTER — converts between AOS Result and ACP wire format
# ═══════════════════════════════════════════════════════════════════════════════


class ACPAdapter:
    """
    Converts between AOS internal types (Result) and ACP wire format.
    This is the bridge between "AOS world" and "every other framework."
    """

    @staticmethod
    def result_to_response(
        result: Any,
        *,
        in_reply_to: str = "",
        agent_name: str = "",
    ) -> ACPResponse:
        """Convert an AOS Result object to an ACPResponse."""
        # Import here to avoid circular imports
        from infrarely.core.result import Result

        if not isinstance(result, Result):
            return ACPResponse(
                in_reply_to=in_reply_to,
                status=ACPStatus.SUCCESS.value,
                output=str(result),
                sender=ACPIdentity(name=agent_name, framework="aos"),
            )

        status = ACPStatus.SUCCESS.value if result.success else ACPStatus.ERROR.value
        error_msg = ""
        error_tp = ""
        if result.error:
            error_msg = result.error.message
            error_tp = (
                result.error.type.value
                if hasattr(result.error.type, "value")
                else str(result.error.type)
            )

        return ACPResponse(
            in_reply_to=in_reply_to,
            sender=ACPIdentity(name=agent_name, framework="aos"),
            status=status,
            output=result.output,
            confidence=result.confidence,
            sources=result.sources,
            duration_ms=result.duration_ms,
            error_message=error_msg,
            error_type=error_tp,
        )

    @staticmethod
    def response_to_result(response: ACPResponse) -> Any:
        """Convert an ACPResponse to an AOS Result."""
        from infrarely.core.result import Result, Error, ErrorType

        if response.success:
            return Result(
                output=response.output,
                success=True,
                confidence=response.confidence,
                sources=response.sources,
                duration_ms=response.duration_ms,
                _agent_name=response.sender.name,
            )
        else:
            # Map error type
            err_type = ErrorType.DELEGATION_FAILED
            if response.error_type:
                try:
                    err_type = ErrorType(response.error_type)
                except (ValueError, KeyError):
                    err_type = ErrorType.DELEGATION_FAILED

            return Result(
                output=response.output,
                success=False,
                confidence=response.confidence,
                error=Error(
                    type=err_type,
                    message=response.error_message
                    or f"External agent error: {response.status}",
                    step="delegate_external",
                ),
                duration_ms=response.duration_ms,
                _agent_name=response.sender.name,
            )

    @staticmethod
    def message_to_task(message: ACPMessage) -> Dict[str, Any]:
        """Convert an incoming ACP message to an AOS-compatible task dict."""
        return {
            "goal": message.task,
            "context": message.context,
            "message_id": message.message_id,
            "sender": message.sender.to_dict(),
            "capabilities_required": message.capabilities_required,
            "timeout_ms": message.timeout_ms,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ACP SERVER — expose an AOS agent as an ACP endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class ACPServer:
    """
    Lightweight HTTP server that exposes an AOS agent as an ACP endpoint.
    Other framework agents (LangChain, CrewAI, etc.) can POST tasks to it.

    Uses only stdlib (http.server) — zero dependencies.

    Example::

        agent = infrarely.agent("tutor")
        server = ACPServer(agent, port=9000)
        server.start()  # Non-blocking, runs in background thread
        # POST http://localhost:9000/acp/v1/task  → agent.run(task)
        server.stop()
    """

    def __init__(
        self,
        agent: Any = None,
        *,
        host: str = "0.0.0.0",
        port: int = 9000,
        auth_token: str = "",
    ):
        self._agent = agent
        self._host = host
        self._port = port
        self._auth_token = auth_token
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._request_count = 0
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def is_running(self) -> bool:
        """Alias for running."""
        return self._running

    @property
    def port(self) -> int:
        return self._port

    @property
    def host(self) -> str:
        return self._host

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    @property
    def address(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _handle_task(self, message: ACPMessage) -> ACPResponse:
        """Handle an incoming ACP task request."""
        with self._lock:
            self._request_count += 1

        if self._agent is None:
            return ACPResponse(
                in_reply_to=message.message_id,
                status=ACPStatus.ERROR.value,
                error_message="No agent configured on this server",
            )

        # Validate message
        errors = message.validate()
        if isinstance(errors, list) and len(errors) > 0:
            return ACPResponse(
                in_reply_to=message.message_id,
                status=ACPStatus.ERROR.value,
                error_message=f"Invalid ACP message: {'; '.join(errors)}",
            )

        # Build context from ACP message context
        context = message.context if message.context else None

        try:
            result = self._agent.run(message.task, context=context)
            return ACPAdapter.result_to_response(
                result,
                in_reply_to=message.message_id,
                agent_name=getattr(self._agent, "name", ""),
            )
        except Exception as e:
            return ACPResponse(
                in_reply_to=message.message_id,
                status=ACPStatus.ERROR.value,
                error_message=f"Agent execution failed: {e}",
                sender=ACPIdentity(
                    name=getattr(self._agent, "name", ""),
                    framework="aos",
                ),
            )

    def _handle_health(self) -> Dict[str, Any]:
        """Handle a health check request."""
        agent_name = getattr(self._agent, "name", "") if self._agent else ""
        agent_alive = getattr(self._agent, "alive", False) if self._agent else False
        return {
            "protocol": ACP_VERSION,
            "status": "healthy" if agent_alive else "degraded",
            "agent": agent_name,
            "framework": "aos",
            "requests_handled": self._request_count,
        }

    def _handle_capabilities(self) -> Dict[str, Any]:
        """Handle a capabilities discovery request."""
        caps: List[str] = []
        if self._agent:
            tools = getattr(self._agent, "_tools", {})
            capabilities = getattr(self._agent, "_capabilities", {})
            caps = list(tools.keys()) + list(capabilities.keys())
        return {
            "protocol": ACP_VERSION,
            "agent": getattr(self._agent, "name", "") if self._agent else "",
            "framework": "aos",
            "capabilities": caps,
        }

    def start(self) -> None:
        """Start the ACP server in a background thread."""
        if self._running:
            return

        from http.server import HTTPServer, BaseHTTPRequestHandler
        import io

        server_ref = self

        class ACPHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Silence default logging

            def do_GET(self):
                if self.path == ACP_HEALTH_PATH:
                    body = json.dumps(server_ref._handle_health())
                    self.send_response(200)
                    self.send_header("Content-Type", ACP_CONTENT_TYPE)
                    self.end_headers()
                    self.wfile.write(body.encode("utf-8"))
                elif self.path == ACP_CAPABILITIES_PATH:
                    body = json.dumps(server_ref._handle_capabilities())
                    self.send_response(200)
                    self.send_header("Content-Type", ACP_CONTENT_TYPE)
                    self.end_headers()
                    self.wfile.write(body.encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path != ACP_PATH:
                    self.send_response(404)
                    self.end_headers()
                    return

                # Auth check
                if server_ref._auth_token:
                    auth = self.headers.get("Authorization", "")
                    if auth != f"Bearer {server_ref._auth_token}":
                        self.send_response(401)
                        self.end_headers()
                        resp = json.dumps({"error": "Unauthorized"})
                        self.wfile.write(resp.encode("utf-8"))
                        return

                content_length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_length).decode("utf-8")

                try:
                    message = ACPMessage.from_json(raw)
                except Exception as e:
                    self.send_response(400)
                    self.send_header("Content-Type", ACP_CONTENT_TYPE)
                    self.end_headers()
                    resp = json.dumps(
                        {
                            "protocol": ACP_VERSION,
                            "status": "error",
                            "error_message": f"Invalid request: {e}",
                        }
                    )
                    self.wfile.write(resp.encode("utf-8"))
                    return

                response = server_ref._handle_task(message)
                body = response.to_json()
                self.send_response(200)
                self.send_header("Content-Type", ACP_CONTENT_TYPE)
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

        self._server = HTTPServer((self._host, self._port), ACPHandler)
        self._server.timeout = 1.0
        self._running = True

        def serve():
            while self._running:
                self._server.handle_request()

        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the ACP server."""
        self._running = False
        if self._server:
            try:
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

_registry: Optional[ACPRegistry] = None
_transport: Optional[ACPTransport] = None
_registry_lock = threading.Lock()
_transport_lock = threading.Lock()


def get_acp_registry() -> ACPRegistry:
    """Get (or create) the global ACP registry singleton."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ACPRegistry()
        return _registry


def get_acp_transport() -> ACPTransport:
    """Get (or create) the global ACP transport singleton."""
    global _transport
    with _transport_lock:
        if _transport is None:
            _transport = ACPTransport()
        return _transport


def _reset_acp() -> None:
    """Reset all ACP singletons (for tests)."""
    global _registry, _transport
    with _registry_lock:
        if _registry is not None:
            _registry.clear()
        _registry = None
    with _transport_lock:
        _transport = None
