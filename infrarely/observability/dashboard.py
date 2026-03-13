"""
aos/dashboard.py — Live Dashboard & Health Server
═══════════════════════════════════════════════════════════════════════════════
Problem 8: Deployment is a separate problem in every other SDK.
SDK includes one-command deploy, health check, and live monitoring.

  infrarely.dashboard.start(port=8080)    → web dashboard
  aos.health.serve(port=8081)       → health endpoint for load balancers
  infrarely.metrics.export("prometheus")  → Prometheus scrape endpoint
"""

from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

from infrarely.observability.observability import get_metrics, get_logger


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH SERVER
# ═══════════════════════════════════════════════════════════════════════════════


class _HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoint."""

    agents_ref = None  # set by HealthServer

    def do_GET(self):
        metrics = get_metrics()
        health = {
            "status": "healthy",
            "uptime_seconds": metrics.uptime_seconds(),
            "total_tasks": metrics.total_tasks(),
            "failure_rate": metrics.failure_rate(),
            "llm_bypass_rate": metrics.llm_bypass_rate(),
            "timestamp": time.time(),
        }
        # Unhealthy if failure rate > 50%
        if metrics.failure_rate() > 50 and metrics.total_tasks() > 10:
            health["status"] = "degraded"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(health, indent=2).encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


class HealthServer:
    """Lightweight health check HTTP server."""

    def __init__(self):
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def serve(self, port: int = 8081) -> None:
        """Start health check endpoint in background."""
        if self._server:
            return
        self._server = HTTPServer(("0.0.0.0", port), _HealthHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        get_logger().info(f"Health server started on port {port}")

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD SERVER
# ═══════════════════════════════════════════════════════════════════════════════


_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>AOS Dashboard</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               background: #0a0a0f; color: #e0e0e0; padding: 20px; }
        h1 { color: #4fc3f7; margin-bottom: 20px; font-size: 24px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 16px; margin-bottom: 24px; }
        .card { background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 12px;
                padding: 20px; }
        .card h3 { color: #8892b0; font-size: 12px; text-transform: uppercase;
                   margin-bottom: 8px; }
        .card .value { font-size: 32px; font-weight: 700; color: #4fc3f7; }
        .card .value.success { color: #66bb6a; }
        .card .value.danger { color: #ef5350; }
        .card .value.warn { color: #ffa726; }
        .traces { background: #1a1a2e; border-radius: 12px; padding: 20px;
                  border: 1px solid #2a2a4e; }
        .traces h2 { color: #4fc3f7; margin-bottom: 12px; font-size: 18px; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; color: #8892b0; font-size: 12px; padding: 8px;
             border-bottom: 1px solid #2a2a4e; text-transform: uppercase; }
        td { padding: 8px; border-bottom: 1px solid #1a1a2e; font-size: 14px; }
        .status-ok { color: #66bb6a; } .status-fail { color: #ef5350; }
        .refresh { color: #8892b0; font-size: 12px; margin-top: 12px; }
    </style>
</head>
<body>
    <h1>AOS InfraRely</h1>
    <div class="grid" id="metrics"></div>
    <div class="traces">
        <h2>Recent Traces</h2>
        <table><thead><tr>
            <th>Trace ID</th><th>Agent</th><th>Goal</th>
            <th>Duration</th><th>Status</th>
        </tr></thead><tbody id="traces"></tbody></table>
    </div>
    <p class="refresh">Auto-refreshes every 5s</p>
    <script>
        async function refresh() {
            try {
                const r = await fetch('/api/dashboard');
                const d = await r.json();
                document.getElementById('metrics').innerHTML = [
                    card('Total Tasks', d.total_tasks, ''),
                    card('Success Rate', (100 - d.failure_rate).toFixed(1) + '%',
                         d.failure_rate < 10 ? 'success' : d.failure_rate < 30 ? 'warn' : 'danger'),
                    card('LLM Bypass', d.llm_bypass_rate.toFixed(1) + '%', 'success'),
                    card('Avg Duration', d.avg_task_duration_ms.toFixed(0) + 'ms', ''),
                    card('Hallucination Risk', d.hallucination_risk.toFixed(1) + '%',
                         d.hallucination_risk < 10 ? 'success' : 'danger'),
                    card('Uptime', formatTime(d.uptime_seconds), ''),
                ].join('');
                const tbody = document.getElementById('traces');
                tbody.innerHTML = (d.traces || []).map(t =>
                    `<tr><td>${t.trace_id}</td><td>${t.agent_name}</td>
                     <td>${t.goal.substring(0,50)}</td>
                     <td>${t.duration_ms.toFixed(0)}ms</td>
                     <td class="${t.success ? 'status-ok' : 'status-fail'}">${t.success ? 'OK' : 'FAIL'}</td></tr>`
                ).join('');
            } catch(e) {}
        }
        function card(title, value, cls) {
            return `<div class="card"><h3>${title}</h3><div class="value ${cls}">${value}</div></div>`;
        }
        function formatTime(s) {
            if (s < 60) return s.toFixed(0) + 's';
            if (s < 3600) return (s/60).toFixed(0) + 'm';
            return (s/3600).toFixed(1) + 'h';
        }
        refresh(); setInterval(refresh, 5000);
    </script>
</body>
</html>"""


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dashboard."""

    trace_store = None  # set externally

    def do_GET(self):
        if self.path == "/api/dashboard":
            self._serve_api()
        elif self.path == "/api/metrics":
            self._serve_metrics()
        elif self.path == "/metrics":
            self._serve_prometheus()
        else:
            self._serve_html()

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(_DASHBOARD_HTML.encode())

    def _serve_api(self):
        metrics = get_metrics()
        data = metrics.export(format="json")
        # Add recent traces
        if self.trace_store:
            data["traces"] = self.trace_store.list_recent(limit=20)
        else:
            data["traces"] = []
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def _serve_metrics(self):
        metrics = get_metrics()
        data = metrics.export(format="json")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def _serve_prometheus(self):
        metrics = get_metrics()
        prom = metrics.export(format="prometheus")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(prom.encode())

    def log_message(self, format, *args):
        pass


class Dashboard:
    """AOS Dashboard — live monitoring web UI."""

    def __init__(self):
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, port: int = 8080, trace_store=None) -> None:
        """Start the dashboard web server in background."""
        if self._server:
            return
        _DashboardHandler.trace_store = trace_store
        self._server = HTTPServer(("0.0.0.0", port), _DashboardHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        get_logger().info(f"Dashboard started at http://localhost:{port}")

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None


# ── Module-level singletons ──────────────────────────────────────────────────

_health = HealthServer()
_dashboard = Dashboard()


def get_health_server() -> HealthServer:
    return _health


def get_dashboard() -> Dashboard:
    return _dashboard
