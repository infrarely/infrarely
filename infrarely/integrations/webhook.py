"""
aos/integrations/webhook.py — Webhook Integration
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
import hashlib
import hmac
import time
from typing import Any, Callable, Dict, List, Optional

from infrarely.integrations import Integration


class WebhookIntegration(Integration):
    """Webhook integration for sending and receiving webhook events."""

    name = "webhook"
    description = "Webhook send/receive integration"

    def __init__(self, *, secret: str = "", **config):
        self._secret = secret
        self._handlers: Dict[str, List[Callable]] = {}
        super().__init__(**config)

    def _setup(self) -> None:
        self._tools = {
            "send_webhook": self.send_webhook,
            "verify_signature": self.verify_signature,
        }

    def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send a webhook event to a URL."""
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)

        body = json.dumps(payload).encode()

        # Add signature if secret is configured
        if self._secret:
            sig = hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()
            req_headers["X-AOS-Signature"] = f"sha256={sig}"

        req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return {
                    "status": resp.status,
                    "body": resp.read().decode(),
                    "success": True,
                }
        except urllib.error.URLError as e:
            return {"success": False, "error": str(e)}

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify a webhook signature."""
        if not self._secret:
            return True
        expected = hmac.new(self._secret.encode(), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)

    def on_event(self, event_type: str, handler: Callable) -> None:
        """Register a handler for webhook events."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def process_event(self, event_type: str, payload: Dict[str, Any]) -> List[Any]:
        """Process an incoming webhook event."""
        results = []
        for handler in self._handlers.get(event_type, []):
            try:
                results.append(handler(payload))
            except Exception as e:
                results.append({"error": str(e)})
        return results
