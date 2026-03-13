"""
aos/integrations/gmail.py — Gmail Integration
"""

from __future__ import annotations

import base64
import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from infrarely.integrations import Integration


class GmailIntegration(Integration):
    """Gmail integration providing email tools."""

    name = "gmail"
    description = "Gmail email integration"
    required_config = ["token"]

    def __init__(self, *, token: str = "", user: str = "me", **config):
        self._token = token or config.get("token", "")
        self._user = user
        super().__init__(**config)

    def _setup(self) -> None:
        self._tools = {
            "send_email": self.send_email,
            "list_messages": self.list_messages,
            "get_message": self.get_message,
            "search_emails": self.search_emails,
        }

    def _api_call(
        self, endpoint: str, method: str = "GET", data: Optional[Dict] = None
    ) -> Any:
        url = f"https://gmail.googleapis.com/gmail/v1/users/{self._user}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            return {"error": str(e)}

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Send an email."""
        message = f"To: {to}\r\nSubject: {subject}\r\n\r\n{body}"
        raw = base64.urlsafe_b64encode(message.encode()).decode()
        return self._api_call("/messages/send", "POST", {"raw": raw})

    def list_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent messages."""
        result = self._api_call(f"/messages?maxResults={limit}")
        return result.get("messages", [])

    def get_message(self, message_id: str) -> Dict[str, Any]:
        """Get a specific message."""
        return self._api_call(f"/messages/{message_id}")

    def search_emails(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search emails."""
        result = self._api_call(f"/messages?q={query}&maxResults={limit}")
        return result.get("messages", [])
