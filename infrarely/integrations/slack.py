"""
aos/integrations/slack.py — Slack Integration
═══════════════════════════════════════════════════════════════════════════════
Pre-built Slack tools for AOS agents.

Usage::

    from infrarely.integrations import slack

    slack_integration = slack.SlackIntegration(token="xoxb-...")
    agent = infrarely.agent("bot", tools=[
        slack_integration.send_message,
        slack_integration.read_channel,
    ])
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from infrarely.integrations import Integration


class SlackIntegration(Integration):
    """Slack integration providing messaging tools."""

    name = "slack"
    description = "Slack workspace integration"
    required_config = ["token"]

    def __init__(self, *, token: str = "", **config):
        self._token = token or config.get("token", "")
        super().__init__(**config)

    def _setup(self) -> None:
        self._tools = {
            "send_message": self.send_message,
            "read_channel": self.read_channel,
            "list_channels": self.list_channels,
            "add_reaction": self.add_reaction,
        }

    def _api_call(self, method: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a Slack API call."""
        url = f"https://slack.com/api/{method}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            return {"ok": False, "error": str(e)}

    def send_message(self, channel: str, text: str) -> Dict[str, Any]:
        """Send a message to a Slack channel."""
        return self._api_call(
            "chat.postMessage",
            {
                "channel": channel,
                "text": text,
            },
        )

    def read_channel(self, channel: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Read recent messages from a Slack channel."""
        result = self._api_call(
            "conversations.history",
            {
                "channel": channel,
                "limit": limit,
            },
        )
        return result.get("messages", [])

    def list_channels(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List available Slack channels."""
        result = self._api_call("conversations.list", {"limit": limit})
        return result.get("channels", [])

    def add_reaction(self, channel: str, timestamp: str, emoji: str) -> Dict[str, Any]:
        """Add a reaction to a message."""
        return self._api_call(
            "reactions.add",
            {
                "channel": channel,
                "timestamp": timestamp,
                "name": emoji,
            },
        )
