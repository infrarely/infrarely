"""infrarely.integrations — Third-party service integrations."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class Integration:
    """
    Base class for InfraRely integrations.

    Each integration provides a set of pre-built tools
    that are decorated with @infrarely.tool.
    """

    name: str = "base"
    description: str = ""
    version: str = "0.1.0"
    required_config: List[str] = []

    def __init__(self, **config: Any) -> None:
        self._config = config
        self._tools: Dict[str, Callable] = {}
        self._setup()

    def _setup(self) -> None:
        """Override in subclasses to register tools."""
        pass

    @property
    def tools(self) -> List[Callable]:
        """Get all tools provided by this integration."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def configure(self, **config: Any) -> "Integration":
        """Update integration configuration."""
        self._config.update(config)
        return self

    def __repr__(self) -> str:
        return f"Integration({self.name!r}, tools={list(self._tools.keys())})"


from infrarely.integrations.github import GitHubIntegration
from infrarely.integrations.gmail import GmailIntegration
from infrarely.integrations.slack import SlackIntegration
from infrarely.integrations.postgres import PostgresIntegration
from infrarely.integrations.notion import NotionIntegration
from infrarely.integrations.webhook import WebhookIntegration
from infrarely.integrations.rest_api import RestAPIIntegration

__all__ = [
    "Integration",
    "GitHubIntegration",
    "GmailIntegration",
    "SlackIntegration",
    "PostgresIntegration",
    "NotionIntegration",
    "WebhookIntegration",
    "RestAPIIntegration",
]
