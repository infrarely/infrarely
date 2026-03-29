"""
infrarely/integrations/notion.py — Notion Integration
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from infrarely.integrations import Integration


class NotionIntegration(Integration):
    """Notion integration providing page and database tools."""

    name = "notion"
    description = "Notion workspace integration"
    required_config = ["token"]

    def __init__(self, *, token: str = "", **config):
        self._token = token or config.get("token", "")
        super().__init__(**config)

    def _setup(self) -> None:
        self._tools = {
            "create_page": self.create_page,
            "search": self.search,
            "get_page": self.get_page,
            "update_page": self.update_page,
            "query_database": self.query_database,
        }

    def _api_call(
        self, endpoint: str, method: str = "GET", data: Optional[Dict] = None
    ) -> Any:
        url = f"https://api.notion.com/v1{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            return {"error": str(e)}

    def create_page(
        self, parent_id: str, title: str, content: str = ""
    ) -> Dict[str, Any]:
        """Create a new Notion page."""
        data = {
            "parent": {"page_id": parent_id},
            "properties": {"title": {"title": [{"text": {"content": title}}]}},
            "children": (
                [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": content}}]},
                    }
                ]
                if content
                else []
            ),
        }
        return self._api_call("/pages", "POST", data)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Notion workspace."""
        result = self._api_call("/search", "POST", {"query": query, "page_size": limit})
        return result.get("results", [])

    def get_page(self, page_id: str) -> Dict[str, Any]:
        """Get a Notion page."""
        return self._api_call(f"/pages/{page_id}")

    def update_page(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Update a Notion page."""
        return self._api_call(f"/pages/{page_id}", "PATCH", {"properties": properties})

    def query_database(
        self, database_id: str, filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Query a Notion database."""
        data: Dict[str, Any] = {}
        if filter:
            data["filter"] = filter
        result = self._api_call(f"/databases/{database_id}/query", "POST", data)
        return result.get("results", [])
