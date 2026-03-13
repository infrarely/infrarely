"""
aos/integrations/rest_api.py — Generic REST API Integration
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from infrarely.integrations import Integration


class RestAPIIntegration(Integration):
    """Generic REST API integration for any HTTP endpoint."""

    name = "rest_api"
    description = "Generic REST API integration"

    def __init__(
        self,
        *,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
        auth_token: str = "",
        **config,
    ):
        self._base_url = base_url.rstrip("/")
        self._headers = headers or {}
        if auth_token:
            self._headers["Authorization"] = f"Bearer {auth_token}"
        super().__init__(**config)

    def _setup(self) -> None:
        self._tools = {
            "get": self.get,
            "post": self.post,
            "put": self.put,
            "delete": self.delete,
        }

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Any:
        url = f"{self._base_url}{path}" if not path.startswith("http") else path
        req_headers = {**self._headers, "Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"text": content, "status": resp.status}
        except urllib.error.URLError as e:
            return {"error": str(e)}

    def get(self, path: str, headers: Optional[Dict] = None) -> Any:
        """Make a GET request."""
        return self._request("GET", path, headers=headers)

    def post(
        self, path: str, data: Optional[Dict] = None, headers: Optional[Dict] = None
    ) -> Any:
        """Make a POST request."""
        return self._request("POST", path, data=data, headers=headers)

    def put(
        self, path: str, data: Optional[Dict] = None, headers: Optional[Dict] = None
    ) -> Any:
        """Make a PUT request."""
        return self._request("PUT", path, data=data, headers=headers)

    def delete(self, path: str, headers: Optional[Dict] = None) -> Any:
        """Make a DELETE request."""
        return self._request("DELETE", path, headers=headers)
