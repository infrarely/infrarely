"""
infrarely/integrations/github.py — GitHub Integration
═══════════════════════════════════════════════════════════════════════════════
Pre-built GitHub tools for InfraRely agents.

Usage::

    from infrarely.integrations import github

    gh = github.GitHubIntegration(token="ghp_...")
    agent = infrarely.agent("bot", tools=[
        gh.create_issue,
        gh.list_issues,
        gh.create_pull_request,
    ])
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from infrarely.integrations import Integration


class GitHubIntegration(Integration):
    """GitHub integration providing repository and issue management tools."""

    name = "github"
    description = "GitHub repository integration"
    required_config = ["token"]

    def __init__(self, *, token: str = "", owner: str = "", repo: str = "", **config):
        self._token = token or config.get("token", "")
        self._owner = owner or config.get("owner", "")
        self._repo = repo or config.get("repo", "")
        super().__init__(**config)

    def _setup(self) -> None:
        self._tools = {
            "create_issue": self.create_issue,
            "list_issues": self.list_issues,
            "get_issue": self.get_issue,
            "create_pull_request": self.create_pull_request,
            "list_pull_requests": self.list_pull_requests,
            "get_repo_info": self.get_repo_info,
        }

    def _api_call(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
    ) -> Any:
        """Make a GitHub API call."""
        url = f"https://api.github.com{endpoint}"
        headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        }
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            return {"error": str(e)}

    def create_issue(
        self,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
        owner: str = "",
        repo: str = "",
    ) -> Dict[str, Any]:
        """Create a GitHub issue."""
        o = owner or self._owner
        r = repo or self._repo
        data: Dict[str, Any] = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        return self._api_call(f"/repos/{o}/{r}/issues", "POST", data)

    def list_issues(
        self,
        state: str = "open",
        limit: int = 10,
        owner: str = "",
        repo: str = "",
    ) -> List[Dict[str, Any]]:
        """List repository issues."""
        o = owner or self._owner
        r = repo or self._repo
        result = self._api_call(f"/repos/{o}/{r}/issues?state={state}&per_page={limit}")
        return result if isinstance(result, list) else []

    def get_issue(
        self,
        issue_number: int,
        owner: str = "",
        repo: str = "",
    ) -> Dict[str, Any]:
        """Get details of a specific issue."""
        o = owner or self._owner
        r = repo or self._repo
        return self._api_call(f"/repos/{o}/{r}/issues/{issue_number}")

    def create_pull_request(
        self,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
        owner: str = "",
        repo: str = "",
    ) -> Dict[str, Any]:
        """Create a pull request."""
        o = owner or self._owner
        r = repo or self._repo
        return self._api_call(
            f"/repos/{o}/{r}/pulls",
            "POST",
            {
                "title": title,
                "head": head,
                "base": base,
                "body": body,
            },
        )

    def list_pull_requests(
        self,
        state: str = "open",
        limit: int = 10,
        owner: str = "",
        repo: str = "",
    ) -> List[Dict[str, Any]]:
        """List pull requests."""
        o = owner or self._owner
        r = repo or self._repo
        result = self._api_call(f"/repos/{o}/{r}/pulls?state={state}&per_page={limit}")
        return result if isinstance(result, list) else []

    def get_repo_info(self, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Get repository information."""
        o = owner or self._owner
        r = repo or self._repo
        return self._api_call(f"/repos/{o}/{r}")
