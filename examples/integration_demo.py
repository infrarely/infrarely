"""
Example: Integration / Plugin System
═══════════════════════════════════════════════════════════════════════════════
Shows how to use pre-built integrations with agents.

Run: python examples/integration_demo.py
"""

import infrarely
from infrarely.integrations.slack import SlackIntegration
from infrarely.integrations.github import GitHubIntegration
from infrarely.integrations.rest_api import RestAPIIntegration

infrarely.configure(llm_provider="openai", log_file_enabled=False)


# ── Create integrations with config ──────────────────────────────────────────
# Note: These use mock implementations. Real usage requires actual API tokens.

slack = SlackIntegration(token="xoxb-mock-token")
github = GitHubIntegration(token="ghp_mock_token", owner="myorg", repo="myrepo")
api = RestAPIIntegration(base_url="https://api.example.com")

# ── Create agent with integrations using .use() ──────────────────────────────
agent = infrarely.agent("integration-bot").use(slack, github)

print("─── Integration Info ───")
print(f"Agent tools: {list(agent.tools.keys())}")

# ── List integration tools ────────────────────────────────────────────────────
print(f"\n─── Slack Tools ({len(slack.tools)}) ───")
for tool in slack.tools:
    print(f"  {tool.__name__}")

print(f"\n─── GitHub Tools ({len(github.tools)}) ───")
for tool in github.tools:
    print(f"  {tool.__name__}")

print(f"\n─── REST API Tools ({len(api.tools)}) ───")
for tool in api.tools:
    print(f"  {tool.__name__}")

# ── Configure integration ─────────────────────────────────────────────────────
print(f"\n─── Integration Configuration ───")
slack.configure(default_channel="#general")
print(f"Slack configured with default channel: #general")

# ── Get specific tool ─────────────────────────────────────────────────────────
send_msg = slack.get_tool("send_message")
if send_msg:
    print(f"\nGot tool: {send_msg.__name__}")

infrarely.shutdown()
