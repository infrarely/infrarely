"""
agent/permission_policy.py  — Gap 3: Tool Permission Policy
Controls which tools a given agent profile may call.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import infrarely.core.app_config as config
from infrarely.observability import logger

_ALWAYS_ALLOWED: Set[str] = set(getattr(config, "ALWAYS_ALLOWED_TOOLS", ["static_responder"]))


@dataclass
class AgentPolicy:
    """Permission profile for one agent identity."""
    agent_id:      str
    allowed_tools: Set[str]   = field(default_factory=set)
    denied_tools:  Set[str]   = field(default_factory=set)
    allow_all:     bool       = False   # superuser flag

    def may_call(self, tool_name: str) -> bool:
        if tool_name in _ALWAYS_ALLOWED:
            return True
        if self.allow_all:
            return True
        if tool_name in self.denied_tools:
            return False
        if self.allowed_tools:
            return tool_name in self.allowed_tools
        return True   # empty allow-list = all permitted


# Default student policy
STUDENT_POLICY = AgentPolicy(
    agent_id      = "student_agent",
    allowed_tools = {
        "static_responder",
        "assignment_tracker",
        "calendar_tool",
        "note_search",
        "study_schedule_generator",
        "exam_topic_predictor",
        "course_material_search",
        "student_profile_manager",
        "practice_question_generator",
    },
)

_POLICIES: Dict[str, AgentPolicy] = {
    "student_agent": STUDENT_POLICY,
}


class ToolPermissionPolicy:
    def __init__(self, policies: Dict[str, AgentPolicy] = None):
        self._policies = policies or _POLICIES

    def check(self, agent_id: str, tool_name: str) -> bool:
        if not getattr(config, "ENABLE_PERMISSION_POLICY", True):
            return True
        policy = self._policies.get(agent_id)
        if policy is None:
            logger.warn(f"PermissionPolicy: no policy for '{agent_id}' — denying '{tool_name}'")
            return False
        allowed = policy.may_call(tool_name)
        if not allowed:
            logger.warn(f"PermissionPolicy: '{agent_id}' denied access to '{tool_name}'")
        return allowed

    def register(self, policy: AgentPolicy):
        self._policies[policy.agent_id] = policy


_default_policy = ToolPermissionPolicy()

def check(agent_id: str, tool_name: str) -> bool:
    return _default_policy.check(agent_id, tool_name)