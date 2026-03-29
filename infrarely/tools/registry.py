"""
tools/registry.py
═══════════════════════════════════════════════════════════════════════════════
Scalable tool registry with metadata.

The StaticResponder is now registered alongside all other tools.
It handles greetings, help, and unknown queries — 0 tokens, offline-safe.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from infrarely.agent.state import ResponseType
from infrarely.tools.base_tool import BaseTool
from infrarely.observability import logger


@dataclass
class ToolMeta:
    name: str
    description: str
    response_type: ResponseType
    tags: List[str] = field(default_factory=list)
    requires_llm: bool = False


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._meta: Dict[str, ToolMeta] = {}

    def register(
        self, tool: BaseTool, meta: Optional[ToolMeta] = None
    ) -> "ToolRegistry":
        name = tool.name
        self._tools[name] = tool
        self._meta[name] = meta or ToolMeta(
            name=name,
            description=tool.description,
            response_type=ResponseType.DETERMINISTIC,
        )
        logger.debug(
            f"ToolRegistry: registered '{name}'",
            type=self._meta[name].response_type.name,
        )
        return self

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_meta(self, name: str) -> Optional[ToolMeta]:
        return self._meta.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def by_tag(self, tag: str) -> List[str]:
        return [n for n, m in self._meta.items() if tag in m.tags]

    def health_check(self) -> Dict[str, str]:
        return {
            name: "circuit_open" if tool._breaker.is_open else "ok"
            for name, tool in self._tools.items()
        }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
