"""
adaptive/capability_discovery.py — Module 5: Capability Discovery Engine
═══════════════════════════════════════════════════════════════════════════════
Automatically identifies new capability opportunities from execution traces.

Pipeline:
  1. Trace mining — find repeated tool sequences
  2. Pattern detection — cluster frequent sequences
  3. Capability suggestion — propose new multi-step workflows

Safety:
  • Suggestions must be reviewed before activation.
  • Auto-activation disabled by default.
  • Maximum capabilities: 50.
"""

from __future__ import annotations
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from infrarely.observability import logger


@dataclass
class ToolSequence:
    """A recorded sequence of tools from one user session."""

    ts: str
    tools: List[str]
    intent: str = ""
    success: bool = True


@dataclass
class CapabilitySuggestion:
    """A suggested new capability based on trace mining."""

    name: str
    description: str
    tool_sequence: List[str]
    occurrences: int
    confidence: float
    approved: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CapabilityDiscoveryEngine:
    """
    Mines execution traces for repeated tool sequences and suggests
    new capabilities.

    Design contract:
      • Never activates capabilities automatically.
      • All suggestions are advisory with confidence scores.
      • Max 50 total capabilities enforced.
    """

    MAX_CAPABILITIES = 50
    MIN_OCCURRENCES = 3  # sequence must appear at least this many times
    MIN_SEQUENCE_LEN = 2  # minimum tools in a sequence
    MAX_SEQUENCE_LEN = 5  # maximum tools in a pattern

    def __init__(self, existing_capability_count: int = 0):
        self._sequences: List[ToolSequence] = []
        self._suggestions: List[CapabilitySuggestion] = []
        self._existing_count = existing_capability_count
        self._auto_activate = False  # disabled by default

    # ── Record tool sequence ──────────────────────────────────────────────────
    def record_sequence(
        self,
        tools: List[str],
        intent: str = "",
        success: bool = True,
    ) -> None:
        if len(tools) < self.MIN_SEQUENCE_LEN:
            return
        seq = ToolSequence(
            ts=datetime.now(timezone.utc).isoformat(),
            tools=tools[: self.MAX_SEQUENCE_LEN],
            intent=intent,
            success=success,
        )
        self._sequences.append(seq)
        if len(self._sequences) > 500:
            self._sequences = self._sequences[-250:]

        # Run discovery every 20 recorded sequences
        if len(self._sequences) % 20 == 0:
            self._mine_patterns()

    # ── Pattern mining ────────────────────────────────────────────────────────
    def _mine_patterns(self):
        """Find frequent tool sequences that aren't already capabilities."""
        # Count tool tuples
        seq_counter: Counter = Counter()
        for seq in self._sequences:
            if seq.success and len(seq.tools) >= self.MIN_SEQUENCE_LEN:
                key = tuple(seq.tools)
                seq_counter[key] += 1

        # Find frequent patterns
        new_suggestions = []
        for tool_tuple, count in seq_counter.most_common(10):
            if count < self.MIN_OCCURRENCES:
                continue

            # Check duplicate
            existing = any(
                tuple(s.tool_sequence) == tool_tuple for s in self._suggestions
            )
            if existing:
                # Update count
                for s in self._suggestions:
                    if tuple(s.tool_sequence) == tool_tuple:
                        s.occurrences = count
                continue

            # Check capacity
            total = self._existing_count + len(self._suggestions) + len(new_suggestions)
            if total >= self.MAX_CAPABILITIES:
                logger.warn(
                    f"CapabilityDiscovery: max capabilities ({self.MAX_CAPABILITIES}) reached"
                )
                break

            name = self._generate_name(list(tool_tuple))
            confidence = min(count / 10, 1.0)
            new_suggestions.append(
                CapabilitySuggestion(
                    name=name,
                    description=f"Auto-discovered workflow: {' → '.join(tool_tuple)}",
                    tool_sequence=list(tool_tuple),
                    occurrences=count,
                    confidence=round(confidence, 2),
                )
            )

        self._suggestions.extend(new_suggestions)
        if new_suggestions:
            logger.info(
                f"CapabilityDiscovery: {len(new_suggestions)} new suggestion(s)",
                names=[s.name for s in new_suggestions],
            )

    def _generate_name(self, tools: List[str]) -> str:
        """Generate a human-readable capability name from tool names."""
        # Map common tool names to short labels
        labels = {
            "exam_topic_predictor": "exam",
            "study_schedule_generator": "schedule",
            "practice_question_generator": "practice",
            "note_search": "notes",
            "assignment_tracker": "assignments",
            "calendar_tool": "calendar",
            "student_profile_manager": "profile",
            "course_material_search": "materials",
            "course_manager": "courses",
        }
        parts = [labels.get(t, t.split("_")[0]) for t in tools]
        return "_".join(parts) + "_workflow"

    # ── Query ─────────────────────────────────────────────────────────────────
    def get_suggestions(self, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "tool_sequence": s.tool_sequence,
                "occurrences": s.occurrences,
                "confidence": s.confidence,
                "approved": s.approved,
            }
            for s in self._suggestions
            if s.confidence >= min_confidence
        ]

    def approve_suggestion(self, name: str) -> bool:
        """Mark a suggestion as approved (for manual activation)."""
        for s in self._suggestions:
            if s.name == name:
                s.approved = True
                logger.info(f"CapabilityDiscovery: approved '{name}'")
                return True
        return False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "sequences_recorded": len(self._sequences),
            "suggestions": len(self._suggestions),
            "approved": sum(1 for s in self._suggestions if s.approved),
            "auto_activate": self._auto_activate,
            "max_capabilities": self.MAX_CAPABILITIES,
            "current_total": self._existing_count + len(self._suggestions),
            "top_suggestions": self.get_suggestions(min_confidence=0.3),
        }
