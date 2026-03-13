"""
adaptive/skill_memory.py — Module 7: Skill Memory
═══════════════════════════════════════════════════════════════════════════════
Stores and retrieves learned procedural knowledge (skills).

Format:
  condition → action → confidence → usage_count

Example:
  IF intent=practice_questions AND topic extracted
  THEN route capability=study_session
  confidence=0.85, uses=12

Rules automatically adjust weight based on success/failure feedback.
Backed by the existing ProceduralMemory in advance_memory.py,
this module adds higher-level skill abstraction with decay and reinforcement.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from infrarely.observability import logger


@dataclass
class Skill:
    """A learned procedural skill."""

    skill_id: str
    condition: Dict[
        str, Any
    ]  # e.g. {"intent": "practice_questions", "has_topic": True}
    action: str  # e.g. "capability:study_session"
    confidence: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total else 0.5

    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if context satisfies this skill's condition."""
        for key, expected in self.condition.items():
            actual = context.get(key)
            if isinstance(expected, bool):
                if bool(actual) != expected:
                    return False
            elif actual != expected:
                return False
        return True


class SkillMemory:
    """
    Stores learned skills (condition→action rules) with reinforcement learning.

    Design:
      • Skills are created from successful execution patterns.
      • Confidence increases on success, decreases on failure.
      • Low-confidence skills are automatically pruned.
      • Maximum 200 skills enforced.
    """

    MAX_SKILLS = 200
    CONFIDENCE_INCREMENT = 0.05
    CONFIDENCE_DECREMENT = 0.10
    PRUNE_THRESHOLD = 0.15

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._next_id = 1
        self._conflict_count = 0

    # ── Skill creation ────────────────────────────────────────────────────────
    def learn(
        self,
        condition: Dict[str, Any],
        action: str,
        initial_confidence: float = 0.5,
    ) -> Skill:
        """
        Learn a new skill or reinforce an existing one.
        If a matching skill already exists, reinforce it.
        """
        # Check for existing matching skill
        for skill in self._skills.values():
            if skill.condition == condition and skill.action == action:
                skill.confidence = min(
                    skill.confidence + self.CONFIDENCE_INCREMENT, 1.0
                )
                skill.usage_count += 1
                skill.last_used = time.time()
                logger.debug(
                    f"SkillMemory: reinforced '{skill.skill_id}' → conf={skill.confidence:.2f}"
                )
                return skill

        # Create new skill
        if len(self._skills) >= self.MAX_SKILLS:
            self._prune()

        skill_id = f"skill_{self._next_id:04d}"
        self._next_id += 1
        skill = Skill(
            skill_id=skill_id,
            condition=condition,
            action=action,
            confidence=initial_confidence,
        )
        self._skills[skill_id] = skill
        logger.debug(
            f"SkillMemory: learned '{skill_id}' — {condition} → {action} (conf={initial_confidence})"
        )
        return skill

    # ── Skill query ───────────────────────────────────────────────────────────
    def best_action(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Find the best matching skill for a given context.
        When multiple skills match (conflict), picks highest
        confidence × (1 + success_rate) and logs the conflict.
        Returns action string or None.
        """
        candidates = [
            s
            for s in self._skills.values()
            if s.matches(context) and s.confidence >= 0.5
        ]
        if not candidates:
            return None

        # Conflict detection: multiple skills with different actions
        unique_actions = {s.action for s in candidates}
        if len(unique_actions) > 1:
            ranked = sorted(
                candidates,
                key=lambda s: s.confidence * (1 + s.success_rate),
                reverse=True,
            )
            logger.info(
                f"SkillMemory: CONFLICT — {len(candidates)} skills match, "
                f"{len(unique_actions)} distinct actions. "
                f"Winner: {ranked[0].skill_id} ({ranked[0].action}, "
                f"conf={ranked[0].confidence:.2f}), "
                f"runners-up: {[s.skill_id for s in ranked[1:3]]}"
            )
            self._conflict_count += 1

        best = max(candidates, key=lambda s: s.confidence * (1 + s.success_rate))
        best.usage_count += 1
        best.last_used = time.time()
        return best.action

    # ── Feedback ──────────────────────────────────────────────────────────────
    def record_outcome(self, action: str, success: bool) -> None:
        """Record success/failure for all skills that recommended this action."""
        for skill in self._skills.values():
            if skill.action == action and skill.last_used > time.time() - 60:
                if success:
                    skill.success_count += 1
                    skill.confidence = min(
                        skill.confidence + self.CONFIDENCE_INCREMENT, 1.0
                    )
                else:
                    skill.failure_count += 1
                    skill.confidence = max(
                        skill.confidence - self.CONFIDENCE_DECREMENT, 0.0
                    )

    # ── Pruning ───────────────────────────────────────────────────────────────
    def _prune(self):
        """Remove lowest-confidence skills to make room."""
        before = len(self._skills)
        self._skills = {
            sid: s
            for sid, s in self._skills.items()
            if s.confidence >= self.PRUNE_THRESHOLD
        }
        # If still full, remove oldest by last_used
        if len(self._skills) >= self.MAX_SKILLS:
            sorted_skills = sorted(self._skills.items(), key=lambda x: x[1].last_used)
            to_remove = len(self._skills) - self.MAX_SKILLS + 10
            for sid, _ in sorted_skills[:to_remove]:
                del self._skills[sid]
        pruned = before - len(self._skills)
        if pruned:
            logger.debug(f"SkillMemory: pruned {pruned} low-confidence skills")

    # ── Query ─────────────────────────────────────────────────────────────────
    def all_skills(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": s.skill_id,
                "condition": s.condition,
                "action": s.action,
                "confidence": round(s.confidence, 3),
                "uses": s.usage_count,
                "success_rate": round(s.success_rate, 3),
            }
            for s in sorted(self._skills.values(), key=lambda x: -x.confidence)
        ]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_skills": len(self._skills),
            "max_skills": self.MAX_SKILLS,
            "conflicts_resolved": self._conflict_count,
            "avg_confidence": (
                round(
                    sum(s.confidence for s in self._skills.values())
                    / len(self._skills),
                    3,
                )
                if self._skills
                else 0.0
            ),
            "top_skills": self.all_skills()[:10],
        }
