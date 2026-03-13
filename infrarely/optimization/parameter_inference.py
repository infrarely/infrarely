"""
adaptive/parameter_inference.py — Module 2: Parameter Inference Engine
═══════════════════════════════════════════════════════════════════════════════
Multi-source parameter extraction with confidence scoring.

Inference sources (priority order):
  1. Query regex (existing — from intent_classifier._extract_params)
  2. Semantic memory facts (last known course, topic, etc.)
  3. Previous session context (last executed capability, recent params)
  4. Domain dictionary (common CS topics, course ID patterns)

Confidence threshold: ≥ 0.75 to auto-fill.
Below threshold → parameter left empty (tool handles gracefully).
Never fabricates values.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from infrarely.observability import logger


@dataclass
class InferredParam:
    """One inferred parameter with provenance."""

    name: str
    value: Any
    confidence: float  # 0.0–1.0
    source: str  # "query" | "memory" | "context" | "domain"


# ── Domain dictionary ─────────────────────────────────────────────────────────
_DOMAIN_TOPICS = {
    "cs": [
        "data structures",
        "algorithms",
        "complexity analysis",
        "programming paradigms",
        "software engineering",
        "databases",
        "operating systems",
        "networks",
        "machine learning",
        "computer architecture",
        "discrete math",
    ],
    "math": [
        "calculus",
        "linear algebra",
        "statistics",
        "probability",
        "differential equations",
        "number theory",
    ],
    "eng": [
        "academic writing",
        "research methods",
        "technical communication",
    ],
}

_COURSE_PATTERN = re.compile(r"\b([A-Z]{2,5}\d{2,4})\b", re.IGNORECASE)


class ParameterInferenceEngine:
    """
    Infers missing tool parameters from multiple sources.
    Never fabricates — only fills when confidence ≥ threshold.
    """

    CONFIDENCE_THRESHOLD = 0.75

    def __init__(self):
        self._last_params: Dict[str, Any] = {}  # last successful params per intent
        self._last_intent: str = ""
        self._last_capability: str = ""
        self._total_inferences = 0  # total inferred (accepted)
        self._total_rejected = 0  # below threshold or safety-blocked
        self._safety_controller = None  # set externally (Gap 6)

    # ── Main API ──────────────────────────────────────────────────────────────
    def infer(
        self,
        query: str,
        existing_params: Dict[str, Any],
        intent: str = "",
        memory_facts: Dict[str, Any] = None,
    ) -> Dict[str, InferredParam]:
        """
        Returns dict of param_name → InferredParam for missing params.
        Only includes params with confidence ≥ threshold.
        """
        inferred: Dict[str, InferredParam] = {}
        memory_facts = memory_facts or {}

        # ── Source 1: Query regex ─────────────────────────────────────────────
        if "course_id" not in existing_params:
            m = _COURSE_PATTERN.search(query.upper())
            if m:
                inferred["course_id"] = InferredParam(
                    name="course_id",
                    value=m.group(1),
                    confidence=0.95,
                    source="query",
                )

        if "topic" not in existing_params:
            topic = self._extract_topic_from_query(query)
            if topic:
                inferred["topic"] = InferredParam(
                    name="topic",
                    value=topic,
                    confidence=0.85,
                    source="query",
                )

        # ── Source 2: Semantic memory ─────────────────────────────────────────
        if "course_id" not in existing_params and "course_id" not in inferred:
            mem_course = memory_facts.get("last_course_id")
            if mem_course:
                inferred["course_id"] = InferredParam(
                    name="course_id",
                    value=mem_course,
                    confidence=0.70,
                    source="memory",
                )

        if "topic" not in existing_params and "topic" not in inferred:
            mem_topic = memory_facts.get("last_topic")
            if mem_topic:
                inferred["topic"] = InferredParam(
                    name="topic",
                    value=mem_topic,
                    confidence=0.65,
                    source="memory",
                )

        # ── Source 3: Previous session context ────────────────────────────────
        if intent and intent in self._last_params:
            prev = self._last_params[intent]
            for param_name, param_val in prev.items():
                if param_name not in existing_params and param_name not in inferred:
                    inferred[param_name] = InferredParam(
                        name=param_name,
                        value=param_val,
                        confidence=0.60,
                        source="context",
                    )

        # ── Source 4: Domain dictionary (last resort) ─────────────────────────
        if "topic" not in existing_params and "topic" not in inferred:
            domain_topic = self._match_domain_topic(query)
            if domain_topic:
                inferred["topic"] = InferredParam(
                    name="topic",
                    value=domain_topic,
                    confidence=0.75,
                    source="domain",
                )

        # ── Filter by confidence threshold + safety controller (Gap 6) ────────
        accepted: Dict[str, InferredParam] = {}
        for k, v in inferred.items():
            if v.confidence < self.CONFIDENCE_THRESHOLD:
                self._total_rejected += 1
                logger.debug(
                    f"ParameterInference: REJECTED '{k}' "
                    f"(conf={v.confidence:.2f} < {self.CONFIDENCE_THRESHOLD})"
                )
                continue
            # Safety controller check (if wired)
            if self._safety_controller:
                if not self._safety_controller.can_infer_parameter(k, v.confidence):
                    self._total_rejected += 1
                    logger.debug(
                        f"ParameterInference: SAFETY BLOCKED '{k}' "
                        f"(conf={v.confidence:.2f})"
                    )
                    continue
            accepted[k] = v
            self._total_inferences += 1

        if accepted:
            logger.debug(
                f"ParameterInference: inferred {len(accepted)} param(s)",
                params={
                    k: f"{v.value} ({v.source}, conf={v.confidence})"
                    for k, v in accepted.items()
                },
            )

        return accepted

    # ── Record successful execution for future context ────────────────────────
    def record_success(self, intent: str, params: Dict[str, Any], capability: str = ""):
        """Store last successful params per intent for future inference."""
        self._last_params[intent] = {
            k: v
            for k, v in params.items()
            if k in ("course_id", "topic", "query", "days", "difficulty")
        }
        self._last_intent = intent
        if capability:
            self._last_capability = capability

    # ── Private helpers ───────────────────────────────────────────────────────
    def _extract_topic_from_query(self, query: str) -> Optional[str]:
        """Extract topic using preposition patterns with word boundary."""
        m = re.search(r"\b(?:about|on|for)\s+([\w\s]{3,40})", query.lower())
        if m:
            topic_val = m.group(1).strip()
            # Don't treat course IDs as topics
            if not re.match(r"^[a-z]{2,4}\d{3,4}$", topic_val):
                return topic_val
        return None

    def _match_domain_topic(self, query: str) -> Optional[str]:
        """Match query words against known domain topics."""
        lower = query.lower()
        for _prefix, topics in _DOMAIN_TOPICS.items():
            for topic in topics:
                if topic in lower:
                    return topic
        return None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "last_intent": self._last_intent,
            "last_capability": self._last_capability,
            "cached_intents": list(self._last_params.keys()),
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "total_inferences": self._total_inferences,
            "total_rejected": self._total_rejected,
        }
