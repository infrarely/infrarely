"""
aos/security.py — Prompt Injection Defense System
═══════════════════════════════════════════════════════════════════════════════
Protection against malicious inputs that try to hijack agent behavior.

Prompt injection is the #1 security concern for enterprise AI.
This module provides layered defense:

1. Pattern-based detection (known attack signatures)
2. Structural analysis (instruction override attempts)
3. Input sanitization
4. Full audit logging for compliance

Usage::

    infrarely.configure(
        security=aos.SecurityPolicy(
            prompt_injection_detection=True,
            injection_action="block",
            audit_all_inputs=True,
        )
    )
"""

from __future__ import annotations

import re
import time
import threading
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# THREAT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


class ThreatLevel(Enum):
    """Classification of detected threats."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InjectionType(Enum):
    """Types of prompt injection attacks."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_HIJACK = "role_hijack"
    DATA_EXFILTRATION = "data_exfiltration"
    JAILBREAK = "jailbreak"
    ENCODING_ATTACK = "encoding_attack"
    DELIMITER_INJECTION = "delimiter_injection"
    CONTEXT_MANIPULATION = "context_manipulation"


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DetectionResult:
    """Result of a prompt injection scan."""

    is_threat: bool = False
    threat_level: ThreatLevel = ThreatLevel.NONE
    injection_type: Optional[InjectionType] = None
    matched_patterns: List[str] = field(default_factory=list)
    confidence: float = 0.0
    details: str = ""
    input_hash: str = ""
    scan_duration_ms: float = 0.0

    @property
    def threat_type(self):
        """Alias for injection_type."""
        return self.injection_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_threat": self.is_threat,
            "threat_level": self.threat_level.value,
            "injection_type": (
                self.injection_type.value if self.injection_type else None
            ),
            "matched_patterns": self.matched_patterns,
            "confidence": self.confidence,
            "details": self.details,
            "input_hash": self.input_hash,
            "scan_duration_ms": self.scan_duration_ms,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY POLICY — Configuration for security behavior
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SecurityPolicy:
    """
    Security policy configuration.

    Usage::

        policy = SecurityPolicy(
            prompt_injection_detection=True,
            injection_action="block",
            audit_all_inputs=True,
        )
        infrarely.configure(security=policy)
    """

    prompt_injection_detection: bool = True
    injection_action: str = "block"  # "block" | "sanitize" | "warn"
    suspicious_patterns: List[str] = field(default_factory=list)
    audit_all_inputs: bool = True
    max_input_length: int = 50000
    block_known_attacks: bool = True
    custom_validators: List[Callable[[str], DetectionResult]] = field(
        default_factory=list
    )
    whitelist_patterns: List[str] = field(default_factory=list)
    on_threat_detected: Optional[Callable[[DetectionResult, str], None]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_injection_detection": self.prompt_injection_detection,
            "injection_action": self.injection_action,
            "audit_all_inputs": self.audit_all_inputs,
            "max_input_length": self.max_input_length,
            "block_known_attacks": self.block_known_attacks,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BUILT-IN ATTACK PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Each pattern: (regex, InjectionType, ThreatLevel, description)
_BUILTIN_PATTERNS: List[Tuple[str, InjectionType, ThreatLevel, str]] = [
    # ── Instruction override attempts ────────────────────────────────────────
    (
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|directives?)",
        InjectionType.INSTRUCTION_OVERRIDE,
        ThreatLevel.CRITICAL,
        "Attempt to override system instructions",
    ),
    (
        r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
        InjectionType.INSTRUCTION_OVERRIDE,
        ThreatLevel.CRITICAL,
        "Attempt to disregard prior instructions",
    ),
    (
        r"forget\s+(everything|all|your)\s*(instructions?|rules?|training|prompts?)?",
        InjectionType.INSTRUCTION_OVERRIDE,
        ThreatLevel.HIGH,
        "Attempt to make agent forget instructions",
    ),
    (
        r"new\s+instructions?\s*:",
        InjectionType.INSTRUCTION_OVERRIDE,
        ThreatLevel.HIGH,
        "Attempt to inject new instructions",
    ),
    (
        r"override\s+(system|your)\s+(prompt|instructions?|rules?)",
        InjectionType.INSTRUCTION_OVERRIDE,
        ThreatLevel.CRITICAL,
        "Direct system prompt override attempt",
    ),
    (
        r"from\s+now\s+on\s*,?\s*(you\s+are|act\s+as|pretend|behave)",
        InjectionType.INSTRUCTION_OVERRIDE,
        ThreatLevel.HIGH,
        "Attempt to redefine agent behavior",
    ),
    # ── Role hijacking ──────────────────────────────────────────────────────
    (
        r"you\s+are\s+now\s+(a|an|the)\s+",
        InjectionType.ROLE_HIJACK,
        ThreatLevel.HIGH,
        "Role reassignment attempt",
    ),
    (
        r"(pretend|act|behave|respond)\s+(as\s+if\s+)?you\s*(are|were)\s+(a|an|the)?",
        InjectionType.ROLE_HIJACK,
        ThreatLevel.MEDIUM,
        "Role-play instruction that may be benign or malicious",
    ),
    (
        r"enter(ing)?\s+(developer|admin|debug|sudo|root|god)\s*mode",
        InjectionType.ROLE_HIJACK,
        ThreatLevel.CRITICAL,
        "Attempt to enter privileged mode",
    ),
    (
        r"\[system\]|\[admin\]|\[developer\]|\[debug\]",
        InjectionType.ROLE_HIJACK,
        ThreatLevel.HIGH,
        "Fake system/admin tag injection",
    ),
    # ── Data exfiltration ───────────────────────────────────────────────────
    (
        r"(send|email|post|transmit|forward|leak|exfil)\s+(all\s+)?(data|memory|context|history|secrets?|keys?|passwords?|credentials?|tokens?)\s+(to|at)",
        InjectionType.DATA_EXFILTRATION,
        ThreatLevel.CRITICAL,
        "Data exfiltration attempt",
    ),
    (
        r"(reveal|show|display|output|print|dump)\s+(your|the|all)\s+(system\s+prompt|instructions?|configuration|secrets?|api\s*keys?|passwords?)",
        InjectionType.DATA_EXFILTRATION,
        ThreatLevel.HIGH,
        "Attempt to reveal system internals",
    ),
    (
        r"what\s+(is|are)\s+your\s+(system\s+prompt|instructions?|rules?|constraints?)",
        InjectionType.DATA_EXFILTRATION,
        ThreatLevel.MEDIUM,
        "Probing for system prompt contents",
    ),
    # ── Jailbreak patterns ──────────────────────────────────────────────────
    (
        r"(DAN|do\s+anything\s+now|absolutely\s+anything)\s*(mode|prompt)?",
        InjectionType.JAILBREAK,
        ThreatLevel.CRITICAL,
        "Known DAN jailbreak pattern",
    ),
    (
        r"(jailbreak|bypass\s+safety|remove\s+filters?|disable\s+safety)",
        InjectionType.JAILBREAK,
        ThreatLevel.CRITICAL,
        "Explicit jailbreak attempt",
    ),
    (
        r"hypothetical(ly)?\s*(scenario|situation)?\s*where\s+.*(no\s+rules|no\s+restrictions|anything\s+goes)",
        InjectionType.JAILBREAK,
        ThreatLevel.HIGH,
        "Hypothetical scenario to bypass restrictions",
    ),
    # ── Encoding / obfuscation attacks ──────────────────────────────────────
    (
        r"base64\s*:\s*[A-Za-z0-9+/=]{20,}",
        InjectionType.ENCODING_ATTACK,
        ThreatLevel.MEDIUM,
        "Suspicious base64 encoded content",
    ),
    (
        r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){5,}",
        InjectionType.ENCODING_ATTACK,
        ThreatLevel.MEDIUM,
        "Hex-encoded content injection",
    ),
    # ── Delimiter injection ─────────────────────────────────────────────────
    (
        r"```\s*(system|assistant|function)\b",
        InjectionType.DELIMITER_INJECTION,
        ThreatLevel.HIGH,
        "Code block delimiter injection",
    ),
    (
        r"<\|?(system|im_start|endoftext|sep)\|?>",
        InjectionType.DELIMITER_INJECTION,
        ThreatLevel.CRITICAL,
        "Model-specific token injection",
    ),
    (
        r"(\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)",
        InjectionType.DELIMITER_INJECTION,
        ThreatLevel.CRITICAL,
        "LLaMA instruction token injection",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT SCANNER — The detection engine
# ═══════════════════════════════════════════════════════════════════════════════


class PromptScanner:
    """
    Scans input text for prompt injection attacks.

    Uses a multi-layer approach:
    1. Pattern matching against known attack signatures
    2. Structural analysis (suspicious character distributions)
    3. Custom user-defined validators
    """

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self._policy = policy or SecurityPolicy()
        self._compiled_patterns: List[
            Tuple[re.Pattern, InjectionType, ThreatLevel, str]
        ] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        self._compiled_patterns.clear()

        # Built-in patterns
        for pattern_str, inj_type, threat, desc in _BUILTIN_PATTERNS:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                self._compiled_patterns.append((compiled, inj_type, threat, desc))
            except re.error:
                pass

        # User-supplied patterns
        for pattern_str in self._policy.suspicious_patterns:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                self._compiled_patterns.append(
                    (
                        compiled,
                        InjectionType.CONTEXT_MANIPULATION,
                        ThreatLevel.HIGH,
                        f"Custom pattern: {pattern_str[:50]}",
                    )
                )
            except re.error:
                pass

    def scan(self, text: str) -> DetectionResult:
        """
        Scan input text for prompt injection attempts.

        Returns a DetectionResult with threat assessment.
        """
        start = time.time()
        input_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[
            :16
        ]

        # Check whitelist first
        for wl_pattern in self._policy.whitelist_patterns:
            try:
                if re.search(wl_pattern, text, re.IGNORECASE):
                    return DetectionResult(
                        is_threat=False,
                        input_hash=input_hash,
                        scan_duration_ms=(time.time() - start) * 1000,
                    )
            except re.error:
                pass

        # Length check
        if len(text) > self._policy.max_input_length:
            return DetectionResult(
                is_threat=True,
                threat_level=ThreatLevel.HIGH,
                injection_type=InjectionType.CONTEXT_MANIPULATION,
                confidence=0.9,
                details=f"Input exceeds maximum length ({len(text)} > {self._policy.max_input_length})",
                input_hash=input_hash,
                scan_duration_ms=(time.time() - start) * 1000,
            )

        # Pattern matching
        matched_patterns: List[str] = []
        max_threat = ThreatLevel.NONE
        detected_type: Optional[InjectionType] = None
        threat_scores: Dict[ThreatLevel, float] = {
            ThreatLevel.LOW: 0.3,
            ThreatLevel.MEDIUM: 0.6,
            ThreatLevel.HIGH: 0.85,
            ThreatLevel.CRITICAL: 0.98,
        }

        for compiled, inj_type, threat, desc in self._compiled_patterns:
            if compiled.search(text):
                matched_patterns.append(desc)
                if threat_scores.get(threat, 0) > threat_scores.get(max_threat, 0):
                    max_threat = threat
                    detected_type = inj_type

        # Structural analysis
        structural_score = self._structural_analysis(text)

        # Combine scores
        pattern_confidence = (
            threat_scores.get(max_threat, 0.0) if matched_patterns else 0.0
        )
        combined_confidence = max(pattern_confidence, structural_score)

        # Run custom validators
        for validator in self._policy.custom_validators:
            try:
                custom_result = validator(text)
                if (
                    custom_result.is_threat
                    and custom_result.confidence > combined_confidence
                ):
                    combined_confidence = custom_result.confidence
                    if custom_result.injection_type:
                        detected_type = custom_result.injection_type
                    if custom_result.matched_patterns:
                        matched_patterns.extend(custom_result.matched_patterns)
                    max_threat = custom_result.threat_level
            except Exception:
                pass

        is_threat = combined_confidence >= 0.5 or len(matched_patterns) > 0

        result = DetectionResult(
            is_threat=is_threat,
            threat_level=max_threat if is_threat else ThreatLevel.NONE,
            injection_type=detected_type,
            matched_patterns=matched_patterns,
            confidence=combined_confidence,
            details=(
                f"Detected {len(matched_patterns)} suspicious patterns"
                if matched_patterns
                else "No threats detected"
            ),
            input_hash=input_hash,
            scan_duration_ms=(time.time() - start) * 1000,
        )

        return result

    def _structural_analysis(self, text: str) -> float:
        """
        Analyze text structure for injection indicators.

        Returns a threat score 0.0-1.0.
        """
        score = 0.0
        text_lower = text.lower()

        # Multiple instruction-like phrases
        instruction_words = [
            "must",
            "always",
            "never",
            "important",
            "remember",
            "instruction",
            "command",
            "directive",
            "rule",
            "override",
        ]
        instruction_count = sum(1 for w in instruction_words if w in text_lower)
        if instruction_count >= 4:
            score = max(score, 0.5)
        elif instruction_count >= 6:
            score = max(score, 0.7)

        # Unusual character concentrations (encoding attacks)
        if len(text) > 100:
            non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
            if non_ascii_ratio > 0.3:
                score = max(score, 0.4)

        # Multiple line breaks with different instructions (multi-part injection)
        segments = text.split("\n\n")
        if len(segments) > 5:
            imperative_segments = sum(
                1
                for s in segments
                if any(
                    w in s.lower() for w in ["you must", "you will", "do not", "always"]
                )
            )
            if imperative_segments >= 3:
                score = max(score, 0.6)

        return score

    def sanitize(self, text: str) -> str:
        """
        Sanitize input by removing or neutralizing detected injection patterns.

        Returns the cleaned text.
        """
        sanitized = text

        # Remove model-specific tokens
        sanitized = re.sub(r"<\|?(system|im_start|endoftext|sep)\|?>", "", sanitized)
        sanitized = re.sub(r"(\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)", "", sanitized)

        # Neutralize instruction overrides by wrapping in quotes
        override_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
            r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
            r"forget\s+(everything|all|your)\s*(instructions?|rules?|training|prompts?)?",
        ]
        for pattern in override_patterns:
            sanitized = re.sub(
                pattern,
                "[REDACTED: instruction override attempt]",
                sanitized,
                flags=re.IGNORECASE,
            )

        return sanitized.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG — Compliance-ready input logging
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: float = field(default_factory=time.time)
    agent_name: str = ""
    input_text: str = ""
    input_hash: str = ""
    detection_result: Optional[DetectionResult] = None
    action_taken: str = ""  # "allowed" | "blocked" | "sanitized" | "warned"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "input_hash": self.input_hash,
            "detection_result": (
                self.detection_result.to_dict() if self.detection_result else None
            ),
            "action_taken": self.action_taken,
            "metadata": self.metadata,
        }


class AuditLog:
    """Thread-safe audit log for all scanned inputs."""

    _instance: Optional["AuditLog"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AuditLog":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._entries: List[AuditEntry] = []
                cls._instance._entry_lock = threading.Lock()
                cls._instance._max_entries = 10000
            return cls._instance

    def log(self, entry: AuditEntry) -> None:
        """Add an audit entry."""
        with self._entry_lock:
            self._entries.append(entry)
            # Cap size
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    def get_entries(
        self,
        agent_name: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Retrieve audit entries with optional filters."""
        with self._entry_lock:
            results = list(self._entries)
        if agent_name:
            results = [e for e in results if e.agent_name == agent_name]
        if action:
            results = [e for e in results if e.action_taken == action]
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def get_threat_entries(self, limit: int = 50) -> List[AuditEntry]:
        """Get only entries where threats were detected."""
        return self.get_entries(action="blocked", limit=limit) + self.get_entries(
            action="sanitized", limit=limit
        )

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._entry_lock:
            self._entries.clear()


def get_audit_log() -> AuditLog:
    """Get the global audit log singleton."""
    return AuditLog()


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY GUARD — Main entry point for input screening
# ═══════════════════════════════════════════════════════════════════════════════


class SecurityGuard:
    """
    Main security interface that screens all agent inputs.

    Integrates with the Agent.run() pipeline to automatically
    scan inputs before execution.
    """

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self._policy = policy or SecurityPolicy()
        self._scanner = PromptScanner(self._policy)
        self._audit = get_audit_log()

    @property
    def policy(self) -> SecurityPolicy:
        return self._policy

    def update_policy(self, policy: SecurityPolicy) -> None:
        """Update the security policy."""
        self._policy = policy
        self._scanner = PromptScanner(policy)

    def screen_input(
        self,
        text: str,
        agent_name: str = "",
    ) -> Tuple[bool, Optional[str], Optional[DetectionResult]]:
        """
        Screen an input before agent processing.

        Returns:
            (allowed: bool, processed_text: str or None, detection: DetectionResult or None)

        If allowed is False, the agent should not process this input.
        If processed_text is not None, use it instead of the original.
        """
        if not self._policy.prompt_injection_detection:
            return True, text, None

        detection = self._scanner.scan(text)

        # Determine action
        action = "allowed"
        processed_text: Optional[str] = text

        if detection.is_threat:
            if self._policy.injection_action == "block":
                action = "blocked"
                processed_text = None
            elif self._policy.injection_action == "sanitize":
                action = "sanitized"
                processed_text = self._scanner.sanitize(text)
            elif self._policy.injection_action == "warn":
                action = "warned"
                processed_text = text
            else:
                action = "blocked"
                processed_text = None

            # Call threat callback
            if self._policy.on_threat_detected:
                try:
                    self._policy.on_threat_detected(detection, text)
                except Exception:
                    pass

        # Audit log
        if self._policy.audit_all_inputs or detection.is_threat:
            self._audit.log(
                AuditEntry(
                    agent_name=agent_name,
                    input_text=text if self._policy.audit_all_inputs else "",
                    input_hash=detection.input_hash,
                    detection_result=detection,
                    action_taken=action,
                )
            )

        allowed = action != "blocked"
        return allowed, processed_text, detection

    def scan(self, text: str) -> DetectionResult:
        """Scan text without taking action. Returns detection result."""
        return self._scanner.scan(text)

    def check(self, text: str) -> DetectionResult:
        """Alias for scan(). Scan text and return detection result."""
        return self.scan(text)


# ── Module-level default guard ───────────────────────────────────────────────

_default_guard: Optional[SecurityGuard] = None
_guard_lock = threading.Lock()


def get_security_guard(policy: Optional[SecurityPolicy] = None) -> SecurityGuard:
    """Get or create the global security guard."""
    global _default_guard
    with _guard_lock:
        if _default_guard is None or policy is not None:
            _default_guard = SecurityGuard(policy)
        return _default_guard
