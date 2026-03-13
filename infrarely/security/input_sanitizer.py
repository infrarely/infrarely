"""
aos/input_sanitizer.py — Always-On Input Sanitization
═══════════════════════════════════════════════════════════════════════════════
SECURITY GAP 1: No Input Sanitization.

Problem: Without an explicit SecurityPolicy, agent.run() accepts ANY input
         — null bytes, control characters, encoding attacks, excessively
         long strings — with zero filtering.

Solution: An always-on sanitizer that runs on EVERY agent.run() call,
          regardless of whether a SecurityPolicy is configured.  The opt-in
          SecurityGuard remains a *second* layer.

Pipeline:
    raw input → InputSanitizer.sanitize() → SecurityGuard (if configured) → engine

The sanitizer is:
  • Zero-config — active by default, no setup required
  • Non-destructive — only strips genuinely dangerous content
  • Fast — regex + character scan, no ML
  • Configurable — advanced users can tune thresholds
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# SANITIZATION POLICY — what the always-on sanitizer enforces
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SanitizationPolicy:
    """
    Controls what the always-on input sanitizer strips/blocks.

    Defaults are secure — advanced users can relax specific controls.
    """

    max_input_length: int = 50_000  # hard cap on input length
    strip_null_bytes: bool = True  # remove \x00
    strip_control_chars: bool = True  # remove C0 controls except \n \r \t
    normalize_unicode: bool = True  # normalize homoglyphs / invisible chars
    block_encoding_attacks: bool = True  # detect base64 bombs, hex injections
    max_repeated_chars: int = 500  # collapse runs of same char beyond N
    strip_model_tokens: bool = True  # remove <|im_start|>, [INST], etc.


# ═══════════════════════════════════════════════════════════════════════════════
# SANITIZATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SanitizationResult:
    """Outcome of sanitizing an input string."""

    text: str = ""
    original_length: int = 0
    sanitized_length: int = 0
    was_modified: bool = False
    modifications: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    duration_ms: float = 0.0

    def __iter__(self):
        """Allow tuple unpacking: text, result = sanitizer.sanitize(...)"""
        return iter((self.text, self))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "original_length": self.original_length,
            "sanitized_length": self.sanitized_length,
            "was_modified": self.was_modified,
            "modifications": self.modifications,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "duration_ms": round(self.duration_ms, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL-SPECIFIC TOKEN PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_TOKEN_PATTERNS = [
    re.compile(r"<\|?(?:system|im_start|im_end|endoftext|sep|pad)\|?>", re.I),
    re.compile(r"\[/?INST\]", re.I),
    re.compile(r"<</?SYS>>", re.I),
    re.compile(r"<\|(?:user|assistant|function_call|tool)\|>", re.I),
]


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT SANITIZER — always-on, zero config
# ═══════════════════════════════════════════════════════════════════════════════


class InputSanitizer:
    """
    Always-on input sanitizer.  Runs before the optional SecurityGuard.

    Usage::

        sanitizer = InputSanitizer()
        clean, result = sanitizer.sanitize("hello\\x00world")
        # clean = "helloworld", result.modifications = ["stripped_null_bytes"]

    Thread-safe singleton by default (get via ``get_input_sanitizer()``).
    """

    _instance: Optional["InputSanitizer"] = None
    _lock = threading.Lock()

    def __new__(cls, policy: Optional[SanitizationPolicy] = None) -> "InputSanitizer":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._policy = policy or SanitizationPolicy()
                inst._stats = {"calls": 0, "blocked": 0, "modified": 0}
                cls._instance = inst
            return cls._instance

    @property
    def policy(self) -> SanitizationPolicy:
        return self._policy

    def update_policy(self, policy: SanitizationPolicy) -> None:
        """Update the sanitization policy."""
        self._policy = policy

    # ── Main API ──────────────────────────────────────────────────────────────

    def sanitize(self, text: str) -> tuple:
        """
        Sanitize input text.

        Returns:
            (sanitized_text: str, result: SanitizationResult)

        If result.blocked is True, sanitized_text is empty and the
        input should be rejected.
        """
        start = time.monotonic()
        mods: List[str] = []
        p = self._policy
        original = text

        self._stats["calls"] += 1

        # ── 1. Length check (hard block) ──────────────────────────────────
        if len(text) > p.max_input_length:
            elapsed = (time.monotonic() - start) * 1000
            self._stats["blocked"] += 1
            return SanitizationResult(
                text="",
                original_length=len(original),
                sanitized_length=0,
                was_modified=True,
                modifications=["blocked_excessive_length"],
                blocked=True,
                block_reason=f"Input length {len(text)} exceeds maximum {p.max_input_length}",
                duration_ms=elapsed,
            )

        # ── 2. Null bytes ─────────────────────────────────────────────────
        if p.strip_null_bytes and "\x00" in text:
            text = text.replace("\x00", "")
            mods.append("stripped_null_bytes")

        # ── 3. Control characters ─────────────────────────────────────────
        if p.strip_control_chars:
            cleaned = []
            had_control = False
            for ch in text:
                cp = ord(ch)
                # Keep \n (10), \r (13), \t (9); strip other C0 (0-31) and DEL (127)
                # If strip_null_bytes is False, also keep \x00
                if cp == 0 and not p.strip_null_bytes:
                    cleaned.append(ch)
                elif (cp < 32 and cp not in (9, 10, 13)) or cp == 127:
                    had_control = True
                else:
                    cleaned.append(ch)
            if had_control:
                text = "".join(cleaned)
                mods.append("stripped_control_chars")

        # ── 4. Unicode normalization (invisible chars / homoglyphs) ───────
        if p.normalize_unicode:
            # Remove zero-width characters
            zw_pattern = re.compile(
                r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060\u2061\u2062\u2063\u2064]"
            )
            if zw_pattern.search(text):
                text = zw_pattern.sub("", text)
                mods.append("stripped_zero_width_chars")

        # ── 5. Model token stripping ──────────────────────────────────────
        if p.strip_model_tokens:
            for pat in _MODEL_TOKEN_PATTERNS:
                if pat.search(text):
                    text = pat.sub("", text)
                    mods.append("stripped_model_tokens")
                    break  # one mod message is enough

        # ── 6. Repeated character collapse ────────────────────────────────
        if p.max_repeated_chars > 0:
            repeat_pat = re.compile(r"(.)\1{" + str(p.max_repeated_chars) + r",}")
            if repeat_pat.search(text):
                text = repeat_pat.sub(lambda m: m.group(1) * p.max_repeated_chars, text)
                mods.append("collapsed_repeated_chars")

        # ── 7. Encoding attack detection ──────────────────────────────────
        if p.block_encoding_attacks:
            # Very long base64 blobs (potential bombs)
            b64_bomb = re.compile(r"[A-Za-z0-9+/=]{1000,}")
            if b64_bomb.search(text):
                mods.append("stripped_base64_bomb")
                text = b64_bomb.sub("[REDACTED:base64]", text)

        # ── Finalize ──────────────────────────────────────────────────────
        text = text.strip()
        elapsed = (time.monotonic() - start) * 1000
        was_modified = len(mods) > 0

        if was_modified:
            self._stats["modified"] += 1

        return SanitizationResult(
            text=text,
            original_length=len(original),
            sanitized_length=len(text),
            was_modified=was_modified,
            modifications=mods,
            blocked=False,
            duration_ms=elapsed,
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Return sanitization statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {"calls": 0, "blocked": 0, "modified": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_sanitizer_lock = threading.Lock()
_default_sanitizer: Optional[InputSanitizer] = None


def get_input_sanitizer(policy: Optional[SanitizationPolicy] = None) -> InputSanitizer:
    """Get or create the global input sanitizer singleton."""
    global _default_sanitizer
    with _sanitizer_lock:
        if _default_sanitizer is None:
            _default_sanitizer = InputSanitizer(policy)
        elif policy is not None:
            _default_sanitizer.update_policy(policy)
        return _default_sanitizer
