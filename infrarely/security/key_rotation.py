"""
infrarely/key_rotation.py — API Key Rotation Without Restart
═══════════════════════════════════════════════════════════════════════════════
SECURITY GAP 3: No API Key Rotation Support.

Problem: ``infrarely.configure(api_key="sk-...")`` sets the key once at startup.
         If the key is compromised, the only option is to restart the
         entire process.  In production, this means downtime.

Solution: ``aos.rotate_api_key("sk-new-key")`` live-swaps the key across
          all agents and the execution engine, with validation and rollback.

Design:
  • Validates the new key format before applying
  • Atomically updates config + all active engines
  • Keeps an audit trail of rotations
  • Supports rollback to previous key on failure
  • Thread-safe
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# KEY ROTATION EVENT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class KeyRotationEvent:
    """Record of a single key rotation."""

    timestamp: float = field(default_factory=time.time)
    old_key_suffix: str = ""  # last 4 chars only (security)
    new_key_suffix: str = ""  # last 4 chars only
    success: bool = True
    error: str = ""
    source: str = ""  # "manual" | "scheduled" | "emergency"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "old_key_suffix": self.old_key_suffix,
            "new_key_suffix": self.new_key_suffix,
            "success": self.success,
            "error": self.error,
            "source": self.source,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# KEY MANAGER — live rotation, validation, audit
# ═══════════════════════════════════════════════════════════════════════════════


class KeyManager:
    """
    Manages API key rotation without downtime.

    Usage::

        import infrarely
        manager = aos.key_manager

        # Rotate key
        result = manager.rotate("sk-new-key-here")
        assert result.success

        # Or use convenience function
        aos.rotate_api_key("sk-new-key-here")

    Thread-safe singleton.
    """

    _instance: Optional["KeyManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "KeyManager":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._history: List[KeyRotationEvent] = []
                inst._max_history = 100
                inst._validators: List[Callable[[str], bool]] = []
                inst._on_rotate_callbacks: List[Callable[[KeyRotationEvent], None]] = []
                inst._previous_key: str = ""
                inst._rotation_lock = threading.Lock()
                cls._instance = inst
            return cls._instance

    # ── Validation ────────────────────────────────────────────────────────────

    def add_validator(self, fn: Callable[[str], bool]) -> None:
        """
        Add a key validator.  Called before rotation proceeds.
        Should return True if the key is acceptable, False otherwise.
        """
        self._validators.append(fn)

    def _validate_key(self, key: str) -> tuple:
        """
        Validate a new API key.

        Returns (valid: bool, reason: str).
        """
        if not key or not isinstance(key, str):
            return False, "Key must be a non-empty string"

        if len(key) < 8:
            return False, "Key too short (minimum 8 characters)"

        if key.isspace():
            return False, "Key cannot be whitespace-only"

        # Run custom validators
        for validator in self._validators:
            try:
                if not validator(key):
                    return False, "Custom validator rejected the key"
            except Exception as e:
                return False, f"Validator error: {e}"

        return True, ""

    # ── Rotation ──────────────────────────────────────────────────────────────

    def rotate(self, new_key: str, *, source: str = "manual") -> KeyRotationEvent:
        """
        Rotate the API key atomically.

        Steps:
          1. Validate the new key
          2. Store the old key for rollback
          3. Update the global config
          4. Notify callbacks
          5. Return rotation event

        Returns a KeyRotationEvent indicating success/failure.
        """
        with self._rotation_lock:
            from infrarely.core.config import get_config

            cfg = get_config()
            old_key = cfg.get("api_key", "") or ""
            old_suffix = old_key[-4:] if len(old_key) >= 4 else "***"
            new_suffix = new_key[-4:] if len(new_key) >= 4 else "***"

            # Validate
            valid, reason = self._validate_key(new_key)
            if not valid:
                event = KeyRotationEvent(
                    old_key_suffix=old_suffix,
                    new_key_suffix=new_suffix,
                    success=False,
                    error=reason,
                    source=source,
                )
                self._history.append(event)
                return event

            # Store old key for rollback
            self._previous_key = old_key

            # Atomically update config
            try:
                cfg.set("api_key", new_key)

                event = KeyRotationEvent(
                    old_key_suffix=old_suffix,
                    new_key_suffix=new_suffix,
                    success=True,
                    source=source,
                )
                self._history.append(event)

                # Notify callbacks
                for cb in self._on_rotate_callbacks:
                    try:
                        cb(event)
                    except Exception:
                        pass

                # Trim history
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history :]

                return event

            except Exception as e:
                # Rollback
                try:
                    cfg.set("api_key", old_key)
                except Exception:
                    pass

                event = KeyRotationEvent(
                    old_key_suffix=old_suffix,
                    new_key_suffix=new_suffix,
                    success=False,
                    error=f"Rotation failed, rolled back: {e}",
                    source=source,
                )
                self._history.append(event)
                return event

    def rollback(self) -> KeyRotationEvent:
        """
        Roll back to the previous API key.

        Returns a KeyRotationEvent indicating success/failure.
        """
        if not self._previous_key:
            return KeyRotationEvent(
                success=False,
                error="No previous key to roll back to",
                source="rollback",
            )
        return self.rotate(self._previous_key, source="rollback")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_rotate(self, callback: Callable[[KeyRotationEvent], None]) -> None:
        """Register a callback that fires after each rotation."""
        self._on_rotate_callbacks.append(callback)

    # ── History / Audit ───────────────────────────────────────────────────────

    def history(self, limit: int = 50) -> List[KeyRotationEvent]:
        """Return recent rotation history."""
        return list(reversed(self._history[-limit:]))

    def last_rotation(self) -> Optional[KeyRotationEvent]:
        """Return the most recent rotation event, or None."""
        return self._history[-1] if self._history else None

    @property
    def current_key_suffix(self) -> str:
        """Last 4 chars of the current key (safe to log)."""
        from infrarely.core.config import get_config

        key = get_config().get("api_key", "") or ""
        return key[-4:] if len(key) >= 4 else "***"

    def reset(self) -> None:
        """Reset history (for testing)."""
        self._history.clear()
        self._previous_key = ""
        self._validators.clear()
        self._on_rotate_callbacks.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_key_manager_lock = threading.Lock()
_default_manager: Optional[KeyManager] = None


def get_key_manager() -> KeyManager:
    """Get or create the global key manager singleton."""
    global _default_manager
    with _key_manager_lock:
        if _default_manager is None:
            _default_manager = KeyManager()
        return _default_manager


def rotate_api_key(new_key: str, *, source: str = "manual") -> KeyRotationEvent:
    """
    Convenience: rotate the API key without restart.

    Usage::

        import infrarely
        event = aos.rotate_api_key("sk-new-key-here")
        assert event.success
    """
    return get_key_manager().rotate(new_key, source=source)
