"""
infrarely/validation.py — Tool Schema Validation & Type Safety
═══════════════════════════════════════════════════════════════════════════════
Automatic validation that tool inputs and outputs match their declared types.

Without this, tools crash at runtime with cryptic TypeErrors.
This module provides:
- Pre-call input type validation
- Safe type coercion (str "10.5" → float 10.5)
- Clear error messages with the exact field that failed
- Post-call output type validation

Usage::

    @infrarely.tool
    def process(amount: float, account: str) -> dict:
        pass

    # Before tool call:
    # 1. Validate input types match declared signature
    # 2. Coerce if safe (str "10.5" → float 10.5)
    # 3. Reject if impossible ("hello" → float)
    # 4. Return typed ErrorType.VALIDATION with exact field
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, get_type_hints


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ValidationError:
    """A single validation error for a specific field."""

    field: str = ""
    expected_type: str = ""
    actual_type: str = ""
    actual_value: Any = None
    message: str = ""
    coerced: bool = False  # was coercion attempted?
    coerced_value: Any = None  # value after coercion (if successful)


@dataclass
class ValidationResult:
    """Result of validating tool inputs."""

    valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    coerced_args: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def args(self) -> Dict[str, Any]:
        """Alias for coerced_args."""
        return self.coerced_args

    @property
    def first_error(self) -> Optional[ValidationError]:
        return self.errors[0] if self.errors else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [
                {
                    "field": e.field,
                    "expected_type": e.expected_type,
                    "actual_type": e.actual_type,
                    "message": e.message,
                }
                for e in self.errors
            ],
            "warnings": self.warnings,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE COERCION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class TypeCoercer:
    """
    Safely coerce values to target types.

    Supports common coercions:
    - str "10.5" → float 10.5
    - str "42" → int 42
    - int 1 → bool True
    - str "true" → bool True
    - list/tuple conversions
    """

    # Strings that map to True/False
    _BOOL_TRUE = {"true", "yes", "1", "on", "t", "y"}
    _BOOL_FALSE = {"false", "no", "0", "off", "f", "n"}

    @classmethod
    def can_coerce(cls, value: Any, target_type: Type) -> Tuple[bool, Any]:
        """
        Try to coerce a value to the target type.

        Returns (success: bool, coerced_value: Any).
        """
        # Already the right type
        if isinstance(value, target_type):
            return True, value

        # None handling
        if value is None:
            return False, None

        try:
            # ── float coercion ──────────────────────────────────────────────
            if target_type is float:
                if isinstance(value, (int, float)):
                    return True, float(value)
                if isinstance(value, str):
                    # Clean up currency, commas, spaces
                    cleaned = (
                        value.strip()
                        .replace(",", "")
                        .replace("$", "")
                        .replace("€", "")
                        .replace("£", "")
                    )
                    return True, float(cleaned)

            # ── int coercion ────────────────────────────────────────────────
            if target_type is int:
                if isinstance(value, float) and value == int(value):
                    return True, int(value)
                if isinstance(value, str):
                    cleaned = value.strip().replace(",", "")
                    f = float(cleaned)
                    if f == int(f):
                        return True, int(f)

            # ── str coercion (anything → str) ───────────────────────────────
            if target_type is str:
                return True, str(value)

            # ── bool coercion ───────────────────────────────────────────────
            if target_type is bool:
                if isinstance(value, (int, float)):
                    return True, bool(value)
                if isinstance(value, str):
                    lower = value.strip().lower()
                    if lower in cls._BOOL_TRUE:
                        return True, True
                    if lower in cls._BOOL_FALSE:
                        return True, False

            # ── list coercion ───────────────────────────────────────────────
            if target_type is list:
                if isinstance(value, (tuple, set, frozenset)):
                    return True, list(value)
                if isinstance(value, str) and value.startswith("["):
                    # Try to parse JSON-like list
                    import json

                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return True, parsed
                    except (json.JSONDecodeError, ValueError):
                        pass

            # ── dict coercion ───────────────────────────────────────────────
            if target_type is dict:
                if isinstance(value, str) and value.startswith("{"):
                    import json

                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            return True, parsed
                    except (json.JSONDecodeError, ValueError):
                        pass

        except (ValueError, TypeError, OverflowError):
            pass

        return False, None

    @classmethod
    def is_impossible(cls, value: Any, target_type: Type) -> bool:
        """Check if coercion is definitely impossible."""
        if isinstance(value, target_type):
            return False

        # str to numeric — impossible if not a number string
        if target_type in (int, float) and isinstance(value, str):
            cleaned = (
                value.strip()
                .replace(",", "")
                .replace("$", "")
                .replace("€", "")
                .replace("£", "")
            )
            if not cleaned:
                return True
            try:
                float(cleaned)
                return False
            except ValueError:
                return True

        return False


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA VALIDATOR — Main validation engine
# ═══════════════════════════════════════════════════════════════════════════════


class SchemaValidator:
    """
    Validates tool function arguments against their type signatures.

    Performs:
    1. Required argument checking
    2. Type validation
    3. Safe coercion where possible
    4. Clear error reporting
    """

    def __init__(self, *, coerce: bool = True, strict: bool = False):
        """
        Parameters
        ----------
        coerce : bool
            If True, attempt safe type coercion (default True).
        strict : bool
            If True, reject any type mismatch even if coercible.
        """
        self._coerce = coerce
        self._strict = strict
        self._coercer = TypeCoercer()

    def validate_call(
        self,
        fn: Callable,
        args: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate arguments against a function's signature.

        Parameters
        ----------
        fn : Callable
            The function to validate against.
        args : dict
            The arguments to validate.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult(coerced_args=dict(args))
        sig = inspect.signature(fn)

        # Get type hints
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            expected_type = hints.get(param_name)

            # Check if argument is provided
            value = args.get(param_name)

            if value is None and param.default is inspect.Parameter.empty:
                # Required argument missing
                if param_name not in args:
                    result.valid = False
                    result.errors.append(
                        ValidationError(
                            field=param_name,
                            expected_type=(
                                str(expected_type) if expected_type else "any"
                            ),
                            actual_type="missing",
                            message=f"Required argument '{param_name}' is missing",
                        )
                    )
                    continue

            if value is None:
                continue  # Optional with None value

            if expected_type is None:
                continue  # No type annotation, skip validation

            # Handle Optional types
            actual_type = _unwrap_optional(expected_type)
            if actual_type is None:
                continue  # Complex type we can't validate

            # Type check
            if isinstance(value, actual_type):
                continue  # Exact match, all good

            # Try coercion
            if self._coerce and not self._strict:
                can_coerce, coerced = TypeCoercer.can_coerce(value, actual_type)
                if can_coerce:
                    result.coerced_args[param_name] = coerced
                    result.warnings.append(
                        f"Coerced '{param_name}': {type(value).__name__} → {actual_type.__name__} "
                        f"({value!r} → {coerced!r})"
                    )
                    continue

            # Validation failed
            result.valid = False
            result.errors.append(
                ValidationError(
                    field=param_name,
                    expected_type=(
                        actual_type.__name__
                        if hasattr(actual_type, "__name__")
                        else str(actual_type)
                    ),
                    actual_type=type(value).__name__,
                    actual_value=value,
                    message=f"Expected {actual_type.__name__}, got {type(value).__name__}: {value!r}",
                )
            )

        return result

    def validate_inputs(
        self, fn: Callable, args: Dict[str, Any], coerce: bool = None
    ) -> "ValidationResult":
        """
        Alias for validate_call(). Validate arguments against a function's signature.

        Parameters
        ----------
        fn : Callable
            The function to validate against.
        args : dict
            The arguments to validate.
        coerce : bool, optional
            Override the default coercion setting.

        Returns
        -------
        ValidationResult
        """
        old_coerce = self._coerce
        if coerce is not None:
            self._coerce = coerce
        try:
            result = self.validate_call(fn, args)
        finally:
            self._coerce = old_coerce
        return result

    def validate_return(
        self,
        fn: Callable,
        return_value: Any,
    ) -> ValidationResult:
        """Validate a function's return value against its type annotation."""
        result = ValidationResult()

        try:
            hints = get_type_hints(fn)
        except Exception:
            return result  # No hints available

        return_type = hints.get("return")
        if return_type is None:
            return result  # No return type annotation

        actual_type = _unwrap_optional(return_type)
        if actual_type is None:
            return result

        if return_value is None:
            # Check if Optional
            if _is_optional(return_type):
                return result
            result.valid = False
            result.errors.append(
                ValidationError(
                    field="return",
                    expected_type=str(return_type),
                    actual_type="NoneType",
                    message=f"Expected return type {return_type}, got None",
                )
            )
            return result

        if not isinstance(return_value, actual_type):
            result.valid = False
            result.errors.append(
                ValidationError(
                    field="return",
                    expected_type=(
                        actual_type.__name__
                        if hasattr(actual_type, "__name__")
                        else str(actual_type)
                    ),
                    actual_type=type(return_value).__name__,
                    actual_value=return_value,
                    message=f"Return type mismatch: expected {actual_type.__name__}, got {type(return_value).__name__}",
                )
            )

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _unwrap_optional(tp: Any) -> Optional[Type]:
    """Unwrap Optional[X] to X. Returns None for complex types."""
    # Handle Optional[X] = Union[X, None]
    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", ())

    if origin is type(None):
        return None

    # typing.Union
    if origin is not None and str(origin) in (
        "<class 'types.UnionType'>",
        "typing.Union",
    ):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
        return None  # Complex union, can't validate simply

    # Plain type
    if isinstance(tp, type):
        return tp

    return None


def _is_optional(tp: Any) -> bool:
    """Check if a type is Optional (Union[X, None])."""
    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", ())
    if origin is not None and str(origin) in (
        "<class 'types.UnionType'>",
        "typing.Union",
    ):
        return type(None) in args
    return False


# ── Module-level singleton ───────────────────────────────────────────────────

_validator: Optional[SchemaValidator] = None


def get_schema_validator(
    *, coerce: bool = True, strict: bool = False
) -> SchemaValidator:
    """Get or create the global schema validator."""
    global _validator
    if _validator is None:
        _validator = SchemaValidator(coerce=coerce, strict=strict)
    return _validator
