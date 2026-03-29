"""
Example: Tool Validation & Type Safety
═══════════════════════════════════════════════════════════════════════════════
Shows automatic type validation and safe coercion of tool inputs.

Run: python examples/validation_demo.py
"""

import infrarely
from infrarely.platform.validation import SchemaValidator, ValidationResult

infrarely.configure(llm_provider="openai", log_file_enabled=False)


# ── Define tools with type annotations ────────────────────────────────────────
@infrarely.tool
def calculate_discount(price: float, percentage: float) -> float:
    """Calculate a discounted price."""
    return price * (1 - percentage / 100)


@infrarely.tool
def format_name(first: str, last: str) -> str:
    """Format a full name."""
    return f"{first.title()} {last.title()}"


@infrarely.tool
def count_items(items: list) -> int:
    """Count items in a list."""
    return len(items)


# ── Direct validator usage ────────────────────────────────────────────────────
print("─── Direct Validator ───\n")
validator = SchemaValidator()

# Valid inputs
result = validator.validate_call(
    calculate_discount, {"price": 100.0, "percentage": 20.0}
)
print(f"Valid inputs:   valid={result.valid}, errors={len(result.errors)}")

# String that can be coerced to float
result = validator.validate_call(
    calculate_discount, {"price": "100.0", "percentage": "20"}
)
print(f"Coercible str:  valid={result.valid}, coerced={result.coerced_args}")

# Invalid inputs
result = validator.validate_call(
    calculate_discount, {"price": "not-a-number", "percentage": 20}
)
print(f"Invalid input:  valid={result.valid}")
for err in result.errors:
    print(f"  → {err.message}")

# ── Agent with validation ─────────────────────────────────────────────────────
print("\n─── Agent Validation Integration ───\n")
agent = infrarely.agent("validated-bot", tools=[calculate_discount, format_name])

# Normal operation (validation passes)
result = agent.run("Calculate discount on 200 with 15 percent")
print(f"Discount: {result.output} (success={result.success})")

# Name formatting
result = agent.run('Format name "alice" "smith"')
print(f"Name: {result.output} (success={result.success})")

# ── Return type validation ────────────────────────────────────────────────────
print("\n─── Return Type Validation ───\n")


def bad_tool(x: int) -> str:
    return 42  # Returns int, declared str


result = validator.validate_return(bad_tool, 42)
print(f"Return validation: valid={result.valid}")
for warn in result.warnings:
    print(f"  Warning: {warn}")

infrarely.shutdown()
