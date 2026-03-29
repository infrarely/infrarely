# InfraRely Verification

InfraRely verification ensures agent outputs are checked before they become system actions or user-visible results.

## Why Verification Is Core

Without verification, agent systems produce:

- Structurally invalid outputs
- Logically inconsistent or contradictory answers
- Ungrounded statements disconnected from known context
- Policy-violating actions that should never execute

Verification in InfraRely is a first-class stage in the execution pipeline.

## Verification in the Stack

Verification is represented across core components:

- `agent/verification.py` — verification orchestration and rule execution
- `agent/pipeline.py` — enforcement point in the request lifecycle
- `agent/reasoning_engine.py` and `agent/context_builder.py` — context shaping for checks
- `agent/tool_validator.py` — tool schema and invocation validation
- `agent/permission_policy.py` — policy compatibility checks

## Verification Layers

InfraRely verification is modeled as layered gates:

1. **Structural checks**
   - Type/shape/schema conformance
2. **Logical checks**
   - Internal consistency and contradiction detection
3. **Knowledge grounding checks**
   - Alignment with available facts or retrieved context
4. **Policy checks**
   - Security, compliance, and permission constraints

## Execution Flow Integration

Typical sequence:

1. Goal enters deterministic routing path
2. Tools/capabilities execute within policy boundaries
3. Result candidate passes through verification gates
4. Failed checks produce recoverable error data and/or retries
5. Verified output is formatted and returned

## Failure Handling Strategy

- Prefer **errors-as-data** over uncaught exceptions
- Attach failure reason categories for observability
- Route recoverable failures to fallback paths when available
- Block unsafe outputs from leaving the runtime

## Verification in Production

- Define strict schemas for all side-effecting tool outputs
- Maintain policy packs per environment and workload class
- Track verification pass/fail rates as reliability KPIs
- Fail closed for high-risk actions, fail soft for low-risk content tasks
