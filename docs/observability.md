# InfraRely Observability

InfraRely observability is designed to make agent behavior inspectable, measurable, and debuggable in production.

## Why Observability Matters for Agents

Agent systems are hard to operate without high-fidelity visibility:

- Decisions are multi-stage and often non-obvious
- Failures can emerge from routing, tools, runtime, or policy gates
- Cost and latency regressions are easy to miss without telemetry

Observability converts agent behavior from opaque to operable.

## Observability Components

InfraRely includes dedicated modules:

- `observability/metrics.py` — performance and reliability metrics
- `observability/logger.py` — structured logging surface
- `observability/token_budget.py` — token usage and budget tracking
- `observability/dashboard.py` — operational visibility layer
- `agent/execution_trace.py` — per-execution trace context

## Key Telemetry Domains

Track at least these domains in production:

1. **Execution reliability**
   - success/failure rate, retry rate, fallback rate
2. **Routing quality**
   - tool hit rate, capability match rate, LLM fallback frequency
3. **Verification health**
   - pass/fail rates by verification layer
4. **Runtime health**
   - queue depth, scheduling latency, deadlock signals
5. **Cost and efficiency**
   - token spend, cache hit rates, knowledge bypass rate

## Trace-First Debugging

Use execution traces as the primary incident primitive:

- Reconstruct request path from router to final output
- Identify stage-specific latency spikes
- Correlate verification failures with policy or tool regressions
- Explain why LLM fallback occurred for a given request

## Operational Best Practices

- Standardize event schemas across agents and tools
- Add per-agent and per-capability reliability dashboards
- Alert on trend shifts, not only absolute thresholds
- Track SLOs for latency, success rate, and verification integrity
- Preserve traces long enough for root-cause analysis cycles
