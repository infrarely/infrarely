# InfraRely Architecture

## Overview

InfraRely is a production-grade AI agent framework built on a **7-layer architecture** that combines a zero-boilerplate SDK with a router-first, tool-centric, minimal-LLM execution engine.

**Core principle:** *"The LLM is a tool, not the controller."*

---

## 7-Layer Stack

```
┌─────────────────────────────────────────┐
│  Layer 7: Autonomous Evolution          │  evolution/
│  Self-improving system: A/B testing,    │
│  architecture proposals, policy guards  │
├─────────────────────────────────────────┤
│  Layer 6: Multi-Agent Runtime           │  runtime/
│  OS-like kernel: scheduler, IPC,        │
│  shared memory, RBAC, deadlock detect   │
├─────────────────────────────────────────┤
│  Layer 5: Adaptive Intelligence         │  adaptive/
│  Self-optimizing routing, failure       │
│  analysis, token optimization           │
├─────────────────────────────────────────┤
│  Layer 4: Verification                  │  agent/verification.py
│  Structural, logical, knowledge,        │
│  policy checks on every result          │
├─────────────────────────────────────────┤
│  Layer 3: Infrastructure                │  agent/, security/
│  Execution depth, permissions,          │
│  tool validation, sandboxing            │
├─────────────────────────────────────────┤
│  Layer 2: Capability Graphs             │  agent/capability_*.py
│  Multi-step workflows with              │
│  dependency resolution                  │
├─────────────────────────────────────────┤
│  Layer 1: Execution Contracts           │  core/, router/
│  Deterministic routing, frozen plans,   │
│  three-gate LLM isolation               │
└─────────────────────────────────────────┘
```

## Execution Pipeline

Every request flows through:

```
Input → Router (intent classification)
      → Reasoning Engine (deterministic rules)
      → Capability Resolver → Compiler → Executor
      → Verification Engine
      → Response Formatter
      → Output
```

**The LLM is only called when:**
1. The router cannot classify intent with confidence ≥ 0.55
2. No tool or capability matches the request
3. Knowledge confidence is below threshold (85%)

## Routing Priority

```
1. Knowledge check (TF-IDF vector search) — bypass LLM if ≥85% confidence
2. Math evaluation — deterministic arithmetic, zero tokens
3. Tool routing — intent classification → registered tools
4. Capability matching — multi-step workflows
5. LLM synthesis — only as final fallback
```

## Memory Architecture

**Three-scope memory (SDK layer):**
- `session` — in-memory per conversation
- `persistent` — SQLite-backed per agent
- `shared` — cross-agent SQLite

**Three-layer memory (execution layer):**
- `working` — sliding window (last 6 turns)
- `structured` — JSON-backed persistent store
- `long-term` — compressed conversation summaries

**Key rule:** The LLM never sees raw memory. It receives a compressed context window (<800 tokens).

## Multi-Agent Runtime (Layer 6)

An OS-like kernel managing multiple agents:

| Component | Purpose |
|-----------|---------|
| Agent Registry | Identity store with status tracking |
| Agent Scheduler | Process scheduling (round-robin, priority, fair) |
| Message Bus | Inter-agent communication (pub-sub) |
| Shared Memory | Cross-agent knowledge sharing |
| Identity & Permissions | RBAC for tool and resource access |
| Resource Isolation | Per-agent CPU, memory, token budgets |
| Capability Market | Service mesh for capability discovery |
| Negotiation Protocol | Task delegation via bidding/auctions |
| Deadlock Detector | Wait-graph cycle detection |
| Lifecycle Manager | Spawn, pause, resume, terminate |

## Evolution System (Layer 7)

Self-improving with safety guardrails:

```
Proposal → Verification → Policy Guard → Apply/Rollback
```

| Component | Purpose |
|-----------|---------|
| Performance Intelligence | System-wide trend detection |
| Failure Intelligence | Root-cause analysis & correlation |
| Capability Evolution | Proposes capability improvements |
| Architecture Optimizer | Proposes structural changes |
| Experimentation Engine | A/B testing for proposed changes |
| Verification Layer | Static/simulation/canary gates |
| Policy Guard | Safety policy enforcement |
| Evolution Memory | Learning history & impact tracking |

## Package Structure

```
infrarely/
├── core/           # Agent, Result, Config, Events, Decorators, Streaming
├── runtime/        # Workflow, async, sandbox, scaling + multi-agent kernel
├── router/         # Intent classification, tool routing
├── agent/          # Execution pipeline, state machine, planning
├── memory/         # Memory subsystem (SDK + execution layers)
├── security/       # Prompt injection, compliance, sanitization
├── observability/  # Metrics, traces, logging, dashboard
├── adaptive/       # Layer 5: self-optimizing intelligence
├── evolution/      # Layer 7: autonomous evolution
├── advanced/       # HITL, evaluation, versioning, marketplace
├── tools/          # Tool base classes and registry
├── capabilities/   # Multi-step capability definitions
├── integrations/   # GitHub, Gmail, Slack, Postgres, etc.
├── internal/       # Execution engine bridges (private)
└── cli/            # Command-line interface
```
