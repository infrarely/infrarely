#!/usr/bin/env python3
"""
infrarely-cli — InfraRely SDK Command-Line Interface
═══════════════════════════════════════════════════════════════════════════════
Interactive and scripted access to the InfraRely — Reliable Agent Infrastructure.

Usage:
    python -m infrarely.cli                     # Interactive REPL
    python -m infrarely.cli run "What is 2+2?"  # One-shot query
    python -m infrarely.cli health              # System health
    python -m infrarely.cli metrics             # Metrics summary
    python -m infrarely.cli traces              # Recent traces
    python -m infrarely.cli agents              # List agents
    python -m infrarely.cli test                # Run test suite
    python -m infrarely.cli info                # SDK info
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import readline  # enables arrow-key history in input()

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import infrarely
from infrarely.observability.observability import get_logger, get_metrics
from infrarely.core.config import get_config

# ─── Colors ───────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_run(args):
    """Run a goal against an agent."""
    agent_name = args.agent or "cli-agent"
    agent = infrarely.agent(agent_name)

    if args.knowledge:
        for kf in args.knowledge:
            if os.path.isfile(kf):
                with open(kf) as f:
                    agent.knowledge.add_data(os.path.basename(kf), f.read())
                print(f"{DIM}  Added knowledge: {kf}{RESET}")
            else:
                agent.knowledge.add_data("inline", kf)
                print(f"{DIM}  Added inline knowledge{RESET}")

    goal = " ".join(args.goal)
    print(f"\n{DIM}  Agent: {agent_name}{RESET}")
    print(f"{DIM}  Goal:  {goal}{RESET}\n")

    t0 = time.time()
    result = agent.run(goal)
    elapsed = (time.time() - t0) * 1000

    if result.success:
        print(f"  {GREEN}{BOLD}{result.output}{RESET}")
    else:
        print(
            f"  {RED}Error: {result.error.message if result.error else 'Unknown'}{RESET}"
        )
        if result.error and result.error.suggestion:
            print(f"  {YELLOW}Suggestion: {result.error.suggestion}{RESET}")

    print(f"\n{DIM}  ─────────────────────────────────────{RESET}")
    print(f"{DIM}  Success:    {result.success}{RESET}")
    print(f"{DIM}  Confidence: {result.confidence:.2f}{RESET}")
    print(f"{DIM}  Used LLM:   {result.used_llm}{RESET}")
    print(f"{DIM}  Sources:    {result.sources}{RESET}")
    print(f"{DIM}  Duration:   {elapsed:.1f}ms{RESET}")
    print(f"{DIM}  Trace ID:   {result.trace_id}{RESET}")

    if args.explain:
        print(f"\n{result.explain()}")


def cmd_health(args):
    """Show system health."""
    agent_name = args.agent
    if agent_name:
        agent = infrarely.agent(agent_name)
        h = agent.health()
        print(f"\n  {BOLD}Agent Health: {agent_name}{RESET}")
        print(f"  Status:         {_color_status(h.status)}")
        print(f"  State:          {h.state}")
        print(f"  Tasks:          {h.task_count}")
        print(f"  Errors:         {h.error_count}")
        print(f"  Uptime:         {h.uptime_seconds:.1f}s")
        print(f"  Memory Keys:    {h.memory_keys}")
        print(f"  Tools:          {h.tools_registered}")
    else:
        h = infrarely.health()
        print(f"\n  {BOLD}System Health{RESET}")
        print(f"  SDK Version:    {h.get('sdk_version', 'unknown')}")
        print(f"  Active Agents:  {h.get('agents', 0)}")
        for name, report in h.get("agent_reports", {}).items():
            print(f"    {CYAN}{name}{RESET}: {report}")
    print()


def cmd_metrics(args):
    """Show metrics summary."""
    m = infrarely.metrics.summary()
    print(f"\n  {BOLD}Metrics Summary{RESET}")
    print(f"  Total Tasks:    {m.get('total_tasks', 0)}")
    print(f"  Successful:     {m.get('successful_tasks', 0)}")
    print(f"  Failed:         {m.get('failed_tasks', 0)}")
    rate = m.get("success_rate", 0)
    print(f"  Success Rate:   {_color_pct(rate)}")
    bypass = m.get("llm_bypass_rate", 0)
    print(f"  LLM Bypass:     {_color_pct(bypass)}")
    print(f"  Avg Duration:   {m.get('avg_duration_ms', 0):.1f}ms")
    print(f"  Total Tokens:   {m.get('total_tokens', 0)}")

    if args.json:
        print(f"\n{DIM}{json.dumps(m, indent=2)}{RESET}")
    print()


def cmd_traces(args):
    """Show recent execution traces."""
    agent_name = args.agent or "cli-agent"
    agent = infrarely.agent(agent_name)
    limit = args.limit or 10
    traces = agent.get_recent_traces(limit=limit)

    if not traces:
        print(f"\n  {DIM}No traces found for agent '{agent_name}'.{RESET}\n")
        return

    print(f"\n  {BOLD}Recent Traces for '{agent_name}' (last {limit}){RESET}\n")
    print(f"  {'Trace ID':<16} {'Goal':<35} {'Status':<6} {'Duration':<10} {'LLM'}")
    print(f"  {'─' * 16} {'─' * 35} {'─' * 6} {'─' * 10} {'─' * 5}")
    for t in traces:
        status = f"{GREEN}OK{RESET}" if t.success else f"{RED}FAIL{RESET}"
        goal = t.goal[:33] + ".." if len(t.goal) > 35 else t.goal
        print(
            f"  {t.trace_id[:14]:<16} {goal:<35} {status:<15} {t.duration_ms:>7.1f}ms  {t.used_llm if hasattr(t, 'used_llm') else '?'}"
        )
    print()


def cmd_agents(args):
    """List all active agents."""
    from infrarely.core.agent import _all_agents

    agents = _all_agents()

    if not agents:
        print(f"\n  {DIM}No active agents.{RESET}\n")
        return

    print(f"\n  {BOLD}Active Agents ({len(agents)}){RESET}\n")
    for name, ag in agents.items():
        try:
            h = ag.health()
            print(
                f"  {CYAN}{name:<20}{RESET} state={h.state:<12} tasks={h.task_count:<5} status={_color_status(h.status)}"
            )
        except Exception as e:
            print(f"  {CYAN}{name:<20}{RESET} {RED}(error: {e}){RESET}")
    print()


def cmd_info(args):
    """Show SDK info."""
    cfg = get_config()
    print(f"\n  {BOLD}InfraRely SDK Info{RESET}")
    print(f"  Version:        {infrarely.__version__}")
    print(f"  Provider:       {cfg.get('llm_provider', 'N/A')}")
    print(f"  Model:          {cfg.get('llm_model', 'N/A')}")
    print(f"  API Key:        {'✔ set' if cfg.get('api_key') else '✘ not set'}")
    print(f"  Log Level:      {cfg.get('log_level', 'N/A')}")
    print(f"  Memory Backend: {cfg.get('memory_backend', 'N/A')}")
    print(f"  Max Retries:    {cfg.get('max_retries', 'N/A')}")
    print(f"  Max Agents:     {cfg.get('max_agents', 'N/A')}")

    from infrarely.core.decorators import ToolRegistry, CapabilityRegistry

    tools = ToolRegistry().all_tools()
    caps = CapabilityRegistry().all_capabilities()
    print(f"  Registered Tools:        {len(tools)}")
    print(f"  Registered Capabilities: {len(caps)}")
    print()


def cmd_logs(args):
    """Show recent log entries."""
    logger = get_logger()
    level = args.level.upper() if args.level else None
    limit = args.limit or 20
    entries = logger.get_entries(level=level, limit=limit)

    if not entries:
        print(f"\n  {DIM}No log entries found.{RESET}\n")
        return

    print(f"\n  {BOLD}Recent Logs{RESET} (level={level or 'ALL'}, limit={limit})\n")
    for e in entries:
        lvl = e.get("level", "?")
        color = {"DEBUG": DIM, "INFO": GREEN, "WARNING": YELLOW, "ERROR": RED}.get(
            lvl, RESET
        )
        ts = e.get("ts", "")[-15:]  # last 15 chars of timestamp
        msg = e.get("message", "")
        extras = {k: v for k, v in e.items() if k not in ("ts", "level", "message")}
        extra_str = (
            f" {DIM}({', '.join(f'{k}={v}' for k, v in extras.items())}){RESET}"
            if extras
            else ""
        )
        print(f"  {DIM}{ts}{RESET} {color}{lvl:7s}{RESET} {msg}{extra_str}")
    print()


def cmd_test(args):
    """Run the test suite."""
    test_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "infrarely_test.py"
    )
    if not os.path.exists(test_file):
        # Try parent dir
        test_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "infrarely_test.py"
        )
    if os.path.exists(test_file):
        section_arg = (
            f"--section {args.section}"
            if hasattr(args, "section") and args.section
            else ""
        )
        os.system(f"{sys.executable} {test_file} {section_arg}")
    else:
        print(
            f"{RED}  infrarely_test.py not found. Run from the infrarely directory.{RESET}"
        )


def cmd_repl(args):
    """Interactive REPL mode."""
    print(
        f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗
║          InfraRely SDK — Interactive Agent Shell                   ║
║          Type a goal or command. Press Ctrl+C to exit.       ║
╚══════════════════════════════════════════════════════════════╝{RESET}

  {DIM}Commands:
    /help           Show this help
    /health         System health
    /metrics        Show metrics
    /traces         Recent traces
    /agents         List agents
    /logs [level]   Show logs
    /info           SDK info
    /explain        Toggle auto-explain
    /agent <name>   Switch agent
    /exit           Exit{RESET}
"""
    )

    agent_name = "cli-agent"
    agent = infrarely.agent(agent_name)
    auto_explain = False

    while True:
        try:
            prompt = f"{CYAN}infrarely:{agent_name}{RESET}> "
            line = input(prompt).strip()
            if not line:
                continue

            # Commands
            if line.startswith("/"):
                parts = line.split()
                cmd = parts[0].lower()

                if cmd == "/exit" or cmd == "/quit":
                    break
                elif cmd == "/help":
                    print(
                        f"""
  {DIM}/health         System health check
  /metrics        Metrics summary
  /traces [N]     Recent N traces (default 5)
  /agents         List active agents
  /logs [level]   Show log entries (DEBUG/INFO/WARNING/ERROR)
  /info           SDK configuration info
  /explain        Toggle auto-explain on results
  /agent <name>   Switch to a different agent
  /memory         Show memory keys
  /knowledge <text>  Add inline knowledge
  /exit           Exit the shell{RESET}
"""
                    )
                elif cmd == "/health":
                    h = agent.health()
                    print(
                        f"  Status: {_color_status(h.status)}, Tasks: {h.task_count}, Errors: {h.error_count}, Uptime: {h.uptime_seconds:.0f}s"
                    )
                elif cmd == "/metrics":
                    m = infrarely.metrics.summary()
                    print(
                        f"  Tasks: {m.get('total_tasks', 0)}, Success: {m.get('success_rate', 0):.0%}, LLM Bypass: {m.get('llm_bypass_rate', 0):.0%}"
                    )
                elif cmd == "/traces":
                    limit = int(parts[1]) if len(parts) > 1 else 5
                    traces = agent.get_recent_traces(limit=limit)
                    for t in traces:
                        status = (
                            f"{GREEN}OK{RESET}" if t.success else f"{RED}FAIL{RESET}"
                        )
                        print(
                            f"  [{t.trace_id[:8]}] {t.goal[:40]:<40} {status} {t.duration_ms:.0f}ms"
                        )
                    if not traces:
                        print(f"  {DIM}No traces yet.{RESET}")
                elif cmd == "/agents":
                    from infrarely.core.agent import _all_agents

                    for n in _all_agents():
                        marker = " ←" if n == agent_name else ""
                        print(f"  {CYAN}{n}{RESET}{marker}")
                elif cmd == "/logs":
                    level = parts[1].upper() if len(parts) > 1 else None
                    entries = get_logger().get_entries(level=level, limit=10)
                    for e in entries:
                        color = {
                            "DEBUG": DIM,
                            "INFO": GREEN,
                            "WARNING": YELLOW,
                            "ERROR": RED,
                        }.get(e["level"], RESET)
                        print(f"  {color}{e['level']:7s}{RESET} {e['message']}")
                    if not entries:
                        print(f"  {DIM}No entries.{RESET}")
                elif cmd == "/info":
                    print(f"  Version: {infrarely.__version__}, Agent: {agent_name}")
                elif cmd == "/explain":
                    auto_explain = not auto_explain
                    print(f"  Auto-explain: {'ON' if auto_explain else 'OFF'}")
                elif cmd == "/agent":
                    if len(parts) > 1:
                        agent_name = parts[1]
                        agent = infrarely.agent(agent_name)
                        print(f"  Switched to agent: {CYAN}{agent_name}{RESET}")
                    else:
                        print(f"  Current: {CYAN}{agent_name}{RESET}")
                elif cmd == "/memory":
                    keys = agent.memory.list_keys()
                    for k in keys:
                        v = agent.memory.get(k)
                        print(f"  {k}: {str(v)[:60]}")
                    if not keys:
                        print(f"  {DIM}No memory entries.{RESET}")
                elif cmd == "/knowledge":
                    text = " ".join(parts[1:])
                    if text:
                        agent.knowledge.add_data("repl_input", text)
                        print(f"  {GREEN}Knowledge added.{RESET}")
                    else:
                        print(f"  Usage: /knowledge <text to add>")
                else:
                    print(f"  {RED}Unknown command: {cmd}. Type /help.{RESET}")
                continue

            # Run as goal
            result = agent.run(line)
            if result.success:
                print(f"  {GREEN}{result.output}{RESET}")
            else:
                print(
                    f"  {RED}Error: {result.error.message if result.error else 'Unknown'}{RESET}"
                )

            print(
                f"  {DIM}[confidence={result.confidence:.2f}, llm={result.used_llm}, {result.duration_ms:.0f}ms]{RESET}"
            )

            if auto_explain:
                print(f"\n{result.explain()}")

        except KeyboardInterrupt:
            print(f"\n  {DIM}Goodbye!{RESET}")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}")

    infrarely.shutdown()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _color_status(status: str) -> str:
    if status == "healthy":
        return f"{GREEN}{status}{RESET}"
    elif status == "degraded":
        return f"{YELLOW}{status}{RESET}"
    return f"{RED}{status}{RESET}"


def _color_pct(val: float) -> str:
    pct = f"{val:.1%}"
    if val >= 0.9:
        return f"{GREEN}{pct}{RESET}"
    elif val >= 0.5:
        return f"{YELLOW}{pct}{RESET}"
    return f"{RED}{pct}{RESET}"


# ═══════════════════════════════════════════════════════════════════════════════
# NEW CLI COMMANDS: init, eval, inspect, security, versions, approvals
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_init(args):
    """Initialize a new InfraRely project with boilerplate."""
    project_name = args.name or "my_agent"
    project_dir = os.path.join(os.getcwd(), project_name)

    if os.path.exists(project_dir):
        print(f"  {RED}Directory '{project_name}' already exists.{RESET}")
        return

    os.makedirs(project_dir, exist_ok=True)

    # Create main.py
    main_py = f'''"""
{project_name} — InfraRely Agent
Generated by: infrarely init
"""
import infrarely

# Configure (set your API key via env: OPENAI_API_KEY)
infrarely.configure(
    llm_provider="openai",
    log_level="INFO",
)

# Define tools
@infrarely.tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {{name}}!"

# Create agent
agent = infrarely.agent("{project_name}", tools=[greet])

# Run
if __name__ == "__main__":
    result = agent.run("Greet Alice")
    print(result.output)
    print(result.explain())
'''

    # Create requirements.txt
    requirements = "infrarely>=0.1.0\n"

    # Create .env.example
    env_example = """# InfraRely Configuration
# Copy to .env and fill in your values
INFRARELY_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
"""

    # Create eval.yaml
    eval_yaml = f"""# Evaluation suite for {project_name}
# Run with: aos eval {project_name}/eval.yaml
name: {project_name}-evals
cases:
  - input: "Greet Alice"
    expected_output: "Hello, Alice!"
    expected_used_llm: false
    expected_confidence_min: 0.8
"""

    with open(os.path.join(project_dir, "main.py"), "w") as f:
        f.write(main_py)
    with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
        f.write(requirements)
    with open(os.path.join(project_dir, ".env.example"), "w") as f:
        f.write(env_example)
    with open(os.path.join(project_dir, "eval.yaml"), "w") as f:
        f.write(eval_yaml)

    print(f"\n  {GREEN}{BOLD}Project '{project_name}' created!{RESET}")
    print(f"\n  {DIM}Files:{RESET}")
    print(f"    {project_name}/main.py          — Agent code")
    print(f"    {project_name}/requirements.txt — Dependencies")
    print(f"    {project_name}/.env.example     — Configuration template")
    print(f"    {project_name}/eval.yaml        — Evaluation suite")
    print(f"\n  {DIM}Next steps:{RESET}")
    print(f"    cd {project_name}")
    print(f"    cp .env.example .env")
    print(f"    python main.py")
    print()


def cmd_eval(args):
    """Run an evaluation suite."""
    eval_path = args.path

    if not eval_path:
        print(f"  {RED}Usage: infrarely eval <path-to-eval.yaml>{RESET}")
        return

    if not os.path.exists(eval_path):
        print(f"  {RED}File not found: {eval_path}{RESET}")
        return

    try:
        from infrarely.platform.evaluation import EvalSuite

        suite = EvalSuite.from_yaml(eval_path)
    except Exception as e:
        print(f"  {RED}Failed to load eval suite: {e}{RESET}")
        return

    agent_name = args.agent or "eval-agent"
    agent = infrarely.agent(agent_name)

    if args.knowledge:
        for kf in args.knowledge:
            if os.path.isfile(kf):
                with open(kf) as f:
                    agent.knowledge.add_data(os.path.basename(kf), f.read())

    print(f"\n  {BOLD}Running evaluation suite: {suite._name}{RESET}")
    print(f"  {DIM}Cases: {len(suite._cases)}, Agent: {agent_name}{RESET}\n")

    results = suite.run(agent)

    # Print results
    for cr in results.case_results:
        status = f"{GREEN}PASS{RESET}" if cr.passed else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {cr.case.input[:50]}")
        if not cr.passed:
            for f_detail in cr.failures:
                print(f"        {YELLOW}{f_detail}{RESET}")

    print(f"\n  {BOLD}Results:{RESET}")
    print(f"  Pass Rate: {_color_pct(results.pass_rate)}")
    print(f"  Passed:    {results.passed_count}/{results.total_count}")
    print(f"  Duration:  {results.total_duration_ms:.0f}ms")

    if results.regression_report:
        print(f"\n  {BOLD}Regression Report:{RESET}")
        print(f"  {results.regression_report}")
    print()


def cmd_inspect(args):
    """Inspect an agent's configuration, tools, and state."""
    agent_name = args.agent or "cli-agent"
    agent = infrarely.agent(agent_name)

    print(f"\n  {BOLD}Agent Inspection: {agent_name}{RESET}")
    print(f"  ─────────────────────────────────────")
    print(f"  State:          {agent.state}")
    print(f"  Description:    {agent.description}")

    # Tools
    tools = agent.tools
    print(f"\n  {BOLD}Tools ({len(tools)}):{RESET}")
    from infrarely.core.decorators import get_tool_registry

    registry = get_tool_registry()
    for tname in tools:
        meta = registry.get_meta(tname)
        if meta:
            print(f"    {CYAN}{tname}{RESET}: {meta.description[:60]}")
            if meta.params:
                for p in meta.params:
                    print(f"      {DIM}param: {p.name} ({p.type_str}){RESET}")
        else:
            print(f"    {CYAN}{tname}{RESET}")

    # Capabilities
    caps = agent.capabilities
    if caps:
        print(f"\n  {BOLD}Capabilities ({len(caps)}):{RESET}")
        for cname in caps:
            print(f"    {CYAN}{cname}{RESET}")

    # Memory
    try:
        keys = agent.memory.list_keys()
        print(f"\n  {BOLD}Memory ({len(keys)} keys):{RESET}")
        for k in keys[:10]:
            v = agent.memory.get(k)
            print(f"    {k}: {str(v)[:50]}")
        if len(keys) > 10:
            print(f"    {DIM}... and {len(keys) - 10} more{RESET}")
    except Exception:
        print(f"\n  {BOLD}Memory:{RESET} {DIM}unavailable{RESET}")

    # HITL rules
    if hasattr(agent, "_hitl_gate") and agent._hitl_gate:
        rules = agent._hitl_gate.rules
        if rules:
            print(f"\n  {BOLD}Approval Rules ({len(rules)}):{RESET}")
            for r in rules:
                print(f"    Tools: {r.tools}, Timeout: {r.timeout}s")

    print()


def cmd_security(args):
    """Show security status and audit log."""
    from infrarely.security.security import get_security_guard, get_audit_log

    guard = get_security_guard()
    audit = get_audit_log()

    print(f"\n  {BOLD}Security Status{RESET}")
    print(f"  ─────────────────────────────────────")

    policy = guard.policy
    print(
        f"  Injection Detection: {'Enabled' if policy.prompt_injection_detection else 'Disabled'}"
    )
    print(f"  Action on Threat:    {policy.injection_action}")
    print(f"  Audit All Inputs:    {policy.audit_all_inputs}")
    print(f"  Max Input Length:    {policy.max_input_length}")

    if args.scan:
        text = " ".join(args.scan)
        result = guard.scan(text)
        print(f"\n  {BOLD}Scan Result:{RESET}")
        if result.is_threat:
            print(f"  {RED}THREAT DETECTED{RESET}")
            print(f"  Level:   {result.threat_level.value}")
            print(
                f"  Type:    {result.injection_type.value if result.injection_type else 'unknown'}"
            )
            print(f"  Score:   {result.confidence:.2f}")
            for p in result.matched_patterns:
                print(f"  Pattern: {YELLOW}{p}{RESET}")
        else:
            print(f"  {GREEN}No threats detected{RESET}")

    entries = audit.get_entries(limit=args.limit or 10)
    if entries:
        print(f"\n  {BOLD}Recent Audit Entries ({len(entries)}):{RESET}")
        for e in entries:
            color = GREEN if e.action_taken == "allowed" else RED
            print(
                f"  {DIM}{time.strftime('%H:%M:%S', time.localtime(e.timestamp))}{RESET} {color}{e.action_taken:10s}{RESET} agent={e.agent_name or '-'}"
            )
    print()


def cmd_versions(args):
    """Manage agent versions."""
    from infrarely.platform.versioning import VersionManager

    vm = VersionManager()

    if args.action == "list":
        agent_name = args.agent or None
        all_versions = vm.list_versions(agent_name)
        if not all_versions:
            print(f"\n  {DIM}No saved versions.{RESET}\n")
            return
        print(f"\n  {BOLD}Saved Versions{RESET}")
        for v in all_versions:
            print(
                f"  {CYAN}{v.tag:<20}{RESET} agent={v.agent_name:<15} {DIM}{time.strftime('%Y-%m-%d %H:%M', time.localtime(v.created_at))}{RESET}"
            )
        print()

    elif args.action == "save":
        if not args.agent or not args.tag:
            print(f"  {RED}Usage: infrarely versions save --agent <name> --tag <tag>{RESET}")
            return
        agent = infrarely.agent(args.agent)
        vm.save(agent, tag=args.tag, description=args.description or "")
        print(f"  {GREEN}Saved version '{args.tag}' for agent '{args.agent}'{RESET}")

    elif args.action == "rollback":
        if not args.agent or not args.tag:
            print(
                f"  {RED}Usage: infrarely versions rollback --agent <name> --tag <tag>{RESET}"
            )
            return
        agent = infrarely.agent(args.agent)
        vm.rollback(agent, tag=args.tag)
        print(
            f"  {GREEN}Rolled back agent '{args.agent}' to version '{args.tag}'{RESET}"
        )

    else:
        print(f"  {DIM}Usage: infrarely versions [list|save|rollback]{RESET}")


def cmd_approvals(args):
    """Manage HITL approval requests."""
    from infrarely.platform.hitl import get_approval_manager

    mgr = get_approval_manager()

    if args.action == "list":
        pending = mgr.get_pending()
        if not pending:
            print(f"\n  {DIM}No pending approval requests.{RESET}\n")
            return
        print(f"\n  {BOLD}Pending Approvals ({len(pending)}){RESET}")
        for r in pending:
            print(
                f"  {CYAN}{r.request_id[:16]}{RESET} agent={r.agent_name} tool={r.tool_name}"
            )
            print(f"    {DIM}Reason: {r.reason[:60]}{RESET}")
        print()

    elif args.action == "approve":
        if not args.request_id:
            print(f"  {RED}Usage: infrarely approvals approve <request_id>{RESET}")
            return
        mgr.approve(args.request_id)
        print(f"  {GREEN}Approved: {args.request_id}{RESET}")

    elif args.action == "reject":
        if not args.request_id:
            print(f"  {RED}Usage: infrarely approvals reject <request_id>{RESET}")
            return
        mgr.reject(args.request_id, reason=args.reason or "")
        print(f"  {RED}Rejected: {args.request_id}{RESET}")

    else:
        print(f"  {DIM}Usage: infrarely approvals [list|approve|reject]{RESET}")


def cmd_install(args):
    """Install a marketplace package."""
    from infrarely.platform.marketplace import get_marketplace

    mp = get_marketplace()
    version = getattr(args, "version", "") or ""
    ok, msg = mp.install(args.package, version=version)
    if ok:
        print(f"\n  {GREEN}{BOLD}{msg}{RESET}\n")
    else:
        print(f"\n  {RED}{msg}{RESET}\n")


def cmd_uninstall(args):
    """Uninstall a marketplace package."""
    from infrarely.platform.marketplace import get_marketplace

    mp = get_marketplace()
    ok, msg = mp.uninstall(args.package)
    if ok:
        print(f"\n  {GREEN}{msg}{RESET}\n")
    else:
        print(f"\n  {RED}{msg}{RESET}\n")


def cmd_publish(args):
    """Publish a capability to the marketplace."""
    from infrarely.platform.marketplace import get_marketplace

    mp = get_marketplace()
    ok, errors = mp.publish(
        name=args.name,
        version=args.version,
        description=args.description,
        author=args.author,
        category=args.category,
        tags=args.tags or [],
    )
    if ok:
        print(f"\n  {GREEN}{BOLD}Published {args.name}@{args.version}{RESET}\n")
    else:
        print(f"\n  {RED}Publish failed:{RESET}")
        for e in errors:
            print(f"    {YELLOW}{e}{RESET}")
        print()


def cmd_marketplace(args):
    """Browse and search the marketplace."""
    from infrarely.platform.marketplace import get_marketplace

    mp = get_marketplace()

    action = args.action

    if action == "search":
        query = args.query or ""
        category = getattr(args, "category", "") or ""
        scope = getattr(args, "scope", "") or ""
        results = mp.search(query, category=category, scope=scope)
        if not results:
            print(f"\n  {DIM}No packages found.{RESET}\n")
            return
        print(f"\n  {BOLD}Search Results ({len(results)}){RESET}\n")
        for m in results:
            verified = f" {GREEN}\u2713{RESET}" if m.verified else ""
            print(f"  {CYAN}{m.name}{RESET}@{m.version}{verified}")
            print(f"    {DIM}{m.description[:70]}{RESET}")
            if m.tags:
                print(f"    {DIM}Tags: {', '.join(m.tags)}{RESET}")
        print()

    elif action == "info":
        name = args.query
        if not name:
            print(f"  {RED}Usage: infrarely marketplace info <package-name>{RESET}")
            return
        meta = mp.info(name)
        if not meta:
            print(f"\n  {RED}Package not found: {name}{RESET}\n")
            return
        installed = mp.is_installed(name)
        print(f"\n  {BOLD}{meta.name}@{meta.version}{RESET}")
        print(f"  {meta.description}")
        print(f"  Author:     {meta.author or 'N/A'}")
        print(f"  Category:   {meta.category}")
        print(f"  License:    {meta.license}")
        print(f"  Downloads:  {meta.downloads}")
        print(f"  Tags:       {', '.join(meta.tags) if meta.tags else 'none'}")
        print(f"  Installed:  {'\u2713 yes' if installed else '\u2717 no'}")
        print()

    elif action == "installed":
        pkgs = mp.list_installed()
        if not pkgs:
            print(f"\n  {DIM}No packages installed.{RESET}\n")
            return
        print(f"\n  {BOLD}Installed Packages ({len(pkgs)}){RESET}\n")
        for p in pkgs:
            print(f"  {CYAN}{p.name}{RESET}@{p.version}")
        print()

    elif action == "outdated":
        outdated = mp.outdated()
        if not outdated:
            print(f"\n  {GREEN}All packages are up to date.{RESET}\n")
            return
        print(f"\n  {BOLD}Outdated Packages ({len(outdated)}){RESET}\n")
        for name, current, latest in outdated:
            print(f"  {CYAN}{name}{RESET} {current} \u2192 {YELLOW}{latest}{RESET}")
        print()

    else:  # list
        pkgs = mp.list_available()
        if not pkgs:
            print(f"\n  {DIM}No packages in marketplace.{RESET}\n")
            return
        print(f"\n  {BOLD}Available Packages ({len(pkgs)}){RESET}\n")
        for m in pkgs:
            verified = f" {GREEN}\u2713{RESET}" if m.verified else ""
            print(
                f"  {CYAN}{m.name}{RESET}@{m.version}{verified}  {DIM}{m.description[:50]}{RESET}"
            )
        print()


def cmd_benchmark(args):
    """Run performance benchmarks."""
    from infrarely.platform.benchmark import run_benchmark, list_baselines

    suite_name = getattr(args, "tasks", "standard-suite-v1")
    vs = getattr(args, "vs", None)
    verbose = getattr(args, "verbose", False)
    use_color = not getattr(args, "no_color", False)
    output_format = getattr(args, "output", "table")

    if verbose:
        print(f"\n  {BOLD}{CYAN}InfraRely Performance Benchmark{RESET}")
        print(f"  Suite: {suite_name}")
        if vs:
            print(f"  Compare vs: {', '.join(vs)}")
        print()

    result = run_benchmark(
        suite=suite_name,
        vs=vs,
        verbose=verbose,
        use_color=use_color,
        output_format=output_format,
    )

    if result["report"]:
        print(result["report"])

    if not vs and output_format == "table":
        avail = list_baselines()
        if avail:
            print(f"  {DIM}Tip: Compare with --vs {' '.join(avail.keys())}{RESET}")
            print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="infrarely",
        description="InfraRely SDK — InfraRely CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  infrarely run "What is 2+2?"              One-shot query
  infrarely run "Explain Mars" --explain    With execution trace
  infrarely run "Find info" -k notes.txt    With knowledge file
  infrarely health                          System health check
  infrarely metrics --json                  Metrics as JSON
  infrarely traces --limit 20               Recent traces
  infrarely agents                          List agents
  infrarely logs --level ERROR              Show error logs
  infrarely info                            SDK info
  infrarely test                            Run test suite
  aos                                 Interactive REPL
""",
    )

    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run a goal")
    p_run.add_argument("goal", nargs="+", help="The goal text")
    p_run.add_argument("--agent", "-a", help="Agent name (default: cli-agent)")
    p_run.add_argument(
        "--explain", "-e", action="store_true", help="Show execution trace"
    )
    p_run.add_argument(
        "--knowledge", "-k", nargs="*", help="Knowledge files or inline text"
    )

    # health
    p_health = sub.add_parser("health", help="System health")
    p_health.add_argument("--agent", "-a", help="Specific agent name")

    # metrics
    p_metrics = sub.add_parser("metrics", help="Metrics summary")
    p_metrics.add_argument("--json", action="store_true", help="Output as JSON")

    # traces
    p_traces = sub.add_parser("traces", help="Recent execution traces")
    p_traces.add_argument("--agent", "-a", help="Agent name")
    p_traces.add_argument("--limit", "-n", type=int, default=10, help="Max traces")

    # agents
    sub.add_parser("agents", help="List active agents")

    # info
    sub.add_parser("info", help="SDK info & config")

    # logs
    p_logs = sub.add_parser("logs", help="Show log entries")
    p_logs.add_argument("--level", "-l", help="Filter by level")
    p_logs.add_argument("--limit", "-n", type=int, default=20, help="Max entries")

    # test
    p_test = sub.add_parser("test", help="Run test suite")
    p_test.add_argument("--section", "-s", type=int, help="Run specific section")

    # init
    p_init = sub.add_parser("init", help="Initialize a new InfraRely project")
    p_init.add_argument("name", nargs="?", help="Project name (default: my_agent)")

    # eval
    p_eval = sub.add_parser("eval", help="Run an evaluation suite")
    p_eval.add_argument("path", nargs="?", help="Path to eval YAML file")
    p_eval.add_argument("--agent", "-a", help="Agent name")
    p_eval.add_argument("--knowledge", "-k", nargs="*", help="Knowledge files")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect agent configuration")
    p_inspect.add_argument("--agent", "-a", help="Agent name")

    # security
    p_security = sub.add_parser("security", help="Security status & audit")
    p_security.add_argument("--scan", nargs="*", help="Scan text for injection")
    p_security.add_argument(
        "--limit", "-n", type=int, default=10, help="Audit entries limit"
    )

    # versions
    p_versions = sub.add_parser("versions", help="Manage agent versions")
    p_versions.add_argument(
        "action", nargs="?", default="list", choices=["list", "save", "rollback"]
    )
    p_versions.add_argument("--agent", "-a", help="Agent name")
    p_versions.add_argument("--tag", "-t", help="Version tag")
    p_versions.add_argument("--description", "-d", help="Version description")

    # approvals
    p_approvals = sub.add_parser("approvals", help="Manage HITL approval requests")
    p_approvals.add_argument(
        "action", nargs="?", default="list", choices=["list", "approve", "reject"]
    )
    p_approvals.add_argument("request_id", nargs="?", help="Approval request ID")
    p_approvals.add_argument("--reason", help="Rejection reason")

    # install
    p_install = sub.add_parser("install", help="Install a marketplace package")
    p_install.add_argument(
        "package", help="Package name (e.g. @community/web-researcher)"
    )
    p_install.add_argument("--version", "-v", help="Specific version")

    # uninstall
    p_uninstall = sub.add_parser("uninstall", help="Uninstall a marketplace package")
    p_uninstall.add_argument("package", help="Package name")

    # publish
    p_publish = sub.add_parser("publish", help="Publish a capability to marketplace")
    p_publish.add_argument("name", help="Package name (e.g. @community/my-cap)")
    p_publish.add_argument("--version", "-v", default="0.1.0", help="Version")
    p_publish.add_argument("--description", "-d", default="", help="Description")
    p_publish.add_argument("--author", default="", help="Author name")
    p_publish.add_argument("--category", default="other", help="Category")
    p_publish.add_argument("--tags", nargs="*", help="Tags")

    # marketplace (search / list / info / outdated)
    p_marketplace = sub.add_parser("marketplace", help="Browse & search marketplace")
    p_marketplace.add_argument(
        "action",
        nargs="?",
        default="list",
        choices=["list", "search", "info", "installed", "outdated"],
    )
    p_marketplace.add_argument(
        "query", nargs="?", default="", help="Search query or package name"
    )
    p_marketplace.add_argument("--category", help="Filter by category")
    p_marketplace.add_argument("--scope", help="Filter by scope")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run performance benchmarks")
    p_bench.add_argument(
        "--vs",
        nargs="*",
        help="Frameworks to compare (langchain, crewai, autogpt, autogen, dspy)",
    )
    p_bench.add_argument(
        "--tasks",
        default="standard-suite-v1",
        help="Benchmark suite name (default: standard-suite-v1)",
    )
    p_bench.add_argument(
        "--output",
        choices=["table", "json", "summary"],
        default="table",
        help="Output format",
    )
    p_bench.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p_bench.add_argument("--no-color", action="store_true", help="Disable color output")

    args = parser.parse_args()

    # Suppress file logging for CLI unless configured
    infrarely.configure(log_level="WARNING", log_file_enabled=False)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    elif args.command == "traces":
        cmd_traces(args)
    elif args.command == "agents":
        cmd_agents(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "logs":
        cmd_logs(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "security":
        cmd_security(args)
    elif args.command == "versions":
        cmd_versions(args)
    elif args.command == "approvals":
        cmd_approvals(args)
    elif args.command == "install":
        cmd_install(args)
    elif args.command == "uninstall":
        cmd_uninstall(args)
    elif args.command == "publish":
        cmd_publish(args)
    elif args.command == "marketplace":
        cmd_marketplace(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        # No subcommand → interactive REPL
        cmd_repl(args)


if __name__ == "__main__":
    main()
