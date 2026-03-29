"""
Example: Human-in-the-Loop Approval Gates
═══════════════════════════════════════════════════════════════════════════════
Shows how to require human approval before executing sensitive operations.

Run: python examples/hitl_approval.py
"""

import infrarely
import threading

# ── Configure ─────────────────────────────────────────────────────────────────
infrarely.configure(llm_provider="openai", log_file_enabled=False)


# ── Define a sensitive tool ───────────────────────────────────────────────────
@infrarely.tool
def process_payment(amount: float, account: str) -> dict:
    """Process a payment to an account."""
    return {"status": "processed", "amount": amount, "account": account}


@infrarely.tool
def send_email(to: str, subject: str) -> str:
    """Send an email."""
    return f"Email sent to {to}: {subject}"


# ── Create agent with approval gates ─────────────────────────────────────────
agent = infrarely.agent("payment-bot", tools=[process_payment, send_email])

# Require approval for payments over $100
agent.require_approval_for(
    tools=["process_payment"],
    when=lambda amount, **_: amount > 100,
    timeout=10,  # 10 seconds for this demo
)


# ── Simulate an approver in a background thread ──────────────────────────────
def auto_approver():
    """Simulates a human approver watching the queue."""
    import time

    time.sleep(1)  # Wait for request to be created
    pending = infrarely.approvals.get_pending()
    for req in pending:
        print(f"\n  [Approver] Reviewing: {req.reason}")
        print(f"  [Approver] ✓ Approved!")
        infrarely.approvals.approve(req.request_id, approved_by="demo-approver")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Small payment — no approval needed (below $100 threshold)
    print("─── Small Payment (no approval needed) ───")
    result = agent.run("Process payment of 50 to account savings")
    print(f"Result: {result.output}")
    print(f"Success: {result.success}")

    # Large payment — approval required
    print("\n─── Large Payment (approval required) ───")
    print("[Agent] Requesting approval for large payment...")

    # Start auto-approver in background
    approver = threading.Thread(target=auto_approver, daemon=True)
    approver.start()

    result = agent.run("Process payment of 5000 to account checking")
    print(f"\nResult: {result.output}")
    print(f"Success: {result.success}")

    infrarely.shutdown()
