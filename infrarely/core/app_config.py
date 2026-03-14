"""
config.py — Central configuration for the Student Life Assistant.
All tuneable knobs live here so the rest of the codebase stays clean.
"""

import os
from dataclasses import dataclass

from infrarely.runtime.paths import LOG_DIR as RUNTIME_LOG_DIR
from infrarely.runtime.paths import LOG_FILE as RUNTIME_LOG_FILE
from infrarely.runtime.paths import TRACE_DIR as RUNTIME_TRACE_DIR

# ─── LLM Backend ─────────────────────────────────────────────────────────────
# Choose ONE backend by setting LLM_BACKEND:
#
#   "ollama"  — fully local, free forever
#               Install: https://ollama.com  then: ollama pull llama3.2
#               No API key needed.
#
#   "groq"    — free API tier, very fast
#               Get key: https://console.groq.com
#               Set env: export GROQ_API_KEY="gsk_..."
#
#   "gemini"  — free API tier (Google)
#               Get key: https://aistudio.google.com/app/apikey
#               Set env: export GEMINI_API_KEY="AIza..."
#
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")  # ollama | groq | gemini

# ─── Model names per backend ──────────────────────────────────────────────────
LLM_MODELS = {
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),  # or mistral, gemma2, etc.
    "groq": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
}
LLM_MODEL = LLM_MODELS.get(LLM_BACKEND, LLM_MODELS["groq"])

# ─── API Keys (read from environment, never hardcoded) ────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── Ollama endpoint (default local install) ──────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Shared LLM Settings ─────────────────────────────────────────────────────
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.3

TOKEN_WARN_THRESHOLD = 300
TOKEN_DAILY_BUDGET = 5_000

# ─── Memory Settings ─────────────────────────────────────────────────────────
WORKING_MEMORY_MAX_TURNS = 6
LONG_TERM_SUMMARY_TRIGGER = 10
LONG_TERM_SUMMARY_MAX_TOKENS = 150
MAX_CONTEXT_TOKENS_FOR_LLM = 800

# ─── Router Settings ─────────────────────────────────────────────────────────
ROUTER_CONFIDENCE_THRESHOLD = 0.55
AMBIGUITY_FALLBACK = "llm_general"

# ─── Fault Injection Settings ────────────────────────────────────────────────
FAULT_INJECTION_ENABLED = False
FAULT_PROBABILITY = 0.3
FAULT_TIMEOUT_SECONDS = 0.5
MAX_RETRIES = 2
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_RESET_SECONDS = 30

# ─── Infrastructure Gaps (Layer 3) ──────────────────────────────────────────
MAX_EXECUTION_DEPTH = 8
ENABLE_PERMISSION_POLICY = True
STRICT_TOOL_VALIDATION = True
MAX_TOOL_DATA_CHARS = 50_000
ENABLE_ERROR_RECOVERY = True
ENABLE_REASONING_ENGINE = True
ENABLE_TOOL_SANDBOX = True
TOOL_TIMEOUT_SECONDS = 5.0
SANDBOX_MAX_OUTPUT_BYTES = 102_400
SCHEDULER_MAX_QUEUE = 50
SCHEDULER_MAX_WORKERS = 1
ENABLE_EXECUTION_TRACE = True
MAX_TRACE_FILES = 200  # Trace retention policy (Gap 3)

# ─── Data Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = str(RUNTIME_LOG_DIR)

PROFILE_FILE = os.path.join(DATA_DIR, "student_profiles.json")
COURSES_FILE = os.path.join(DATA_DIR, "courses.json")
ASSIGNMENTS_FILE = os.path.join(DATA_DIR, "assignments.json")
NOTES_FILE = os.path.join(DATA_DIR, "notes.json")
CALENDAR_FILE = os.path.join(DATA_DIR, "calendar_events.json")
SUMMARY_FILE = os.path.join(DATA_DIR, "conversation_summaries.json")

# ─── Observability ───────────────────────────────────────────────────────────
ENABLE_RICH_LOGGING = True
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE = str(RUNTIME_LOG_FILE)
TRACE_DIR = str(RUNTIME_TRACE_DIR)
# ── Layer 5: Adaptive Agent Intelligence ─────────────────────────────────
ENABLE_ADAPTIVE = True
ADAPTIVE_ROUTING_OPTIMIZER = True
ADAPTIVE_PARAM_INFERENCE = True
ADAPTIVE_OPTIMIZATION_EVERY = 100  # analyse every N capability executions
ADAPTIVE_FAILURE_REPORT_EVERY = 50  # write failure report every N failures
ADAPTIVE_TOKEN_LLM_RATIO_SOFT = 0.10  # soft warning when LLM ratio exceeds this
ADAPTIVE_TOKEN_LLM_RATIO_HARD = 0.20  # hard block when LLM ratio exceeds this
ADAPTIVE_SKILL_MAX = 200
ADAPTIVE_CAPABILITY_MAX = 50
ADAPTIVE_QUALITY_THRESHOLD = 0.60  # flag capabilities below this score

# ── Layer 6: Multi-Agent Runtime ─────────────────────────────────────────
ENABLE_RUNTIME = True
RUNTIME_MAX_AGENTS = 50  # max registered agents
RUNTIME_MESSAGE_RATE_LIMIT = 50  # messages/sec per agent (Gap 2)
RUNTIME_MESSAGE_BUS_CAPACITY = 2000  # total messages across all inboxes
RUNTIME_MESSAGE_INBOX_CAP = 100  # per-agent inbox depth
RUNTIME_MESSAGE_TTL_MS = 60_000  # message time-to-live in ms (Gap 1)
RUNTIME_SCHEDULER_QUEUE = 200  # max queued tasks
RUNTIME_SCHEDULER_STRATEGY = "capability"  # capability | priority | round_robin
RUNTIME_SHARED_MEMORY_MAX = 1000  # max shared memory entries
RUNTIME_SHARED_MEMORY_LOCK_TIMEOUT = 30.0  # lock auto-expire seconds (Gap 4)
RUNTIME_GLOBAL_TOKEN_CEILING = 10_000  # total tokens across all agents (Gap 7)
RUNTIME_AGENT_IDLE_TIMEOUT_MS = 300_000  # 5 min idle → GC (Gap 9)
RUNTIME_AGENT_MAX_RESTARTS = 3  # crash recovery restart limit (Gap 8)
RUNTIME_NEGOTIATION_BID_TIMEOUT_MS = 10_000  # bidding window (Gap 10)
RUNTIME_NEGOTIATION_MAX_CONCURRENT = 20
RUNTIME_GC_INTERVAL_MS = 60_000  # lifecycle GC sweep interval

# ── Layer 6 Gap Modules ──────────────────────────────────────────────────
# GAP 1 — Task Graph
RUNTIME_TASK_GRAPH_MAX_NODES = 500
RUNTIME_TASK_GRAPH_MAX_RETRIES = 3
# GAP 2 — Deadlock Detector
RUNTIME_DEADLOCK_TIMEOUT_S = 30.0
RUNTIME_DEADLOCK_CHECK_INTERVAL_MS = 5_000
# GAP 3 — Priority Scheduler
RUNTIME_PRIORITY_QUEUE_SIZE = 200
RUNTIME_PRIORITY_AGING_BOOST = 1
RUNTIME_PRIORITY_STARVATION_THRESHOLD = 10
# GAP 4 — Capability Reputation
RUNTIME_REPUTATION_WINDOW = 100
RUNTIME_REPUTATION_SUSPENSION_THRESHOLD = 0.3
# GAP 5 — State Persistence
RUNTIME_PERSISTENCE_MAX_SNAPSHOTS = 50
# GAP 6 — Distributed Scalability
RUNTIME_DISTRIBUTED_MAX_NODES = 10
RUNTIME_DISTRIBUTED_HEARTBEAT_TIMEOUT_S = 30.0
RUNTIME_DISTRIBUTED_NODE_CAPACITY = 50
# GAP 7 — Capability Load Balancer
RUNTIME_LB_STRATEGY = "round_robin"  # round_robin | least_loaded | random
RUNTIME_LB_MAX_CONCURRENT_PER_AGENT = 10
# GAP 8 — Negotiation Timeouts
RUNTIME_NEGOTIATION_BID_TIMEOUT = 10_000  # ms
RUNTIME_NEGOTIATION_AUCTION_CLOSE = 15_000  # ms
# GAP 9 — Global Resource Governance
RUNTIME_CLUSTER_TOKEN_CEILING = 100_000
RUNTIME_CLUSTER_CPU_CEILING_S = 600.0
RUNTIME_AGENT_DEFAULT_TOKEN_QUOTA = 2_000
RUNTIME_AGENT_DEFAULT_CPU_QUOTA_S = 30.0
# GAP 10 — Security Sandbox
RUNTIME_SANDBOX_ENABLED = True
RUNTIME_SANDBOX_MAX_VIOLATIONS = 5

# ─── Default Student ─────────────────────────────────────────────────────────
DEFAULT_STUDENT_ID = "student_1"


@dataclass
class AppConfig:
    """Runtime-mutable config bundle passed through the agent."""

    fault_injection: bool = False
    verbose: bool = False
    student_id: str = DEFAULT_STUDENT_ID
    token_count_session: int = 0
    llm_call_count_session: int = 0
    tool_call_count_session: int = 0

    def reset_session_counters(self):
        self.token_count_session = 0
        self.llm_call_count_session = 0
        self.tool_call_count_session = 0
