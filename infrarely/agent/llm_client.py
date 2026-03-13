"""
agent/llm_client.py
═══════════════════════════════════════════════════════════════════════════════
Unified LLM client — Ollama | Groq | Gemini.

Two public functions
────────────────────
  llm_call(...)       — make one inference call; returns (text, tokens)
  check_llm_health()  — probe backend at startup; returns LLMStatus

Error classification
─────────────────────
  ConnectionRefusedError / urlopen Errno 111
    → Ollama not running.  Fix: `ollama serve`

  HTTP 404
    → Model not pulled.  Fix: `ollama pull <model>`

  Missing API key
    → env var not set.  Fix: export GROQ_API_KEY / GEMINI_API_KEY

  All failures → (clean human-readable message, 0 tokens)
    Never exposes raw Python exception text to the user.

Usage
─────
    from infrarely.agent.llm_client import llm_call, check_llm_health
    status = check_llm_health()
    if not status.ok:
        print(status.fix_hint)
    text, tokens = llm_call(messages=[...], reason="practice_questions")
"""

from __future__ import annotations
import json
import socket
from dataclasses import dataclass
from typing import List, Dict, Tuple

try:
    # Package mode: python -m student_agent.main
    from .. import infrarely.core.app_config as config
    from ..observability import logger
except ImportError:
    # Script fallback
    import infrarely.core.app_config as config
    from infrarely.observability import logger


# ── Token governance hook (CP5) ────────────────────────────────────────────
_token_governor = None


def register_token_governor(budget):
    """Register a TokenBudget instance to receive all LLM call records."""
    global _token_governor
    _token_governor = budget
    logger.info("Token governor registered for LLM client")


def _record_governance(tokens: int, reason: str):
    """Record LLM usage with the registered token governor."""
    if _token_governor is not None:
        _token_governor.record(tokens, reason)


# ── LLM health status ─────────────────────────────────────────────────────────
@dataclass
class LLMStatus:
    ok:        bool
    backend:   str
    model:     str
    message:   str   = ""
    fix_hint:  str   = ""

    def __str__(self) -> str:
        if self.ok:
            return f"✅ LLM ready  [{self.backend}] {self.model}"
        return f"⚠️  LLM unavailable  [{self.backend}]  {self.message}\n   → {self.fix_hint}"


def check_llm_health() -> LLMStatus:
    """
    Probe the configured LLM backend without making an inference call.
    Call this once at startup so problems are surfaced immediately,
    not mid-conversation.

    Returns an LLMStatus.  Never raises.
    """
    backend = config.LLM_BACKEND.lower()
    model   = config.LLM_MODEL

    try:
        if backend == "ollama":
            return _probe_ollama(model)
        elif backend == "groq":
            return _probe_groq(model)
        elif backend == "gemini":
            return _probe_gemini(model)
        else:
            return LLMStatus(
                ok=False, backend=backend, model=model,
                message=f"Unknown backend '{backend}'",
                fix_hint="Set LLM_BACKEND=ollama | groq | gemini  in config.py or env",
            )
    except Exception as exc:
        return LLMStatus(
            ok=False, backend=backend, model=model,
            message=str(exc),
            fix_hint="Check your config.py and backend installation.",
        )


# ── main inference call ───────────────────────────────────────────────────────
def llm_call(
    messages:   List[Dict],
    system:     str = "",
    max_tokens: int = config.LLM_MAX_TOKENS,
    reason:     str = "unknown",
) -> Tuple[str, int]:
    """
    Single inference call.  Returns (response_text, total_tokens).
    On any failure returns a clean, human-readable message and 0 tokens.
    Never exposes raw exception text to the caller.
    """
    backend = config.LLM_BACKEND.lower()
    try:
        if backend == "ollama":
            text, tokens = _call_ollama(messages, system, max_tokens, reason)
        elif backend == "groq":
            text, tokens = _call_groq(messages, system, max_tokens, reason)
        elif backend == "gemini":
            text, tokens = _call_gemini(messages, system, max_tokens, reason)
        else:
            raise ValueError(f"Unknown LLM_BACKEND '{backend}'")
        _record_governance(tokens, reason)
        return text, tokens
    except Exception as exc:
        clean_msg = _classify_error(exc, backend)
        logger.error(f"LLM call failed [{backend}/{reason}]: {exc}")
        return clean_msg, 0


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA  (local, fully free)
# Install: https://ollama.com
# Pull a model: ollama pull llama3.2
# ─────────────────────────────────────────────────────────────────────────────
def _call_ollama(messages, system, max_tokens, reason) -> Tuple[str, int]:
    import urllib.request

    # Prepend system message if provided
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload = json.dumps({
        "model":   config.LLM_MODEL,
        "messages": full_messages,
        "stream":  False,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": max_tokens,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    text = data.get("message", {}).get("content", "")

    # Ollama reports token counts in eval_count / prompt_eval_count
    prompt_tokens     = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    total             = prompt_tokens + completion_tokens

    logger.llm_log(prompt_tokens, completion_tokens, reason=reason,
                   backend="ollama", model=config.LLM_MODEL)
    return text, total


# ─────────────────────────────────────────────────────────────────────────────
# GROQ  (free tier, OpenAI-compatible)
# Get key: https://console.groq.com
# export GROQ_API_KEY="gsk_..."
# pip install groq
# ─────────────────────────────────────────────────────────────────────────────
def _call_groq(messages, system, max_tokens, reason) -> Tuple[str, int]:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. "
            "Get a free key at https://console.groq.com and run:\n"
            "  export GROQ_API_KEY='gsk_...'"
        )

    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "groq package not installed. Run:\n  pip install groq"
        )

    client = Groq(api_key=config.GROQ_API_KEY)

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=config.LLM_TEMPERATURE,
    )

    text             = resp.choices[0].message.content
    prompt_tokens    = resp.usage.prompt_tokens
    completion_tokens = resp.usage.completion_tokens
    total            = prompt_tokens + completion_tokens

    logger.llm_log(prompt_tokens, completion_tokens, reason=reason,
                   backend="groq", model=config.LLM_MODEL)
    return text, total


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI  (free tier via Google AI Studio)
# Get key: https://aistudio.google.com/app/apikey
# export GEMINI_API_KEY="AIza..."
# pip install google-generativeai
# ─────────────────────────────────────────────────────────────────────────────
def _call_gemini(messages, system, max_tokens, reason) -> Tuple[str, int]:
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. "
            "Get a free key at https://aistudio.google.com/app/apikey and run:\n"
            "  export GEMINI_API_KEY='AIza...'"
        )

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed. Run:\n"
            "  pip install google-generativeai"
        )

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction=system or None,
    )

    # Convert OpenAI-style messages to Gemini format
    gemini_history = []
    last_user_msg  = ""
    for m in messages:
        role    = "user" if m["role"] == "user" else "model"
        content = m["content"]
        if m == messages[-1] and role == "user":
            last_user_msg = content   # send separately as the prompt
        else:
            gemini_history.append({"role": role, "parts": [content]})

    chat = model.start_chat(history=gemini_history)
    resp = chat.send_message(
        last_user_msg,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=config.LLM_TEMPERATURE,
        ),
    )

    text = resp.text

    # Gemini usage metadata (may be None on free tier)
    try:
        prompt_tokens     = resp.usage_metadata.prompt_token_count
        completion_tokens = resp.usage_metadata.candidates_token_count
    except Exception:
        prompt_tokens, completion_tokens = 0, 0

    total = prompt_tokens + completion_tokens
    logger.llm_log(prompt_tokens, completion_tokens, reason=reason,
                   backend="gemini", model=config.LLM_MODEL)
    return text, total


# ── error classifier ──────────────────────────────────────────────────────────
def _classify_error(exc: Exception, backend: str) -> str:
    """
    Convert any exception into a clean, actionable user-facing message.
    No raw Python exception text ever reaches the user.
    """
    msg = str(exc).lower()

    # Connection refused (Errno 111) — service not running
    if "111" in msg or "connection refused" in msg:
        if backend == "ollama":
            return (
                "Ollama is not running.\n"
                "Fix: open a terminal and run  ollama serve\n"
                "     then in another terminal: ollama pull llama3.2"
            )
        return f"{backend.title()} service is unreachable. Check your network or API key."

    # Model not found
    if "404" in msg or "not found" in msg or "pull" in msg:
        return (
            f"Model '{config.LLM_MODEL}' not found in Ollama.\n"
            f"Fix: ollama pull {config.LLM_MODEL}"
        )

    # Timeout
    if "timeout" in msg or "timed out" in msg:
        return (
            f"LLM request timed out ({backend}).\n"
            "The model may still be loading — try again in a moment."
        )

    # Auth / API key
    if "401" in msg or "403" in msg or "api key" in msg or "unauthorized" in msg:
        if backend == "groq":
            return "Invalid Groq API key.\nFix: export GROQ_API_KEY='gsk_...'"
        if backend == "gemini":
            return "Invalid Gemini API key.\nFix: export GEMINI_API_KEY='AIza...'"
        return "Authentication failed. Check your API key."

    # Rate limit
    if "429" in msg or "rate limit" in msg:
        return f"Rate limit hit on {backend}. Wait a moment and try again."

    # Generic fallback — still no raw exception
    return (
        f"LLM unavailable ({backend}). "
        "Practice questions and open-ended answers need a running LLM. "
        "All other features work without one."
    )


# ── startup probes ────────────────────────────────────────────────────────────
def _probe_ollama(model: str) -> LLMStatus:
    """Check if Ollama is listening and the model is available."""
    import urllib.request, urllib.error

    base = config.OLLAMA_BASE_URL  # e.g. http://localhost:11434

    # 1. Is Ollama running at all?
    try:
        host = base.replace("http://", "").replace("https://", "").split(":")[0]
        port = int(base.split(":")[-1]) if ":" in base.split("//")[-1] else 11434
        with socket.create_connection((host, port), timeout=2):
            pass
    except (ConnectionRefusedError, OSError):
        return LLMStatus(
            ok=False, backend="ollama", model=model,
            message="Ollama is not running (port not open)",
            fix_hint=(
                "Start Ollama:  ollama serve\n"
                f"   Pull model:  ollama pull {model}"
            ),
        )

    # 2. Is the specific model available?
    try:
        req  = urllib.request.Request(f"{base}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data   = json.loads(resp.read())
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            model_base = model.split(":")[0]
            if model_base not in models:
                return LLMStatus(
                    ok=False, backend="ollama", model=model,
                    message=f"Model '{model}' not pulled  (available: {models or 'none'})",
                    fix_hint=f"ollama pull {model}",
                )
    except Exception as e:
        # Tags endpoint failed — Ollama is up but something else is wrong
        return LLMStatus(
            ok=False, backend="ollama", model=model,
            message=f"Ollama responded but model check failed: {e}",
            fix_hint=f"ollama pull {model}",
        )

    return LLMStatus(ok=True, backend="ollama", model=model,
                     message="Ollama running, model available")


def _probe_groq(model: str) -> LLMStatus:
    if not config.GROQ_API_KEY:
        return LLMStatus(
            ok=False, backend="groq", model=model,
            message="GROQ_API_KEY is not set",
            fix_hint="export GROQ_API_KEY='gsk_...'  (get key at console.groq.com)",
        )
    try:
        from groq import Groq
    except ImportError:
        return LLMStatus(
            ok=False, backend="groq", model=model,
            message="groq package not installed",
            fix_hint="pip install groq",
        )
    # Lightweight auth check — list models endpoint
    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        client.models.list()
        return LLMStatus(ok=True, backend="groq", model=model,
                         message="API key valid, Groq reachable")
    except Exception as e:
        return LLMStatus(
            ok=False, backend="groq", model=model,
            message=str(e),
            fix_hint="Check your GROQ_API_KEY at console.groq.com",
        )


def _probe_gemini(model: str) -> LLMStatus:
    if not config.GEMINI_API_KEY:
        return LLMStatus(
            ok=False, backend="gemini", model=model,
            message="GEMINI_API_KEY is not set",
            fix_hint="export GEMINI_API_KEY='AIza...'  (get key at aistudio.google.com)",
        )
    try:
        import google.generativeai as genai
    except ImportError:
        return LLMStatus(
            ok=False, backend="gemini", model=model,
            message="google-generativeai package not installed",
            fix_hint="pip install google-generativeai",
        )
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        genai.list_models()
        return LLMStatus(ok=True, backend="gemini", model=model,
                         message="API key valid, Gemini reachable")
    except Exception as e:
        return LLMStatus(
            ok=False, backend="gemini", model=model,
            message=str(e),
            fix_hint="Check your GEMINI_API_KEY at aistudio.google.com/app/apikey",
        )


# ── backend implementations ───────────────────────────────────────────────────
def _call_ollama(messages, system, max_tokens, reason) -> Tuple[str, int]:
    import urllib.request

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload = json.dumps({
        "model":    config.LLM_MODEL,
        "messages": full_messages,
        "stream":   False,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": max_tokens,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    text              = data.get("message", {}).get("content", "")
    prompt_tokens     = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    total             = prompt_tokens + completion_tokens

    logger.llm_log(prompt_tokens, completion_tokens,
                   reason=reason, backend="ollama", model=config.LLM_MODEL)
    return text, total


def _call_groq(messages, system, max_tokens, reason) -> Tuple[str, int]:
    from groq import Groq

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    resp = Groq(api_key=config.GROQ_API_KEY).chat.completions.create(
        model=config.LLM_MODEL,
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=config.LLM_TEMPERATURE,
    )
    text              = resp.choices[0].message.content
    prompt_tokens     = resp.usage.prompt_tokens
    completion_tokens = resp.usage.completion_tokens
    total             = prompt_tokens + completion_tokens

    logger.llm_log(prompt_tokens, completion_tokens,
                   reason=reason, backend="groq", model=config.LLM_MODEL)
    return text, total


def _call_gemini(messages, system, max_tokens, reason) -> Tuple[str, int]:
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    gmodel = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        system_instruction=system or None,
    )

    gemini_history = []
    last_user_msg  = ""
    for m in messages:
        role    = "user" if m["role"] == "user" else "model"
        content = m["content"]
        if m is messages[-1] and role == "user":
            last_user_msg = content
        else:
            gemini_history.append({"role": role, "parts": [content]})

    chat = gmodel.start_chat(history=gemini_history)
    resp = chat.send_message(
        last_user_msg,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=config.LLM_TEMPERATURE,
        ),
    )
    text = resp.text
    try:
        prompt_tokens     = resp.usage_metadata.prompt_token_count
        completion_tokens = resp.usage_metadata.candidates_token_count
    except Exception:
        prompt_tokens, completion_tokens = 0, 0

    total = prompt_tokens + completion_tokens
    logger.llm_log(prompt_tokens, completion_tokens,
                   reason=reason, backend="gemini", model=config.LLM_MODEL)
    return text, total