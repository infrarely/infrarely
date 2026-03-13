"""
aos/nlconfig.py — Natural Language Agent Configuration
═══════════════════════════════════════════════════════════════════════════════
Describe your agent in plain English and AOS builds the configuration
automatically.

    agent = aos.agent.from_description(\"\"\"
        A customer support agent for an e-commerce store.
        It should answer questions about orders, products, and returns.
        It must never discuss competitor products.
        It should escalate to a human if the customer is angry.
        It should always be polite and professional.
    \"\"\")
    # → Auto-generates tools, capabilities, guardrails, knowledge schema

Zero external dependencies — pure pattern-matching and rule-based NLP.

Architecture:
  DescriptionParser     — Extracts structured intent from plain English
  AgentBlueprint        — Intermediate representation of parsed config
  GuardrailRule         — A safety/content guardrail extracted from description
  EscalationTrigger     — When to hand off to a human
  TopicScope            — Allowed and blocked topics
  PersonalityProfile    — Tone, style, behaviour traits
  BlueprintCompiler     — Compiles blueprint into Agent config + post-init hooks
  NLConfigurator        — High-level façade: description string → configured Agent
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

NLCONFIG_VERSION = "1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class Tone(Enum):
    """Personality tone."""

    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"
    TECHNICAL = "technical"
    NEUTRAL = "neutral"


class AgentRole(Enum):
    """High-level agent role categories."""

    SUPPORT = "support"
    RESEARCH = "research"
    WRITING = "writing"
    CODING = "coding"
    EDUCATION = "education"
    SALES = "sales"
    DATA = "data"
    ASSISTANT = "assistant"
    MODERATION = "moderation"
    CUSTOM = "custom"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GuardrailRule:
    """A safety or content guardrail extracted from the description."""

    rule_type: str  # "block_topic", "require_tone", "content_filter", "custom"
    description: str  # human-readable rule
    pattern: str = ""  # regex or keyword pattern
    action: str = "block"  # "block" | "warn" | "rewrite"
    severity: str = "high"  # "low" | "medium" | "high" | "critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_type": self.rule_type,
            "description": self.description,
            "pattern": self.pattern,
            "action": self.action,
            "severity": self.severity,
        }


@dataclass
class EscalationTrigger:
    """When to escalate to a human."""

    condition: str  # description of the condition
    trigger_type: str  # "sentiment", "keyword", "confidence", "topic", "custom"
    keywords: List[str] = field(default_factory=list)
    threshold: float = 0.0  # for confidence-based triggers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "trigger_type": self.trigger_type,
            "keywords": list(self.keywords),
            "threshold": self.threshold,
        }


@dataclass
class TopicScope:
    """Allowed and blocked topics."""

    allowed: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": list(self.allowed), "blocked": list(self.blocked)}


@dataclass
class PersonalityProfile:
    """Tone, style, and behaviour traits."""

    tones: List[str] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    communication_style: str = "neutral"
    formality: str = "medium"  # "low" | "medium" | "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tones": list(self.tones),
            "traits": list(self.traits),
            "communication_style": self.communication_style,
            "formality": self.formality,
        }


@dataclass
class SuggestedTool:
    """A tool suggested by the NL configurator."""

    name: str
    description: str
    category: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": dict(self.parameters),
        }


@dataclass
class SuggestedCapability:
    """A capability suggested by the NL configurator."""

    name: str
    description: str
    category: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
        }


@dataclass
class KnowledgeSchema:
    """Suggested knowledge domains."""

    domains: List[str] = field(default_factory=list)
    suggested_sources: List[str] = field(default_factory=list)
    auto_populate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": list(self.domains),
            "suggested_sources": list(self.suggested_sources),
            "auto_populate": self.auto_populate,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT BLUEPRINT — intermediate representation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AgentBlueprint:
    """
    Complete parsed configuration from a natural language description.

    This is the intermediate representation between raw text and a
    configured AOS Agent.
    """

    # Identity
    name: str = ""
    description: str = ""
    role: str = "custom"
    domain: str = ""

    # Configuration
    personality: PersonalityProfile = field(default_factory=PersonalityProfile)
    topics: TopicScope = field(default_factory=TopicScope)
    guardrails: List[GuardrailRule] = field(default_factory=list)
    escalation_triggers: List[EscalationTrigger] = field(default_factory=list)
    suggested_tools: List[SuggestedTool] = field(default_factory=list)
    suggested_capabilities: List[SuggestedCapability] = field(default_factory=list)
    knowledge_schema: KnowledgeSchema = field(default_factory=KnowledgeSchema)

    # Overrides
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    source_description: str = ""
    parsed_at: float = field(default_factory=time.time)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "domain": self.domain,
            "personality": self.personality.to_dict(),
            "topics": self.topics.to_dict(),
            "guardrails": [g.to_dict() for g in self.guardrails],
            "escalation_triggers": [e.to_dict() for e in self.escalation_triggers],
            "suggested_tools": [t.to_dict() for t in self.suggested_tools],
            "suggested_capabilities": [
                c.to_dict() for c in self.suggested_capabilities
            ],
            "knowledge_schema": self.knowledge_schema.to_dict(),
            "config_overrides": dict(self.config_overrides),
            "source_description": self.source_description,
            "parsed_at": self.parsed_at,
            "confidence": self.confidence,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DESCRIPTION PARSER — rule-based NLP
# ═══════════════════════════════════════════════════════════════════════════════

# ── Role detection patterns ───────────────────────────────────────────────────
_ROLE_PATTERNS: Dict[str, List[str]] = {
    "support": [
        r"customer\s+support",
        r"help\s+desk",
        r"support\s+agent",
        r"service\s+agent",
        r"customer\s+service",
        r"helpdesk",
        r"customer\s+care",
        r"support\s+bot",
    ],
    "research": [
        r"research",
        r"investigate",
        r"analyst",
        r"find\s+information",
        r"gather\s+data",
        r"look\s+up",
        r"deep\s+dive",
    ],
    "writing": [
        r"writ(?:e|ing|er)",
        r"content\s+creation",
        r"copywrite",
        r"blog",
        r"article",
        r"essay",
        r"draft",
    ],
    "coding": [
        r"cod(?:e|ing)",
        r"program",
        r"develop",
        r"software",
        r"debug",
        r"engineer",
        r"implement",
    ],
    "education": [
        r"tutor",
        r"teach",
        r"educat",
        r"learn",
        r"lesson",
        r"course",
        r"student",
        r"instruct",
    ],
    "sales": [
        r"sales",
        r"sell",
        r"commerce",
        r"e-commerce",
        r"store",
        r"shop",
        r"retail",
        r"product",
    ],
    "data": [
        r"data\s+analy",
        r"dashboard",
        r"report",
        r"metric",
        r"statistic",
        r"database",
        r"sql",
    ],
    "assistant": [
        r"assistant",
        r"personal\s+assist",
        r"virtual\s+assist",
        r"secretary",
        r"scheduler",
        r"organiz",
    ],
    "moderation": [
        r"moderat",
        r"content\s+filter",
        r"spam",
        r"abuse",
        r"flagg",
        r"review\s+content",
    ],
}

# ── Domain keywords ───────────────────────────────────────────────────────────
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "e-commerce": [
        "order",
        "product",
        "return",
        "shipping",
        "cart",
        "checkout",
        "e-commerce",
        "ecommerce",
        "store",
        "shop",
        "purchase",
    ],
    "healthcare": [
        "health",
        "medical",
        "patient",
        "doctor",
        "diagnosis",
        "symptom",
        "treatment",
        "clinic",
        "hospital",
    ],
    "finance": [
        "finance",
        "bank",
        "invest",
        "stock",
        "portfolio",
        "payment",
        "transaction",
        "accounting",
        "tax",
    ],
    "education": [
        "student",
        "course",
        "lesson",
        "grade",
        "assignment",
        "curriculum",
        "lecture",
        "exam",
        "tutor",
    ],
    "technology": [
        "software",
        "code",
        "api",
        "deploy",
        "server",
        "database",
        "cloud",
        "devops",
    ],
    "legal": [
        "legal",
        "law",
        "contract",
        "compliance",
        "regulation",
        "attorney",
        "court",
        "patent",
    ],
    "hr": [
        "hr",
        "human resource",
        "recruit",
        "hiring",
        "employee",
        "onboard",
        "performance review",
        "benefit",
    ],
    "marketing": [
        "marketing",
        "campaign",
        "brand",
        "seo",
        "social media",
        "advertisement",
        "content marketing",
    ],
    "real-estate": [
        "real estate",
        "property",
        "listing",
        "mortgage",
        "rent",
        "tenant",
        "landlord",
    ],
}

# ── Tone keywords ─────────────────────────────────────────────────────────────
_TONE_KEYWORDS: Dict[str, List[str]] = {
    "professional": ["professional", "business-like", "corporate"],
    "friendly": ["friendly", "warm", "personable", "approachable"],
    "formal": ["formal", "official", "dignified"],
    "casual": ["casual", "relaxed", "informal", "laid-back"],
    "empathetic": ["empathetic", "compassionate", "understanding", "caring"],
    "technical": ["technical", "precise", "detailed", "accurate"],
    "polite": ["polite", "courteous", "respectful", "civil"],
}

# ── Negative/blocking patterns ────────────────────────────────────────────────
_BLOCK_PATTERNS: List[Tuple[str, str]] = [
    (
        r"(?:must|should|shall)\s+never\s+(?:discuss|mention|talk\s+about|reference|bring\s+up)\s+(.+?)(?:\.|$)",
        "block_topic",
    ),
    (
        r"(?:do|does)\s+not\s+(?:discuss|mention|talk\s+about|reference)\s+(.+?)(?:\.|$)",
        "block_topic",
    ),
    (
        r"(?:avoid|refuse|decline)\s+(?:discussing|mentioning|talking\s+about)\s+(.+?)(?:\.|$)",
        "block_topic",
    ),
    (
        r"never\s+(?:discuss|mention|talk\s+about|reference)\s+(.+?)(?:\.|$)",
        "block_topic",
    ),
    (r"(?:no|without)\s+(?:discussing|mentioning)\s+(.+?)(?:\.|$)", "block_topic"),
]

# ── Escalation patterns ──────────────────────────────────────────────────────
_ESCALATION_PATTERNS: List[Tuple[str, str, List[str]]] = [
    (
        r"escalat\w*\s+(?:to\s+)?(?:a\s+)?human\s+(?:if|when)\s+.*(angry|upset|frustrated|furious|irate|hostile)",
        "sentiment",
        ["angry", "upset", "frustrated", "furious", "irate", "hostile"],
    ),
    (
        r"escalat\w*\s+(?:to\s+)?(?:a\s+)?human\s+(?:if|when)\s+.*((?:can(?:'t|not)|unable)\s+(?:answer|help|resolve|handle))",
        "confidence",
        [],
    ),
    (
        r"(?:hand\s+off|transfer|redirect)\s+(?:to\s+)?(?:a\s+)?(?:human|person|agent|manager)\s+(?:if|when)\s+(.*?)(?:\.|$)",
        "keyword",
        [],
    ),
    (
        r"escalat\w*\s+(?:to\s+)?(?:a\s+)?(?:human|person|agent|manager)\s+(?:if|when)\s+(.*?)(?:\.|$)",
        "keyword",
        [],
    ),
    (
        r"(?:involve|engage|contact)\s+(?:a\s+)?(?:human|person|supervisor)\s+(?:for|if|when)\s+(.*?)(?:\.|$)",
        "keyword",
        [],
    ),
]

# ── Allowed topic patterns ────────────────────────────────────────────────────
_ALLOWED_PATTERNS: List[str] = [
    r"(?:answer|handle|address|respond\s+to)\s+(?:questions?\s+(?:about|regarding|on)\s+)(.+?)(?:\.|$)",
    r"(?:help\s+(?:with|users?\s+with)\s+)(.+?)(?:\.|$)",
    r"(?:assist\s+(?:with|in)\s+)(.+?)(?:\.|$)",
    r"(?:manage|handle)\s+(.+?)(?:\s+(?:queries|requests|questions|issues))(?:\.|$)",
    r"(?:provide\s+(?:information|help|guidance|support)\s+(?:about|for|on|regarding)\s+)(.+?)(?:\.|$)",
]

# ── Tool suggestion map per role/domain ───────────────────────────────────────
_TOOL_SUGGESTIONS: Dict[str, List[Dict[str, str]]] = {
    "support": [
        {
            "name": "lookup_order",
            "description": "Look up order details by order ID",
            "category": "data",
            "parameters": '{"order_id": "str"}',
        },
        {
            "name": "check_status",
            "description": "Check the status of a request",
            "category": "data",
            "parameters": '{"request_id": "str"}',
        },
        {
            "name": "create_ticket",
            "description": "Create a support ticket",
            "category": "action",
            "parameters": '{"subject": "str", "description": "str"}',
        },
    ],
    "e-commerce": [
        {
            "name": "search_products",
            "description": "Search the product catalog",
            "category": "data",
            "parameters": '{"query": "str"}',
        },
        {
            "name": "get_order",
            "description": "Get order details",
            "category": "data",
            "parameters": '{"order_id": "str"}',
        },
        {
            "name": "process_return",
            "description": "Process a product return",
            "category": "action",
            "parameters": '{"order_id": "str", "reason": "str"}',
        },
        {
            "name": "track_shipment",
            "description": "Track a shipment",
            "category": "data",
            "parameters": '{"tracking_id": "str"}',
        },
    ],
    "research": [
        {
            "name": "web_search",
            "description": "Search the web for information",
            "category": "data",
            "parameters": '{"query": "str"}',
        },
        {
            "name": "summarize_article",
            "description": "Summarize a long article",
            "category": "transform",
            "parameters": '{"text": "str"}',
        },
    ],
    "coding": [
        {
            "name": "run_code",
            "description": "Execute code in a sandbox",
            "category": "action",
            "parameters": '{"code": "str", "language": "str"}',
        },
        {
            "name": "lint_code",
            "description": "Lint and check code quality",
            "category": "analysis",
            "parameters": '{"code": "str"}',
        },
        {
            "name": "search_docs",
            "description": "Search documentation",
            "category": "data",
            "parameters": '{"query": "str"}',
        },
    ],
    "education": [
        {
            "name": "quiz_student",
            "description": "Generate a quiz question",
            "category": "action",
            "parameters": '{"topic": "str", "difficulty": "str"}',
        },
        {
            "name": "explain_concept",
            "description": "Explain a concept simply",
            "category": "transform",
            "parameters": '{"concept": "str"}',
        },
        {
            "name": "grade_answer",
            "description": "Grade a student's answer",
            "category": "analysis",
            "parameters": '{"question": "str", "answer": "str"}',
        },
    ],
    "sales": [
        {
            "name": "search_products",
            "description": "Search the product catalog",
            "category": "data",
            "parameters": '{"query": "str"}',
        },
        {
            "name": "get_pricing",
            "description": "Get pricing information",
            "category": "data",
            "parameters": '{"product_id": "str"}',
        },
        {
            "name": "create_quote",
            "description": "Create a sales quote",
            "category": "action",
            "parameters": '{"items": "list", "customer_id": "str"}',
        },
    ],
    "data": [
        {
            "name": "run_query",
            "description": "Run a data query",
            "category": "data",
            "parameters": '{"query": "str"}',
        },
        {
            "name": "generate_chart",
            "description": "Generate a chart from data",
            "category": "action",
            "parameters": '{"data": "dict", "chart_type": "str"}',
        },
    ],
}

# ── Capability suggestion map ─────────────────────────────────────────────────
_CAPABILITY_SUGGESTIONS: Dict[str, List[Dict[str, str]]] = {
    "support": [
        {
            "name": "ticket_resolution",
            "description": "End-to-end ticket resolution workflow",
            "category": "workflow",
        },
        {
            "name": "customer_lookup",
            "description": "Find customer info across systems",
            "category": "data",
        },
    ],
    "research": [
        {
            "name": "deep_research",
            "description": "Multi-source research with synthesis",
            "category": "workflow",
        },
        {
            "name": "fact_check",
            "description": "Verify claims against multiple sources",
            "category": "analysis",
        },
    ],
    "education": [
        {
            "name": "adaptive_lesson",
            "description": "Generate adaptive learning content",
            "category": "workflow",
        },
        {
            "name": "progress_tracking",
            "description": "Track student progress over time",
            "category": "data",
        },
    ],
}


class DescriptionParser:
    """
    Parses a natural language description into an AgentBlueprint.

    Uses pattern matching, keyword extraction, and rule-based NLP.
    No external dependencies.
    """

    def parse(self, description: str, *, name: str = "") -> AgentBlueprint:
        """
        Parse a natural language description into a structured blueprint.

        Parameters
        ----------
        description : str
            Plain English description of the desired agent.
        name : str, optional
            Agent name override. If empty, auto-generated from role.

        Returns
        -------
        AgentBlueprint
        """
        text = self._normalize(description)

        # Extract each component
        role = self._detect_role(text)
        domain = self._detect_domain(text)
        personality = self._extract_personality(text)
        topics = self._extract_topics(text)
        guardrails = self._extract_guardrails(text)
        escalation = self._extract_escalation(text)
        tools = self._suggest_tools(role, domain, text)
        capabilities = self._suggest_capabilities(role, domain)
        knowledge = self._extract_knowledge_schema(text, domain, topics)

        # Auto-generate name
        if not name:
            name = self._generate_name(role, domain)

        # Compute confidence based on how much we extracted
        confidence = self._compute_confidence(
            role, domain, personality, topics, guardrails, escalation
        )

        return AgentBlueprint(
            name=name,
            description=description.strip(),
            role=role,
            domain=domain,
            personality=personality,
            topics=topics,
            guardrails=guardrails,
            escalation_triggers=escalation,
            suggested_tools=tools,
            suggested_capabilities=capabilities,
            knowledge_schema=knowledge,
            source_description=description.strip(),
            confidence=confidence,
        )

    # ── Normalization ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize whitespace and case for matching."""
        return re.sub(r"\s+", " ", text.strip()).lower()

    # ── Role detection ────────────────────────────────────────────────────

    @staticmethod
    def _detect_role(text: str) -> str:
        """Detect the primary role from the description."""
        best_role = "custom"
        best_score = 0

        for role, patterns in _ROLE_PATTERNS.items():
            score = 0
            for pat in patterns:
                matches = re.findall(pat, text, re.IGNORECASE)
                # Multi-word phrase patterns get higher weight
                weight = 3 if r"\s" in pat else 1
                score += len(matches) * weight
            if score > best_score:
                best_score = score
                best_role = role

        return best_role

    # ── Domain detection ──────────────────────────────────────────────────

    @staticmethod
    def _detect_domain(text: str) -> str:
        """Detect the domain from keyword frequency."""
        best_domain = ""
        best_score = 0

        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_domain

    # ── Personality ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_personality(text: str) -> PersonalityProfile:
        """Extract personality traits from the description."""
        tones: List[str] = []
        traits: List[str] = []

        for tone, keywords in _TONE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    tones.append(tone)
                    break

        # Extract trait phrases
        trait_patterns = [
            r"(?:should|must)\s+(?:be|remain|stay)\s+([\w\s]+?)(?:\.|,|$)",
            r"(?:always)\s+([\w\s]+?)(?:\.|,|$)",
        ]
        for pat in trait_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                trait = m.group(1).strip()
                if trait and len(trait.split()) <= 5:
                    traits.append(trait)

        # Determine formality
        formality = "medium"
        if any(t in tones for t in ["formal", "professional"]):
            formality = "high"
        elif any(t in tones for t in ["casual", "friendly"]):
            formality = "low"

        style = "neutral"
        if tones:
            style = tones[0]

        return PersonalityProfile(
            tones=tones,
            traits=traits,
            communication_style=style,
            formality=formality,
        )

    # ── Topic extraction ──────────────────────────────────────────────────

    @staticmethod
    def _extract_topics(text: str) -> TopicScope:
        """Extract allowed and blocked topics."""
        allowed: List[str] = []
        blocked: List[str] = []

        # Allowed topics
        for pat in _ALLOWED_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                topic = m.group(1).strip().rstrip(",.")
                if topic and len(topic.split()) <= 8:
                    # Split on "and" and commas
                    parts = re.split(r",\s*|\s+and\s+", topic)
                    for p in parts:
                        p = p.strip()
                        if p:
                            allowed.append(p)

        # Blocked topics
        for pat, _ in _BLOCK_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                topic = m.group(1).strip().rstrip(",.")
                if topic and len(topic.split()) <= 8:
                    parts = re.split(r",\s*|\s+and\s+", topic)
                    for p in parts:
                        p = p.strip()
                        if p:
                            blocked.append(p)

        return TopicScope(allowed=allowed, blocked=blocked)

    # ── Guardrail extraction ──────────────────────────────────────────────

    @staticmethod
    def _extract_guardrails(text: str) -> List[GuardrailRule]:
        """Extract guardrail rules from the description."""
        rules: List[GuardrailRule] = []

        # Block-topic guardrails
        for pat, rule_type in _BLOCK_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                topic = m.group(1).strip().rstrip(",.")
                if topic:
                    rules.append(
                        GuardrailRule(
                            rule_type=rule_type,
                            description=f"Never discuss: {topic}",
                            pattern=re.escape(topic),
                            action="block",
                            severity="high",
                        )
                    )

        # Tone requirements as guardrails
        tone_reqs = re.findall(
            r"(?:should|must)\s+(?:always\s+)?(?:be|remain|stay)\s+(polite|professional|respectful|courteous|civil|formal)(.*?)(?:\.|$)",
            text,
            re.IGNORECASE,
        )
        for tone_word, _ in tone_reqs:
            rules.append(
                GuardrailRule(
                    rule_type="require_tone",
                    description=f"Must maintain {tone_word} tone",
                    pattern=tone_word.lower(),
                    action="warn",
                    severity="medium",
                )
            )

        return rules

    # ── Escalation extraction ─────────────────────────────────────────────

    @staticmethod
    def _extract_escalation(text: str) -> List[EscalationTrigger]:
        """Extract escalation triggers from the description."""
        triggers: List[EscalationTrigger] = []

        for pat, trigger_type, default_keywords in _ESCALATION_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                condition = m.group(0).strip()
                keywords = list(default_keywords) if default_keywords else []

                # Extract keywords from the matched condition
                if not keywords and trigger_type == "keyword":
                    captured = m.group(1).strip() if m.lastindex else ""
                    if captured:
                        keywords = [
                            w.strip()
                            for w in re.split(r",|\s+and\s+", captured)
                            if w.strip()
                        ]

                triggers.append(
                    EscalationTrigger(
                        condition=condition,
                        trigger_type=trigger_type,
                        keywords=keywords,
                        threshold=0.3 if trigger_type == "confidence" else 0.0,
                    )
                )

        return triggers

    # ── Tool suggestions ──────────────────────────────────────────────────

    @staticmethod
    def _suggest_tools(role: str, domain: str, text: str) -> List[SuggestedTool]:
        """Suggest tools based on role, domain, and description."""
        tools: List[SuggestedTool] = []
        seen: Set[str] = set()

        # From role
        for entry in _TOOL_SUGGESTIONS.get(role, []):
            if entry["name"] not in seen:
                tools.append(
                    SuggestedTool(
                        name=entry["name"],
                        description=entry["description"],
                        category=entry.get("category", ""),
                    )
                )
                seen.add(entry["name"])

        # From domain
        for entry in _TOOL_SUGGESTIONS.get(domain, []):
            if entry["name"] not in seen:
                tools.append(
                    SuggestedTool(
                        name=entry["name"],
                        description=entry["description"],
                        category=entry.get("category", ""),
                    )
                )
                seen.add(entry["name"])

        return tools

    # ── Capability suggestions ────────────────────────────────────────────

    @staticmethod
    def _suggest_capabilities(role: str, domain: str) -> List[SuggestedCapability]:
        """Suggest capabilities based on role and domain."""
        caps: List[SuggestedCapability] = []
        seen: Set[str] = set()

        for source in [role, domain]:
            for entry in _CAPABILITY_SUGGESTIONS.get(source, []):
                if entry["name"] not in seen:
                    caps.append(
                        SuggestedCapability(
                            name=entry["name"],
                            description=entry["description"],
                            category=entry.get("category", ""),
                        )
                    )
                    seen.add(entry["name"])

        return caps

    # ── Knowledge schema ──────────────────────────────────────────────────

    @staticmethod
    def _extract_knowledge_schema(
        text: str,
        domain: str,
        topics: TopicScope,
    ) -> KnowledgeSchema:
        """Build a knowledge schema from parsed information."""
        domains: List[str] = []
        sources: List[str] = []

        if domain:
            domains.append(domain)

        # Add allowed topics as knowledge domains
        for topic in topics.allowed:
            if topic not in domains:
                domains.append(topic)

        # Suggest knowledge sources based on patterns
        source_patterns = [
            (
                r"(?:using|from|based\s+on)\s+(?:the\s+)?([\w\s]+?)\s+(?:database|api|system|data|documentation|docs)",
                lambda m: m.group(1).strip(),
            ),
        ]
        for pat, extractor in source_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                source = extractor(m)
                if source and source not in sources:
                    sources.append(source)

        return KnowledgeSchema(
            domains=domains,
            suggested_sources=sources,
        )

    # ── Name generation ───────────────────────────────────────────────────

    @staticmethod
    def _generate_name(role: str, domain: str) -> str:
        """Auto-generate a meaningful agent name."""
        parts: List[str] = []
        if domain:
            parts.append(domain.replace("-", "_"))
        if role and role != "custom":
            parts.append(role)
        else:
            parts.append("agent")

        suffix = uuid.uuid4().hex[:4]
        return "_".join(parts) + f"_{suffix}"

    # ── Confidence scoring ────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        role: str,
        domain: str,
        personality: PersonalityProfile,
        topics: TopicScope,
        guardrails: List[GuardrailRule],
        escalation: List[EscalationTrigger],
    ) -> float:
        """Compute a confidence score for the parsing quality."""
        score = 0.0

        # Role identified (not custom)
        if role != "custom":
            score += 0.25

        # Domain identified
        if domain:
            score += 0.15

        # Personality extracted
        if personality.tones:
            score += 0.15
        if personality.traits:
            score += 0.05

        # Topics extracted
        if topics.allowed:
            score += 0.15
        if topics.blocked:
            score += 0.05

        # Guardrails
        if guardrails:
            score += 0.10

        # Escalation
        if escalation:
            score += 0.10

        return min(score, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# BLUEPRINT COMPILER — converts blueprint → Agent config
# ═══════════════════════════════════════════════════════════════════════════════


class BlueprintCompiler:
    """
    Compiles an AgentBlueprint into parameters for Agent.__init__
    and post-init hooks (guardrails, escalation, knowledge).
    """

    def compile(self, blueprint: AgentBlueprint) -> Dict[str, Any]:
        """
        Compile a blueprint into an Agent configuration dict.

        Returns a dict with keys:
          - agent_kwargs: kwargs for Agent.__init__
          - post_init_hooks: list of callable(agent) for post-creation setup
          - blueprint: the original blueprint
        """
        agent_kwargs: Dict[str, Any] = {
            "name": blueprint.name,
            "description": blueprint.description,
        }

        if blueprint.config_overrides:
            agent_kwargs["config"] = dict(blueprint.config_overrides)

        # Collect post-init hooks
        hooks: List[Callable] = []

        # Guardrails → stored on agent for inspection
        if (
            blueprint.guardrails
            or blueprint.escalation_triggers
            or blueprint.personality.tones
        ):

            def apply_blueprint_metadata(
                agent: Any, bp: AgentBlueprint = blueprint
            ) -> None:
                """Attach blueprint metadata to the agent."""
                agent._blueprint = bp
                agent._guardrails = bp.guardrails
                agent._escalation_triggers = bp.escalation_triggers
                agent._personality = bp.personality
                agent._topic_scope = bp.topics

            hooks.append(apply_blueprint_metadata)

        # Knowledge schema → add domains as knowledge data hints
        if blueprint.knowledge_schema.domains:

            def apply_knowledge_hints(
                agent: Any, ks: KnowledgeSchema = blueprint.knowledge_schema
            ) -> None:
                """Seed agent knowledge with domain hints."""
                for domain in ks.domains:
                    try:
                        agent.knowledge.add_data(
                            f"domain_{domain}",
                            f"This agent specializes in: {domain}",
                        )
                    except Exception:
                        pass

            hooks.append(apply_knowledge_hints)

        return {
            "agent_kwargs": agent_kwargs,
            "post_init_hooks": hooks,
            "blueprint": blueprint,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NL CONFIGURATOR — high-level façade
# ═══════════════════════════════════════════════════════════════════════════════


class NLConfigurator:
    """
    High-level façade: description string → configured Agent.

    Usage::

        configurator = NLConfigurator()
        agent = configurator.create_agent(\"\"\"
            A customer support agent for an e-commerce store.
            It should answer questions about orders, products, and returns.
            It must never discuss competitor products.
            It should escalate to a human if the customer is angry.
            It should always be polite and professional.
        \"\"\")

    The returned agent is a standard AOS Agent with:
      - Appropriate name and description
      - Blueprint metadata (.blueprint, .guardrails, etc.)
      - Knowledge seeded with domain hints
      - Suggested tools and capabilities (accessible via blueprint)
    """

    def __init__(self) -> None:
        self._parser = DescriptionParser()
        self._compiler = BlueprintCompiler()

    def parse(self, description: str, *, name: str = "") -> AgentBlueprint:
        """Parse a description into a blueprint (without creating an agent)."""
        return self._parser.parse(description, name=name)

    def compile(self, blueprint: AgentBlueprint) -> Dict[str, Any]:
        """Compile a blueprint into Agent config."""
        return self._compiler.compile(blueprint)

    def create_agent(
        self,
        description: str,
        *,
        name: str = "",
        tools: Optional[list] = None,
        capabilities: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a fully configured Agent from a natural language description.

        Parameters
        ----------
        description : str
            Plain English description of the agent.
        name : str, optional
            Override auto-generated name.
        tools : list, optional
            Additional tools to add.
        capabilities : list, optional
            Additional capabilities to add.
        config : dict, optional
            Additional config overrides.

        Returns
        -------
        Agent
            A configured AOS Agent.
        """
        # Lazy import to avoid circular deps
        from infrarely.core.agent import Agent

        # Parse
        blueprint = self._parser.parse(description, name=name)

        # Compile
        compiled = self._compiler.compile(blueprint)
        kwargs = compiled["agent_kwargs"]

        # Merge caller overrides
        if tools:
            kwargs["tools"] = tools
        if capabilities:
            kwargs["capabilities"] = capabilities
        if config:
            existing_config = kwargs.get("config", {})
            existing_config.update(config)
            kwargs["config"] = existing_config

        # Create agent
        agent = Agent(**kwargs)

        # Apply post-init hooks
        for hook in compiled["post_init_hooks"]:
            try:
                hook(agent)
            except Exception:
                pass

        return agent


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_nl_configurator: Optional[NLConfigurator] = None
_nl_lock = threading.Lock()


def get_nl_configurator() -> NLConfigurator:
    """Return the global NLConfigurator singleton."""
    global _nl_configurator
    with _nl_lock:
        if _nl_configurator is None:
            _nl_configurator = NLConfigurator()
        return _nl_configurator


def _reset_nl_configurator() -> None:
    """Reset singleton (for testing)."""
    global _nl_configurator
    with _nl_lock:
        _nl_configurator = None
