"""
infrarely/knowledge.py — Knowledge API
═══════════════════════════════════════════════════════════════════════════════
Ground agents in facts. Knowledge is queried BEFORE LLM.
If confidence >= threshold (default from config: 0.85), LLM is bypassed entirely.

Sources:
  infrarely.knowledge.add_database(name, source, schema)
  infrarely.knowledge.add_documents(name, path, refresh_hours)
  infrarely.knowledge.add_data(name, data)
  infrarely.knowledge.add_api(name, endpoint, refresh_minutes)
  infrarely.knowledge.query(question) → KnowledgeResult

Philosophy: LLM is the LAST resort, not the first.
  "Hallucination is an infrastructure failure, not a model failure."
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class KnowledgeChunk:
    """A single piece of knowledge with provenance."""

    content: str
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class KnowledgeResult:
    """Result from a knowledge query."""

    chunks: List[KnowledgeChunk] = field(default_factory=list)
    confidence: float = 0.0
    source_names: List[str] = field(default_factory=list)
    query: str = ""
    duration_ms: float = 0.0
    decision: str = (
        "no_knowledge"  # "bypass_llm" | "ground_llm" | "low_confidence" | "no_knowledge"
    )

    def __bool__(self) -> bool:
        return self.confidence > 0 and len(self.chunks) > 0

    def __str__(self) -> str:
        if not self.chunks:
            return "No knowledge found."
        return "\n".join(c.content for c in self.chunks[:5])


@dataclass
class KnowledgeSource:
    """A registered knowledge source."""

    name: str
    source_type: str  # "database" | "documents" | "data" | "api"
    description: str = ""
    chunk_count: int = 0
    is_authoritative: bool = False
    refresh_seconds: int = 86400  # 24h default
    last_indexed: Optional[str] = None
    source_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY VECTOR INDEX (TF-IDF based, no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════════


class _SimpleVectorIndex:
    """
    Lightweight TF-IDF based search index. No numpy, no external deps.
    Fits in 8GB RAM. Good enough for 100k chunks.
    """

    def __init__(self):
        self._chunks: List[KnowledgeChunk] = []
        self._term_freq: Dict[str, Dict[int, float]] = defaultdict(
            dict
        )  # term → {chunk_idx: freq}
        self._doc_count: Dict[str, int] = defaultdict(
            int
        )  # term → num docs containing it
        self._lock = threading.Lock()

    # Stopwords that carry no topical signal — removing them prevents
    # question-form words ("what", "do", "how") from diluting query coverage.
    _STOPWORDS: Set[str] = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "must",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "him",
        "her",
        "them",
        "his",
        "their",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "and",
        "but",
        "or",
        "if",
        "because",
        "as",
        "until",
        "while",
        "just",
        "also",
        "any",
        "many",
        "much",
        "tell",
        "me",
        "please",
        "give",
        "get",
        "find",
        "show",
        "know",
        "think",
        "want",
    }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Word tokenizer with stopword removal for topical signal."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        # Remove very short tokens and stopwords
        return [
            t for t in tokens if len(t) > 1 and t not in _SimpleVectorIndex._STOPWORDS
        ]

    def add(self, chunk: KnowledgeChunk) -> None:
        with self._lock:
            idx = len(self._chunks)
            self._chunks.append(chunk)
            tokens = self._tokenize(chunk.content)
            if not tokens:
                return
            # Term frequency for this document
            tf_counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_counts[token] += 1
            seen_terms: Set[str] = set()
            for token, count in tf_counts.items():
                self._term_freq[token][idx] = count / len(tokens)
                if token not in seen_terms:
                    self._doc_count[token] += 1
                    seen_terms.add(token)

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.01
    ) -> List[Tuple[KnowledgeChunk, float]]:
        """TF-IDF search with cosine similarity. Returns [(chunk, score), ...]."""
        with self._lock:
            if not self._chunks:
                return []
            tokens = self._tokenize(query)
            if not tokens:
                return []

            n_docs = len(self._chunks)

            # Build query TF-IDF vector
            query_tf: Dict[str, float] = defaultdict(float)
            for token in tokens:
                query_tf[token] += 1.0
            for token in query_tf:
                query_tf[token] /= len(tokens)

            # Calculate query-side IDF weights
            query_idf: Dict[str, float] = {}
            for token in query_tf:
                if token in self._term_freq:
                    query_idf[token] = math.log(
                        1 + n_docs / (1 + self._doc_count.get(token, 0))
                    )
                else:
                    query_idf[token] = 0.0

            # Query vector magnitude
            query_mag_sq = 0.0
            for token in query_tf:
                w = query_tf[token] * query_idf.get(token, 0)
                query_mag_sq += w * w
            query_mag = math.sqrt(query_mag_sq) if query_mag_sq > 0 else 0.0

            if query_mag == 0:
                # No query terms matched any document at all
                return []

            # Score each matching document using cosine similarity
            scores: Dict[int, float] = defaultdict(float)
            doc_mags: Dict[int, float] = defaultdict(float)

            matched_query_terms = set()
            for token in tokens:
                if token not in self._term_freq:
                    continue
                matched_query_terms.add(token)
                idf = math.log(1 + n_docs / (1 + self._doc_count.get(token, 0)))
                q_weight = query_tf[token] * idf
                for idx, tf in self._term_freq[token].items():
                    d_weight = tf * idf
                    scores[idx] += q_weight * d_weight

            if not scores:
                return []

            # Calculate document magnitudes for matched docs
            for idx in scores:
                mag_sq = 0.0
                for token in self._term_freq:
                    if idx in self._term_freq[token]:
                        idf = math.log(1 + n_docs / (1 + self._doc_count.get(token, 0)))
                        w = self._term_freq[token][idx] * idf
                        mag_sq += w * w
                doc_mags[idx] = math.sqrt(mag_sq) if mag_sq > 0 else 1.0

            # Cosine similarity = dot / (|q| * |d|)
            for idx in scores:
                scores[idx] /= query_mag * doc_mags[idx]

            # Apply query coverage penalty:
            # If only 1 out of 5 query terms matched, penalize heavily.
            # This prevents "quantum computing" from matching documents
            # that only share the word "the" or a common filler.
            unique_query_terms = set(tokens)
            coverage = len(matched_query_terms) / len(unique_query_terms)
            # Scale by sqrt(coverage) so partial matches still work
            coverage_factor = math.sqrt(coverage)

            for idx in scores:
                scores[idx] *= coverage_factor

            # Clamp to [0, 1]
            for idx in scores:
                scores[idx] = min(1.0, max(0.0, scores[idx]))

            # Sort by score descending
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results = []
            for idx, score in ranked[:top_k]:
                if score >= min_score:
                    results.append((self._chunks[idx], score))
            return results

    @property
    def size(self) -> int:
        return len(self._chunks)

    def clear(self):
        with self._lock:
            self._chunks.clear()
            self._term_freq.clear()
            self._doc_count.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE MANAGER — the public API
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeManager:
    """
    Global knowledge manager. Accessed via infrarely.knowledge.

    Registers sources, indexes content, answers queries.
    Knowledge is always queried before LLM (Rule 3).
    """

    _instance: Optional["KnowledgeManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "KnowledgeManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sources: Dict[str, KnowledgeSource] = {}
                cls._instance._index = _SimpleVectorIndex()
                # Read threshold from config; fall back to 0.85 if unset
                try:
                    from infrarely.core.config import get_config

                    cls._instance._threshold = get_config().get(
                        "knowledge_threshold", 0.85
                    )
                except Exception:
                    cls._instance._threshold = 0.85
                cls._instance._initialized = False
            return cls._instance

    def set_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for LLM bypass."""
        self._threshold = max(0.0, min(1.0, threshold))

    # ── Add Database ──────────────────────────────────────────────────────────

    def add_database(
        self,
        name: str,
        source: Any = None,
        schema: Optional[Dict[str, Any]] = None,
        description: str = "",
        data: Optional[List[Dict]] = None,
    ) -> None:
        """
        Register a database as a knowledge source.

        Args:
            name: Unique name for this source
            source: Database connection (optional)
            schema: Dict describing the schema
            description: Human description
            data: Pre-loaded data rows (list of dicts)
        """
        self._sources[name] = KnowledgeSource(
            name=name,
            source_type="database",
            description=description or f"Database: {name}",
            is_authoritative=True,
        )

        chunk_count = 0
        if data:
            for row in data:
                text = " | ".join(f"{k}: {v}" for k, v in row.items())
                self._index.add(
                    KnowledgeChunk(
                        content=text,
                        source=name,
                        confidence=1.0,
                        metadata=row,
                    )
                )
                chunk_count += 1

        if schema:
            schema_text = f"Schema for {name}: " + ", ".join(
                f"{k} ({v})" for k, v in schema.items()
            )
            self._index.add(
                KnowledgeChunk(
                    content=schema_text,
                    source=name,
                    confidence=0.5,
                    metadata={"type": "schema"},
                )
            )
            chunk_count += 1

        self._sources[name].chunk_count = chunk_count
        self._sources[name].last_indexed = datetime.now(timezone.utc).isoformat()

    # ── Add Documents ─────────────────────────────────────────────────────────

    def add_documents(
        self,
        name: str,
        path: str = "",
        content: Optional[List[str]] = None,
        refresh_hours: int = 24,
        description: str = "",
    ) -> None:
        """
        Register documents (files or text list) as a knowledge source.

        Args:
            name: Unique name
            path: Folder path containing .txt, .md, .json files
            content: Direct list of text content
            refresh_hours: How often to re-index
        """
        self._sources[name] = KnowledgeSource(
            name=name,
            source_type="documents",
            description=description or f"Documents: {name}",
            refresh_seconds=refresh_hours * 3600,
        )

        chunk_count = 0

        if content:
            for text in content:
                for para in self._split_text(text):
                    self._index.add(
                        KnowledgeChunk(
                            content=para,
                            source=name,
                            confidence=0.9,
                        )
                    )
                    chunk_count += 1

        if path and os.path.isdir(path):
            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    if fname.endswith((".txt", ".md")):
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        for para in self._split_text(text):
                            self._index.add(
                                KnowledgeChunk(
                                    content=para,
                                    source=name,
                                    confidence=0.85,
                                    metadata={"file": fname},
                                )
                            )
                            chunk_count += 1
                    elif fname.endswith(".json"):
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                text = (
                                    json.dumps(item, indent=0)
                                    if isinstance(item, dict)
                                    else str(item)
                                )
                                self._index.add(
                                    KnowledgeChunk(
                                        content=text,
                                        source=name,
                                        confidence=0.9,
                                        metadata={"file": fname},
                                    )
                                )
                                chunk_count += 1
                        elif isinstance(data, dict):
                            for key, val in data.items():
                                text = (
                                    f"{key}: {json.dumps(val)}"
                                    if isinstance(val, (dict, list))
                                    else f"{key}: {val}"
                                )
                                self._index.add(
                                    KnowledgeChunk(
                                        content=text,
                                        source=name,
                                        confidence=0.9,
                                        metadata={"file": fname, "key": key},
                                    )
                                )
                                chunk_count += 1
                except Exception:
                    continue  # Never crash on bad files

        self._sources[name].chunk_count = chunk_count
        self._sources[name].last_indexed = datetime.now(timezone.utc).isoformat()

    # ── Add Data ──────────────────────────────────────────────────────────────

    def add_data(
        self,
        name: str,
        data: Any,
        description: str = "",
    ) -> None:
        """
        Register plain Python data (dict, list) as a knowledge source.

        Args:
            name: Unique name
            data: Dict, list, or any JSON-serializable object
        """
        self._sources[name] = KnowledgeSource(
            name=name,
            source_type="data",
            description=description or f"Data: {name}",
            is_authoritative=True,
        )

        chunk_count = 0

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text = " | ".join(f"{k}: {v}" for k, v in item.items())
                else:
                    text = str(item)
                self._index.add(
                    KnowledgeChunk(
                        content=text,
                        source=name,
                        confidence=1.0,
                        metadata=item if isinstance(item, dict) else {},
                    )
                )
                chunk_count += 1
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, (dict, list)):
                    text = f"{key}: {json.dumps(val)}"
                else:
                    text = f"{key}: {val}"
                self._index.add(
                    KnowledgeChunk(
                        content=text,
                        source=name,
                        confidence=1.0,
                        metadata={"key": key, "value": val},
                    )
                )
                chunk_count += 1
        else:
            text = str(data)
            self._index.add(
                KnowledgeChunk(
                    content=text,
                    source=name,
                    confidence=0.9,
                )
            )
            chunk_count += 1

        self._sources[name].chunk_count = chunk_count
        self._sources[name].last_indexed = datetime.now(timezone.utc).isoformat()

    # ── Add API ───────────────────────────────────────────────────────────────

    def add_api(
        self,
        name: str,
        endpoint: str = "",
        refresh_minutes: int = 60,
        fetch_fn: Optional[Callable] = None,
        data: Optional[Any] = None,
        description: str = "",
    ) -> None:
        """
        Register an API as a knowledge source.

        Args:
            name: Unique name
            endpoint: API URL
            refresh_minutes: How often to re-fetch
            fetch_fn: Custom function to call the API
            data: Pre-fetched data
        """
        self._sources[name] = KnowledgeSource(
            name=name,
            source_type="api",
            description=description or f"API: {endpoint or name}",
            refresh_seconds=refresh_minutes * 60,
        )

        # If pre-fetched data provided, index it immediately
        if data:
            self.add_data(f"{name}_data", data)
            self._sources[name].chunk_count = self._sources.get(
                f"{name}_data", KnowledgeSource(name="", source_type="")
            ).chunk_count
            self._sources[name].last_indexed = datetime.now(timezone.utc).isoformat()

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> KnowledgeResult:
        """
        Query knowledge sources. Returns ranked results with confidence.

        Decision gate:
          confidence >= 0.45 → "bypass_llm" (LLM not needed)
          confidence >= 0.25 → "ground_llm" (pass chunks to LLM with grounding)
          confidence <  0.25 → "low_confidence" (flag as knowledge gap)
          no results          → "no_knowledge"
        """
        start = time.monotonic()
        results = self._index.search(question, top_k=top_k)
        elapsed = (time.monotonic() - start) * 1000

        if not results:
            return KnowledgeResult(
                query=question,
                duration_ms=elapsed,
                decision="no_knowledge",
            )

        chunks = []
        source_names = set()
        total_confidence = 0.0
        for chunk, score in results:
            adjusted_confidence = min(1.0, score * chunk.confidence)
            chunks.append(
                KnowledgeChunk(
                    content=chunk.content,
                    source=chunk.source,
                    confidence=adjusted_confidence,
                    metadata=chunk.metadata,
                    chunk_id=chunk.chunk_id,
                )
            )
            source_names.add(chunk.source)
            total_confidence += adjusted_confidence

        avg_confidence = total_confidence / len(chunks) if chunks else 0.0
        # Boost confidence if authoritative sources dominate
        auth_count = sum(
            1
            for sn in source_names
            if sn in self._sources and self._sources[sn].is_authoritative
        )
        if auth_count > 0 and len(source_names) > 0:
            auth_boost = 0.05 * (auth_count / len(source_names))
            avg_confidence = min(1.0, avg_confidence + auth_boost)

        # Decision gate
        if avg_confidence >= self._threshold:
            decision = "bypass_llm"
        elif avg_confidence >= 0.25:
            decision = "ground_llm"
        else:
            decision = "low_confidence"

        return KnowledgeResult(
            chunks=chunks,
            confidence=round(avg_confidence, 3),
            source_names=list(source_names),
            query=question,
            duration_ms=elapsed,
            decision=decision,
        )

    # ── Source management ─────────────────────────────────────────────────────

    def list_sources(self) -> List[KnowledgeSource]:
        return list(self._sources.values())

    def remove_source(self, name: str) -> bool:
        if name in self._sources:
            del self._sources[name]
            return True
        return False

    def clear(self) -> None:
        """Remove all knowledge sources and indexed data."""
        self._sources.clear()
        self._index.clear()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _split_text(text: str, max_chunk: int = 500) -> List[str]:
        """Split text into chunks by paragraph, then by size."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= max_chunk:
                chunks.append(para)
            else:
                # Split long paragraphs by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chunk:
                        current = f"{current} {sent}".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
                if current:
                    chunks.append(current)
        return chunks if chunks else [text[:max_chunk]] if text.strip() else []


# ── Module-level singleton ────────────────────────────────────────────────────
_knowledge = KnowledgeManager()


def get_knowledge_manager() -> KnowledgeManager:
    return _knowledge
