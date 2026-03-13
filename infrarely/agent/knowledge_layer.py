"""
agent/knowledge_layer.py — CAPABILITY 3: Knowledge Layer
═══════════════════════════════════════════════════════════════════════════════
Agent-native knowledge infrastructure — first-class, not bolted on.

PROBLEM SOLVED:
  Agents hallucinate because they reason over nothing but LLM weights.
  RAG alone is insufficient — it retrieves text, not structured knowledge.
  No framework provides a knowledge layer that is agent-native, structured,
  validated, and integrated into the execution pipeline.

THIS IMPLEMENTATION:
  KnowledgeRegistry  → what sources exist + metadata
  KnowledgeIngester  → parse sources → structured chunks
  KnowledgeIndex     → vector store (in-memory / ChromaDB)
  KnowledgeResolver  → query → ranked chunks → agent
  KnowledgeValidator → freshness + confidence scoring

DECISION GATE (CRITICAL — replaces LLM default behavior):
  if confidence >= 0.85 AND is_authoritative:
    → Return knowledge directly. NO LLM CALL.
  if confidence >= 0.70:
    → Pass chunks to LLM with strict grounding instruction
  if confidence < 0.70:
    → Flag as LOW_CONFIDENCE → trigger knowledge gap event

RULES (from AOS spec):
  RULE 3 — KNOWLEDGE BEFORE GENERATION
    Before any LLM call, the Knowledge Resolver MUST be queried.
    If confidence > 0.85 from knowledge retrieval, LLM is bypassed entirely.
    Hallucination is an infrastructure failure, not a model failure.

Hardware: Fits entirely in 8GB RAM, no GPU required.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from infrarely.observability import logger


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE SOURCE TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class SourceType(Enum):
    """Types of knowledge sources with their TTL policies."""

    DOCUMENTS = "documents"  # PDFs, markdown, text files — TTL: 30 days
    APIS = "apis"  # REST endpoints — TTL: 1 hour
    DATABASES = "databases"  # SQL/NoSQL — TTL: real-time
    MEMORY = "memory"  # Agent execution history — TTL: no expiry
    WEB_FEEDS = "web_feeds"  # Web content — TTL: 24 hours

    @property
    def ttl_seconds(self) -> int:
        """Return TTL in seconds for each source type."""
        _ttls = {
            "documents": 30 * 24 * 3600,  # 30 days
            "apis": 3600,  # 1 hour
            "databases": 0,  # real-time (always fresh)
            "memory": -1,  # no expiry
            "web_feeds": 24 * 3600,  # 24 hours
        }
        return _ttls.get(self.value, 3600)


class ConfidenceLevel(Enum):
    """Confidence thresholds for the decision gate."""

    HIGH = "high"  # >= 0.85 — bypass LLM entirely
    MEDIUM = "medium"  # >= 0.70 — grounded LLM synthesis
    LOW = "low"  # < 0.70  — flag as gap
    NONE = "none"  # no knowledge found


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class KnowledgeSource:
    """A registered knowledge source."""

    source_id: str = field(default_factory=lambda: f"src_{uuid.uuid4().hex[:8]}")
    name: str = ""
    source_type: SourceType = SourceType.DOCUMENTS
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_authoritative: bool = False  # True for verified DB/API sources
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_indexed: Optional[str] = None
    chunk_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, SourceType):
                d[k] = v.value
            else:
                d[k] = v
        return d


@dataclass
class KnowledgeChunk:
    """A single piece of structured knowledge."""

    chunk_id: str = field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:8]}")
    source_id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ttl_expires: Optional[str] = None  # ISO timestamp when this chunk expires

    def is_fresh(self) -> bool:
        """Check if this chunk is still within its TTL."""
        if self.ttl_expires is None:
            return True  # No expiry set
        try:
            expiry = datetime.fromisoformat(self.ttl_expires)
            return datetime.now(timezone.utc) < expiry
        except (ValueError, TypeError):
            return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "ttl_expires": self.ttl_expires,
        }


@dataclass
class KnowledgeResult:
    """
    Result from the KnowledgeResolver.
    This is THE object the decision gate evaluates.
    """

    chunks: List[KnowledgeChunk] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)  # source IDs for citation
    confidence: float = 0.0
    requires_llm_synthesis: bool = True
    is_authoritative: bool = False
    query: str = ""
    level: ConfidenceLevel = ConfidenceLevel.NONE
    knowledge_gaps: List[str] = field(default_factory=list)
    resolution_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "source_refs": self.source_refs,
            "confidence": round(self.confidence, 4),
            "requires_llm_synthesis": self.requires_llm_synthesis,
            "is_authoritative": self.is_authoritative,
            "level": self.level.value,
            "knowledge_gaps": self.knowledge_gaps,
            "resolution_ms": round(self.resolution_ms, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT VECTOR INDEX — In-memory, no GPU required
# ═══════════════════════════════════════════════════════════════════════════════


class SimpleEmbedder:
    """
    Lightweight TF-IDF-based embedder for environments without sentence-transformers.
    Uses bag-of-words with IDF weighting — fast, deterministic, zero dependencies.

    SYNONYM EXPANSION (Rule 3 compliance):
      TF-IDF cannot match "teaches" → "instructor". This creates a class of
      silent confidence failures where the system KNOWS the answer but the
      confidence gate doesn't know it knows. Every synonym gap silently routes
      to LLM — breaking Rule 3 without any error or warning.

      Fix: A deterministic synonym/alias expansion table. When tokenizing,
      each token is expanded to include all its synonyms. This means
      "who teaches CS301" → ["who", "teaches", "instructor", "professor",
      "faculty", "cs301"] — closing the vocabulary gap at zero cost.

    When sentence-transformers is available, automatically upgrades to
    all-MiniLM-L6-v2 for better semantic understanding.
    """

    # ── Bidirectional synonym clusters ─────────────────────────────────────
    # Each list is a cluster of semantically equivalent terms.
    # When ANY token in a cluster appears, ALL tokens in that cluster are added.
    # This is bidirectional by construction — "teaches" adds "instructor" AND
    # "instructor" adds "teaches".
    SYNONYM_CLUSTERS: List[List[str]] = [
        # People / roles
        [
            "teaches",
            "taught",
            "instructor",
            "professor",
            "teacher",
            "faculty",
            "lecturer",
        ],
        ["student", "learner", "enrollee", "pupil"],
        ["author", "creator", "writer", "built", "made", "developed", "wrote"],
        # Academic
        ["course", "class", "subject", "module"],
        ["assignment", "homework", "task", "project", "work"],
        ["exam", "test", "quiz", "midterm", "final", "assessment", "evaluation"],
        ["grade", "score", "mark", "gpa", "rating", "result"],
        ["enroll", "enrolled", "register", "registered", "signup", "joined"],
        ["schedule", "timetable", "calendar", "agenda"],
        ["semester", "term", "quarter", "session"],
        ["lecture", "class", "section", "meeting"],
        ["syllabus", "curriculum", "outline", "plan"],
        ["credit", "credits", "units", "hours"],
        ["major", "concentration", "specialization", "field", "discipline"],
        ["degree", "diploma", "certificate", "program"],
        ["prerequisite", "prereq", "requirement", "required"],
        # Time / deadlines
        ["deadline", "due", "due_date", "duedate", "submission"],
        ["when", "date", "time", "schedule", "timing"],
        ["start", "begin", "begins", "starts", "commences"],
        ["end", "finish", "ends", "finishes", "concludes"],
        # Cost / money
        ["cost", "price", "fee", "tuition", "charge", "amount", "payment"],
        ["free", "complimentary", "no_cost", "gratis"],
        # Location
        ["where", "location", "room", "building", "place", "venue", "hall"],
        ["online", "virtual", "remote", "distance"],
        # Status / properties
        ["available", "open", "offered", "active"],
        ["name", "title", "label", "called"],
        ["description", "about", "summary", "overview", "details", "info"],
        # Actions
        ["get", "fetch", "retrieve", "find", "lookup", "show", "list", "display"],
        ["add", "create", "new", "insert", "submit"],
        ["update", "edit", "modify", "change"],
        ["delete", "remove", "drop", "cancel"],
        # Question words → intent mapping
        ["who", "whom", "whose", "person", "instructor", "professor", "teacher"],
        ["what", "which", "describe"],
        ["how", "method", "way", "process", "steps"],
        ["why", "reason", "cause", "because"],
        # Notes / content
        ["note", "notes", "notebook", "memo", "jotting"],
        ["summary", "recap", "overview", "brief", "abstract"],
        # Student-specific
        ["gpa", "grade_point", "average", "cumulative"],
        ["attendance", "present", "absent", "participated"],
        ["office_hours", "officehours", "consultation", "availability"],
    ]

    def __init__(self):
        self._use_transformer = False
        self._model = None
        self._idf: Dict[str, float] = {}
        self._vocab: List[str] = []
        self._doc_count = 0
        self._lock = threading.Lock()

        # Build synonym lookup from clusters
        self._synonym_map: Dict[str, Set[str]] = {}
        for cluster in self.SYNONYM_CLUSTERS:
            normalized = [t.lower() for t in cluster]
            for token in normalized:
                if token not in self._synonym_map:
                    self._synonym_map[token] = set()
                self._synonym_map[token].update(normalized)
                self._synonym_map[token].discard(token)  # don't include self

        # Try loading sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_transformer = True
            logger.info(
                "KnowledgeLayer: using sentence-transformers (all-MiniLM-L6-v2)"
            )
        except ImportError:
            logger.info(
                "KnowledgeLayer: using lightweight TF-IDF embedder with synonym expansion (no GPU)"
            )

    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        if self._use_transformer and self._model:
            return self._model.encode(text, show_progress_bar=False).tolist()
        return self._tfidf_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        if self._use_transformer and self._model:
            return self._model.encode(texts, show_progress_bar=False).tolist()
        return [self._tfidf_embed(t) for t in texts]

    def generate_query_variants(self, text: str) -> List[str]:
        """
        Generate focused query variants by substituting synonym clusters.

        For "Who teaches CS301?":
          → "Who teaches CS301?"      (original)
          → "Who instructor CS301?"   (teaches→instructor)
          → "Who professor CS301?"    (teaches→professor)

        Each variant is short and focused, scoring well against documents
        containing that specific word. Multi-query fusion then takes the
        max score per chunk across all variants.

        This is MORE effective than embedding synonym tokens into one vector
        (which dilutes TF-IDF weights). Each variant maintains concentrated
        signal on the substituted dimension.
        """
        if self._use_transformer:
            return [text]  # transformer handles semantics natively

        base_tokens = self._tokenize(text)
        variants = [text]  # always include original
        seen_variants: Set[str] = {text.lower()}

        for i, token in enumerate(base_tokens):
            synonyms = self._synonym_map.get(token, set())
            for syn in synonyms:
                # Build variant by replacing token with synonym
                new_tokens = list(base_tokens)
                new_tokens[i] = syn
                variant = " ".join(new_tokens)
                if variant.lower() not in seen_variants:
                    variants.append(variant)
                    seen_variants.add(variant.lower())

        return variants

    def _tfidf_embed(self, text: str) -> List[float]:
        """Simple TF-IDF embedding using current vocabulary."""
        tokens = self._tokenize(text)
        if not self._vocab:
            return [0.0] * 128  # empty vocab → zero vector

        vec = [0.0] * len(self._vocab)
        tf: Dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        total = len(tokens) or 1
        for i, word in enumerate(self._vocab):
            if word in tf:
                tf_val = tf[word] / total
                idf_val = self._idf.get(word, 1.0)
                vec[i] = tf_val * idf_val

        # Normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def build_vocabulary(self, documents: List[str]):
        """Build IDF vocabulary from a corpus of documents."""
        with self._lock:
            if self._use_transformer:
                return  # not needed for transformer

            self._doc_count = len(documents)
            word_doc_freq: Dict[str, int] = defaultdict(int)

            for doc in documents:
                tokens = set(self._tokenize(doc))
                for t in tokens:
                    word_doc_freq[t] += 1

            # Keep top 512 words by document frequency
            sorted_words = sorted(word_doc_freq.items(), key=lambda x: -x[1])
            self._vocab = [w for w, _ in sorted_words[:512]]

            # Compute IDF
            n = max(self._doc_count, 1)
            self._idf = {
                w: math.log(n / (1 + freq))
                for w, freq in word_doc_freq.items()
                if w in set(self._vocab)
            }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer (no synonym expansion)."""
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9_]+\b", text)
        # Remove very short and very common stop words
        _stop = {
            "the",
            "a",
            "an",
            "is",
            "it",
            "in",
            "on",
            "to",
            "and",
            "of",
            "for",
            "with",
            "as",
        }
        return [t for t in tokens if len(t) > 1 and t not in _stop]

    def _tokenize_with_synonyms(self, text: str) -> List[str]:
        """
        Tokenize with synonym expansion.
        Each token is expanded by its synonym cluster, giving TF-IDF
        semantic awareness without any ML model.

        Example:
            "Who teaches CS301?"
            → base:     ["who", "teaches", "cs301"]
            → expanded: ["who", "teaches", "instructor", "professor",
                         "teacher", "faculty", "lecturer", "cs301",
                         "person", "whom", "whose"]

        This closes the vocabulary gap so "teaches" matches "instructor"
        in the indexed documents.
        """
        base_tokens = self._tokenize(text)
        expanded: List[str] = list(base_tokens)  # keep originals
        seen = set(base_tokens)

        for token in base_tokens:
            synonyms = self._synonym_map.get(token, set())
            for syn in synonyms:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)

        return expanded


class VectorIndex:
    """
    In-memory vector index with cosine similarity search.
    Backed by simple lists — works for up to ~50K chunks efficiently.

    Can be replaced with ChromaDB for larger deployments.
    """

    def __init__(self):
        self._embedder = SimpleEmbedder()
        self._chunks: Dict[str, KnowledgeChunk] = {}  # chunk_id → chunk
        self._embeddings: Dict[str, List[float]] = {}  # chunk_id → embedding
        self._lock = threading.Lock()
        self._use_chromadb = False
        self._chroma_collection = None

        # Try ChromaDB
        try:
            import chromadb

            client = chromadb.Client()
            self._chroma_collection = client.get_or_create_collection(
                name="knowledge_index",
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chromadb = True
            logger.info("KnowledgeLayer: using ChromaDB for vector storage")
        except ImportError:
            logger.info("KnowledgeLayer: using in-memory vector index")

    def add(self, chunk: KnowledgeChunk) -> None:
        """Add a chunk to the index."""
        with self._lock:
            embedding = self._embedder.embed(chunk.content)
            chunk.embedding = embedding

            if self._use_chromadb and self._chroma_collection:
                self._chroma_collection.upsert(
                    ids=[chunk.chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.content],
                    metadatas=[
                        {
                            "source_id": chunk.source_id,
                            "created_at": chunk.created_at,
                            "ttl_expires": chunk.ttl_expires or "",
                        }
                    ],
                )
            else:
                self._chunks[chunk.chunk_id] = chunk
                self._embeddings[chunk.chunk_id] = embedding

    def add_batch(self, chunks: List[KnowledgeChunk]) -> None:
        """Add a batch of chunks to the index."""
        if not chunks:
            return

        texts = [c.content for c in chunks]

        with self._lock:
            if self._use_chromadb and self._chroma_collection:
                embeddings = self._embedder.embed_batch(texts)
                self._chroma_collection.upsert(
                    ids=[c.chunk_id for c in chunks],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[
                        {
                            "source_id": c.source_id,
                            "created_at": c.created_at,
                            "ttl_expires": c.ttl_expires or "",
                        }
                        for c in chunks
                    ],
                )
            else:
                # Build vocabulary FIRST so embeddings are meaningful
                all_docs = [c.content for c in self._chunks.values()] + texts
                self._embedder.build_vocabulary(all_docs)

                # Embed new chunks with updated vocabulary
                embeddings = self._embedder.embed_batch(texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb
                    self._chunks[chunk.chunk_id] = chunk
                    self._embeddings[chunk.chunk_id] = emb

                # Re-embed existing chunks with updated vocabulary
                new_ids = {ch.chunk_id for ch in chunks}
                for cid, c in self._chunks.items():
                    if cid not in new_ids:
                        new_emb = self._embedder.embed(c.content)
                        c.embedding = new_emb
                        self._embeddings[cid] = new_emb

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeChunk, float]]:
        """
        Search for chunks most similar to the query.
        Returns list of (chunk, similarity_score) sorted by score descending.
        """
        if self._use_chromadb and self._chroma_collection:
            return self._search_chromadb(query, top_k)
        return self._search_memory(query, top_k)

    def _search_memory(
        self, query: str, top_k: int
    ) -> List[Tuple[KnowledgeChunk, float]]:
        """
        Hybrid search: TF-IDF cosine similarity + concept-level synonym matching.

        Pure TF-IDF fails for synonym queries because:
        1. "teaches" ≠ "instructor" at the token level → zero cosine on that dim
        2. With small corpora, structural keys (instructor, name) have low IDF
           because they appear in every document → synonym substitution barely
           helps cosine similarity

        Fix: Concept-level matching. For each BASE query token, check if the
        token itself OR any of its synonyms appears in the document. The
        denominator is the number of base tokens (e.g. 3 for "who teaches
        CS301"), NOT the expanded count (12+). This avoids dilution from
        large synonym clusters while still closing the vocabulary gap.

        Example — "Who teaches CS301?" against CS301 doc:
          - "who"     → synonyms include "instructor" → doc has "instructor" → ✓
          - "teaches" → synonyms include "instructor" → doc has "instructor" → ✓
          - "cs301"   → direct match → doc has "cs301" → ✓
          → concept_score = 3/3 = 1.0

        hybrid_score = 0.4 * cosine_sim + 0.6 * concept_match_ratio
        """
        if not self._chunks:
            return []

        # Precompute query data
        query_emb = self._embedder.embed(query)
        query_base_tokens = self._embedder._tokenize(query)

        # Build per-token synonym sets (once, not per-doc)
        token_forms: List[Set[str]] = []
        for token in query_base_tokens:
            forms = {token} | self._embedder._synonym_map.get(token, set())
            token_forms.append(forms)

        n_concepts = len(query_base_tokens) or 1

        scores: List[Tuple[str, float]] = []

        for chunk_id, emb in self._embeddings.items():
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue

            # Score 1: TF-IDF cosine similarity
            cosine_score = self._cosine_sim(query_emb, emb)

            # Score 2: Concept-level synonym matching
            doc_tokens = set(self._embedder._tokenize(chunk.content))
            matched_concepts = sum(
                1 for forms in token_forms if forms & doc_tokens
            )
            concept_score = matched_concepts / n_concepts

            # Hybrid blend
            hybrid = 0.4 * cosine_score + 0.6 * concept_score

            scores.append((chunk_id, hybrid))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, score in scores[:top_k]:
            chunk = self._chunks.get(chunk_id)
            if chunk and chunk.is_fresh():
                results.append((chunk, score))

        return results

    def _search_chromadb(
        self, query: str, top_k: int
    ) -> List[Tuple[KnowledgeChunk, float]]:
        """ChromaDB vector search."""
        try:
            query_emb = self._embedder.embed(query)
            result = self._chroma_collection.query(
                query_embeddings=[query_emb],
                n_results=min(top_k, self._chroma_collection.count() or 1),
            )

            chunks = []
            if result and result["documents"] and result["documents"][0]:
                for i, doc in enumerate(result["documents"][0]):
                    meta = result["metadatas"][0][i] if result["metadatas"] else {}
                    distance = result["distances"][0][i] if result["distances"] else 1.0
                    similarity = max(0.0, 1.0 - distance)

                    chunk = KnowledgeChunk(
                        chunk_id=result["ids"][0][i],
                        source_id=meta.get("source_id", ""),
                        content=doc,
                        created_at=meta.get("created_at", ""),
                        ttl_expires=meta.get("ttl_expires", None) or None,
                    )
                    if chunk.is_fresh():
                        chunks.append((chunk, similarity))
            return chunks
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    def remove_by_source(self, source_id: str) -> int:
        """Remove all chunks from a specific source."""
        removed = 0
        with self._lock:
            if self._use_chromadb and self._chroma_collection:
                try:
                    self._chroma_collection.delete(where={"source_id": source_id})
                except Exception:
                    pass
            else:
                to_remove = [
                    cid for cid, c in self._chunks.items() if c.source_id == source_id
                ]
                for cid in to_remove:
                    del self._chunks[cid]
                    self._embeddings.pop(cid, None)
                    removed += 1
        return removed

    @property
    def count(self) -> int:
        if self._use_chromadb and self._chroma_collection:
            return self._chroma_collection.count()
        return len(self._chunks)

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            # Pad shorter vector
            max_len = max(len(a), len(b))
            a = a + [0.0] * (max_len - len(a))
            b = b + [0.0] * (max_len - len(b))

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE REGISTRY — What sources exist + metadata
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeRegistry:
    """
    Central registry of all knowledge sources.
    Every source must be registered before its chunks can be indexed.
    """

    def __init__(self):
        self._sources: Dict[str, KnowledgeSource] = {}
        self._lock = threading.Lock()

    def register(self, source: KnowledgeSource) -> str:
        """Register a knowledge source. Returns source_id."""
        with self._lock:
            self._sources[source.source_id] = source
            logger.info(
                f"KnowledgeRegistry: registered source '{source.name}'",
                source_id=source.source_id,
                type=source.source_type.value,
            )
            return source.source_id

    def get(self, source_id: str) -> Optional[KnowledgeSource]:
        return self._sources.get(source_id)

    def get_by_name(self, name: str) -> Optional[KnowledgeSource]:
        for s in self._sources.values():
            if s.name == name:
                return s
        return None

    def list_sources(self) -> List[KnowledgeSource]:
        return list(self._sources.values())

    def unregister(self, source_id: str) -> bool:
        with self._lock:
            return self._sources.pop(source_id, None) is not None

    def is_authoritative(self, source_id: str) -> bool:
        src = self._sources.get(source_id)
        return src.is_authoritative if src else False

    def source_count(self) -> int:
        return len(self._sources)


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE INGESTER — Parse sources into structured chunks
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeIngester:
    """
    Parses raw content into structured KnowledgeChunks.
    Supports: plain text, JSON, markdown, key-value data.

    Chunking strategy:
      - Documents: split by paragraphs/sections, max 500 chars per chunk
      - JSON: each top-level key becomes a chunk
      - Memory: each episode/fact becomes a chunk
    """

    MAX_CHUNK_SIZE = 500

    def ingest_text(
        self,
        text: str,
        source: KnowledgeSource,
        metadata: Dict[str, Any] = None,
    ) -> List[KnowledgeChunk]:
        """Ingest plain text or markdown into chunks."""
        metadata = metadata or {}
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text.strip())

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para or len(para) < 10:
                continue

            # Further split if paragraph is too long
            sub_chunks = self._split_long_text(para, self.MAX_CHUNK_SIZE)
            for j, sub in enumerate(sub_chunks):
                ttl = self._compute_ttl(source.source_type)
                chunks.append(
                    KnowledgeChunk(
                        source_id=source.source_id,
                        content=sub,
                        metadata={
                            **metadata,
                            "paragraph_index": i,
                            "sub_index": j,
                            "source_name": source.name,
                        },
                        ttl_expires=ttl,
                    )
                )

        # Update source metadata
        source.chunk_count += len(chunks)
        source.last_indexed = datetime.now(timezone.utc).isoformat()

        return chunks

    def ingest_json(
        self,
        data: Any,
        source: KnowledgeSource,
        metadata: Dict[str, Any] = None,
    ) -> List[KnowledgeChunk]:
        """
        Ingest structured JSON data into search-friendly chunks.

        Instead of raw JSON (which creates poor TF-IDF matches), converts
        each record to natural language key-value format:

            {"id": "CS301", "instructor": "Dr. Smith"}
            → "id CS301. instructor Dr. Smith."

        This makes chunks directly matchable against natural language queries.
        """
        metadata = metadata or {}
        chunks = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                content = (
                    self._json_to_searchable_text(item)
                    if isinstance(item, dict)
                    else str(item)
                )
                ttl = self._compute_ttl(source.source_type)
                chunks.append(
                    KnowledgeChunk(
                        source_id=source.source_id,
                        content=content[: self.MAX_CHUNK_SIZE],
                        metadata={**metadata, "index": i, "source_name": source.name},
                        ttl_expires=ttl,
                    )
                )
        elif isinstance(data, dict):
            for key, value in data.items():
                content = f"{key}: {json.dumps(value, indent=2, default=str)}"
                ttl = self._compute_ttl(source.source_type)
                chunks.append(
                    KnowledgeChunk(
                        source_id=source.source_id,
                        content=content[: self.MAX_CHUNK_SIZE],
                        metadata={**metadata, "key": key, "source_name": source.name},
                        ttl_expires=ttl,
                    )
                )
        else:
            chunks = self.ingest_text(str(data), source, metadata)

        source.chunk_count += len(chunks)
        source.last_indexed = datetime.now(timezone.utc).isoformat()
        return chunks

    def ingest_memory(
        self,
        episodes: List[Dict[str, Any]],
        source: KnowledgeSource,
    ) -> List[KnowledgeChunk]:
        """Ingest agent execution history as knowledge chunks."""
        chunks = []
        for ep in episodes:
            content = json.dumps(ep, indent=2, default=str)
            chunks.append(
                KnowledgeChunk(
                    source_id=source.source_id,
                    content=content[: self.MAX_CHUNK_SIZE],
                    metadata={
                        "type": ep.get("type", "episode"),
                        "source_name": source.name,
                    },
                    # Memory chunks never expire
                    ttl_expires=None,
                )
            )

        source.chunk_count += len(chunks)
        source.last_indexed = datetime.now(timezone.utc).isoformat()
        return chunks

    def _split_long_text(self, text: str, max_size: int) -> List[str]:
        """Split text into chunks at sentence boundaries."""
        if len(text) <= max_size:
            return [text]

        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_size:
                current = f"{current} {sent}".strip()
            else:
                if current:
                    chunks.append(current)
                current = sent[:max_size]

        if current:
            chunks.append(current)

        return chunks or [text[:max_size]]

    @staticmethod
    def _json_to_searchable_text(item: dict) -> str:
        """
        Convert a JSON dict to search-friendly natural language text.

        {"id": "CS301", "name": "Data Structures", "instructor": "Dr. Smith"}
        → "id CS301. name Data Structures. instructor Dr. Smith."

        This makes the content directly matchable by TF-IDF against natural
        language queries like "Who teaches CS301?" (because "instructor" is
        a plain token in the content, not hidden in JSON syntax).
        """
        parts = []
        for key, value in item.items():
            val_str = (
                str(value)
                if not isinstance(value, (dict, list))
                else json.dumps(value, default=str)
            )
            parts.append(f"{key} {val_str}")
        return ". ".join(parts) + "."

    @staticmethod
    def _compute_ttl(source_type: SourceType) -> Optional[str]:
        """Compute TTL expiry timestamp for a chunk based on source type."""
        ttl = source_type.ttl_seconds
        if ttl < 0:  # no expiry
            return None
        if ttl == 0:  # real-time (always fresh by default)
            return None
        from datetime import timedelta

        expiry = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        return expiry.isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE VALIDATOR — Freshness + confidence scoring
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeValidator:
    """
    Validates and scores knowledge chunks for freshness and confidence.

    Freshness TTL per source type:
      documents  : 30 days
      APIs       : 1 hour
      databases  : real-time
      memory     : no expiry
      web feeds  : 24 hours

    Confidence decay: reduce score for stale chunks proportionally.
    Contradiction detection: flag chunks from same source with conflicting data.
    """

    def validate_freshness(self, chunks: List[KnowledgeChunk]) -> List[KnowledgeChunk]:
        """Filter out stale chunks."""
        return [c for c in chunks if c.is_fresh()]

    def score_confidence(
        self,
        chunks: List[Tuple[KnowledgeChunk, float]],
        registry: KnowledgeRegistry,
    ) -> Tuple[float, bool]:
        """
        Compute confidence score using top-chunk-dominant scoring.

        OLD approach (broken): weighted average across all top-k chunks.
        This dragged confidence down when only 1-2 chunks matched well.
        "Who teaches CS301?" matched the CS301 chunk at 0.8 but averaged
        with 4 low-scoring chunks → final confidence 0.42.

        NEW approach: 70% weight on best chunk, 30% on rest.
        The BEST matching chunk dominates the confidence score because
        knowledge queries typically have ONE correct answer chunk.
        Low-scoring tail chunks don't dilute the signal.

        Returns: (confidence, is_authoritative)
        """
        if not chunks:
            return 0.0, False

        is_authoritative = False

        # Score each chunk
        scored = []
        for chunk, similarity in chunks:
            freshness = self._freshness_factor(chunk)
            auth_bonus = 1.0
            if registry.is_authoritative(chunk.source_id):
                auth_bonus = 1.2
                is_authoritative = True
            effective_score = similarity * freshness * auth_bonus
            scored.append(effective_score)

        # Sort descending
        scored.sort(reverse=True)

        # Top-chunk-dominant scoring: 70% best chunk, 30% weighted rest
        best_score = scored[0]
        if len(scored) > 1:
            rest_avg = sum(scored[1:]) / len(scored[1:])
            confidence = 0.70 * best_score + 0.30 * rest_avg
        else:
            confidence = best_score

        # Normalize to 0-1
        confidence = min(1.0, max(0.0, confidence))

        return confidence, is_authoritative

    def detect_contradictions(
        self, chunks: List[KnowledgeChunk]
    ) -> List[Dict[str, Any]]:
        """
        Detect potential contradictions between chunks from the same source.
        Returns list of contradiction reports.
        """
        contradictions = []

        # Group by source
        by_source: Dict[str, List[KnowledgeChunk]] = defaultdict(list)
        for c in chunks:
            by_source[c.source_id].append(c)

        # Simple heuristic: check for negation patterns
        _negation_pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("available", "unavailable"),
            ("active", "inactive"),
        ]

        for source_id, source_chunks in by_source.items():
            if len(source_chunks) < 2:
                continue
            for i, c1 in enumerate(source_chunks):
                for c2 in source_chunks[i + 1 :]:
                    text1 = c1.content.lower()
                    text2 = c2.content.lower()
                    for pos, neg in _negation_pairs:
                        if (pos in text1 and neg in text2) or (
                            neg in text1 and pos in text2
                        ):
                            contradictions.append(
                                {
                                    "source_id": source_id,
                                    "chunk_1": c1.chunk_id,
                                    "chunk_2": c2.chunk_id,
                                    "pattern": f"{pos}/{neg}",
                                }
                            )

        return contradictions

    @staticmethod
    def _freshness_factor(chunk: KnowledgeChunk) -> float:
        """Compute freshness decay factor (1.0 = fresh, 0.0 = expired)."""
        if chunk.ttl_expires is None:
            return 1.0
        try:
            expiry = datetime.fromisoformat(chunk.ttl_expires)
            now = datetime.now(timezone.utc)
            if now >= expiry:
                return 0.0
            created = datetime.fromisoformat(chunk.created_at)
            total_life = (expiry - created).total_seconds()
            remaining = (expiry - now).total_seconds()
            if total_life <= 0:
                return 1.0
            return remaining / total_life
        except (ValueError, TypeError):
            return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# GROUNDING INSTRUCTION — Passed to LLM when synthesis is needed
# ═══════════════════════════════════════════════════════════════════════════════

GROUNDING_INSTRUCTION = (
    "You are a synthesis engine. You may ONLY use the provided knowledge chunks. "
    "You may NOT add facts not present in the chunks. "
    "Every claim must cite a source_ref. "
    "If the chunks are insufficient, respond: KNOWLEDGE_INSUFFICIENT"
)


def build_grounded_prompt(
    query: str,
    result: KnowledgeResult,
) -> str:
    """
    Build a grounded LLM prompt with knowledge chunks and citation requirements.
    Used when confidence is in the MEDIUM range (0.70–0.85).
    """
    lines = [
        GROUNDING_INSTRUCTION,
        "",
        f"QUERY: {query}",
        "",
        "KNOWLEDGE CHUNKS (use ONLY these):",
    ]

    for i, chunk in enumerate(result.chunks):
        source_ref = chunk.source_id
        lines.append(f"  [{i+1}] (source: {source_ref})")
        lines.append(f"      {chunk.content[:300]}")
        lines.append("")

    lines.append("SOURCE REFERENCES:")
    for ref in result.source_refs:
        lines.append(f"  - {ref}")

    lines.append("")
    lines.append(
        "RESPOND with a synthesis of the above chunks. "
        "Cite sources as [source_ref]. "
        "If insufficient: respond KNOWLEDGE_INSUFFICIENT"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE RESOLVER — The core query engine
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeResolver:
    """
    The execution protocol for knowledge queries.

    Step 1: Embed query
    Step 2: Query index → top-k=5 chunks with scores
    Step 3: Filter by freshness (TTL check)
    Step 4: Score confidence = weighted average
    Step 5: Return KnowledgeResult with decision gate classification

    DECISION GATE:
      confidence >= 0.85 AND authoritative → NO LLM CALL
      confidence >= 0.70 → grounded LLM synthesis
      confidence < 0.70 → LOW_CONFIDENCE flag
    """

    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.70

    def __init__(
        self,
        registry: KnowledgeRegistry,
        index: VectorIndex,
        validator: KnowledgeValidator,
    ):
        self._registry = registry
        self._index = index
        self._validator = validator
        self._gap_queue: List[Dict[str, Any]] = []  # knowledge gap events
        self._stats = {
            "queries": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "llm_bypassed": 0,
            "gaps_detected": 0,
        }

    def resolve(self, query: str, top_k: int = 5) -> KnowledgeResult:
        """
        Execute the knowledge resolution protocol.
        Returns a KnowledgeResult with confidence-gated classification.
        """
        t_start = time.monotonic()
        self._stats["queries"] += 1

        # Step 1-2: Query vector index
        raw_results = self._index.search(query, top_k=top_k)

        # Step 3: Filter by freshness
        fresh_results = [
            (chunk, score) for chunk, score in raw_results if chunk.is_fresh()
        ]

        if not fresh_results:
            result = KnowledgeResult(
                query=query,
                confidence=0.0,
                level=ConfidenceLevel.NONE,
                requires_llm_synthesis=True,
                knowledge_gaps=[f"No knowledge found for: {query[:100]}"],
                resolution_ms=(time.monotonic() - t_start) * 1000,
            )
            self._stats["low_confidence"] += 1
            self._stats["gaps_detected"] += 1
            self._gap_queue.append(
                {
                    "query": query,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": "no_results",
                }
            )
            return result

        # Step 4: Score confidence
        confidence, is_authoritative = self._validator.score_confidence(
            fresh_results, self._registry
        )

        # Step 5: Classify and build result
        chunks = [chunk for chunk, _ in fresh_results]
        source_refs = list(set(c.source_id for c in chunks))

        # Decision gate
        if confidence >= self.HIGH_CONFIDENCE and is_authoritative:
            level = ConfidenceLevel.HIGH
            requires_synthesis = False
            self._stats["high_confidence"] += 1
            self._stats["llm_bypassed"] += 1
        elif confidence >= self.MEDIUM_CONFIDENCE:
            level = ConfidenceLevel.MEDIUM
            requires_synthesis = len(chunks) > 3 or confidence < self.HIGH_CONFIDENCE
            self._stats["medium_confidence"] += 1
        else:
            level = ConfidenceLevel.LOW
            requires_synthesis = True
            self._stats["low_confidence"] += 1
            self._gap_queue.append(
                {
                    "query": query,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": "low_confidence",
                    "confidence": confidence,
                }
            )
            self._stats["gaps_detected"] += 1

        # Check for contradictions
        contradictions = self._validator.detect_contradictions(chunks)
        knowledge_gaps = []
        if contradictions:
            knowledge_gaps.append(
                f"Contradictions detected in {len(contradictions)} chunk pair(s)"
            )

        result = KnowledgeResult(
            chunks=chunks,
            source_refs=source_refs,
            confidence=confidence,
            requires_llm_synthesis=requires_synthesis,
            is_authoritative=is_authoritative,
            query=query,
            level=level,
            knowledge_gaps=knowledge_gaps,
            resolution_ms=(time.monotonic() - t_start) * 1000,
        )

        logger.debug(
            f"KnowledgeResolver: query='{query[:40]}' "
            f"confidence={confidence:.3f} level={level.value} "
            f"chunks={len(chunks)} authoritative={is_authoritative}",
        )

        return result

    def get_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Return queued knowledge gap events for ingestion pipeline."""
        gaps = list(self._gap_queue)
        return gaps

    def clear_gaps(self) -> int:
        """Clear processed knowledge gaps. Returns count cleared."""
        count = len(self._gap_queue)
        self._gap_queue.clear()
        return count

    def snapshot(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "index_size": self._index.count,
            "sources": self._registry.source_count(),
            "pending_gaps": len(self._gap_queue),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE LAYER — Unified facade
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeLayer:
    """
    Unified facade for the entire Knowledge Layer infrastructure.

    Usage:
      layer = KnowledgeLayer()
      # Register sources
      layer.register_source("courses", SourceType.DATABASES, is_authoritative=True)
      # Ingest data
      layer.ingest_json("courses", courses_data)
      # Query
      result = layer.query("What are my CS301 assignments?")
      # Decision gate
      if result.level == ConfidenceLevel.HIGH:
          # Use directly — no LLM needed
      elif result.level == ConfidenceLevel.MEDIUM:
          # Pass to LLM with grounding
          prompt = build_grounded_prompt(query, result)
    """

    def __init__(self):
        self._registry = KnowledgeRegistry()
        self._index = VectorIndex()
        self._validator = KnowledgeValidator()
        self._ingester = KnowledgeIngester()
        self._resolver = KnowledgeResolver(self._registry, self._index, self._validator)
        self._source_names: Dict[str, str] = {}  # name → source_id

        logger.info("KnowledgeLayer: initialized")

    # ── Source management ─────────────────────────────────────────────────────

    def register_source(
        self,
        name: str,
        source_type: SourceType,
        is_authoritative: bool = False,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Register a knowledge source. Returns source_id."""
        source = KnowledgeSource(
            name=name,
            source_type=source_type,
            is_authoritative=is_authoritative,
            metadata=metadata or {},
        )
        source_id = self._registry.register(source)
        self._source_names[name] = source_id
        return source_id

    def get_source_id(self, name: str) -> Optional[str]:
        return self._source_names.get(name)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_text(
        self,
        source_name: str,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> int:
        """Ingest text content. Returns number of chunks created."""
        source_id = self._source_names.get(source_name)
        if not source_id:
            logger.error(f"KnowledgeLayer: source '{source_name}' not registered")
            return 0

        source = self._registry.get(source_id)
        if not source:
            return 0

        chunks = self._ingester.ingest_text(text, source, metadata)
        self._index.add_batch(chunks)

        logger.info(
            f"KnowledgeLayer: ingested {len(chunks)} chunks from '{source_name}'",
        )
        return len(chunks)

    def ingest_json(
        self,
        source_name: str,
        data: Any,
        metadata: Dict[str, Any] = None,
    ) -> int:
        """Ingest JSON data. Returns number of chunks created."""
        source_id = self._source_names.get(source_name)
        if not source_id:
            logger.error(f"KnowledgeLayer: source '{source_name}' not registered")
            return 0

        source = self._registry.get(source_id)
        if not source:
            return 0

        chunks = self._ingester.ingest_json(data, source, metadata)
        self._index.add_batch(chunks)

        logger.info(
            f"KnowledgeLayer: ingested {len(chunks)} JSON chunks from '{source_name}'",
        )
        return len(chunks)

    def ingest_memory(
        self,
        source_name: str,
        episodes: List[Dict[str, Any]],
    ) -> int:
        """Ingest agent execution memory. Returns number of chunks created."""
        source_id = self._source_names.get(source_name)
        if not source_id:
            # Auto-register memory source
            self.register_source(source_name, SourceType.MEMORY, is_authoritative=False)
            source_id = self._source_names.get(source_name)

        source = self._registry.get(source_id)
        if not source:
            return 0

        chunks = self._ingester.ingest_memory(episodes, source)
        self._index.add_batch(chunks)
        return len(chunks)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, query: str, top_k: int = 5) -> KnowledgeResult:
        """
        Query the knowledge layer.
        Returns KnowledgeResult with confidence-gated classification.

        RULE 3: This must be called BEFORE any LLM call.
        """
        return self._resolver.resolve(query, top_k=top_k)

    def query_with_decision(self, query: str) -> Tuple[KnowledgeResult, str]:
        """
        Query and return the decision gate action.

        Returns: (result, action)
        action is one of:
          "bypass_llm"     — confidence >= 0.85, return knowledge directly
          "grounded_llm"   — confidence >= 0.70, pass to LLM with grounding
          "low_confidence"  — confidence < 0.70, flag as gap
          "no_knowledge"    — no results found
        """
        result = self.query(query)

        if result.level == ConfidenceLevel.HIGH and not result.requires_llm_synthesis:
            return result, "bypass_llm"
        elif result.level == ConfidenceLevel.MEDIUM:
            return result, "grounded_llm"
        elif result.level == ConfidenceLevel.LOW:
            return result, "low_confidence"
        else:
            return result, "no_knowledge"

    # ── Grounding ─────────────────────────────────────────────────────────────

    def build_grounded_prompt(
        self, query: str, result: Optional[KnowledgeResult] = None
    ) -> str:
        """Build a grounded LLM prompt from knowledge results."""
        if result is None:
            result = self.query(query)
        return build_grounded_prompt(query, result)

    # ── Knowledge gap management ──────────────────────────────────────────────

    def get_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Get detected knowledge gaps for the ingestion pipeline."""
        return self._resolver.get_knowledge_gaps()

    def clear_gaps(self) -> int:
        return self._resolver.clear_gaps()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            "resolver": self._resolver.snapshot(),
            "sources": [s.to_dict() for s in self._registry.list_sources()],
            "index_size": self._index.count,
        }

    def list_sources(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._registry.list_sources()]


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: Auto-ingest from data directory
# ═══════════════════════════════════════════════════════════════════════════════


def create_knowledge_layer_from_data(data_dir: str) -> KnowledgeLayer:
    """
    Create and populate a KnowledgeLayer from the data directory.
    Registers all JSON files as authoritative database sources.
    """
    layer = KnowledgeLayer()

    json_files = {
        "student_profiles": "student_profiles.json",
        "courses": "courses.json",
        "assignments": "assignments.json",
        "notes": "notes.json",
        "calendar_events": "calender_events.json",
        "conversation_summaries": "conversation_summaries.json",
    }

    for name, filename in json_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                layer.register_source(
                    name=name,
                    source_type=SourceType.DATABASES,
                    is_authoritative=True,
                    metadata={"file": filename},
                )
                count = layer.ingest_json(name, data)
                logger.debug(
                    f"KnowledgeLayer: auto-ingested {count} chunks from {filename}",
                )
            except Exception as e:
                logger.error(f"KnowledgeLayer: failed to ingest {filename}: {e}")

    return layer
