"""Sparse (lexical) retrieval over code chunks using BM25.

This module implements keyword-based retrieval using the Okapi BM25 algorithm
via ``rank_bm25``.  It complements :class:`DenseRetriever` by excelling at
exact-match and keyword-heavy queries (e.g. function names, error messages).

Tokenisation is intentionally code-aware: it preserves ``snake_case`` sub-tokens,
``camelCase`` splits, and strips common programming punctuation so that BM25
ranks on semantically meaningful terms.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokeniser tuned for source code
# ---------------------------------------------------------------------------

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Produce lowercased sub-word tokens suitable for BM25 over code.

    Strategy
    --------
    1. Replace underscores and common separators with spaces.
    2. Insert spaces at camelCase boundaries.
    3. Strip non-alphanumeric chars.
    4. Lowercase and split.
    5. Drop single-character tokens (noise for BM25).
    """
    text = text.replace("_", " ").replace("-", " ").replace(".", " ")
    text = _CAMEL_BOUNDARY.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text.lower())
    return [tok for tok in text.split() if len(tok) > 1]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseSearchResult:
    """A single result returned by :meth:`SparseRetriever.retrieve`.

    Attributes:
        chunk_id:  Unique identifier carried over from the ``CodeChunk``.
        content:   Raw source text of the chunk.
        score:     BM25 relevance score (unnormalised — higher is better).
        metadata:  Flat dictionary of chunk metadata.
    """

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sparse retriever
# ---------------------------------------------------------------------------


class SparseRetriever:
    """Lexical retriever backed by BM25-Okapi.

    The retriever maintains an in-memory inverted index built via
    ``rank_bm25.BM25Okapi``.  Call :meth:`index_chunks` to ingest
    ``CodeChunk`` objects, then :meth:`retrieve` to query.
    """

    def __init__(self) -> None:
        self._bm25: Any | None = None
        self._corpus_tokens: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._contents: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_indexed(self) -> bool:
        """``True`` after :meth:`index_chunks` has run successfully."""
        return self._indexed

    @property
    def corpus_size(self) -> int:
        """Number of documents in the BM25 index."""
        return len(self._chunk_ids)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: Sequence[Any]) -> int:
        """Tokenise and build the BM25 index from *chunks*.

        Args:
            chunks: Iterable of ``CodeChunk`` (or any object with
                    ``chunk_id``, ``content``, and ``metadata``).

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            logger.info("index_chunks called with empty chunk list — nothing to index")
            return 0

        start_time = time.perf_counter()

        corpus_tokens: list[list[str]] = []
        chunk_ids: list[str] = []
        contents: list[str] = []
        metadata_list: list[dict[str, Any]] = []

        for chunk in chunks:
            text: str = chunk.content
            if not text or not text.strip():
                logger.debug(
                    "Skipping empty chunk during sparse indexing",
                    extra={"chunk_id": chunk.chunk_id},
                )
                continue

            tokens = _tokenize(text)
            if not tokens:
                logger.debug(
                    "Chunk produced no tokens after tokenisation",
                    extra={"chunk_id": chunk.chunk_id},
                )
                continue

            corpus_tokens.append(tokens)
            chunk_ids.append(chunk.chunk_id)
            contents.append(text)
            metadata_list.append(_chunk_metadata_to_dict(chunk.metadata))

        if not corpus_tokens:
            logger.warning("All chunks were empty or produced no tokens — nothing indexed")
            return 0

        # Build BM25 ---------------------------------------------------------
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 is required for SparseRetriever.  "
                "Install it with:  pip install rank-bm25"
            ) from exc

        self._bm25 = BM25Okapi(corpus_tokens)
        self._corpus_tokens = corpus_tokens
        self._chunk_ids = chunk_ids
        self._contents = contents
        self._metadata = metadata_list
        self._indexed = True

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Sparse (BM25) indexing complete",
            extra={
                "indexed_chunks": len(chunk_ids),
                "skipped_chunks": len(chunks) - len(chunk_ids),
                "avg_tokens_per_chunk": round(
                    sum(len(t) for t in corpus_tokens) / len(corpus_tokens), 1
                ),
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        return len(chunk_ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
    ) -> list[SparseSearchResult]:
        """Return the *top_k* most lexically relevant chunks for *query*.

        Args:
            query:  Natural-language or code query string.
            top_k:  Maximum number of results.

        Returns:
            Ordered list of :class:`SparseSearchResult` (highest score first).

        Raises:
            RuntimeError: If :meth:`index_chunks` has not been called.
        """
        if not self._indexed or self._bm25 is None:
            raise RuntimeError(
                "SparseRetriever has not been indexed yet.  "
                "Call index_chunks() before retrieve()."
            )

        if not query or not query.strip():
            logger.warning("retrieve called with empty query — returning no results")
            return []

        start_time = time.perf_counter()

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning(
                "Query produced no tokens after tokenisation",
                extra={"raw_query": query[:120]},
            )
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Argsort descending and take top_k ----------------------------------
        # numpy is already a project dependency (used everywhere in embeddings)
        import numpy as np

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[SparseSearchResult] = []
        for idx in top_indices:
            idx_int = int(idx)
            score_val = float(scores[idx_int])
            if score_val <= 0.0:
                # BM25 scores of 0 mean no term overlap — not useful.
                break
            results.append(
                SparseSearchResult(
                    chunk_id=self._chunk_ids[idx_int],
                    content=self._contents[idx_int],
                    score=score_val,
                    metadata=dict(self._metadata[idx_int]),
                )
            )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Sparse (BM25) retrieval complete",
            extra={
                "query_preview": query[:80],
                "query_tokens": query_tokens[:10],
                "top_k": top_k,
                "returned": len(results),
                "best_score": round(results[0].score, 4) if results else None,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402  (keep imports near usage for clarity)


def _chunk_metadata_to_dict(meta: Any) -> dict[str, Any]:
    """Flatten a ``ChunkMetadata`` dataclass into a serialisable dict."""
    try:
        return {
            "file_path": str(meta.file_path),
            "chunk_type": meta.chunk_type,
            "start_line": meta.start_line,
            "end_line": meta.end_line,
            "symbol_name": meta.symbol_name,
            "language": meta.language,
        }
    except AttributeError:
        try:
            raw = vars(meta)
        except TypeError:
            return {}
        return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}


__all__ = ["SparseRetriever", "SparseSearchResult"]
