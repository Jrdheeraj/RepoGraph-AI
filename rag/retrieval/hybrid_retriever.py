"""Hybrid retrieval combining dense (semantic) and sparse (lexical) signals.

This module fuses results from :class:`DenseRetriever` and
:class:`SparseRetriever` using **Reciprocal Rank Fusion (RRF)**, a
score-aggregation method that is robust to the incompatible score
distributions produced by cosine-similarity (dense) vs. BM25 (sparse).

RRF reference
-------------
Cormack, Clarke & Buettcher — *Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods*, SIGIR 2009.

    rrf_score(d) = Σ  1 / (k + rank_i(d))
                   i
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)

# Default RRF constant — 60 is the value used in the original paper and in
# Elasticsearch / OpenSearch hybrid search implementations.
_DEFAULT_RRF_K = 60


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridSearchResult:
    """A single result returned by :meth:`HybridRetriever.retrieve`.

    Attributes:
        chunk_id:     Unique chunk identifier.
        content:      Raw source text.
        score:        Fused RRF score (higher is better).
        dense_score:  Original cosine-similarity score from the dense
                      retriever, or ``None`` if the chunk was only found
                      by the sparse retriever.
        sparse_score: Original BM25 score, or ``None`` if only found by
                      the dense retriever.
        metadata:     Flat dictionary of chunk metadata.
    """

    chunk_id: str
    content: str
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Fuses dense and sparse retrieval via Reciprocal Rank Fusion.

    Parameters:
        dense_retriever:  An initialised :class:`DenseRetriever`.
        sparse_retriever: An initialised :class:`SparseRetriever`.
        rrf_k:            The ranking constant *k* in the RRF formula.
                          Higher values smooth out rank differences.
        dense_weight:     Multiplicative weight applied to dense RRF
                          contributions.  Defaults to ``1.0``.
        sparse_weight:    Multiplicative weight applied to sparse RRF
                          contributions.  Defaults to ``1.0``.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        *,
        rrf_k: int = _DEFAULT_RRF_K,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("Retriever weights must be non-negative")

        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
    ) -> list[HybridSearchResult]:
        """Run both retrievers and fuse results with RRF.

        Args:
            query:         The search query.
            top_k:         Number of final fused results to return.
            dense_top_k:   Candidates to pull from the dense retriever.
                           Defaults to ``top_k * 3`` so RRF has headroom.
            sparse_top_k:  Candidates to pull from the sparse retriever.
                           Defaults to ``top_k * 3``.

        Returns:
            Ordered list of :class:`HybridSearchResult` (highest RRF score
            first).
        """
        if not query or not query.strip():
            logger.warning("retrieve called with empty query — returning no results")
            return []

        start_time = time.perf_counter()

        candidate_k = top_k * 3
        dense_k = dense_top_k if dense_top_k is not None else candidate_k
        sparse_k = sparse_top_k if sparse_top_k is not None else candidate_k

        # --- Fetch candidates from both retrievers -----------------------
        dense_results = self._dense.retrieve(query, top_k=dense_k)
        sparse_results = self._sparse.retrieve(query, top_k=sparse_k)

        # --- Build per-chunk accumulators --------------------------------
        # _ChunkAccumulator is a mutable helper; we key by chunk_id.
        accumulators: dict[str, _ChunkAccumulator] = {}

        for rank, hit in enumerate(dense_results, start=1):
            acc = accumulators.setdefault(
                hit.chunk_id,
                _ChunkAccumulator(
                    chunk_id=hit.chunk_id,
                    content=hit.content,
                    metadata=hit.metadata,
                ),
            )
            acc.dense_rank = rank
            acc.dense_score = hit.score

        for rank, hit in enumerate(sparse_results, start=1):
            acc = accumulators.setdefault(
                hit.chunk_id,
                _ChunkAccumulator(
                    chunk_id=hit.chunk_id,
                    content=hit.content,
                    metadata=hit.metadata,
                ),
            )
            acc.sparse_rank = rank
            acc.sparse_score = hit.score

        # --- Compute RRF scores ------------------------------------------
        for acc in accumulators.values():
            rrf = 0.0
            if acc.dense_rank is not None:
                rrf += self._dense_weight / (self._rrf_k + acc.dense_rank)
            if acc.sparse_rank is not None:
                rrf += self._sparse_weight / (self._rrf_k + acc.sparse_rank)
            acc.rrf_score = rrf

        # --- Sort and truncate -------------------------------------------
        sorted_accs = sorted(
            accumulators.values(),
            key=lambda a: a.rrf_score,
            reverse=True,
        )[:top_k]

        results = [
            HybridSearchResult(
                chunk_id=acc.chunk_id,
                content=acc.content,
                score=acc.rrf_score,
                dense_score=acc.dense_score,
                sparse_score=acc.sparse_score,
                metadata=acc.metadata,
            )
            for acc in sorted_accs
        ]

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Hybrid retrieval complete",
            extra={
                "query_preview": query[:80],
                "dense_candidates": len(dense_results),
                "sparse_candidates": len(sparse_results),
                "unique_candidates": len(accumulators),
                "returned": len(results),
                "best_rrf_score": round(results[0].score, 6) if results else None,
                "rrf_k": self._rrf_k,
                "dense_weight": self._dense_weight,
                "sparse_weight": self._sparse_weight,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        return results


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------


class _ChunkAccumulator:
    """Mutable helper used during RRF score computation."""

    __slots__ = (
        "chunk_id",
        "content",
        "metadata",
        "dense_rank",
        "dense_score",
        "sparse_rank",
        "sparse_score",
        "rrf_score",
    )

    def __init__(
        self,
        chunk_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = metadata
        self.dense_rank: int | None = None
        self.dense_score: float | None = None
        self.sparse_rank: int | None = None
        self.sparse_score: float | None = None
        self.rrf_score: float = 0.0


__all__ = ["HybridRetriever", "HybridSearchResult"]
