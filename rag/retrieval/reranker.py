"""Cross-encoder reranker for retrieval result refinement.

This module provides a second-stage reranker that rescores retrieval
candidates using a cross-encoder model (``BAAI/bge-reranker-v2-m3``).
Unlike the bi-encoder used for dense retrieval, a cross-encoder jointly
attends to both the query and the document, producing substantially
more accurate relevance judgements at the cost of higher latency.

Typical usage is to run :class:`HybridRetriever` (or any retriever) first
to obtain ~30–50 candidates, then pass those through :meth:`Reranker.rerank`
to surface the best ~5–10.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankedResult:
    """A reranked retrieval result.

    Attributes:
        chunk_id:        Unique chunk identifier.
        content:         Raw source text of the chunk.
        rerank_score:    Relevance score produced by the cross-encoder
                         (logit; higher is better).
        original_score:  Score from the upstream retriever, preserved for
                         observability and score-blending experiments.
        metadata:        Flat dictionary of chunk metadata.
    """

    chunk_id: str
    content: str
    rerank_score: float
    original_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Cross-encoder reranker using ``BAAI/bge-reranker-v2-m3``.

    The underlying model is loaded lazily on the first call to
    :meth:`rerank` so that importing this module is always cheap.

    Parameters:
        model_name: HuggingFace model identifier.  Defaults to
                    ``BAAI/bge-reranker-v2-m3``.
        device:     Torch device string (``"cpu"`` or ``"cuda"``).
        batch_size: Number of query–document pairs scored per forward
                    pass.
    """

    _DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._cross_encoder: Any | None = None
        self._backend: str | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the cross-encoder on first use and cache the instance."""
        if self._cross_encoder is not None:
            return self._cross_encoder

        logger.info(
            "Loading reranker cross-encoder model",
            extra={"model_name": self._model_name, "device": self._device},
        )

        # Strategy: try FlagEmbedding first (purpose-built for BGE rerankers),
        # then fall back to sentence-transformers CrossEncoder, which is the
        # most widely-installed alternative.
        model = self._try_flag_reranker() or self._try_cross_encoder()

        if model is None:
            raise RuntimeError(
                "Neither FlagEmbedding (FlagReranker) nor "
                "sentence-transformers (CrossEncoder) is installed.  "
                "Install one with:  pip install FlagEmbedding  OR  "
                "pip install sentence-transformers"
            )

        self._cross_encoder = model
        logger.info(
            "Loaded reranker model",
            extra={"model_name": self._model_name, "backend": self._backend},
        )
        return model

    def _try_flag_reranker(self) -> Any | None:
        """Attempt to load ``FlagEmbedding.FlagReranker``."""
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            return None

        try:
            model = FlagReranker(self._model_name, device=self._device)
        except TypeError:
            # Older versions may not accept a device kwarg.
            model = FlagReranker(self._model_name)

        self._backend = "FlagEmbedding"
        return model

    def _try_cross_encoder(self) -> Any | None:
        """Attempt to load ``sentence_transformers.CrossEncoder``."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return None

        model = CrossEncoder(self._model_name, device=self._device)
        self._backend = "sentence-transformers"
        return model

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: Sequence[Any],
        *,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Score *results* against *query* with the cross-encoder and return
        the top-ranked subset.

        Each element of *results* must expose ``chunk_id``, ``content``, and
        ``metadata`` attributes (or dict keys).  The ``score`` attribute (or
        key) is carried forward as ``original_score``.

        Args:
            query:   The original search query.
            results: Candidate retrieval results to rerank.
            top_k:   Number of results to return.  ``None`` returns all
                     results (still re-sorted by rerank score).

        Returns:
            Ordered list of :class:`RerankedResult` (highest rerank score
            first).
        """
        if not results:
            logger.info("rerank called with empty result set — nothing to rerank")
            return []

        if not query or not query.strip():
            logger.warning("rerank called with empty query — returning empty results")
            return []

        start_time = time.perf_counter()
        model = self._load_model()

        # --- Normalise inputs into uniform dicts -------------------------
        candidates = [_normalise_candidate(r) for r in results]

        # --- Build query–document pairs ----------------------------------
        pairs: list[list[str]] = [
            [query, c["content"]] for c in candidates
        ]

        # --- Score in batches --------------------------------------------
        scores = self._compute_scores(model, pairs)

        # --- Attach scores and sort --------------------------------------
        scored: list[RerankedResult] = []
        for candidate, rerank_score in zip(candidates, scores):
            scored.append(
                RerankedResult(
                    chunk_id=candidate["chunk_id"],
                    content=candidate["content"],
                    rerank_score=float(rerank_score),
                    original_score=candidate.get("original_score"),
                    metadata=candidate.get("metadata", {}),
                )
            )

        scored.sort(key=lambda r: r.rerank_score, reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Reranking complete",
            extra={
                "query_preview": query[:80],
                "candidates_in": len(candidates),
                "returned": len(scored),
                "best_rerank_score": round(scored[0].rerank_score, 4) if scored else None,
                "backend": self._backend,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        return scored

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        model: Any,
        pairs: list[list[str]],
    ) -> list[float]:
        """Run the cross-encoder over *pairs* and return raw scores."""
        if self._backend == "FlagEmbedding":
            return self._score_flag(model, pairs)
        return self._score_cross_encoder(model, pairs)

    def _score_flag(
        self,
        model: Any,
        pairs: list[list[str]],
    ) -> list[float]:
        """Score with ``FlagReranker.compute_score``."""
        all_scores: list[float] = []
        for batch_start in range(0, len(pairs), self._batch_size):
            batch = pairs[batch_start : batch_start + self._batch_size]
            batch_scores = model.compute_score(batch)
            # FlagReranker returns a float for single pairs, list for batches.
            if isinstance(batch_scores, (int, float)):
                all_scores.append(float(batch_scores))
            else:
                all_scores.extend(float(s) for s in batch_scores)
        return all_scores

    def _score_cross_encoder(
        self,
        model: Any,
        pairs: list[list[str]],
    ) -> list[float]:
        """Score with ``CrossEncoder.predict``."""
        import numpy as np

        all_scores: list[float] = []
        for batch_start in range(0, len(pairs), self._batch_size):
            batch = pairs[batch_start : batch_start + self._batch_size]
            batch_scores = model.predict(batch, show_progress_bar=False)
            arr = np.atleast_1d(batch_scores)
            all_scores.extend(float(s) for s in arr)
        return all_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_candidate(result: Any) -> dict[str, Any]:
    """Accept either a dataclass/object or plain dict and return a dict."""
    if isinstance(result, dict):
        return {
            "chunk_id": result["chunk_id"],
            "content": result["content"],
            "original_score": result.get("score") or result.get("rerank_score"),
            "metadata": result.get("metadata", {}),
        }
    # Dataclass / named-attribute object
    return {
        "chunk_id": result.chunk_id,
        "content": result.content,
        "original_score": getattr(result, "score", None)
        or getattr(result, "rerank_score", None),
        "metadata": getattr(result, "metadata", {}),
    }


__all__ = ["Reranker", "RerankedResult"]
