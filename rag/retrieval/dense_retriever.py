"""Dense retrieval over code chunks using BGE embeddings and FAISS.

This module provides semantic vector search for the RepoGraph RAG pipeline.
Chunks are embedded with :class:`BGEEmbedder` and indexed in a
:class:`FAISSVectorStore`.  The retriever converts each upstream
:class:`CodeChunk` into a metadata-rich FAISS document and returns scored
results at query time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rag.embeddings.bge_embedder import BGEEmbedder
from rag.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseSearchResult:
    """A single result returned by :meth:`DenseRetriever.retrieve`.

    Attributes:
        chunk_id:  Unique identifier carried over from the ``CodeChunk``.
        content:   Raw source text of the chunk.
        score:     Cosine-similarity score produced by the FAISS inner-product
                   index (vectors are L2-normalised before insertion).
        metadata:  Flat dictionary that preserves every field from
                   ``ChunkMetadata`` so downstream consumers never need to
                   import the parser package.
    """

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------


class DenseRetriever:
    """Semantic retriever backed by BGE-M3 embeddings and a FAISS index.

    Parameters:
        embedder:      A pre-configured :class:`BGEEmbedder` instance.
        vector_store:  A pre-configured :class:`FAISSVectorStore` instance.
                       May already contain vectors (e.g. loaded from disk).
    """

    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_store: FAISSVectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        # In-memory lookup so retrieve() can return chunk *content* without
        # requiring an external database round-trip.
        self._content_by_id: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: Sequence[Any]) -> int:
        """Embed and index a sequence of ``CodeChunk`` objects.

        Each chunk's ``content`` is embedded via :class:`BGEEmbedder` and
        stored in the FAISS index together with its metadata.

        Args:
            chunks: Iterable of ``CodeChunk`` (or any object exposing
                    ``chunk_id``, ``content``, and ``metadata`` attributes).

        Returns:
            The number of chunks successfully indexed.
        """
        if not chunks:
            logger.info("index_chunks called with empty chunk list — nothing to index")
            return 0

        start_time = time.perf_counter()

        texts: list[str] = []
        chunk_ids: list[str] = []
        metadata_dicts: list[dict[str, Any]] = []

        for chunk in chunks:
            text = chunk.content
            if not text or not text.strip():
                logger.debug(
                    "Skipping empty chunk during dense indexing",
                    extra={"chunk_id": chunk.chunk_id},
                )
                continue

            meta = _chunk_metadata_to_dict(chunk.metadata)
            meta["content"] = text  # stash content inside FAISS metadata

            texts.append(text)
            chunk_ids.append(chunk.chunk_id)
            metadata_dicts.append(meta)
            self._content_by_id[chunk.chunk_id] = text

        if not texts:
            logger.warning("All chunks were empty — nothing indexed")
            return 0

        # Batch-embed --------------------------------------------------------
        embeddings: np.ndarray = self._embedder.embed_batch(texts)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise RuntimeError(
                f"Embedder returned unexpected shape {embeddings.shape} "
                f"for {len(texts)} texts"
            )

        # Insert into FAISS --------------------------------------------------
        stored_ids = self._vector_store.add_embeddings(
            embeddings,
            chunk_ids=chunk_ids,
            metadata=metadata_dicts,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Dense indexing complete",
            extra={
                "indexed_chunks": len(stored_ids),
                "skipped_chunks": len(chunks) - len(stored_ids),
                "embedding_dim": int(embeddings.shape[1]),
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        return len(stored_ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
    ) -> list[DenseSearchResult]:
        """Return the *top_k* most semantically similar chunks for *query*.

        Args:
            query:  Natural-language or code query string.
            top_k:  Maximum number of results to return.

        Returns:
            Ordered list of :class:`DenseSearchResult` (highest score first).
        """
        if not query or not query.strip():
            logger.warning("retrieve called with empty query — returning no results")
            return []

        start_time = time.perf_counter()

        query_embedding: np.ndarray = self._embedder.embed_text(query)

        if query_embedding.size == 0:
            logger.error(
                "Embedder produced empty vector for query",
                extra={"query_preview": query[:120]},
            )
            return []

        raw_results: list[dict[str, Any]] = self._vector_store.search(
            query_embedding, top_k=top_k,
        )

        results: list[DenseSearchResult] = []
        for hit in raw_results:
            chunk_id: str = hit["chunk_id"]
            metadata: dict[str, Any] = dict(hit.get("metadata", {}))

            # Resolve content — prefer the metadata stash, fall back to the
            # in-memory map, and finally to an empty string.
            content = metadata.pop("content", None) or self._content_by_id.get(chunk_id, "")

            results.append(
                DenseSearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    score=float(hit["score"]),
                    metadata=metadata,
                )
            )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Dense retrieval complete",
            extra={
                "query_preview": query[:80],
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


def _chunk_metadata_to_dict(meta: Any) -> dict[str, Any]:
    """Flatten a ``ChunkMetadata`` dataclass into a JSON-friendly dict.

    Handles the ``Path`` field by converting to ``str`` and leaves unknown
    attribute sets untouched by falling back to ``vars()``.
    """
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
        # Graceful degradation for unexpected metadata shapes.
        try:
            raw = vars(meta)
        except TypeError:
            return {}
        return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}


__all__ = ["DenseRetriever", "DenseSearchResult"]
