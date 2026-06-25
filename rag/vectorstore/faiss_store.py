from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)

SearchResult = dict[str, Any]


class FAISSVectorStore:
    """FAISS-backed vector store using cosine similarity."""

    INDEX_FILE_NAME = "index.faiss"
    METADATA_FILE_NAME = "metadata.pkl"

    def __init__(self, index_path: str | Path | None = None) -> None:
        self.index_path = Path(index_path) if index_path is not None else None
        self.index: Any | None = None
        self.dimension: int | None = None
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.chunk_ids: list[str] = []
        self.chunk_metadata: list[dict[str, Any]] = []

    def create_index(self, dimension: int) -> None:
        """Create an empty FAISS index for normalized vectors."""
        if dimension <= 0:
            raise ValueError("FAISS index dimension must be greater than zero.")

        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required to use FAISSVectorStore.") from exc

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.embeddings = np.empty((0, dimension), dtype=np.float32)
        self.chunk_ids = []
        self.chunk_metadata = []

        logger.info("Created FAISS vector index", extra={"dimension": dimension})

    def add_embeddings(
        self,
        embeddings: np.ndarray | Sequence[Sequence[float]],
        *,
        chunk_ids: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Normalize and add a batch of embeddings to the index."""
        vectors = _as_2d_float32(embeddings)
        if vectors.shape[0] == 0:
            logger.debug("Skipping empty FAISS insertion batch")
            return []

        if self.index is None:
            self.create_index(vectors.shape[1])

        if self.dimension != vectors.shape[1]:
            raise ValueError(f"Embedding dimension {vectors.shape[1]} does not match index dimension {self.dimension}.")

        ids = list(chunk_ids) if chunk_ids is not None else [str(uuid4()) for _ in range(vectors.shape[0])]
        if len(ids) != vectors.shape[0]:
            raise ValueError("chunk_ids length must match number of embeddings.")

        metadata_items = list(metadata) if metadata is not None else [{} for _ in range(vectors.shape[0])]
        if len(metadata_items) != vectors.shape[0]:
            raise ValueError("metadata length must match number of embeddings.")

        normalized_vectors = _normalize(vectors)
        self.index.add(normalized_vectors)
        self.embeddings = (
            normalized_vectors
            if self.embeddings.size == 0
            else np.vstack([self.embeddings, normalized_vectors]).astype(np.float32)
        )
        self.chunk_ids.extend(ids)
        self.chunk_metadata.extend(metadata_items)

        logger.info(
            "Added embeddings to FAISS index",
            extra={"batch_size": len(ids), "total_vectors": len(self.chunk_ids), "dimension": self.dimension},
        )
        return ids

    def search(self, query_embedding: np.ndarray | Sequence[float], *, top_k: int = 5) -> list[SearchResult]:
        """Search the index and return score, chunk_id, and metadata."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.index is None or self.index.ntotal == 0:
            logger.debug("Search requested against an empty FAISS index")
            return []

        query = _as_2d_float32(query_embedding)
        if query.shape[0] != 1:
            raise ValueError("search expects a single query embedding.")
        if self.dimension != query.shape[1]:
            raise ValueError(f"Query dimension {query.shape[1]} does not match index dimension {self.dimension}.")

        limit = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(_normalize(query), limit)

        results: list[SearchResult] = []
        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue
            results.append(
                {
                    "score": float(score),
                    "chunk_id": self.chunk_ids[int(index_position)],
                    "metadata": self.chunk_metadata[int(index_position)],
                }
            )

        logger.debug(
            "Completed FAISS search",
            extra={"top_k": top_k, "returned": len(results), "total_vectors": self.index.ntotal},
        )
        return results

    def save(self, index_path: str | Path | None = None) -> None:
        """Persist the FAISS index and sidecar metadata."""
        path = self._resolve_index_path(index_path)
        if self.index is None or self.dimension is None:
            raise RuntimeError("Cannot save FAISS store before an index is created.")

        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required to save FAISSVectorStore.") from exc

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / self.INDEX_FILE_NAME))
        with (path / self.METADATA_FILE_NAME).open("wb") as metadata_file:
            pickle.dump(
                {
                    "dimension": self.dimension,
                    "embeddings": self.embeddings,
                    "chunk_ids": self.chunk_ids,
                    "chunk_metadata": self.chunk_metadata,
                },
                metadata_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        self.index_path = path
        logger.info("Saved FAISS vector store", extra={"index_path": str(path), "total_vectors": len(self.chunk_ids)})

    def load(self, index_path: str | Path | None = None) -> None:
        """Load a persisted FAISS index and metadata sidecar."""
        path = self._resolve_index_path(index_path)
        index_file = path / self.INDEX_FILE_NAME
        metadata_file = path / self.METADATA_FILE_NAME

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file does not exist: {index_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"FAISS metadata file does not exist: {metadata_file}")

        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required to load FAISSVectorStore.") from exc

        self.index = faiss.read_index(str(index_file))
        with metadata_file.open("rb") as metadata_handle:
            payload = pickle.load(metadata_handle)

        self.dimension = int(payload["dimension"])
        self.embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        self.chunk_ids = list(payload["chunk_ids"])
        self.chunk_metadata = list(payload["chunk_metadata"])
        self.index_path = path

        if self.index.d != self.dimension:
            raise ValueError("Loaded FAISS index dimension does not match metadata dimension.")
        if self.index.ntotal != len(self.chunk_ids):
            raise ValueError("Loaded FAISS index vector count does not match metadata count.")

        logger.info("Loaded FAISS vector store", extra={"index_path": str(path), "total_vectors": self.index.ntotal})

    def delete_index(self) -> None:
        """Clear the in-memory index and remove persisted sidecar files when configured."""
        total_vectors = len(self.chunk_ids)
        self.index = None
        self.dimension = None
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.chunk_ids = []
        self.chunk_metadata = []

        deleted_files = 0
        if self.index_path is not None:
            for file_name in (self.INDEX_FILE_NAME, self.METADATA_FILE_NAME):
                file_path = self.index_path / file_name
                if file_path.exists():
                    file_path.unlink()
                    deleted_files += 1

        logger.info(
            "Deleted FAISS vector index",
            extra={"deleted_vectors": total_vectors, "deleted_files": deleted_files},
        )

    def _resolve_index_path(self, index_path: str | Path | None) -> Path:
        path = Path(index_path) if index_path is not None else self.index_path
        if path is None:
            raise ValueError("index_path must be provided.")
        return path


def _as_2d_float32(vectors: np.ndarray | Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Embeddings must be a 1D or 2D numeric array.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


__all__ = ["FAISSVectorStore", "SearchResult"]
