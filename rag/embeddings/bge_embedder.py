from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """Lazy BGE-M3 embedder backed by FlagEmbedding."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str = "cpu",
        use_fp16: bool = False,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self._model: Any | None = None
        self._backend: str | None = None

    def load_model(self) -> Any:
        """Load the BGE model on first use and return the cached instance."""
        if self._model is not None:
            return self._model

        logger.info(
            "Loading BGE embedding model",
            extra={"model_name": self.model_name, "device": self.device, "use_fp16": self.use_fp16},
        )
        try:
            from FlagEmbedding import BGEM3FlagModel

            try:
                self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16, devices=self.device)
            except TypeError:
                try:
                    self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16, device=self.device)
                except TypeError:
                    self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
            self._backend = "FlagEmbedding"
        except ImportError:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "FlagEmbedding or sentence-transformers is required to use BGEEmbedder."
                ) from exc

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._backend = "sentence-transformers"

        logger.info(
            "Loaded BGE embedding model",
            extra={"model_name": self.model_name, "backend": self._backend},
        )
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text value as a normalized numpy vector."""
        vectors = self.embed_batch([text])
        if vectors.size == 0:
            return np.array([], dtype=np.float32)
        return vectors[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts as normalized numpy vectors."""
        clean_texts = [text for text in texts if text and text.strip()]
        if not clean_texts:
            logger.debug("Skipping empty BGE embedding batch")
            return np.empty((0, 0), dtype=np.float32)

        model = self.load_model()
        logger.debug(
            "Embedding text batch with BGE",
            extra={"model_name": self.model_name, "batch_size": len(clean_texts)},
        )
        if self._backend == "sentence-transformers":
            dense_vectors = model.encode(
                clean_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.asarray(dense_vectors, dtype=np.float32)

        output = model.encode(
            clean_texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"] if isinstance(output, dict) else output
        return _normalize(np.asarray(dense_vectors, dtype=np.float32))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


__all__ = ["BGEEmbedder"]
