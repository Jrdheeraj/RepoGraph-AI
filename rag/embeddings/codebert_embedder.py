from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class CodeBERTEmbedder:
    """Lazy CodeBERT embedder using Transformers mean pooling."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        *,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def load_model(self) -> tuple[Any, Any]:
        """Load tokenizer/model on first use and return cached instances."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required to use CodeBERTEmbedder.") from exc

        logger.info(
            "Loading CodeBERT embedding model",
            extra={"model_name": self.model_name, "device": self.device},
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.to(torch.device(self.device))
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        logger.info("Loaded CodeBERT embedding model", extra={"model_name": self.model_name})
        return tokenizer, model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text value as a numpy vector."""
        vectors = self.embed_batch([text])
        if vectors.size == 0:
            return np.array([], dtype=np.float32)
        return vectors[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts using mean pooling over token embeddings."""
        clean_texts = [text for text in texts if text and text.strip()]
        if not clean_texts:
            logger.debug("Skipping empty CodeBERT embedding batch")
            return np.empty((0, 0), dtype=np.float32)

        tokenizer, model = self.load_model()
        torch = self._torch
        if torch is None:
            raise RuntimeError("CodeBERT torch runtime was not initialized.")

        vectors: list[np.ndarray] = []
        logger.debug(
            "Embedding text batch with CodeBERT",
            extra={"model_name": self.model_name, "batch_size": len(clean_texts)},
        )

        for start in range(0, len(clean_texts), self.batch_size):
            batch = clean_texts[start : start + self.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                output = model(**encoded)
                pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"], torch)
                vectors.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)


def _mean_pool(token_embeddings: Any, attention_mask: Any, torch: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * mask
    summed = masked_embeddings.sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


__all__ = ["CodeBERTEmbedder"]
