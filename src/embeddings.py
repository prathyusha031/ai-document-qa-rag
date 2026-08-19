"""
embeddings.py — Wraps the sentence-transformers embedding model.

The model (all-MiniLM-L6-v2) is loaded lazily on first use, because loading
it takes several seconds. It produces 384-dimensional vectors locally, so
there is no embedding API cost.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from . import config

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when the embedding model cannot be loaded or used."""


class EmbeddingModel:
    """Lazy wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None  # loaded on first use

    def _load(self):
        """Load the SentenceTransformer model (once)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model '%s' ...", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded.")
            except Exception as exc:
                logger.exception("Failed to load embedding model '%s'", self.model_name)
                raise EmbeddingError(
                    f"The embedding model '{self.model_name}' could not be loaded. "
                    "Check your internet connection (it downloads on first use) "
                    "and that sentence-transformers is installed."
                ) from exc
        return self._model

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Create one embedding vector per document text."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = self._load()
        try:
            return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise EmbeddingError("Embedding generation failed. Please try again.") from exc

    def embed_query(self, text: str) -> np.ndarray:
        """Create a single embedding vector for a user question."""
        return self.embed_documents([text])[0]
