"""
shared/embeddings.py
--------------------
Sentence-transformer embedding wrapper used by Task B's retrieval engine.
Lazily loads the model on first use to avoid startup overhead.

Usage:
    from shared.embeddings import EmbeddingModel
    em = EmbeddingModel()
    vec = em.encode("Great jollof rice, would recommend!")
    vecs = em.encode_batch(["review 1", "review 2"])
"""

from __future__ import annotations
import numpy as np
from typing import Union
from functools import lru_cache


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Wraps sentence-transformers for easy encoding.
    Model is downloaded once and cached locally by the HuggingFace hub.
    """

    _instance = None  # Singleton pattern to avoid reloading

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[EmbeddingModel] Loading {MODEL_NAME} ...")
            self._model = SentenceTransformer(MODEL_NAME)
            print("[EmbeddingModel] Ready.")

    def encode(self, text: str) -> np.ndarray:
        """Encode a single string → 384-dim float32 vector."""
        self._load()
        return self._model.encode(text, normalize_embeddings=True)

    def encode_batch(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode a list of strings → (N, 384) float32 array.
        Normalised for cosine similarity via dot product.
        """
        self._load()
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalised vectors."""
        return float(np.dot(a, b))

    def top_k_similar(
        self,
        query: np.ndarray,
        corpus: np.ndarray,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Returns (index, score) pairs for the top-k most similar vectors
        in corpus to the query vector.
        """
        scores = corpus @ query
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_k_idx]