from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, text: str) -> list[float]:
        """Return a fixed-width vector for text."""


class HashingEmbedder:
    """Deterministic offline embedder for development, tests, and fallback memory search."""

    def __init__(self, dim: int = 384, model: str = "hashing-embedder-v1"):
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dim = dim
        self.model = model

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return normalize(vector)


class SentenceTransformerEmbedder:
    """Adapter for a real local embedding model when sentence-transformers is installed."""

    def __init__(self, model_name: str, dim: int | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbedder"
            ) from exc

        self.model = model_name
        self._model = SentenceTransformer(model_name)
        detected_dim = self._model.get_sentence_embedding_dimension()
        self.dim = int(dim or detected_dim)
        if self.dim != int(detected_dim):
            raise ValueError(
                f"configured dimension {self.dim} does not match model dimension {detected_dim}"
            )

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(value / norm) for value in vector]


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
    left_norm: float | None = None,
    right_norm: float | None = None,
) -> float:
    if len(left) != len(right):
        raise ValueError("cannot compare vectors with different dimensions")
    left_size = vector_norm(left) if left_norm is None else left_norm
    right_size = vector_norm(right) if right_norm is None else right_norm
    if left_size == 0 or right_size == 0:
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return float(dot / (left_size * right_size))
