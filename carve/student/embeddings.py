from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from carve.schemas import Trace


class EmbeddingBackend(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class EmbeddedEvents:
    model_name: str
    event_ids: list[str]
    vectors: list[list[float]]


class HashEmbeddingBackend:
    """Deterministic local embedding backend for tests and offline smoke runs."""

    def __init__(self, dim: int = 64, model_name: str = "hash"):
        self.dim = dim
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            values = []
            counter = 0
            while len(values) < self.dim:
                digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
                values.extend((byte / 255.0) * 2.0 - 1.0 for byte in digest)
                counter += 1
            vectors.append(values[: self.dim])
        return vectors


class QwenEmbeddingBackend:
    """Qwen-compatible embedding backend with deterministic fallback.

    The paper path uses this class with a local/server Qwen embedding model. In
    lightweight environments without sentence-transformers or model weights, it
    falls back to a deterministic backend while preserving the same vector API.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", fallback: EmbeddingBackend | None = None):
        self.model_name = model_name
        self.fallback = fallback or HashEmbeddingBackend(dim=64, model_name="hash")
        self.dim = self.fallback.dim
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
            detected_dim = int(self._model.get_sentence_embedding_dimension() or self.dim)
            self.dim = detected_dim
        except Exception:
            self._model = None
        self.used_fallback = self._model is None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            return self.fallback.embed(texts)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in row] for row in vectors]


def embed_trace_events(trace: Trace, backend: EmbeddingBackend) -> EmbeddedEvents:
    texts = [event.content for event in trace.events]
    return EmbeddedEvents(
        model_name=backend.model_name,
        event_ids=[event.event_id for event in trace.events],
        vectors=backend.embed(texts),
    )
