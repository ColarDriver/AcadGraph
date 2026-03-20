"""
Async embedding client — wraps an OpenAI-compatible Embedding API.

Supports batching and caching for efficiency.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Sequence

from openai import AsyncOpenAI

from acadgraph.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Async wrapper around an OpenAI-compatible embedding endpoint."""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._client = AsyncOpenAI(
            base_url=self.config.api_base,
            api_key=self.config.api_key,
        )
        self._cache: dict[str, list[float]] = {}
        self._semaphore = asyncio.Semaphore(3)  # Embedding calls are heavier

    async def __aenter__(self) -> "EmbeddingClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Results are cached in-memory."""
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        result = await self.embed_batch([text])
        self._cache[key] = result[0]
        return result[0]

    async def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of texts, splitting into sub-batches if needed."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            async with self._semaphore:
                response = await self._client.embeddings.create(
                    model=self.config.model,
                    input=batch,
                )
            embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(embeddings)

        # Cache results
        for text, emb in zip(texts, all_embeddings):
            self._cache[self._cache_key(text)] = emb

        return all_embeddings

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    @property
    def dim(self) -> int:
        """Return the embedding dimensionality."""
        return self.config.dim

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
