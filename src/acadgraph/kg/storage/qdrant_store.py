"""
Qdrant vector storage for semantic retrieval.

Manages only two collections for this project profile:
- entities: Embeddings of academic entities
- claims: Embeddings of claims/evidence-style statements
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from acadgraph.config import QdrantConfig
from acadgraph.kg.interfaces import VectorIndex
from acadgraph.kg.ontology import CLAIM_COLLECTION, ENTITY_COLLECTION

logger = logging.getLogger(__name__)

# Collection definitions
COLLECTIONS = {
    ENTITY_COLLECTION: "METHOD, DATASET, METRIC, TASK, MODEL, FRAMEWORK, CONCEPT embeddings",
    CLAIM_COLLECTION: "Claim text embeddings",
}


class QdrantKGStore(VectorIndex):
    """Qdrant vector storage for the Three-Layer KG."""

    def __init__(self, config: QdrantConfig | None = None, embedding_dim: int = 4096):
        self.config = config or QdrantConfig()
        self.embedding_dim = embedding_dim
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        """Connect to Qdrant."""
        self._client = AsyncQdrantClient(
            host=self.config.host,
            port=self.config.port,
        )
        logger.info("Connected to Qdrant at %s:%d", self.config.host, self.config.port)

    async def close(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            await self._client.close()
            logger.info("Qdrant connection closed")

    @property
    def client(self) -> AsyncQdrantClient:
        assert self._client is not None, "Not connected. Call connect() first."
        return self._client

    async def init_collections(self) -> None:
        """Create all required collections if they don't exist."""
        existing = await self.client.get_collections()
        existing_names = {c.name for c in existing.collections}

        for name in COLLECTIONS:
            if name not in existing_names:
                await self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", name)
            else:
                logger.debug("Collection already exists: %s", name)

    async def upsert_embedding(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a single embedding."""
        await self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=self._to_int_id(point_id),
                    vector=vector,
                    payload={**(payload or {}), "_original_id": point_id},
                )
            ],
        )

    async def upsert_embeddings_batch(
        self,
        collection: str,
        point_ids: Sequence[str],
        vectors: Sequence[list[float]],
        payloads: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Batch insert/update embeddings."""
        if payloads is None:
            payloads = [{}] * len(point_ids)

        points = [
            PointStruct(
                id=self._to_int_id(pid),
                vector=vec,
                payload={**pl, "_original_id": pid},
            )
            for pid, vec, pl in zip(point_ids, vectors, payloads)
        ]
        await self.client.upsert(collection_name=collection, points=points)

    async def search_similar(
        self,
        collection: str,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors in a collection.

        Returns a list of dicts with 'id', 'score', and 'payload'.
        """
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=val))
                for key, val in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = await self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=k,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": hit.payload.get("_original_id", str(hit.id)) if hit.payload else str(hit.id),
                "score": hit.score,
                "payload": hit.payload or {},
            }
            for hit in results
        ]

    async def delete_by_paper_id(self, collection: str, paper_id: str) -> None:
        """Delete all points associated with a paper."""
        await self.client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
        )

    async def get_collection_count(self, collection: str) -> int:
        """Get the number of points in a collection."""
        info = await self.client.get_collection(collection)
        return info.points_count or 0

    @staticmethod
    def _to_int_id(string_id: str) -> int:
        """Convert a string ID to an integer for Qdrant (using hash)."""
        return abs(hash(string_id)) % (2**63)
