"""Abstract interfaces for KG repository and vector index backends."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from acadgraph.kg.schema import (
    ArgumentationGraph,
    CitationEdge,
    ClaimEvidenceLedger,
    Entity,
    EntityRelation,
    EvolutionStep,
)


@runtime_checkable
class KgRepository(Protocol):
    """Core graph-storage capabilities required by builders and query engines."""

    async def connect(self) -> None:
        """Initialize graph database connection."""

    async def close(self) -> None:
        """Close graph database connection."""

    async def init_schema(self) -> None:
        """Initialize schema/constraints/indexes."""

    async def upsert_paper(
        self,
        paper_id: str,
        title: str,
        year: int | None = None,
        venue: str = "",
        authors: list[str] | None = None,
    ) -> None:
        """Insert or update a paper node."""

    async def upsert_entity(self, entity: Entity) -> str:
        """Insert or update an entity."""

    async def upsert_relation(self, relation: EntityRelation) -> None:
        """Insert or update an entity relation."""

    async def add_citation(self, edge: CitationEdge) -> bool:
        """Insert or update a citation edge and report success."""

    async def store_argumentation(self, paper_id: str, arg: ArgumentationGraph) -> None:
        """Store a full argumentation graph for one paper."""

    async def get_claim_evidence_ledger(self, paper_id: str) -> ClaimEvidenceLedger:
        """Query claim-evidence ledger by paper."""

    async def find_nearest_competitors(self, paper_id: str, k: int = 10) -> list[dict[str, Any]]:
        """Find nearest competitor papers."""

    async def get_evolution_chain(self, method_id: str) -> list[EvolutionStep]:
        """Get method evolution chain."""

    async def get_papers_by_year_range(
        self,
        paper_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> set[str]:
        """Filter candidate papers by publication year range."""

    async def find_evolution_related_papers(
        self,
        paper_ids: list[str],
        k: int = 20,
    ) -> list[dict[str, Any]]:
        """Expand candidates using evolution/citation graph links."""

    async def get_stats(self) -> dict[str, int]:
        """Get storage-level statistics."""

    async def paper_exists(self, paper_id: str) -> bool:
        """Check whether a paper exists."""


@runtime_checkable
class VectorIndex(Protocol):
    """Core vector-index capabilities required by builders and query engines."""

    async def connect(self) -> None:
        """Initialize vector database connection."""

    async def close(self) -> None:
        """Close vector database connection."""

    async def init_collections(self) -> None:
        """Initialize required collections."""

    async def upsert_embedding(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Insert/update one embedding point."""

    async def upsert_embeddings_batch(
        self,
        collection: str,
        point_ids: Sequence[str],
        vectors: Sequence[list[float]],
        payloads: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Insert/update multiple embedding points."""

    async def search_similar(
        self,
        collection: str,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search by vector similarity."""

    async def delete_by_paper_id(self, collection: str, paper_id: str) -> None:
        """Delete points linked to a paper id."""

    async def get_collection_count(self, collection: str) -> int:
        """Return number of points in a collection."""
