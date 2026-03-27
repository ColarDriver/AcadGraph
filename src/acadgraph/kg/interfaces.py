"""Abstract interfaces for KG repository and vector index backends."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from acadgraph.kg.ontology import RelationMetadata
from acadgraph.kg.schema import (
    ArgumentationGraph,
    CitationEdge,
    ClaimEvidenceLedger,
    Entity,
    EntityRelation,
    EvolutionStep,
    SectionTreeNode,
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

    async def upsert_section_tree(
        self,
        paper_id: str,
        section_tree: list[SectionTreeNode],
    ) -> None:
        """Store hierarchical section tree for a paper."""

    async def get_section_tree(self, paper_id: str) -> list[dict[str, Any]]:
        """Get a paper's section tree for Tree Search retrieval."""

    async def link_claim_to_section(
        self,
        claim_id: str,
        section_node_id: str,
    ) -> None:
        """Link a claim to the section it appears in."""

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

    async def get_edges_by_source_rule(
        self,
        source_rule: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return relation edges carrying a specific source_rule metadata value."""

    async def paper_exists(self, paper_id: str) -> bool:
        """Check whether a paper exists."""

    async def link_paper_entity(
        self,
        paper_id: str,
        entity_id: str,
        relation_type: str,
        confidence: float = 1.0,
        properties: dict[str, Any] | None = None,
        source_rule: str | None = None,
        confidence_source: str | None = None,
        metadata: RelationMetadata | None = None,
    ) -> bool:
        """Create/merge a Paper->Entity relation edge."""

    async def link_method_claim(
        self,
        method_id: str,
        claim_id: str,
        source_paper_id: str,
        confidence: float = 0.7,
        source_rule: str | None = None,
        confidence_source: str | None = None,
        metadata: RelationMetadata | None = None,
    ) -> bool:
        """Create/merge a METHOD->CLAIM cross-layer edge."""

    async def link_dataset_evidence(
        self,
        dataset_id: str,
        evidence_id: str,
        source_paper_id: str,
        confidence: float = 0.9,
        source_rule: str | None = None,
        confidence_source: str | None = None,
        evidence_span: str | None = None,
        metadata: RelationMetadata | None = None,
    ) -> bool:
        """Create/merge a DATASET->EVIDENCE cross-layer edge."""

    async def add_method_evolution(
        self,
        from_method_id: str,
        to_method_id: str,
        source_paper_id: str,
        delta_description: str = "",
        year: int | None = None,
        confidence: float = 1.0,
    ) -> bool:
        """Create EVOLVES_FROM edge between method entities."""

    async def get_related_methods(
        self,
        paper_id: str,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Return this paper's methods and earlier comparable methods."""

    async def link_citation_gap(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        gap_id: str,
        confidence: float = 0.8,
        source_rule: str | None = None,
    ) -> bool:
        """Create SUPPORTS_GAP edge from a citation to a Gap node."""

    async def get_gap_context(self, problem_id: str) -> dict[str, Any]:
        """Get full gap context for a Problem."""

    async def traverse_evidence_chain(self, claim_id: str) -> dict[str, Any]:
        """Traverse the full evidence chain for a Claim."""

    # Innovation Path Mining
    async def find_gaps_for_methods(self, method_names: list[str]) -> list[dict[str, Any]]:
        """Find Gap/Problem nodes associated with given methods."""

    async def find_addressing_ideas(self, gap_ids: list[str]) -> list[dict[str, Any]]:
        """Find CoreIdeas that address given Gaps."""

    async def find_cross_domain_bridges(
        self, method_a_names: list[str], method_b_names: list[str]
    ) -> list[dict[str, Any]]:
        """Find shared concepts/tasks/datasets between two method groups."""

    async def find_unsupported_gaps(self, method_names: list[str]) -> list[dict[str, Any]]:
        """Find Gaps with no ADDRESSED_BY CoreIdea for given methods."""

    async def get_component_evidence(self, method_names: list[str]) -> list[dict[str, Any]]:
        """Get evidence strength breakdown per method component."""

    async def find_bridge_papers(
        self, method_a_names: list[str], method_b_names: list[str]
    ) -> list[dict[str, Any]]:
        """Find papers referencing methods from both domains."""


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
