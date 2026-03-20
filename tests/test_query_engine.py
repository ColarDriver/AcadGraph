"""Tests for query_engine.py — retrieval paths and trend filtering."""

import asyncio

from acadgraph.kg.query_engine import KGQueryEngine


class _StubEmbedding:
    async def embed(self, _text: str):
        return [0.1, 0.2, 0.3]


class _StubQdrant:
    def __init__(self):
        self.calls = []

    async def search_similar(self, collection, _query_vector, k=10, filters=None):
        self.calls.append((collection, k, filters))
        if collection == "entities":
            # first/second entities call both return entity-like payloads
            return [
                {
                    "id": "ent-1",
                    "score": 0.92,
                    "payload": {"paper_id": "p1", "entity_type": "METHOD", "name": "MethodA", "section": "method"},
                },
                {
                    "id": "ent-2",
                    "score": 0.87,
                    "payload": {"paper_id": "p2", "entity_type": "METHOD", "name": "MethodB", "section": "introduction"},
                },
                {
                    "id": "ent-3",
                    "score": 0.80,
                    "payload": {"paper_id": "p3", "entity_type": "DATASET", "name": "DataC", "section": "experiments"},
                },
            ]
        if collection == "claims":
            return [
                {"id": "cl-1", "score": 0.70, "payload": {"paper_id": "p1", "claim_type": "PERFORMANCE", "evidence_type": "EXPERIMENT"}},
            ]
        return []

    async def get_collection_count(self, _collection):
        return 0


class _StubNeo4j:
    def __init__(self):
        self.year_filter_calls = []

    async def get_papers_by_year_range(self, paper_ids, start_year, end_year):
        self.year_filter_calls.append((set(paper_ids), start_year, end_year))
        # Keep only p2 in range for this test.
        return {"p2"}

    async def find_evolution_related_papers(self, paper_ids, k=20):
        # Return one candidate expanded from seed papers.
        if not paper_ids:
            return []
        return [
            {
                "paper_id": "p5",
                "relation_confidence": 0.9,
                "link_count": 2,
            }
        ]

    # Methods needed by type checks/other flows but unused here.
    async def get_stats(self):
        return {}


class _StubLLM:
    async def complete_json(self, _prompt: str):
        return {}


def _build_engine():
    return KGQueryEngine(
        neo4j=_StubNeo4j(),
        qdrant=_StubQdrant(),
        llm=_StubLLM(),
        embedding=_StubEmbedding(),
    )


def test_get_research_trends_applies_year_range_filter():
    """year_range should filter related paper statistics using Neo4j year data."""
    engine = _build_engine()

    result = asyncio.run(engine.get_research_trends("test domain", year_range=(2020, 2021)))

    assert result["year_range"] == (2020, 2021)
    assert result["related_paper_count"] == 1
    assert result["total_related_entities"] == 1
    assert [item["paper_id"] for item in result["top_entities"]] == ["p2"]

    neo4j = engine.neo4j
    assert len(neo4j.year_filter_calls) == 1
    paper_ids, start_year, end_year = neo4j.year_filter_calls[0]
    assert paper_ids == {"p1", "p2", "p3"}
    assert (start_year, end_year) == (2020, 2021)


def test_enhanced_recall_includes_evolution_path_candidates():
    """Path3 should add evolution-related papers into merged recall results."""
    engine = _build_engine()

    results = asyncio.run(engine.enhanced_recall("new method", k=10))

    paper_ids = [item["paper_id"] for item in results]
    assert "p5" in paper_ids

    p5 = next(item for item in results if item["paper_id"] == "p5")
    assert p5["path_scores"].get("evolution", 0) > 0
