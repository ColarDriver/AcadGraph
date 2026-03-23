"""Tests for entity deduplication cache lock and precise argumentation mapping."""

import asyncio
from types import SimpleNamespace

import pytest

from acadgraph.kg.extract.entities import EntityExtractor
from acadgraph.kg.schema import Entity, EntityType


class TestEntityCacheLock:
    """Verify that EntityExtractor uses asyncio.Lock for thread safety."""

    def test_has_cache_lock(self):
        """EntityExtractor should have a _cache_lock attribute."""
        from unittest.mock import MagicMock

        ext = EntityExtractor(
            llm=MagicMock(),
            embedding=MagicMock(),
            qdrant=None,
        )
        assert hasattr(ext, "_cache_lock")
        assert isinstance(ext._cache_lock, asyncio.Lock)

    def test_dedup_deterministic_under_sequential_calls(self):
        """Sequential dedup calls should produce consistent results."""
        from unittest.mock import MagicMock

        ext = EntityExtractor(
            llm=MagicMock(),
            embedding=MagicMock(),
            qdrant=None,
        )
        e1 = Entity(name="BERT", entity_type=EntityType.METHOD, description="language model")
        e2 = Entity(name="BERT", entity_type=EntityType.METHOD, description="bidirectional LM")

        async def _run():
            # First call: caches BERT
            r1 = await ext.deduplicate_cross_paper([e1])
            # Second call: should merge (same name)
            r2 = await ext.deduplicate_cross_paper([e2])
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert len(r1) == 1
        assert len(r2) == 1
        # Second should reuse the first entity's ID:
        assert r2[0].entity_id == r1[0].entity_id


class TestPreciseArgumentationMapping:
    """Verify that ArgumentationGraph carries mapping metadata."""

    def test_argumentation_graph_accepts_mapping_attrs(self):
        """The graph should accept _problem_gap_map as dynamic attr."""
        from acadgraph.kg.schema import ArgumentationGraph

        ag = ArgumentationGraph(paper_id="test")
        ag._problem_gap_map = {0: [0]}  # type: ignore[attr-defined]
        ag._gap_idea_map = {0: [0]}  # type: ignore[attr-defined]
        ag._idea_claim_map = {0: [0, 1]}  # type: ignore[attr-defined]

        assert ag._problem_gap_map == {0: [0]}
        assert ag._gap_idea_map == {0: [0]}
        assert ag._idea_claim_map == {0: [0, 1]}

    def test_getattr_fallback_for_missing_maps(self):
        """getattr with default should work when maps are not set."""
        from acadgraph.kg.schema import ArgumentationGraph

        ag = ArgumentationGraph(paper_id="test")
        assert getattr(ag, "_problem_gap_map", {}) == {}
        assert getattr(ag, "_gap_idea_map", {}) == {}
        assert getattr(ag, "_idea_claim_map", {}) == {}
