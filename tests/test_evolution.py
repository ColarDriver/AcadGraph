"""Tests for evolution chain cycle detection."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from acadgraph.kg.extract.evolution import CitationEvolutionBuilder


class TestEvolutionCycleDetection:
    """Verify that evolution chain building rejects cycles."""

    def _make_builder(self, evolution_links: list[dict]) -> CitationEvolutionBuilder:
        """Create a builder with a mock LLM that returns predefined links."""
        builder = CitationEvolutionBuilder.__new__(CitationEvolutionBuilder)
        builder._llm = AsyncMock()
        builder._llm.complete_json = AsyncMock(
            return_value={"evolution_links": evolution_links}
        )
        return builder

    def test_normal_chain_is_built(self):
        """Non-cyclic links should produce chains."""
        builder = self._make_builder([
            {"from_method": "MethodA", "to_method": "MethodB", "delta_description": "improved"},
        ])
        methods = [
            {"method_id": "m1", "method_name": "MethodA", "paper_id": "p1", "year": 2020, "description": "base"},
            {"method_id": "m2", "method_name": "MethodB", "paper_id": "p2", "year": 2021, "description": "improved"},
        ]
        chains = asyncio.run(builder.build_evolution_chains(methods))
        assert len(chains) == 1

    def test_self_loop_is_skipped(self):
        """A→A self-loops should be rejected."""
        builder = self._make_builder([
            {"from_method": "MethodA", "to_method": "MethodA", "delta_description": "self"},
        ])
        methods = [
            {"method_id": "m1", "method_name": "MethodA", "paper_id": "p1", "year": 2020, "description": "base"},
            {"method_id": "m2", "method_name": "MethodB", "paper_id": "p2", "year": 2021, "description": "other"},
        ]
        chains = asyncio.run(builder.build_evolution_chains(methods))
        assert len(chains) == 0

    def test_reverse_edge_cycle_is_skipped(self):
        """A→B then B→A should skip the second edge as it creates a cycle."""
        builder = self._make_builder([
            {"from_method": "MethodA", "to_method": "MethodB", "delta_description": "forward"},
            {"from_method": "MethodB", "to_method": "MethodA", "delta_description": "backward"},
        ])
        methods = [
            {"method_id": "m1", "method_name": "MethodA", "paper_id": "p1", "year": 2020, "description": "base"},
            {"method_id": "m2", "method_name": "MethodB", "paper_id": "p2", "year": 2021, "description": "improved"},
        ]
        chains = asyncio.run(builder.build_evolution_chains(methods))
        # Only the first A→B should be created
        assert len(chains) == 1
        assert chains[0].root_method == "MethodA"


class TestEvolutionMinimumMethods:
    """Edge case: fewer than 2 methods should return empty."""

    def test_single_method(self):
        builder = CitationEvolutionBuilder.__new__(CitationEvolutionBuilder)
        methods = [
            {"method_id": "m1", "method_name": "MethodA", "paper_id": "p1", "year": 2020, "description": "only one"},
        ]
        chains = asyncio.run(builder.build_evolution_chains(methods))
        assert chains == []
