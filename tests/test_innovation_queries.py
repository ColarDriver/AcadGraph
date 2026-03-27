"""Tests for innovation path mining query methods."""

import asyncio

from acadgraph.kg.ontology import (
    CLAIM_COLLECTION,
    DOC_KIND_CLAIM,
    DOC_KIND_ENTITY,
    DOC_KIND_EVIDENCE,
    DOC_KIND_SECTION,
    ENTITY_COLLECTION,
)
from acadgraph.kg.query_engine import KGQueryEngine
from acadgraph.kg.schema import (
    ComponentEvidence,
    CrossDomainBridge,
    InnovationPath,
)


class _StubEmbedding:
    async def embed(self, _text: str):
        return [0.1, 0.2, 0.3]


class _StubQdrant:
    def __init__(self):
        self.calls = []

    async def search_similar(self, collection, _query_vector, k=10, filters=None):
        self.calls.append((collection, k, filters))
        if collection == ENTITY_COLLECTION:
            return [
                {
                    "id": "ent-1",
                    "score": 0.92,
                    "payload": {"paper_id": "p1", "entity_type": "METHOD", "name": "SAM", "section": "method"},
                },
            ]
        if collection == CLAIM_COLLECTION:
            return [
                {
                    "id": "cl-1",
                    "score": 0.85,
                    "payload": {"paper_id": "p1", "doc_kind": DOC_KIND_CLAIM, "claim_type": "PERFORMANCE"},
                },
            ]
        return []

    async def get_collection_count(self, _collection):
        return 0


class _StubNeo4j:
    """Mock Neo4j with innovation path mining methods."""

    async def find_gaps_for_methods(self, method_names):
        if "sam" in [m.lower() for m in method_names]:
            return [
                {
                    "method": "SAM",
                    "problem": "Lack of GUI understanding",
                    "problem_id": "prob-1",
                    "failure_mode": "Cannot handle GUI elements",
                    "constraint": "GUI-specific layout",
                    "gap_id": "gap-1",
                    "paper_id": "p1",
                    "title": "SAM Paper",
                }
            ]
        return []

    async def find_addressing_ideas(self, gap_ids):
        if "gap-1" in gap_ids:
            return [
                {
                    "gap_id": "gap-1",
                    "idea_id": "ci-1",
                    "mechanism": "Visual grounding for GUI",
                    "novelty_type": "NEW_MECHANISM",
                    "key_innovation": "Ground visual elements to UI semantics",
                    "paper_id": "p2",
                    "claim_count": 3,
                }
            ]
        return []

    async def find_unsupported_gaps(self, method_names):
        return [
            {
                "gap_id": "gap-2",
                "failure_mode": "No RL-based interaction",
                "constraint": "Agent decision making",
                "problem": "GUI navigation is sequential",
                "problem_id": "prob-2",
                "paper_id": "p3",
                "title": "GUI Agent Paper",
            }
        ]

    async def find_cross_domain_bridges(self, method_a, method_b):
        return [
            {"entity_type": "Task", "name": "UI Navigation", "papers_a_count": 3},
            {"entity_type": "Concept", "name": "Visual Grounding", "papers_a_count": 2},
        ]

    async def find_bridge_papers(self, method_a, method_b):
        return [
            {
                "paper_id": "bridge-1",
                "title": "Visual Grounding for GUI Agents",
                "year": 2024,
                "venue": "NeurIPS",
                "methods_a": ["SAM"],
                "methods_b": ["PPO"],
            }
        ]

    async def get_component_evidence(self, method_names):
        if not method_names:
            return []
        return [
            {
                "method": "SAM",
                "paper_id": "p1",
                "claim_count": 5,
                "evidence_count": 3,
                "strengths": ["FULL", "PARTIAL"],
                "claim_details": [
                    {"claim_id": "c1", "text": "SAM achieves SOTA", "severity": "P0", "has_evidence": True},
                    {"claim_id": "c2", "text": "SAM generalizes to GUI", "severity": "P0", "has_evidence": False},
                ],
            }
        ]

    async def get_evolution_chain(self, method_id):
        return []

    async def get_papers_by_year_range(self, paper_ids, start_year, end_year):
        return set()

    async def find_evolution_related_papers(self, paper_ids, k=20):
        return []

    async def get_edges_by_source_rule(self, source_rule, limit=50):
        return []

    async def get_stats(self):
        return {}

    async def paper_exists(self, paper_id):
        return False


class _StubLLM:
    async def complete_json(self, _prompt: str):
        return {"suggested_combination": "Combine SAM visual grounding with RL for GUI agent training"}

    async def generate(self, _prompt: str):
        return "[]"


def _build_engine():
    return KGQueryEngine(
        neo4j=_StubNeo4j(),
        qdrant=_StubQdrant(),
        llm=_StubLLM(),
        embedding=_StubEmbedding(),
    )


class TestFindInnovationPaths:
    """Tests for find_innovation_paths."""

    def test_returns_innovation_path(self):
        engine = _build_engine()
        result = asyncio.run(engine.find_innovation_paths(
            "GUI agent with visual grounding",
            ["SAM", "PPO"],
            k=5,
        ))
        assert isinstance(result, InnovationPath)
        assert result.source_methods == ["SAM", "PPO"]
        assert len(result.gaps) >= 1
        assert result.gaps[0]["method"] == "SAM"

    def test_finds_addressing_ideas(self):
        engine = _build_engine()
        result = asyncio.run(engine.find_innovation_paths(
            "GUI agent", ["SAM"], k=5,
        ))
        assert len(result.addressing_ideas) >= 1
        assert result.addressing_ideas[0]["mechanism"] == "Visual grounding for GUI"

    def test_finds_unaddressed_gaps(self):
        engine = _build_engine()
        result = asyncio.run(engine.find_innovation_paths(
            "GUI agent", ["SAM"], k=5,
        ))
        assert len(result.unaddressed_gaps) >= 1
        assert "RL" in result.unaddressed_gaps[0]["failure_mode"]

    def test_llm_suggestion_populated(self):
        engine = _build_engine()
        result = asyncio.run(engine.find_innovation_paths(
            "GUI agent", ["SAM"], k=5,
        ))
        assert "SAM" in result.suggested_combination


class TestFindCrossDomainBridges:
    """Tests for find_cross_domain_bridges."""

    def test_returns_bridge(self):
        engine = _build_engine()
        result = asyncio.run(engine.find_cross_domain_bridges(
            "SAM", "PPO",
        ))
        assert isinstance(result, CrossDomainBridge)
        assert result.method_a == "SAM"
        assert result.method_b == "PPO"

    def test_finds_shared_concepts(self):
        engine = _build_engine()
        result = asyncio.run(engine.find_cross_domain_bridges(
            "SAM", "PPO",
        ))
        assert len(result.shared_concepts) >= 1
        names = [sc["name"] for sc in result.shared_concepts]
        assert "UI Navigation" in names

    def test_finds_bridge_papers(self):
        engine = _build_engine()
        result = asyncio.run(engine.find_cross_domain_bridges(
            "SAM", "PPO",
        ))
        assert len(result.bridge_papers) >= 1
        assert result.bridge_papers[0]["venue"] == "NeurIPS"


class TestAnalyzeComponentEvidence:
    """Tests for analyze_component_evidence."""

    def test_returns_component_evidence(self):
        engine = _build_engine()
        results = asyncio.run(engine.analyze_component_evidence(["SAM"]))
        assert len(results) >= 1
        assert isinstance(results[0], ComponentEvidence)
        assert results[0].method_name == "SAM"

    def test_counts_correct(self):
        engine = _build_engine()
        results = asyncio.run(engine.analyze_component_evidence(["SAM"]))
        ce = results[0]
        assert ce.claim_count == 5
        assert ce.evidence_count == 3
        assert ce.paper_count == 1

    def test_strength_calculation(self):
        engine = _build_engine()
        results = asyncio.run(engine.analyze_component_evidence(["SAM"]))
        ce = results[0]
        # FULL=1.0, PARTIAL=0.5  avg = 0.75
        assert ce.avg_support_strength == 0.75

    def test_unsupported_claims_detected(self):
        engine = _build_engine()
        results = asyncio.run(engine.analyze_component_evidence(["SAM"]))
        ce = results[0]
        assert len(ce.unsupported_claims) >= 1
        assert ce.unsupported_claims[0]["severity"] == "P0"


class TestEmptyInputs:
    """Edge cases with empty inputs."""

    def test_empty_methods_returns_empty_innovation(self):
        engine = _build_engine()
        result = asyncio.run(engine.find_innovation_paths("idea", [], k=5))
        assert isinstance(result, InnovationPath)
        assert result.gaps == []

    def test_empty_component_evidence(self):
        engine = _build_engine()
        results = asyncio.run(engine.analyze_component_evidence([]))
        assert results == []
