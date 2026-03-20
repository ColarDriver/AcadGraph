"""Tests for incremental_builder.py — pipeline orchestration unit tests."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from acadgraph.kg.incremental_builder import (
    CROSS_LAYER_GAP_THRESHOLD,
    IncrementalKGBuilder,
    METHOD_CLAIM_KEYWORD_MIN_LENGTH,
)
from acadgraph.kg.schema import (
    ArgumentationGraph,
    BuildResult,
    Claim,
    ClaimSeverity,
    ClaimType,
    CoreIdea,
    Entity,
    EntityType,
    Evidence,
    EvidenceType,
    Gap,
    NoveltyType,
    PaperSource,
    Problem,
)


# ---- Stubs/Mocks ----

class _StubNeo4j:
    """Minimal stub satisfying KgRepository interface for unit testing."""

    def __init__(self):
        self.papers: dict[str, dict] = {}
        self.entities: dict[str, dict] = {}
        self.argumentations: list[tuple] = []
        self.method_claims: list[tuple] = []
        self.dataset_evidences: list[tuple] = []

    async def paper_exists(self, paper_id: str) -> bool:
        return paper_id in self.papers

    async def upsert_paper(self, paper_id, title, year=None, venue="", authors=None):
        self.papers[paper_id] = {"title": title, "year": year, "venue": venue}

    async def upsert_entity(self, entity) -> str:
        self.entities[entity.entity_id] = entity
        return entity.entity_id

    async def upsert_relation(self, relation):
        pass

    async def add_citation(self, edge) -> bool:
        return True

    async def store_argumentation(self, paper_id, arg):
        self.argumentations.append((paper_id, arg))

    async def link_paper_entity(self, paper_id, entity_id, relation_type, confidence=1.0, **kw):
        return True

    async def link_method_claim(self, method_id, claim_id, source_paper_id, confidence=0.7, **kw):
        self.method_claims.append((method_id, claim_id, confidence))
        return True

    async def link_dataset_evidence(self, dataset_id, evidence_id, source_paper_id, confidence=0.9, **kw):
        self.dataset_evidences.append((dataset_id, evidence_id))
        return True

    async def get_related_methods(self, paper_id, limit=40):
        return []

    async def add_method_evolution(self, *args, **kwargs):
        return True

    async def get_stats(self):
        return {}

    async def init_schema(self):
        pass

    async def connect(self):
        pass

    async def close(self):
        pass


class _StubQdrant:
    """Minimal stub satisfying VectorIndex interface."""

    def __init__(self):
        self.upserted: list[tuple] = []

    async def upsert_embedding(self, collection, point_id, vector, payload=None):
        self.upserted.append((collection, point_id))

    async def upsert_embeddings_batch(self, collection, point_ids, vectors, payloads=None):
        for pid in point_ids:
            self.upserted.append((collection, pid))

    async def search_similar(self, collection, query_vector, k=10, filters=None):
        return []

    async def get_collection_count(self, collection):
        return 0

    async def delete_by_paper_id(self, collection, paper_id):
        pass

    async def init_collections(self):
        pass

    async def connect(self):
        pass

    async def close(self):
        pass


class _StubLLM:
    async def complete(self, prompt, system_prompt="", **kwargs):
        return ""

    async def complete_json(self, prompt, system_prompt="", **kwargs):
        return {}

    async def close(self):
        pass


class _StubEmbedding:
    async def embed(self, text):
        return [0.1] * 32

    async def embed_batch(self, texts, batch_size=32):
        return [[0.1] * 32 for _ in texts]

    @property
    def dim(self):
        return 32

    async def close(self):
        pass


@pytest.fixture
def builder():
    return IncrementalKGBuilder(
        neo4j_store=_StubNeo4j(),
        qdrant_store=_StubQdrant(),
        llm=_StubLLM(),
        embedding=_StubEmbedding(),
    )


# ---- Tests ----

def test_keyword_extraction_filters_short_tokens(builder):
    """_keyword_tokens should filter tokens shorter than METHOD_CLAIM_KEYWORD_MIN_LENGTH."""
    keywords = builder._keyword_tokens("an the BERT model for NLP")
    # 'an' and 'the' are stopwords or too short; 'BERT', 'model', 'for' depends on threshold
    assert all(len(k) > METHOD_CLAIM_KEYWORD_MIN_LENGTH for k in keywords)


def test_method_claim_confidence_basic(builder):
    """Confidence should be > 0 when method name appears in claim text."""
    conf = builder._method_claim_confidence(
        "SparseFlow",
        "SparseFlow achieves state-of-the-art performance",
    )
    assert conf > 0.0


def test_method_claim_confidence_no_overlap(builder):
    """Confidence should be low/zero when there's no keyword overlap."""
    conf = builder._method_claim_confidence(
        "UnrelatedMethodName",
        "We improve efficiency via pruning",
    )
    assert conf < 0.5


def test_normalize_text(builder):
    """_normalize_text should lowercase and collapse whitespace."""
    result = builder._normalize_text("  Hello   World  \n FOO  ")
    assert result == "hello world foo"


def test_add_paper_skips_existing_paper(builder):
    """Pipeline should skip papers that already exist in the graph."""
    builder.neo4j.papers["existing_123"] = {"title": "Existing"}

    paper = PaperSource(paper_id="existing_123", title="Existing", text="some text")
    result = asyncio.run(builder.add_paper(paper))

    assert isinstance(result, BuildResult)
    assert result.entities_added == 0


def test_cross_layer_link_method_to_claim(builder):
    """_cross_layer_link should create METHOD→CLAIM edges when keywords overlap."""
    from types import SimpleNamespace

    entities = [
        Entity(name="SparseFlow", entity_type=EntityType.METHOD, description="sparse attention"),
    ]
    entity_result = SimpleNamespace(entities=entities)
    arg_graph = ArgumentationGraph(
        paper_id="test_001",
        claims=[
            Claim(
                text="SparseFlow achieves SOTA on LRA",
                claim_type=ClaimType.PERFORMANCE,
                severity=ClaimSeverity.P0,
            ),
        ],
    )
    asyncio.run(builder._cross_layer_link("test_001", entity_result, arg_graph))
    assert len(builder.neo4j.method_claims) > 0


def test_cross_layer_link_dataset_to_evidence(builder):
    """_cross_layer_link should create DATASET→EVIDENCE edges when dataset names match."""
    from types import SimpleNamespace

    entities = [
        Entity(name="ImageNet", entity_type=EntityType.DATASET, description="large-scale image dataset"),
    ]
    entity_result = SimpleNamespace(entities=entities)
    arg_graph = ArgumentationGraph(
        paper_id="test_001",
        evidences=[
            Evidence(
                evidence_type=EvidenceType.EXPERIMENT,
                result_summary="95% accuracy",
                datasets=["ImageNet"],
            ),
        ],
    )
    asyncio.run(builder._cross_layer_link("test_001", entity_result, arg_graph))
    assert len(builder.neo4j.dataset_evidences) > 0
