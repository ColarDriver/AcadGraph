"""Knowledge Graph sub-package."""

from acadgraph.kg.interfaces import KgRepository, VectorIndex

__all__ = [
    "KgRepository",
    "VectorIndex",
    "ENTITY_COLLECTION",
    "CLAIM_COLLECTION",
    "DOC_KIND_ENTITY",
    "DOC_KIND_SECTION",
    "DOC_KIND_CLAIM",
    "DOC_KIND_EVIDENCE",
    "DOC_KIND_CITATION",
    "KG_LAYER_L1",
    "KG_LAYER_L2",
    "KG_LAYER_L3",
]

from acadgraph.kg.ontology import (
    CLAIM_COLLECTION,
    ENTITY_COLLECTION,
    DOC_KIND_CLAIM,
    DOC_KIND_CITATION,
    DOC_KIND_ENTITY,
    DOC_KIND_EVIDENCE,
    DOC_KIND_SECTION,
    KG_LAYER_L1,
    KG_LAYER_L2,
    KG_LAYER_L3,
)
