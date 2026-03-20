"""KG ontology constants and semantic validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Shared vector collections in the current deployment profile.
ENTITY_COLLECTION = "entities"
CLAIM_COLLECTION = "claims"

# Payload typing contract for shared-vector buckets.
DOC_KIND_ENTITY = "entity"
DOC_KIND_SECTION = "section"
DOC_KIND_CLAIM = "claim"
DOC_KIND_EVIDENCE = "evidence"
DOC_KIND_CITATION = "citation"

KG_LAYER_L1 = "L1"
KG_LAYER_L2 = "L2"
KG_LAYER_L3 = "L3"

# Graph relation constraints.
ALLOWED_PAPER_ENTITY_RELATIONS = {"PROPOSES", "INTRODUCES", "USES"}

# Common relation metadata keys.
REL_META_SOURCE_RULE = "source_rule"
REL_META_CONFIDENCE_SOURCE = "confidence_source"
REL_META_EVIDENCE_SPAN = "evidence_span"

DEFAULT_SOURCE_RULE = "unknown.rule"
DEFAULT_CONFIDENCE_SOURCE = "unknown.source"


def _normalize_token(value: str | None, fallback: str) -> str:
    """Normalize metadata token fields to stable non-empty strings."""
    if value is None:
        return fallback
    cleaned = value.strip()
    return cleaned or fallback


def normalize_evidence_span(value: str | None) -> str:
    """Normalize evidence-span metadata to a stable non-empty string."""
    if not value:
        return "unknown"
    cleaned = value.strip()
    return cleaned or "unknown"


@dataclass(frozen=True)
class RelationMetadata:
    """Normalized metadata attached to semantically constrained relation edges."""

    source_rule: str
    confidence_source: str
    evidence_span: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_rule", _normalize_token(self.source_rule, DEFAULT_SOURCE_RULE))
        object.__setattr__(
            self,
            "confidence_source",
            _normalize_token(self.confidence_source, DEFAULT_CONFIDENCE_SOURCE),
        )
        object.__setattr__(self, "evidence_span", normalize_evidence_span(self.evidence_span))

    def as_neo4j_params(self) -> dict[str, str]:
        """Render metadata as Neo4j relation-property parameters."""
        return {
            REL_META_SOURCE_RULE: self.source_rule,
            REL_META_CONFIDENCE_SOURCE: self.confidence_source,
            REL_META_EVIDENCE_SPAN: self.evidence_span,
        }


def make_relation_metadata(
    *,
    source_rule: str,
    confidence_source: str,
    evidence_span: str | None = None,
) -> RelationMetadata:
    """Construct normalized relation metadata with ontology defaults."""
    return RelationMetadata(
        source_rule=source_rule,
        confidence_source=confidence_source,
        evidence_span=normalize_evidence_span(evidence_span),
    )


def make_vector_payload(*, doc_kind: str, kg_layer: str, base: dict[str, Any]) -> dict[str, Any]:
    """Attach mandatory ontology metadata to a vector payload."""
    payload = dict(base)
    payload["doc_kind"] = doc_kind
    payload["kg_layer"] = kg_layer
    return payload


def validate_paper_entity_relation(relation_type: str) -> bool:
    """Validate if a Paper->Entity relation type is allowed by ontology."""
    return relation_type in ALLOWED_PAPER_ENTITY_RELATIONS


def validate_confidence(confidence: float) -> bool:
    """Validate confidence range used by semantic relation edges."""
    return 0.0 <= confidence <= 1.0
