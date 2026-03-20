"""
Semantic Entity Extraction.

Extracts 7 types of academic entities from paper sections:
METHOD, DATASET, METRIC, TASK, MODEL, FRAMEWORK, CONCEPT

Uses section-aware extraction: different entity types are expected
from different sections (e.g., DATASET from experiments, METHOD from method).
"""

from __future__ import annotations

import logging
from typing import Any

from acadgraph.embedding_client import EmbeddingClient
from acadgraph.kg.prompts.loader import load_prompt, render_prompt
from acadgraph.kg.schema import (
    Entity,
    EntityRelation,
    EntityType,
    EntityExtractionResult,
    ParsedPaper,
    RelationType,
    generate_id,
)
from acadgraph.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Mapping from string type names to EntityType enum
_TYPE_MAP: dict[str, EntityType] = {
    "METHOD": EntityType.METHOD,
    "DATASET": EntityType.DATASET,
    "METRIC": EntityType.METRIC,
    "TASK": EntityType.TASK,
    "MODEL": EntityType.MODEL,
    "FRAMEWORK": EntityType.FRAMEWORK,
    "CONCEPT": EntityType.CONCEPT,
}

# Mapping from string relation names to RelationType enum
_REL_MAP: dict[str, RelationType] = {
    "APPLIED_ON": RelationType.APPLIED_ON,
    "EVALUATED_ON": RelationType.EVALUATED_ON,
    "MEASURED_BY": RelationType.MEASURED_BY,
    "USES": RelationType.USES,
    "OUTPERFORMS": RelationType.OUTPERFORMS,
    "EXTENDS": RelationType.EXTENDS,
    "COMPONENT_OF": RelationType.COMPONENT_OF,
}

# Load prompts from Markdown files
SYSTEM_PROMPT = load_prompt("entity", "entity_extraction_system")
_EXTRACTION_TEMPLATE = load_prompt("entity", "entity_extraction")

# Section-specific entity type focus
SECTION_ENTITY_FOCUS: dict[str, list[str]] = {
    "abstract": ["TASK", "CONCEPT", "METHOD"],
    "introduction": ["TASK", "CONCEPT", "METHOD", "MODEL"],
    "related_work": ["METHOD", "CONCEPT", "MODEL", "FRAMEWORK"],
    "method": ["METHOD", "FRAMEWORK", "MODEL", "CONCEPT"],
    "experiments": ["DATASET", "METRIC", "MODEL", "METHOD"],
    "conclusion": ["TASK", "CONCEPT"],
    "full_text": ["METHOD", "DATASET", "METRIC", "TASK", "MODEL", "FRAMEWORK", "CONCEPT"],
}


def get_extraction_prompt(section_name: str, text: str, max_chars: int = 8000) -> str:
    """Build the extraction prompt for a given section."""
    entity_types = SECTION_ENTITY_FOCUS.get(section_name, SECTION_ENTITY_FOCUS["full_text"])
    types_str = ", ".join(entity_types)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"

    return render_prompt(
        _EXTRACTION_TEMPLATE,
        entity_types=types_str,
        section_name=section_name,
        text=text,
    )

class EntityExtractor:
    """Semantic entity extractor — section-aware, schema-constrained."""

    def __init__(self, llm: LLMClient, embedding: EmbeddingClient):
        self._llm = llm
        self._embedding = embedding
        # In-memory entity name → entity mapping for cross-paper dedup
        self._entity_cache: dict[str, Entity] = {}

    async def extract(self, parsed_paper: ParsedPaper) -> EntityExtractionResult:
        """
        Extract entities from all sections of a parsed paper.

        Per-section extraction strategy:
        - Abstract + Intro → TASK, CONCEPT, high-level METHOD
        - Method → METHOD (detailed), FRAMEWORK, MODEL
        - Experiments → DATASET, METRIC, MODEL (baselines)
        - Related Work → METHOD (prior), CONCEPT
        """
        all_entities: list[Entity] = []
        all_relations: list[EntityRelation] = []

        for section_name, section_text in parsed_paper.sections.items():
            if not section_text or len(section_text.strip()) < 50:
                continue

            try:
                entities, relations = await self._extract_from_section(
                    section_name=section_name,
                    section_text=section_text,
                    paper_id=parsed_paper.paper_id,
                )
                all_entities.extend(entities)
                all_relations.extend(relations)
            except Exception as e:
                logger.warning(
                    "Entity extraction failed for section '%s' of paper '%s': %s",
                    section_name, parsed_paper.paper_id, e,
                )

        # Deduplicate within this paper
        deduped = self._deduplicate_within_paper(all_entities)

        # Update relation IDs after dedup
        name_to_id = {e.name.lower(): e.entity_id for e in deduped}
        updated_relations = []
        for rel in all_relations:
            src = name_to_id.get(rel.source_id.lower())
            dst = name_to_id.get(rel.target_id.lower())
            if src and dst:
                rel.source_id = src
                rel.target_id = dst
                updated_relations.append(rel)

        return EntityExtractionResult(
            paper_id=parsed_paper.paper_id,
            entities=deduped,
            relations=updated_relations,
        )

    async def _extract_from_section(
        self, section_name: str, section_text: str, paper_id: str
    ) -> tuple[list[Entity], list[EntityRelation]]:
        """Extract entities and relations from a single section."""
        prompt = get_extraction_prompt(section_name, section_text)
        result = await self._llm.complete_json(prompt, system_prompt=SYSTEM_PROMPT)

        entities: list[Entity] = []
        relations: list[EntityRelation] = []

        # Parse entities
        for raw_entity in result.get("entities", []):
            entity_type_str = raw_entity.get("type", "CONCEPT").upper()
            entity_type = _TYPE_MAP.get(entity_type_str, EntityType.CONCEPT)

            entity = Entity(
                entity_id=generate_id(entity_type.value.lower()),
                entity_type=entity_type,
                name=raw_entity.get("name", "").strip(),
                description=raw_entity.get("description", "").strip(),
                attributes=raw_entity.get("attributes", {}),
                source_paper_id=paper_id,
                source_section=section_name,
            )
            if entity.name:
                entities.append(entity)

        # Parse relations (using entity names as temporary IDs)
        for raw_rel in result.get("relations", []):
            rel_type_str = raw_rel.get("relation", "USES").upper()
            rel_type = _REL_MAP.get(rel_type_str, RelationType.USES)

            relation = EntityRelation(
                source_id=raw_rel.get("source", ""),
                target_id=raw_rel.get("target", ""),
                relation_type=rel_type,
                properties={"description": raw_rel.get("description", "")},
                source_paper_id=paper_id,
            )
            if relation.source_id and relation.target_id:
                relations.append(relation)

        return entities, relations

    def _deduplicate_within_paper(self, entities: list[Entity]) -> list[Entity]:
        """Deduplicate entities within a single paper by name similarity."""
        seen: dict[str, Entity] = {}

        for entity in entities:
            normalized = entity.name.strip().lower()
            if normalized in seen:
                # Merge: keep the one with more description
                existing = seen[normalized]
                if len(entity.description) > len(existing.description):
                    entity.entity_id = existing.entity_id
                    seen[normalized] = entity
                # Merge attributes
                existing.attributes.update(entity.attributes)
            else:
                seen[normalized] = entity

        return list(seen.values())

    async def deduplicate_cross_paper(
        self, new_entities: list[Entity], threshold: float = 0.95
    ) -> list[Entity]:
        """
        Cross-paper entity deduplication via embedding similarity + name matching.

        If a new entity has embedding similarity > threshold with a cached entity,
        they are merged (the existing entity ID is reused).
        """
        if not self._entity_cache:
            # First paper — no dedup needed, just cache
            for entity in new_entities:
                self._entity_cache[entity.name.lower()] = entity
            return new_entities

        result: list[Entity] = []
        for entity in new_entities:
            normalized = entity.name.strip().lower()

            # Exact name match
            if normalized in self._entity_cache:
                cached = self._entity_cache[normalized]
                entity.entity_id = cached.entity_id
                # Merge descriptions if new one is longer
                if len(entity.description) > len(cached.description):
                    cached.description = entity.description
                result.append(entity)
                continue

            # Embedding-based similarity check
            try:
                entity_emb = await self._embedding.embed(
                    f"{entity.entity_type.value}: {entity.name}. {entity.description}"
                )
                best_sim = 0.0
                best_match: Entity | None = None

                for cached in self._entity_cache.values():
                    if cached.entity_type != entity.entity_type:
                        continue
                    cached_emb = await self._embedding.embed(
                        f"{cached.entity_type.value}: {cached.name}. {cached.description}"
                    )
                    sim = self._cosine_sim(entity_emb, cached_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = cached

                if best_sim >= threshold and best_match:
                    entity.entity_id = best_match.entity_id
                    logger.debug("Dedup merged '%s' → '%s' (sim=%.3f)", entity.name, best_match.name, best_sim)
                else:
                    self._entity_cache[normalized] = entity

            except Exception as e:
                logger.warning("Embedding dedup failed for '%s': %s", entity.name, e)
                self._entity_cache[normalized] = entity

            result.append(entity)

        return result

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        a_np = np.array(a)
        b_np = np.array(b)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / (norm_a * norm_b))
