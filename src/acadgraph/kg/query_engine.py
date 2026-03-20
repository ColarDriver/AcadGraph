"""
KG Query Engine — Unified query interface for the Three-Layer KG.

Provides high-level query APIs for:
- Gap Mining: finding competitive space, structural alignment, gap generation
- Evidence Auditing: claim-evidence ledger, external verification
- Enhanced Recall: multi-path retrieval combining graph + vector
- Evolution Tracking: method evolution timelines and research trends
"""

from __future__ import annotations

import json
import logging
from typing import Any

from acadgraph.embedding_client import EmbeddingClient
from acadgraph.kg.interfaces import KgRepository, VectorIndex
from acadgraph.kg.ontology import (
    CLAIM_COLLECTION,
    ENTITY_COLLECTION,
    DOC_KIND_CLAIM,
    DOC_KIND_ENTITY,
    DOC_KIND_EVIDENCE,
    DOC_KIND_SECTION,
)
from acadgraph.kg.schema import (
    ClaimEvidenceLedger,
    CompetitionSpace,
    EvolutionTimeline,
    GapStatement,
    NoveltyMap,
)
from acadgraph.llm_client import LLMClient

logger = logging.getLogger(__name__)


class KGQueryEngine:
    """Knowledge graph reasoning and query engine."""

    def __init__(
        self,
        neo4j: KgRepository,
        qdrant: VectorIndex,
        llm: LLMClient,
        embedding: EmbeddingClient,
    ):
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.llm = llm
        self.embedding = embedding

    # ========================================================================
    # Gap Mining Queries
    # ========================================================================

    async def find_competition_space(
        self, idea: str, k: int = 20
    ) -> CompetitionSpace:
        """
        Build the competitive literature space for an idea.

        Three-path retrieval:
        1. Qdrant: Semantic search for similar entities/claims
        2. Neo4j: Citation network expansion (co-citation, bibliographic coupling)
        3. Merge and rank by relevance
        """
        # Path 1: Semantic retrieval via Qdrant
        idea_embedding = await self.embedding.embed(idea)

        # Search similar entities
        entity_hits = await self.qdrant.search_similar(
            ENTITY_COLLECTION, idea_embedding, k=k * 2,
            filters={"doc_kind": DOC_KIND_ENTITY},
        )

        # Search similar claims
        claim_hits = await self.qdrant.search_similar(
            CLAIM_COLLECTION, idea_embedding, k=k,
            filters={"doc_kind": DOC_KIND_CLAIM},
        )

        # Search similar sections
        section_hits = await self.qdrant.search_similar(
            ENTITY_COLLECTION, idea_embedding, k=k,
            filters={"doc_kind": DOC_KIND_SECTION},
        )

        # Collect unique paper IDs with scores
        paper_scores: dict[str, float] = {}
        for hit in entity_hits + claim_hits + section_hits:
            pid = hit["payload"].get("paper_id", "")
            if pid:
                score = hit["score"]
                paper_scores[pid] = max(paper_scores.get(pid, 0), score)

        # Path 2: Citation network expansion (via Neo4j)
        top_paper_ids = sorted(paper_scores, key=paper_scores.get, reverse=True)[:10]
        for pid in top_paper_ids:
            neighbors = await self.neo4j.find_nearest_competitors(pid, k=5)
            for nb in neighbors:
                nb_id = nb.get("paper_id", "")
                if nb_id and nb_id not in paper_scores:
                    paper_scores[nb_id] = 0.3  # Lower score for citation-based

        # Path 3: Sort and return top-k
        sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        nearest = []
        for pid, score in sorted_papers:
            nearest.append({
                "paper_id": pid,
                "score": score,
            })

        return CompetitionSpace(
            query_idea=idea,
            nearest_papers=nearest,
        )

    async def structural_alignment(
        self, idea: str, competitor_paper_ids: list[str]
    ) -> NoveltyMap:
        """
        Project idea and competitors onto a unified coordinate system.

        Dimensions: Problem, Setting, Assumption, Mechanism,
                    Supervision, Metric, FailureMode
        """
        dimensions = [
            "Problem", "Setting", "Assumption", "Mechanism",
            "Supervision", "Metric", "FailureMode",
        ]

        # Get competitor data from Neo4j
        competitor_data = []
        for pid in competitor_paper_ids[:10]:
            ledger = await self.neo4j.get_claim_evidence_ledger(pid)
            competitor_data.append({
                "paper_id": pid,
                "claims": [e.claim_text for e in ledger.entries[:5]],
            })

        # Use LLM to project onto dimensions
        prompt = f"""Given this research idea and its competitors, project each onto these dimensions:
{json.dumps(dimensions)}

## Research Idea:
{idea}

## Competitors:
{json.dumps(competitor_data, indent=2, ensure_ascii=False)}

Return JSON:
```json
{{
  "idea_projection": {{"Problem": "...", "Setting": "...", ...}},
  "competitor_projections": [
    {{"paper_id": "...", "Problem": "...", "Setting": "...", ...}}
  ],
  "unique_dimensions": ["dimensions where ideas is unique"]
}}
```
"""

        try:
            result = await self.llm.complete_json(prompt)
            return NoveltyMap(
                dimensions=dimensions,
                idea_projection=result.get("idea_projection", {}),
                competitor_projections=result.get("competitor_projections", []),
                unique_dimensions=result.get("unique_dimensions", []),
            )
        except Exception as e:
            logger.error("Structural alignment failed: %s", e)
            return NoveltyMap(dimensions=dimensions)

    async def generate_gap_statement(
        self, idea: str, novelty_map: NoveltyMap | None = None
    ) -> GapStatement:
        """
        Generate a falsifiable gap statement.

        Template:
        "现有方法在 [Setting S] 下可以解决 [Problem P],
         但在 [Constraint F] 下仍然失败,
         因为它们缺少 [Mechanism M];
         据检索，尚无方法同时满足 [A, B, C]。"
        """
        context = ""
        if novelty_map and novelty_map.idea_projection:
            context = f"\nNovelty analysis:\n{json.dumps(novelty_map.idea_projection, indent=2)}"
            context += f"\nUnique dimensions: {novelty_map.unique_dimensions}"

        prompt = f"""Generate a rigorous, falsifiable gap statement for this research idea.

## Research Idea:
{idea}
{context}

## Gap Statement Template:
"Existing methods can solve [Problem P] under [Setting S],
 but still fail under [Constraint F],
 because they lack [Mechanism M];
 according to our retrieval, no existing method simultaneously satisfies [A, B, C]."

Return JSON:
```json
{{
  "statement": "full gap statement",
  "problem": "the problem P",
  "setting": "the setting S",
  "failure_constraint": "the constraint/failure mode F",
  "missing_mechanism": "the mechanism M they lack",
  "novelty_checklist": ["A", "B", "C"],
  "supporting_evidence": ["evidence supporting this gap"]
}}
```
"""

        try:
            result = await self.llm.complete_json(prompt)
            return GapStatement(
                statement=result.get("statement", ""),
                problem=result.get("problem", ""),
                setting=result.get("setting", ""),
                failure_constraint=result.get("failure_constraint", ""),
                missing_mechanism=result.get("missing_mechanism", ""),
                novelty_checklist=result.get("novelty_checklist", []),
                supporting_evidence=result.get("supporting_evidence", []),
            )
        except Exception as e:
            logger.error("Gap statement generation failed: %s", e)
            return GapStatement()

    # ========================================================================
    # Evidence Auditing Queries
    # ========================================================================

    async def get_claim_evidence_ledger(
        self, paper_id: str
    ) -> ClaimEvidenceLedger:
        """Get the Claim-Evidence ledger for a paper from Neo4j."""
        return await self.neo4j.get_claim_evidence_ledger(paper_id)

    async def verify_claims_against_literature(
        self, claims: list[dict[str, str]], k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Verify claims against external literature.

        For each claim, search for similar claims and evidence in
        the KG to find support/contradiction from other papers.
        """
        results = []
        for claim_data in claims:
            claim_text = claim_data.get("text", "")
            if not claim_text:
                continue

            claim_embedding = await self.embedding.embed(claim_text)

            # Search for similar claims in other papers
            similar_claims = await self.qdrant.search_similar(
                CLAIM_COLLECTION, claim_embedding, k=k, filters={"doc_kind": DOC_KIND_CLAIM}
            )

            # Search for related evidence
            similar_evidence = await self.qdrant.search_similar(
                CLAIM_COLLECTION, claim_embedding, k=k, filters={"doc_kind": DOC_KIND_EVIDENCE}
            )

            results.append({
                "claim": claim_text,
                "similar_claims": [
                    {
                        "paper_id": h["payload"].get("paper_id"),
                        "score": h["score"],
                        "claim_type": h["payload"].get("claim_type"),
                    }
                    for h in similar_claims
                ],
                "related_evidence": [
                    {
                        "paper_id": h["payload"].get("paper_id"),
                        "score": h["score"],
                        "evidence_type": h["payload"].get("evidence_type"),
                    }
                    for h in similar_evidence
                ],
            })

        return results

    # ========================================================================
    # Enhanced Recall Queries
    # ========================================================================

    async def enhanced_recall(
        self, idea: str, k: int = 20
    ) -> list[dict[str, Any]]:
        """
        Multi-path recall combining graph and vector retrieval.

        Path 1: Idea → similar entities → entity→paper (Qdrant + Neo4j)
        Path 2: Idea → similar claims → claim→paper (Qdrant + Neo4j)
        Path 3: Idea → evolution chains → method-evolution related papers (Neo4j)
        Path 4: Idea → section similarity (Qdrant)
        """
        idea_embedding = await self.embedding.embed(idea)
        all_results: dict[str, dict[str, Any]] = {}

        def _upsert_result(
            pid: str,
            score: float,
            recall_path: str,
            path_key: str,
            path_payload: str | int | float | None,
        ) -> None:
            if not pid:
                return

            normalized_score = max(float(score), 0.0)
            if pid not in all_results:
                all_results[pid] = {
                    "paper_id": pid,
                    "score": normalized_score,
                    "recall_path": recall_path,
                    "path_scores": {recall_path: normalized_score},
                    "path_matches": {},
                }
            else:
                all_results[pid]["score"] = max(all_results[pid]["score"], normalized_score)
                all_results[pid].setdefault("path_scores", {})
                all_results[pid]["path_scores"][recall_path] = max(
                    all_results[pid]["path_scores"].get(recall_path, 0.0),
                    normalized_score,
                )

            if path_payload is not None and path_payload != "":
                all_results[pid].setdefault("path_matches", {})
                all_results[pid]["path_matches"].setdefault(path_key, [])
                if path_payload not in all_results[pid]["path_matches"][path_key]:
                    all_results[pid]["path_matches"][path_key].append(path_payload)

        # Path 1: Entity-based recall
        entity_hits = await self.qdrant.search_similar(ENTITY_COLLECTION, idea_embedding, k=k)
        for hit in entity_hits:
            payload = hit.get("payload", {})
            _upsert_result(
                pid=payload.get("paper_id", ""),
                score=hit.get("score", 0.0),
                recall_path="entity",
                path_key="matched_entities",
                path_payload=payload.get("name", ""),
            )

        # Path 2: Claim-based recall (strict type filter in shared collection)
        claim_hits = await self.qdrant.search_similar(CLAIM_COLLECTION, idea_embedding, k=k, filters={"doc_kind": DOC_KIND_CLAIM})
        for hit in claim_hits:
            payload = hit.get("payload", {})
            _upsert_result(
                pid=payload.get("paper_id", ""),
                score=hit.get("score", 0.0),
                recall_path="claim",
                path_key="matched_claim_types",
                path_payload=payload.get("claim_type", ""),
            )

        # Path 3: Evolution-chain-based expansion from top semantic seeds.
        seed_ids = sorted(
            ((pid, data.get("score", 0.0)) for pid, data in all_results.items()),
            key=lambda x: x[1],
            reverse=True,
        )[: max(5, k // 2)]
        evolution_candidates = await self.neo4j.find_evolution_related_papers(
            [pid for pid, _ in seed_ids],
            k=max(k, 10),
        )
        for evo in evolution_candidates:
            pid = evo.get("paper_id", "")
            relation_conf = float(evo.get("relation_confidence", 0.5) or 0.5)
            link_count = int(evo.get("link_count", 1) or 1)
            score = min(0.95, 0.35 + 0.12 * relation_conf + 0.08 * link_count)
            _upsert_result(
                pid=pid,
                score=score,
                recall_path="evolution",
                path_key="evolution_link_count",
                path_payload=link_count,
            )

        # Path 4: Section-based recall (strict type filter in shared collection)
        section_hits = await self.qdrant.search_similar(ENTITY_COLLECTION, idea_embedding, k=k, filters={"doc_kind": DOC_KIND_SECTION})
        for hit in section_hits:
            payload = hit.get("payload", {})
            _upsert_result(
                pid=payload.get("paper_id", ""),
                score=hit.get("score", 0.0),
                recall_path="section",
                path_key="matched_sections",
                path_payload=payload.get("section", ""),
            )

        # Merge scores with light path-diversity bonus.
        for data in all_results.values():
            path_scores = data.get("path_scores", {})
            path_bonus = 0.05 * max(0, len(path_scores) - 1)
            merged_score = max(path_scores.values()) if path_scores else data.get("score", 0.0)
            data["score"] = min(1.0, merged_score + path_bonus)

            # Backward-compatible shortcuts for existing CLI fields.
            matches = data.get("path_matches", {})
            if matches.get("matched_entities"):
                data["matched_entity"] = matches["matched_entities"][0]
            if matches.get("matched_sections"):
                data["matched_section"] = matches["matched_sections"][0]

        # Sort by score and return top-k
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:k]

    # ========================================================================
    # Evolution Queries
    # ========================================================================

    async def get_method_evolution(self, method_name: str) -> EvolutionTimeline:
        """Get the evolution timeline for a method."""
        # Find the method entity by searching Qdrant
        method_embedding = await self.embedding.embed(f"METHOD: {method_name}")
        hits = await self.qdrant.search_similar(
            ENTITY_COLLECTION, method_embedding, k=5,
            filters={"entity_type": "METHOD"},
        )

        if not hits:
            return EvolutionTimeline(method_name=method_name)

        method_id = hits[0]["id"]
        steps = await self.neo4j.get_evolution_chain(method_id)

        return EvolutionTimeline(
            method_name=method_name,
            steps=steps,
        )

    async def get_research_trends(
        self, domain: str, year_range: tuple[int, int] | None = None
    ) -> dict[str, Any]:
        """
        Get research trends in a domain.

        Uses entity frequency analysis + claim type distribution.
        """
        # Search for entities in this domain
        domain_embedding = await self.embedding.embed(f"research domain: {domain}")
        entity_hits = await self.qdrant.search_similar(
            ENTITY_COLLECTION, domain_embedding, k=50,
            filters={"doc_kind": DOC_KIND_ENTITY},
        )

        # Optional year-range filtering on related paper IDs.
        filtered_paper_ids: set[str] | None = None
        normalized_year_range: tuple[int, int] | None = None
        if year_range:
            start_year, end_year = year_range
            if start_year > end_year:
                start_year, end_year = end_year, start_year
            normalized_year_range = (start_year, end_year)

            candidate_paper_ids = {
                hit.get("payload", {}).get("paper_id", "")
                for hit in entity_hits
                if hit.get("payload", {}).get("paper_id", "")
            }
            filtered_paper_ids = await self.neo4j.get_papers_by_year_range(
                list(candidate_paper_ids),
                start_year,
                end_year,
            )

        # Aggregate by entity type after optional year filtering.
        entity_type_counts: dict[str, int] = {}
        paper_ids: set[str] = set()
        filtered_hits: list[dict[str, Any]] = []
        for hit in entity_hits:
            payload = hit.get("payload", {})
            pid = payload.get("paper_id", "")
            if filtered_paper_ids is not None and pid not in filtered_paper_ids:
                continue

            filtered_hits.append(hit)
            et = payload.get("entity_type", "UNKNOWN")
            entity_type_counts[et] = entity_type_counts.get(et, 0) + 1
            if pid:
                paper_ids.add(pid)

        return {
            "domain": domain,
            "year_range": normalized_year_range,
            "total_related_entities": len(filtered_hits),
            "entity_type_distribution": entity_type_counts,
            "related_paper_count": len(paper_ids),
            "top_entities": [
                {
                    "name": h.get("payload", {}).get("name", ""),
                    "type": h.get("payload", {}).get("entity_type", ""),
                    "paper_id": h.get("payload", {}).get("paper_id", ""),
                    "score": h.get("score", 0.0),
                }
                for h in filtered_hits[:10]
            ],
        }

    # ========================================================================
    # KG Statistics
    # ========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get overall KG statistics."""
        neo4j_stats = await self.neo4j.get_stats()
        qdrant_stats = {}
        for collection in [ENTITY_COLLECTION, CLAIM_COLLECTION]:
            try:
                count = await self.qdrant.get_collection_count(collection)
                qdrant_stats[collection] = count
            except Exception:
                qdrant_stats[collection] = 0

        return {
            "neo4j": neo4j_stats,
            "qdrant": qdrant_stats,
        }
