"""
Neo4j graph storage for the academic KG.

Manages CRUD operations for all three layers of the knowledge graph:
- Semantic entities and their relations
- Citation edges with intent
- Argumentation chains (Problem → Gap → Claim → Evidence)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from acadgraph.config import Neo4jConfig
from acadgraph.kg.interfaces import KgRepository
from acadgraph.kg.ontology import (
    RelationMetadata,
    make_relation_metadata,
    REL_META_CONFIDENCE_SOURCE,
    REL_META_EVIDENCE_SPAN,
    REL_META_SOURCE_RULE,
    validate_confidence,
    validate_paper_entity_relation,
)
from acadgraph.kg.schema import (
    ArgumentationGraph,
    Baseline,
    CitationEdge,
    ClaimEvidenceLedger,
    ClaimEvidenceLedgerEntry,
    ClaimEvidenceLink,
    CoreIdea,
    Entity,
    EntityRelation,
    Evidence,
    EvolutionStep,
    Gap,
    Limitation,
    Problem,
    SupportStrength,
)

logger = logging.getLogger(__name__)

# Regex for validating safe Neo4j property keys (prevents Cypher injection)
_SAFE_PROPERTY_KEY = re.compile(r"^[a-z_][a-z0-9_]*$", re.IGNORECASE)


class Neo4jKGStore(KgRepository):
    """Neo4j graph storage — manages the three-layer knowledge graph."""

    def __init__(self, config: Neo4jConfig | None = None):
        self.config = config or Neo4jConfig()
        self._driver: AsyncDriver | None = None

    async def __aenter__(self) -> "Neo4jKGStore":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Connect to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
        )
        logger.info("Connected to Neo4j at %s", self.config.uri)

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j connection closed")

    @property
    def driver(self) -> AsyncDriver:
        assert self._driver is not None, "Not connected. Call connect() first."
        return self._driver

    # ========================================================================
    # Schema Initialization
    # ========================================================================

    async def init_schema(self) -> None:
        """Create constraints, indexes, and fulltext indexes."""
        queries = [
            # Entity constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE d.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (me:Metric) REQUIRE me.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (mo:Model) REQUIRE mo.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Framework) REQUIRE f.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.entity_id IS UNIQUE",
            # Paper constraint
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            # Argumentation constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Problem) REQUIRE pr.problem_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gap) REQUIRE g.gap_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ci:CoreIdea) REQUIRE ci.idea_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cl:Claim) REQUIRE cl.claim_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidence_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Baseline) REQUIRE b.baseline_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Limitation) REQUIRE l.limitation_id IS UNIQUE",
            # Indexes
            "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.year)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Claim) ON (c.severity, c.claim_type)",
            # Fulltext indexes for semantic search
            (
                "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
                "FOR (n:Method|Dataset|Task|Metric|Model|Framework|Concept) "
                "ON EACH [n.name, n.description]"
            ),
            (
                "CREATE FULLTEXT INDEX claim_text IF NOT EXISTS "
                "FOR (n:Claim) ON EACH [n.text]"
            ),
        ]

        async with self.driver.session() as session:
            for q in queries:
                try:
                    await session.run(q)
                except Exception as e:
                    logger.warning("Schema init query failed (may already exist): %s — %s", q[:60], e)

        logger.info("Neo4j schema initialized")

    # ========================================================================
    # Entities
    # ========================================================================

    async def upsert_entity(self, entity: Entity) -> str:
        """Insert or update a semantic entity. Returns entity_id."""
        label = entity.entity_type.value.capitalize()

        params: dict[str, Any] = {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "description": entity.description,
            "source_paper_id": entity.source_paper_id,
            "source_section": entity.source_section,
            "confidence": entity.confidence,
        }

        # Build extra SET clauses from custom attributes (with safe key validation)
        extra_sets: list[str] = []
        for k, v in entity.attributes.items():
            if not isinstance(v, (str, int, float, bool)):
                continue
            if not _SAFE_PROPERTY_KEY.match(k):
                logger.warning("Skipping unsafe attribute key: %r", k)
                continue
            params[f"attr_{k}"] = v
            extra_sets.append(f"e.{k} = $attr_{k}")

        extra_clause = ""
        if extra_sets:
            extra_clause = "SET " + ", ".join(extra_sets)

        query = f"""
        MERGE (e:{label} {{entity_id: $entity_id}})
        SET e.name = $name,
            e.description = $description,
            e.source_paper_id = $source_paper_id,
            e.source_section = $source_section,
            e.confidence = $confidence
        {extra_clause}
        RETURN e.entity_id AS id
        """

        async with self.driver.session() as session:
            result = await session.run(query, params)
            record = await result.single()
            return record["id"] if record else entity.entity_id

    async def upsert_relation(self, relation: EntityRelation) -> None:
        """Insert or update a relation between entities."""
        rel_type = relation.relation_type.value
        query = f"""
        MATCH (src {{entity_id: $src_id}})
        MATCH (dst {{entity_id: $dst_id}})
        MERGE (src)-[r:{rel_type}]->(dst)
        SET r.source_paper_id = $source_paper_id,
            r.confidence = $confidence
        """
        params = {
            "src_id": relation.source_id,
            "dst_id": relation.target_id,
            "source_paper_id": relation.source_paper_id,
            "confidence": relation.confidence,
        }
        for k, v in relation.properties.items():
            if not isinstance(v, (str, int, float, bool)):
                continue
            if not _SAFE_PROPERTY_KEY.match(k):
                logger.warning("Skipping unsafe relation property key: %r", k)
                continue
            param_key = f"prop_{k}"
            params[param_key] = v
            query += f"\nSET r.{k} = ${param_key}"

        async with self.driver.session() as session:
            await session.run(query, params)

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
        """Create/merge an edge between Paper and entity."""
        if not validate_paper_entity_relation(relation_type):
            logger.warning("Rejected invalid Paper->Entity relation_type: %s", relation_type)
            return False

        query = f"""
        MATCH (p:Paper {{paper_id: $paper_id}})
        MATCH (e {{entity_id: $entity_id}})
        MERGE (p)-[r:{relation_type}]->(e)
        SET r.source_paper_id = $paper_id,
            r.confidence = $confidence,
            r.source_rule = $source_rule,
            r.confidence_source = $confidence_source
        RETURN count(r) AS linked
        """
        resolved = metadata or make_relation_metadata(
            source_rule=source_rule or "ontology.paper_entity_relation",
            confidence_source=confidence_source or "builder.default",
        )
        params: dict[str, Any] = {
            "paper_id": paper_id,
            "entity_id": entity_id,
            "confidence": confidence,
            **resolved.as_neo4j_params(),
        }

        if properties:
            for key, value in properties.items():
                if not isinstance(value, (str, int, float, bool)):
                    continue
                if not _SAFE_PROPERTY_KEY.match(key):
                    logger.warning("Skipping unsafe link_paper_entity property key: %r", key)
                    continue
                param_key = f"prop_{key}"
                params[param_key] = value
                query = query.replace(
                    "RETURN count(r) AS linked",
                    f"SET r.{key} = ${param_key}\n        RETURN count(r) AS linked",
                )

        async with self.driver.session() as session:
            result = await session.run(query, params)
            record = await result.single()
            return bool(record and record.get("linked", 0) > 0)

    async def upsert_paper(self, paper_id: str, title: str, year: int | None = None,
                           venue: str = "", authors: list[str] | None = None) -> None:
        """Insert or update a Paper node."""
        query = """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.title = $title,
            p.year = $year,
            p.venue = $venue,
            p.authors = $authors
        """
        async with self.driver.session() as session:
            await session.run(query, {
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "venue": venue,
                "authors": authors or [],
            })

    # ========================================================================
    # Citations
    # ========================================================================

    async def add_citation(self, edge: CitationEdge) -> bool:
        """Add a citation edge with intent."""
        query = """
        MERGE (citing:Paper {paper_id: $citing_id})
        ON CREATE SET citing.title = coalesce($citing_title, $citing_id)
        MERGE (cited:Paper {paper_id: $cited_id})
        ON CREATE SET cited.title = coalesce($cited_title, $cited_id)
        MERGE (citing)-[r:CITES {intent: $intent}]->(cited)
        SET r.context = $context,
            r.section = $section,
            r.confidence = $confidence
        RETURN count(r) AS linked
        """
        async with self.driver.session() as session:
            result = await session.run(query, {
                "citing_id": edge.citing_paper_id,
                "cited_id": edge.cited_paper_id,
                "intent": edge.intent.value,
                "context": edge.context,
                "section": edge.section,
                "confidence": edge.confidence,
                "citing_title": edge.citing_paper_id,
                "cited_title": edge.cited_paper_id,
            })
            record = await result.single()
            return bool(record and record.get("linked", 0) > 0)

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
        query = """
        MATCH (prev:Method {entity_id: $from_method_id})
        MATCH (cur:Method {entity_id: $to_method_id})
        MERGE (cur)-[r:EVOLVES_FROM]->(prev)
        SET r.source_paper_id = $source_paper_id,
            r.confidence = $confidence
        FOREACH (_ IN CASE WHEN $delta_description <> '' THEN [1] ELSE [] END |
            SET r.delta_description = $delta_description
        )
        FOREACH (_ IN CASE WHEN $year IS NOT NULL THEN [1] ELSE [] END |
            SET r.year = $year
        )
        RETURN count(r) AS linked
        """
        async with self.driver.session() as session:
            result = await session.run(query, {
                "from_method_id": from_method_id,
                "to_method_id": to_method_id,
                "source_paper_id": source_paper_id,
                "delta_description": delta_description,
                "year": year,
                "confidence": confidence,
            })
            record = await result.single()
            return bool(record and record.get("linked", 0) > 0)

    async def get_related_methods(
        self,
        paper_id: str,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Return this paper's methods and earlier comparable methods."""
        query = """
        MATCH (paper:Paper {paper_id: $paper_id})-[:PROPOSES]->(m:Method)
        WITH collect(DISTINCT m) AS current_methods
        OPTIONAL MATCH (other:Paper)-[:PROPOSES]->(candidate:Method)
        WHERE other.paper_id <> $paper_id
          AND (other.year IS NULL OR other.year <= coalesce($paper_year, other.year))
        WITH current_methods, collect(DISTINCT candidate)[0..$limit] AS historical_methods
        WITH [m IN (current_methods + historical_methods) WHERE m IS NOT NULL] AS methods
        UNWIND methods AS method
        WITH DISTINCT method
        OPTIONAL MATCH (p:Paper)-[:PROPOSES]->(method)
        RETURN method.entity_id AS method_id,
               method.name AS method_name,
               method.description AS description,
               p.paper_id AS paper_id,
               p.year AS year
        """
        async with self.driver.session() as session:
            year_result = await session.run(
                "MATCH (p:Paper {paper_id: $paper_id}) RETURN p.year AS year",
                {"paper_id": paper_id},
            )
            year_record = await year_result.single()
            paper_year = year_record.get("year") if year_record else None

            result = await session.run(
                query,
                {
                    "paper_id": paper_id,
                    "paper_year": paper_year,
                    "limit": limit,
                },
            )
            return await result.data()

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
        """Create cross-layer METHOD -> CLAIM link."""
        if not validate_confidence(confidence):
            logger.warning("Rejected METHOD->CLAIM with out-of-range confidence: %s", confidence)
            return False

        query = """
        MATCH (m:Method {entity_id: $method_id})
        MATCH (c:Claim {claim_id: $claim_id})
        MERGE (m)-[r:SUPPORTS_CLAIM]->(c)
        SET r.source_paper_id = $source_paper_id,
            r.confidence = $confidence,
            r.source_rule = $source_rule,
            r.confidence_source = $confidence_source
        RETURN count(r) AS linked
        """
        resolved = metadata or make_relation_metadata(
            source_rule=source_rule or "heuristic.method_claim_overlap",
            confidence_source=confidence_source or "builder.heuristic",
        )
        async with self.driver.session() as session:
            result = await session.run(query, {
                "method_id": method_id,
                "claim_id": claim_id,
                "source_paper_id": source_paper_id,
                "confidence": confidence,
                REL_META_SOURCE_RULE: resolved.source_rule,
                REL_META_CONFIDENCE_SOURCE: resolved.confidence_source,
            })
            record = await result.single()
            return bool(record and record.get("linked", 0) > 0)

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
        """Create cross-layer DATASET -> EVIDENCE link."""
        if not validate_confidence(confidence):
            logger.warning("Rejected DATASET->EVIDENCE with out-of-range confidence: %s", confidence)
            return False

        query = """
        MATCH (d:Dataset {entity_id: $dataset_id})
        MATCH (e:Evidence {evidence_id: $evidence_id})
        MERGE (d)-[r:USED_IN_EVIDENCE]->(e)
        SET r.source_paper_id = $source_paper_id,
            r.confidence = $confidence,
            r.source_rule = $source_rule,
            r.confidence_source = $confidence_source,
            r.evidence_span = $evidence_span
        RETURN count(r) AS linked
        """
        resolved = metadata or make_relation_metadata(
            source_rule=source_rule or "heuristic.dataset_evidence_match",
            confidence_source=confidence_source or "builder.heuristic",
            evidence_span=evidence_span or "datasets_field",
        )
        async with self.driver.session() as session:
            result = await session.run(query, {
                "dataset_id": dataset_id,
                "evidence_id": evidence_id,
                "source_paper_id": source_paper_id,
                "confidence": confidence,
                REL_META_SOURCE_RULE: resolved.source_rule,
                REL_META_CONFIDENCE_SOURCE: resolved.confidence_source,
                REL_META_EVIDENCE_SPAN: resolved.evidence_span,
            })
            record = await result.single()
            return bool(record and record.get("linked", 0) > 0)

    async def get_evolution_chain(self, method_id: str) -> list[EvolutionStep]:
        """Get the evolution chain for a method."""
        query = """
        MATCH path = (m:Method {entity_id: $method_id})-[:EVOLVES_FROM*0..]->(ancestor:Method)
        WITH nodes(path) AS methods
        UNWIND methods AS method
        OPTIONAL MATCH (p:Paper)-[:PROPOSES]->(method)
        OPTIONAL MATCH (method)-[r:EVOLVES_FROM]->(:Method)
        RETURN method.entity_id AS method_id,
               method.name AS method_name,
               p.paper_id AS paper_id,
               p.year AS year,
               coalesce(r.delta_description, "") AS delta_description
        ORDER BY p.year
        """
        async with self.driver.session() as session:
            result = await session.run(query, {"method_id": method_id})
            records = await result.data()
            return [
                EvolutionStep(
                    method_id=r["method_id"],
                    method_name=r["method_name"],
                    paper_id=r.get("paper_id", ""),
                    year=r.get("year", 0),
                    delta_description=r.get("delta_description", ""),
                )
                for r in records
            ]

    # ========================================================================
    # Argumentation
    # ========================================================================

    async def store_argumentation(self, paper_id: str, arg: ArgumentationGraph) -> None:
        """Store the full argumentation graph for a paper.

        Uses Cypher UNWIND for batch writes to minimize round-trips.
        """
        import json as _json

        async with self.driver.session() as session:
            # ---- Problems (batch) ----
            if arg.problems:
                await session.run("""
                    UNWIND $items AS item
                    MERGE (pr:Problem {problem_id: item.problem_id})
                    SET pr.description = item.description,
                        pr.scope = item.scope,
                        pr.importance_signal = item.importance_signal,
                        pr.source_paper_id = $paper_id
                    WITH pr
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (p)-[:HAS_PROBLEM]->(pr)
                """, {
                    "paper_id": paper_id,
                    "items": [
                        {
                            "problem_id": prob.problem_id,
                            "description": prob.description,
                            "scope": prob.scope,
                            "importance_signal": prob.importance_signal,
                        }
                        for prob in arg.problems
                    ],
                })

            # ---- Gaps (batch) ----
            if arg.gaps:
                await session.run("""
                    UNWIND $items AS item
                    MERGE (g:Gap {gap_id: item.gap_id})
                    SET g.failure_mode = item.failure_mode,
                        g.constraint = item.constraint,
                        g.prior_methods_failing = item.prior_methods,
                        g.source_paper_id = $paper_id
                """, {
                    "paper_id": paper_id,
                    "items": [
                        {
                            "gap_id": gap.gap_id,
                            "failure_mode": gap.failure_mode,
                            "constraint": gap.constraint,
                            "prior_methods": gap.prior_methods_failing,
                        }
                        for gap in arg.gaps
                    ],
                })

            # ---- Problem → Gap links (precise mapping or fallback cartesian) ----
            if arg.problems and arg.gaps:
                problem_gap_map = getattr(arg, "_problem_gap_map", {})
                if problem_gap_map:
                    pairs = [
                        {"prob_id": arg.problems[pidx].problem_id, "gap_id": arg.gaps[gidx].gap_id}
                        for pidx, gap_indices in problem_gap_map.items()
                        for gidx in gap_indices
                        if pidx < len(arg.problems) and gidx < len(arg.gaps)
                    ]
                else:
                    # Fallback: cartesian (legacy behavior)
                    pairs = [
                        {"prob_id": p.problem_id, "gap_id": g.gap_id}
                        for p in arg.problems
                        for g in arg.gaps
                    ]
                if pairs:
                    await session.run("""
                        UNWIND $pairs AS pair
                        MATCH (pr:Problem {problem_id: pair.prob_id})
                        MATCH (g:Gap {gap_id: pair.gap_id})
                        MERGE (pr)-[:HAS_GAP]->(g)
                    """, {"pairs": pairs})

            # ---- Core Ideas (batch) ----
            if arg.core_ideas:
                await session.run("""
                    UNWIND $items AS item
                    MERGE (ci:CoreIdea {idea_id: item.idea_id})
                    SET ci.mechanism = item.mechanism,
                        ci.novelty_type = item.novelty_type,
                        ci.key_innovation = item.key_innovation,
                        ci.source_paper_id = $paper_id
                """, {
                    "paper_id": paper_id,
                    "items": [
                        {
                            "idea_id": idea.idea_id,
                            "mechanism": idea.mechanism,
                            "novelty_type": idea.novelty_type.value,
                            "key_innovation": idea.key_innovation,
                        }
                        for idea in arg.core_ideas
                    ],
                })

            # ---- Gap → CoreIdea links (precise mapping or fallback cartesian) ----
            if arg.gaps and arg.core_ideas:
                gap_idea_map = getattr(arg, "_gap_idea_map", {})
                if gap_idea_map:
                    pairs = [
                        {"gap_id": arg.gaps[gidx].gap_id, "idea_id": arg.core_ideas[iidx].idea_id}
                        for gidx, idea_indices in gap_idea_map.items()
                        for iidx in idea_indices
                        if gidx < len(arg.gaps) and iidx < len(arg.core_ideas)
                    ]
                else:
                    pairs = [
                        {"gap_id": g.gap_id, "idea_id": i.idea_id}
                        for g in arg.gaps
                        for i in arg.core_ideas
                    ]
                if pairs:
                    await session.run("""
                        UNWIND $pairs AS pair
                        MATCH (g:Gap {gap_id: pair.gap_id})
                        MATCH (ci:CoreIdea {idea_id: pair.idea_id})
                        MERGE (g)-[:ADDRESSED_BY]->(ci)
                    """, {"pairs": pairs})

            # ---- Claims (batch) ----
            if arg.claims:
                await session.run("""
                    UNWIND $items AS item
                    MERGE (cl:Claim {claim_id: item.claim_id})
                    SET cl.text = item.text,
                        cl.claim_type = item.claim_type,
                        cl.severity = item.severity,
                        cl.source_section = item.source_section,
                        cl.claim_hash = item.claim_hash,
                        cl.source_paper_id = $paper_id
                """, {
                    "paper_id": paper_id,
                    "items": [
                        {
                            "claim_id": claim.claim_id,
                            "text": claim.text,
                            "claim_type": claim.claim_type.value,
                            "severity": claim.severity.value,
                            "source_section": claim.source_section,
                            "claim_hash": claim.claim_hash,
                        }
                        for claim in arg.claims
                    ],
                })

            # ---- CoreIdea → Claim links (precise mapping or fallback cartesian) ----
            if arg.core_ideas and arg.claims:
                idea_claim_map = getattr(arg, "_idea_claim_map", {})
                if idea_claim_map:
                    pairs = [
                        {"idea_id": arg.core_ideas[iidx].idea_id, "claim_id": arg.claims[cidx].claim_id}
                        for iidx, claim_indices in idea_claim_map.items()
                        for cidx in claim_indices
                        if iidx < len(arg.core_ideas) and cidx < len(arg.claims)
                    ]
                else:
                    pairs = [
                        {"idea_id": i.idea_id, "claim_id": c.claim_id}
                        for i in arg.core_ideas
                        for c in arg.claims
                    ]
                if pairs:
                    await session.run("""
                        UNWIND $pairs AS pair
                        MATCH (ci:CoreIdea {idea_id: pair.idea_id})
                        MATCH (cl:Claim {claim_id: pair.claim_id})
                        MERGE (ci)-[:MAKES_CLAIM]->(cl)
                    """, {"pairs": pairs})

            # ---- Evidence (batch) — includes numeric_results + consistency ----
            if arg.evidences:
                consistency_issues = getattr(arg, "numeric_consistency_issues", [])
                consistency_json = _json.dumps(consistency_issues, ensure_ascii=False) if consistency_issues else "[]"
                await session.run("""
                    UNWIND $items AS item
                    MERGE (e:Evidence {evidence_id: item.evidence_id})
                    SET e.evidence_type = item.evidence_type,
                        e.result_summary = item.result_summary,
                        e.datasets = item.datasets,
                        e.metrics = item.metrics,
                        e.numeric_results = item.numeric_results,
                        e.numeric_consistency_issues = $consistency_json,
                        e.source_paper_id = $paper_id
                """, {
                    "paper_id": paper_id,
                    "consistency_json": consistency_json,
                    "items": [
                        {
                            "evidence_id": evid.evidence_id,
                            "evidence_type": evid.evidence_type.value,
                            "result_summary": evid.result_summary,
                            "datasets": evid.datasets,
                            "metrics": evid.metrics,
                            "numeric_results": _json.dumps(
                                evid.numeric_results, ensure_ascii=False
                            ) if evid.numeric_results else "{}",
                        }
                        for evid in arg.evidences
                    ],
                })

            # ---- Claim-Evidence links (batch) ----
            if arg.claim_evidence_links:
                await session.run("""
                    UNWIND $items AS item
                    MATCH (cl:Claim {claim_id: item.claim_id})
                    MATCH (e:Evidence {evidence_id: item.evidence_id})
                    MERGE (cl)-[r:SUPPORTED_BY]->(e)
                    SET r.strength = item.strength,
                        r.explanation = item.explanation
                """, {
                    "items": [
                        {
                            "claim_id": link.claim_id,
                            "evidence_id": link.evidence_id,
                            "strength": link.strength.value,
                            "explanation": link.explanation,
                        }
                        for link in arg.claim_evidence_links
                    ],
                })

            # ---- Baselines (batch) ----
            if arg.baselines:
                await session.run("""
                    UNWIND $items AS item
                    MERGE (b:Baseline {baseline_id: item.baseline_id})
                    SET b.method_name = item.method_name,
                        b.paper_ref = item.paper_ref,
                        b.source_paper_id = $paper_id
                """, {
                    "paper_id": paper_id,
                    "items": [
                        {
                            "baseline_id": bl.baseline_id,
                            "method_name": bl.method_name,
                            "paper_ref": bl.paper_ref,
                        }
                        for bl in arg.baselines
                    ],
                })

            # ---- Limitations (batch) ----
            if arg.limitations:
                await session.run("""
                    UNWIND $items AS item
                    MERGE (l:Limitation {limitation_id: item.limitation_id})
                    SET l.text = item.text,
                        l.scope = item.scope,
                        l.acknowledged_by_author = item.acknowledged,
                        l.source_paper_id = $paper_id
                    WITH l
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (p)-[:HAS_LIMITATION]->(l)
                """, {
                    "paper_id": paper_id,
                    "items": [
                        {
                            "limitation_id": lim.limitation_id,
                            "text": lim.text,
                            "scope": lim.scope,
                            "acknowledged": lim.acknowledged_by_author,
                        }
                        for lim in arg.limitations
                    ],
                })

    async def get_claim_evidence_ledger(self, paper_id: str) -> ClaimEvidenceLedger:
        """Build a Claim-Evidence ledger for a paper."""
        query = """
        MATCH (cl:Claim {source_paper_id: $paper_id})
        OPTIONAL MATCH (cl)-[r:SUPPORTED_BY]->(e:Evidence)
        RETURN cl.claim_id AS claim_id,
               cl.text AS claim_text,
               cl.claim_type AS claim_type,
               cl.severity AS severity,
               collect(DISTINCT {
                   evidence_id: e.evidence_id,
                   result_summary: e.result_summary,
                   strength: r.strength
               }) AS evidences
        """
        async with self.driver.session() as session:
            result = await session.run(query, {"paper_id": paper_id})
            records = await result.data()

        entries = []
        for r in records:
            actual_evidence = []
            best_strength = SupportStrength.UNVERIFIABLE
            for ev in r["evidences"]:
                if ev.get("evidence_id"):
                    actual_evidence.append(ev["result_summary"] or "")
                    strength = ev.get("strength", "UNVERIFIABLE")
                    if strength == "FULL":
                        best_strength = SupportStrength.FULL
                    elif strength == "PARTIAL" and best_strength != SupportStrength.FULL:
                        best_strength = SupportStrength.PARTIAL
                    elif strength == "REFUTED":
                        best_strength = SupportStrength.REFUTED

            entries.append(ClaimEvidenceLedgerEntry(
                claim_text=r["claim_text"],
                claim_type=r.get("claim_type", "PERFORMANCE"),
                severity=r.get("severity", "P1"),
                actual_evidence=actual_evidence,
                support_status=best_strength,
            ))

        return ClaimEvidenceLedger(paper_id=paper_id, entries=entries)

    # ========================================================================
    # Query Helpers
    # ========================================================================

    async def find_nearest_competitors(self, paper_id: str, k: int = 10) -> list[dict[str, Any]]:
        """Find papers that share the most entities with a given paper."""
        query = """
        MATCH (p1:Paper {paper_id: $paper_id})-[:PROPOSES|INTRODUCES]->(e)
        MATCH (p2:Paper)-[:PROPOSES|INTRODUCES]->(e)
        WHERE p2.paper_id <> $paper_id
        WITH p2, count(DISTINCT e) AS shared_entities
        ORDER BY shared_entities DESC
        LIMIT $k
        RETURN p2.paper_id AS paper_id, p2.title AS title, 
               p2.year AS year, shared_entities
        """
        async with self.driver.session() as session:
            result = await session.run(query, {"paper_id": paper_id, "k": k})
            return await result.data()

    async def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper already exists in the graph."""
        query = "MATCH (p:Paper {paper_id: $paper_id}) RETURN count(p) AS cnt"
        async with self.driver.session() as session:
            result = await session.run(query, {"paper_id": paper_id})
            record = await result.single()
            return record["cnt"] > 0 if record else False

    async def get_papers_by_year_range(self, paper_ids: list[str], start_year: int, end_year: int) -> set[str]:
        """Filter a candidate paper ID set to a closed year range [start, end]."""
        if not paper_ids:
            return set()

        query = """
        MATCH (p:Paper)
        WHERE p.paper_id IN $paper_ids
          AND p.year IS NOT NULL
          AND p.year >= $start_year
          AND p.year <= $end_year
        RETURN p.paper_id AS paper_id
        """
        async with self.driver.session() as session:
            result = await session.run(
                query,
                {
                    "paper_ids": paper_ids,
                    "start_year": start_year,
                    "end_year": end_year,
                },
            )
            rows = await result.data()

        return {row.get("paper_id", "") for row in rows if row.get("paper_id")}

    async def find_evolution_related_papers(self, paper_ids: list[str], k: int = 20) -> list[dict[str, Any]]:
        """Expand candidates via EVOLVES_FROM/CITES_FOR_FOUNDATION links."""
        if not paper_ids:
            return []

        query = """
        MATCH (seed:Paper)-[r:CITES]->(target:Paper)
        WHERE seed.paper_id IN $paper_ids
          AND r.intent IN ["EVOLVES_FROM", "CITES_FOR_FOUNDATION"]
          AND target.paper_id <> seed.paper_id
        RETURN target.paper_id AS paper_id,
               max(coalesce(r.confidence, 0.5)) AS relation_confidence,
               count(*) AS link_count
        ORDER BY link_count DESC, relation_confidence DESC
        LIMIT $k
        """
        async with self.driver.session() as session:
            result = await session.run(query, {"paper_ids": paper_ids, "k": k})
            return await result.data()

    async def get_edges_by_source_rule(
        self,
        source_rule: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return edges that are tagged with the given source_rule metadata."""
        query = """
        MATCH (src)-[r]->(dst)
        WHERE r.source_rule = $source_rule
        RETURN labels(src) AS src_labels,
               coalesce(src.paper_id, src.entity_id, src.claim_id, src.evidence_id, '') AS src_id,
               type(r) AS relation,
               labels(dst) AS dst_labels,
               coalesce(dst.paper_id, dst.entity_id, dst.claim_id, dst.evidence_id, '') AS dst_id,
               r.confidence AS confidence,
               r.source_rule AS source_rule,
               r.confidence_source AS confidence_source,
               r.evidence_span AS evidence_span
        ORDER BY coalesce(r.confidence, 0.0) DESC
        LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(
                query,
                {
                    "source_rule": source_rule,
                    "limit": max(1, int(limit)),
                },
            )
            return await result.data()

    async def get_stats(self) -> dict[str, int]:
        """Get counts of all node types."""
        labels = [
            "Paper", "Method", "Dataset", "Task", "Metric", "Model",
            "Framework", "Concept", "Problem", "Gap", "CoreIdea",
            "Claim", "Evidence", "Baseline", "Limitation",
        ]
        stats: dict[str, int] = {}
        async with self.driver.session() as session:
            for label in labels:
                result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                record = await result.single()
                stats[label] = record["cnt"] if record else 0
        return stats

    # ========================================================================
    # Cross-Layer Linking (CITATION → GAP)
    # ========================================================================

    async def link_citation_gap(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        gap_id: str,
        confidence: float = 0.8,
        source_rule: str | None = None,
    ) -> bool:
        """Create a SUPPORTS_GAP edge from a citation to a Gap node."""
        query = """
        MATCH (citing:Paper {paper_id: $citing_paper_id})-[c:CITES]->(cited:Paper {paper_id: $cited_paper_id})
        MATCH (g:Gap {gap_id: $gap_id})
        MERGE (c)-[r:SUPPORTS_GAP]->(g)
        SET r.confidence = $confidence,
            r.source_rule = $source_rule
        RETURN count(r) AS linked
        """
        async with self.driver.session() as session:
            result = await session.run(query, {
                "citing_paper_id": citing_paper_id,
                "cited_paper_id": cited_paper_id,
                "gap_id": gap_id,
                "confidence": confidence,
                "source_rule": source_rule or "heuristic.citation_gap_embedding",
            })
            record = await result.single()
            return bool(record and record.get("linked", 0) > 0)

    # ========================================================================
    # Additional Query Methods (from design doc)
    # ========================================================================

    async def get_gap_context(self, problem_id: str) -> dict[str, Any]:
        """Get the full gap context for a Problem node."""
        query = """
        MATCH (pr:Problem {problem_id: $problem_id})
        OPTIONAL MATCH (pr)-[:HAS_GAP]->(g:Gap)
        OPTIONAL MATCH (g)-[:ADDRESSED_BY]->(ci:CoreIdea)
        RETURN pr.description AS problem,
               pr.scope AS scope,
               pr.importance_signal AS importance_signal,
               collect(DISTINCT {
                   gap_id: g.gap_id,
                   failure_mode: g.failure_mode,
                   constraint: g.constraint,
                   prior_methods: g.prior_methods_failing,
                   core_idea: ci.mechanism,
                   novelty_type: ci.novelty_type,
                   key_innovation: ci.key_innovation
               }) AS gaps
        """
        async with self.driver.session() as session:
            result = await session.run(query, {"problem_id": problem_id})
            record = await result.single()
            if not record:
                return {}
            return dict(record)

    async def traverse_evidence_chain(self, claim_id: str) -> dict[str, Any]:
        """Traverse the full evidence chain for a Claim."""
        query = """
        MATCH (cl:Claim {claim_id: $claim_id})
        OPTIONAL MATCH (cl)-[s:SUPPORTED_BY]->(e:Evidence)
        OPTIONAL MATCH (ci:CoreIdea)-[:MAKES_CLAIM]->(cl)
        OPTIONAL MATCH (g:Gap)-[:ADDRESSED_BY]->(ci)
        OPTIONAL MATCH (pr:Problem)-[:HAS_GAP]->(g)
        RETURN cl.text AS claim_text,
               cl.claim_type AS claim_type,
               cl.severity AS severity,
               collect(DISTINCT {
                   evidence_id: e.evidence_id,
                   evidence_type: e.evidence_type,
                   result_summary: e.result_summary,
                   strength: s.strength
               }) AS evidences,
               ci.mechanism AS core_mechanism,
               ci.key_innovation AS key_innovation,
               g.failure_mode AS gap_failure,
               pr.description AS problem_description
        """
        async with self.driver.session() as session:
            result = await session.run(query, {"claim_id": claim_id})
            record = await result.single()
            if not record:
                return {}
            return dict(record)
