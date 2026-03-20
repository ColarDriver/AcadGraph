"""
Neo4j graph storage for the Three-Layer KG.

Manages CRUD operations for all three layers of the knowledge graph:
- Layer 1: Semantic entities and their relations
- Layer 2: Citation edges with intent
- Layer 3: Argumentation chains (Problem → Gap → Claim → Evidence)
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from acadgraph.config import Neo4jConfig
from acadgraph.kg.interfaces import KgRepository
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


class Neo4jKGStore(KgRepository):
    """Neo4j graph storage — manages the three-layer knowledge graph."""

    def __init__(self, config: Neo4jConfig | None = None):
        self.config = config or Neo4jConfig()
        self._driver: AsyncDriver | None = None

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
            # Layer 1 constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE d.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (me:Metric) REQUIRE me.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (mo:Model) REQUIRE mo.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Framework) REQUIRE f.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.entity_id IS UNIQUE",
            # Paper constraint
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            # Layer 3 constraints
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
        ]

        async with self.driver.session() as session:
            for q in queries:
                try:
                    await session.run(q)
                except Exception as e:
                    logger.warning("Schema init query failed (may already exist): %s — %s", q[:60], e)

        logger.info("Neo4j schema initialized")

    # ========================================================================
    # Layer 1: Entities
    # ========================================================================

    async def upsert_entity(self, entity: Entity) -> str:
        """Insert or update a Layer 1 entity. Returns entity_id."""
        label = entity.entity_type.value.capitalize()
        query = f"""
        MERGE (e:{label} {{entity_id: $entity_id}})
        SET e.name = $name,
            e.description = $description,
            e.source_paper_id = $source_paper_id,
            e.source_section = $source_section,
            e.confidence = $confidence
        RETURN e.entity_id AS id
        """
        params = {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "description": entity.description,
            "source_paper_id": entity.source_paper_id,
            "source_section": entity.source_section,
            "confidence": entity.confidence,
        }
        # Add custom attributes
        for k, v in entity.attributes.items():
            if isinstance(v, (str, int, float, bool)):
                params[k] = v
                query = query.replace(
                    "RETURN e.entity_id AS id",
                    f"SET e.{k} = ${k}\nRETURN e.entity_id AS id",
                )

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
            if isinstance(v, (str, int, float, bool)):
                params[k] = v
                query += f"\nSET r.{k} = ${k}"

        async with self.driver.session() as session:
            await session.run(query, params)

    async def link_paper_entity(
        self,
        paper_id: str,
        entity_id: str,
        relation_type: str,
        confidence: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create/merge an edge between Paper and entity."""
        query = f"""
        MATCH (p:Paper {{paper_id: $paper_id}})
        MATCH (e {{entity_id: $entity_id}})
        MERGE (p)-[r:{relation_type}]->(e)
        SET r.source_paper_id = $paper_id,
            r.confidence = $confidence
        RETURN count(r) AS linked
        """
        params: dict[str, Any] = {
            "paper_id": paper_id,
            "entity_id": entity_id,
            "confidence": confidence,
        }

        if properties:
            for key, value in properties.items():
                if isinstance(value, (str, int, float, bool)):
                    params[key] = value
                    query = query.replace(
                        "RETURN count(r) AS linked",
                        f"SET r.{key} = ${key}\n        RETURN count(r) AS linked",
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
    # Layer 2: Citations
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
    ) -> bool:
        """Create cross-layer METHOD -> CLAIM link."""
        query = """
        MATCH (m:Method {entity_id: $method_id})
        MATCH (c:Claim {claim_id: $claim_id})
        MERGE (m)-[r:SUPPORTS_CLAIM]->(c)
        SET r.source_paper_id = $source_paper_id,
            r.confidence = $confidence
        RETURN count(r) AS linked
        """
        async with self.driver.session() as session:
            result = await session.run(query, {
                "method_id": method_id,
                "claim_id": claim_id,
                "source_paper_id": source_paper_id,
                "confidence": confidence,
            })
            record = await result.single()
            return bool(record and record.get("linked", 0) > 0)

    async def link_dataset_evidence(
        self,
        dataset_id: str,
        evidence_id: str,
        source_paper_id: str,
        confidence: float = 0.9,
    ) -> bool:
        """Create cross-layer DATASET -> EVIDENCE link."""
        query = """
        MATCH (d:Dataset {entity_id: $dataset_id})
        MATCH (e:Evidence {evidence_id: $evidence_id})
        MERGE (d)-[r:USED_IN_EVIDENCE]->(e)
        SET r.source_paper_id = $source_paper_id,
            r.confidence = $confidence
        RETURN count(r) AS linked
        """
        async with self.driver.session() as session:
            result = await session.run(query, {
                "dataset_id": dataset_id,
                "evidence_id": evidence_id,
                "source_paper_id": source_paper_id,
                "confidence": confidence,
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
    # Layer 3: Argumentation
    # ========================================================================

    async def store_argumentation(self, paper_id: str, arg: ArgumentationGraph) -> None:
        """Store the full argumentation graph for a paper."""
        async with self.driver.session() as session:
            # Problems
            for prob in arg.problems:
                await session.run("""
                    MERGE (pr:Problem {problem_id: $problem_id})
                    SET pr.description = $description,
                        pr.scope = $scope,
                        pr.importance_signal = $importance_signal,
                        pr.source_paper_id = $paper_id
                    WITH pr
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (p)-[:HAS_PROBLEM]->(pr)
                """, {
                    "problem_id": prob.problem_id,
                    "description": prob.description,
                    "scope": prob.scope,
                    "importance_signal": prob.importance_signal,
                    "paper_id": paper_id,
                })

            # Gaps
            for gap in arg.gaps:
                await session.run("""
                    MERGE (g:Gap {gap_id: $gap_id})
                    SET g.failure_mode = $failure_mode,
                        g.constraint = $constraint,
                        g.prior_methods_failing = $prior_methods,
                        g.source_paper_id = $paper_id
                """, {
                    "gap_id": gap.gap_id,
                    "failure_mode": gap.failure_mode,
                    "constraint": gap.constraint,
                    "prior_methods": gap.prior_methods_failing,
                    "paper_id": paper_id,
                })

            # Link Problem → Gap
            if arg.problems and arg.gaps:
                for prob in arg.problems:
                    for gap in arg.gaps:
                        await session.run("""
                            MATCH (pr:Problem {problem_id: $prob_id})
                            MATCH (g:Gap {gap_id: $gap_id})
                            MERGE (pr)-[:HAS_GAP]->(g)
                        """, {"prob_id": prob.problem_id, "gap_id": gap.gap_id})

            # Core Ideas
            for idea in arg.core_ideas:
                await session.run("""
                    MERGE (ci:CoreIdea {idea_id: $idea_id})
                    SET ci.mechanism = $mechanism,
                        ci.novelty_type = $novelty_type,
                        ci.key_innovation = $key_innovation,
                        ci.source_paper_id = $paper_id
                """, {
                    "idea_id": idea.idea_id,
                    "mechanism": idea.mechanism,
                    "novelty_type": idea.novelty_type.value,
                    "key_innovation": idea.key_innovation,
                    "paper_id": paper_id,
                })

            # Link Gap → CoreIdea
            if arg.gaps and arg.core_ideas:
                for gap in arg.gaps:
                    for idea in arg.core_ideas:
                        await session.run("""
                            MATCH (g:Gap {gap_id: $gap_id})
                            MATCH (ci:CoreIdea {idea_id: $idea_id})
                            MERGE (g)-[:ADDRESSED_BY]->(ci)
                        """, {"gap_id": gap.gap_id, "idea_id": idea.idea_id})

            # Claims
            for claim in arg.claims:
                await session.run("""
                    MERGE (cl:Claim {claim_id: $claim_id})
                    SET cl.text = $text,
                        cl.claim_type = $claim_type,
                        cl.severity = $severity,
                        cl.source_section = $source_section,
                        cl.claim_hash = $claim_hash,
                        cl.source_paper_id = $paper_id
                """, {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "claim_type": claim.claim_type.value,
                    "severity": claim.severity.value,
                    "source_section": claim.source_section,
                    "claim_hash": claim.claim_hash,
                    "paper_id": paper_id,
                })

            # Link CoreIdea → Claim
            if arg.core_ideas and arg.claims:
                for idea in arg.core_ideas:
                    for claim in arg.claims:
                        await session.run("""
                            MATCH (ci:CoreIdea {idea_id: $idea_id})
                            MATCH (cl:Claim {claim_id: $claim_id})
                            MERGE (ci)-[:MAKES_CLAIM]->(cl)
                        """, {"idea_id": idea.idea_id, "claim_id": claim.claim_id})

            # Evidence
            for evid in arg.evidences:
                await session.run("""
                    MERGE (e:Evidence {evidence_id: $evidence_id})
                    SET e.evidence_type = $evidence_type,
                        e.result_summary = $result_summary,
                        e.datasets = $datasets,
                        e.metrics = $metrics,
                        e.source_paper_id = $paper_id
                """, {
                    "evidence_id": evid.evidence_id,
                    "evidence_type": evid.evidence_type.value,
                    "result_summary": evid.result_summary,
                    "datasets": evid.datasets,
                    "metrics": evid.metrics,
                    "paper_id": paper_id,
                })

            # Claim-Evidence links
            for link in arg.claim_evidence_links:
                await session.run("""
                    MATCH (cl:Claim {claim_id: $claim_id})
                    MATCH (e:Evidence {evidence_id: $evidence_id})
                    MERGE (cl)-[r:SUPPORTED_BY]->(e)
                    SET r.strength = $strength,
                        r.explanation = $explanation
                """, {
                    "claim_id": link.claim_id,
                    "evidence_id": link.evidence_id,
                    "strength": link.strength.value,
                    "explanation": link.explanation,
                })

            # Baselines
            for bl in arg.baselines:
                await session.run("""
                    MERGE (b:Baseline {baseline_id: $baseline_id})
                    SET b.method_name = $method_name,
                        b.paper_ref = $paper_ref,
                        b.source_paper_id = $paper_id
                """, {
                    "baseline_id": bl.baseline_id,
                    "method_name": bl.method_name,
                    "paper_ref": bl.paper_ref,
                    "paper_id": paper_id,
                })

            # Limitations
            for lim in arg.limitations:
                await session.run("""
                    MERGE (l:Limitation {limitation_id: $limitation_id})
                    SET l.text = $text,
                        l.scope = $scope,
                        l.acknowledged_by_author = $acknowledged,
                        l.source_paper_id = $paper_id
                    WITH l
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (p)-[:HAS_LIMITATION]->(l)
                """, {
                    "limitation_id": lim.limitation_id,
                    "text": lim.text,
                    "scope": lim.scope,
                    "acknowledged": lim.acknowledged_by_author,
                    "paper_id": paper_id,
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
