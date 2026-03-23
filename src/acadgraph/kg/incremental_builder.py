"""
Incremental KG Builder — Adds papers to the KG incrementally.

New papers are processed through the full pipeline without
rebuilding the existing graph. Supports batch processing with
concurrency control and shared deduplication caches.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from typing import Sequence

from acadgraph.embedding_client import EmbeddingClient
from acadgraph.kg.interfaces import KgRepository, VectorIndex
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
    RelationMetadata,
    make_relation_metadata,
    make_vector_payload,
)
from acadgraph.kg.extract.entities import EntityExtractor
from acadgraph.kg.extract.evolution import CitationEvolutionBuilder
from acadgraph.kg.extract.argumentation import ArgumentationExtractor
from acadgraph.kg.extract.peer_review import PeerReviewExtractor
from acadgraph.kg.paper_parser import PaperParser
from acadgraph.kg.schema import (
    BatchBuildResult,
    BuildResult,
    PaperSource,
)
from acadgraph.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---- Tuning Constants ----
SECTION_EMBED_MAX_CHARS = 2000
SECTION_MIN_LENGTH = 50
CITATION_CONTEXT_MIN_LENGTH = 10
METHOD_CLAIM_KEYWORD_MIN_LENGTH = 3
CROSS_LAYER_GAP_THRESHOLD = 0.80


class IncrementalKGBuilder:
    """Incremental knowledge graph builder."""

    @staticmethod
    def _relation_metadata(
        *,
        source_rule: str,
        confidence_source: str,
        evidence_span: str | None = None,
    ) -> RelationMetadata:
        """Create normalized relation metadata objects from ontology helpers."""
        return make_relation_metadata(
            source_rule=source_rule,
            confidence_source=confidence_source,
            evidence_span=evidence_span,
        )

    def __init__(
        self,
        neo4j_store: KgRepository,
        qdrant_store: VectorIndex,
        llm: LLMClient,
        embedding: EmbeddingClient,
    ):
        self.neo4j = neo4j_store
        self.qdrant = qdrant_store
        self.llm = llm
        self.embedding = embedding

        # Sub-components
        self.parser = PaperParser(llm_client=llm)
        self.entity_extractor = EntityExtractor(llm=llm, embedding=embedding, qdrant=qdrant_store)
        self.citation_builder = CitationEvolutionBuilder(llm=llm)
        self.argumentation_extractor = ArgumentationExtractor(llm=llm)
        self.peer_review_extractor = PeerReviewExtractor()

    async def _call_neo4j_optional(self, method_name: str, *args, **kwargs):
        """Call optional Neo4j methods without breaking alternate implementations."""
        method = getattr(self.neo4j, method_name, None)
        if method is None:
            return None

        signature = inspect.signature(method)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if accepts_var_kwargs:
            filtered_kwargs = kwargs
        else:
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }

        return await method(*args, **filtered_kwargs)

    async def add_paper(self, paper_source: PaperSource) -> BuildResult:
        """
        Incrementally add a single paper to the knowledge graph.

        Pipeline:
        1. Parse → ParsedPaper
        2. Entity Extraction → Neo4j + Qdrant
        3. Citation Classification → Neo4j
        4. Argumentation Chain Extraction → Neo4j + Qdrant
        5. Cross-layer linking
        """
        start_time = time.time()
        result = BuildResult(paper_id=paper_source.paper_id)

        # Check for duplicates
        if await self.neo4j.paper_exists(paper_source.paper_id):
            logger.info("Paper %s already exists, skipping", paper_source.paper_id)
            result.errors.append("Paper already exists in graph")
            return result

        try:
            # ---- Step 1: Parse ----
            logger.info("[1/6] Parsing paper %s", paper_source.paper_id)
            if paper_source.pdf_path:
                parsed = await self.parser.parse(paper_source.pdf_path)
            elif paper_source.text:
                if paper_source.content_meta:
                    parsed = await self.parser.parse_from_jsonl_record(
                        content=paper_source.text,
                        content_meta=paper_source.content_meta,
                        abstract=paper_source.abstract,
                    )
                else:
                    parsed = await self.parser.parse_from_text(paper_source.text)
            else:
                raise ValueError("PaperSource must have either pdf_path or text")

            # Fill in metadata
            parsed.paper_id = paper_source.paper_id
            parsed.title = paper_source.title or parsed.title
            parsed.year = paper_source.year or parsed.year
            parsed.venue = paper_source.venue or parsed.venue
            parsed.authors = paper_source.authors or parsed.authors
            parsed.openreview_id = paper_source.paper_id  # OpenReview ID

            # Create Paper node
            await self.neo4j.upsert_paper(
                paper_id=parsed.paper_id,
                title=parsed.title,
                year=parsed.year,
                venue=parsed.venue,
                authors=parsed.authors,
            )

            # ---- Step 2: Entity Extraction ----
            logger.info("[2/6] Extracting entities for %s", parsed.paper_id)
            entity_result = await self.entity_extractor.extract(parsed)

            # Cross-paper deduplication
            entity_result.entities = await self.entity_extractor.deduplicate_cross_paper(
                entity_result.entities
            )

            # Store entities in Neo4j
            for entity in entity_result.entities:
                await self.neo4j.upsert_entity(entity)
                result.entities_added += 1

            # Store entity relations
            for relation in entity_result.relations:
                await self.neo4j.upsert_relation(relation)
                result.relations_added += 1

            # Store entity embeddings in Qdrant
            await self._store_entity_embeddings(entity_result.entities)

            # Link Paper → entities
            for entity in entity_result.entities:
                rel_type = self._paper_entity_relation_type(entity.entity_type.value)
                metadata = self._relation_metadata(
                    source_rule="ontology.paper_entity_relation",
                    confidence_source="builder.default",
                )
                linked = await self._call_neo4j_optional(
                    "link_paper_entity",
                    paper_id=parsed.paper_id,
                    entity_id=entity.entity_id,
                    relation_type=rel_type,
                    confidence=1.0,
                    source_rule=metadata.source_rule,
                    confidence_source=metadata.confidence_source,
                    metadata=metadata,
                )
                if linked is None:
                    # Keep backward compatibility when using older store implementations.
                    continue
                elif linked:
                    result.relations_added += 1
                else:
                    msg = (
                        f"Failed to link paper {parsed.paper_id} to entity "
                        f"{entity.entity_id} with relation {rel_type}"
                    )
                    logger.warning(msg)
                    result.errors.append(msg)

            # ---- Step 3: Citation Classification ----
            logger.info("[3/6] Classifying citations for %s", parsed.paper_id)
            citations = await self.citation_builder.classify_citations(parsed)

            for citation in citations:
                try:
                    linked = await self.neo4j.add_citation(citation)
                    if linked is False:
                        msg = (
                            f"Citation edge not created: "
                            f"{citation.citing_paper_id} -> {citation.cited_paper_id} "
                            f"({citation.intent.value})"
                        )
                        logger.warning(msg)
                        result.errors.append(msg)
                    else:
                        result.citations_added += 1
                except Exception as e:
                    msg = (
                        f"Failed to add citation "
                        f"{citation.citing_paper_id}->{citation.cited_paper_id}: {e}"
                    )
                    logger.warning(msg)
                    result.errors.append(msg)

            # Store citation context embeddings in Qdrant
            await self._store_citation_embeddings(citations)

            # Enrich citations from Semantic Scholar (best effort)
            try:
                enriched = await self.citation_builder.enrich_from_semantic_scholar(
                    parsed.paper_id
                )
                for ec in enriched:
                    try:
                        await self.neo4j.add_citation(ec)
                        result.citations_added += 1
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Semantic Scholar enrichment skipped for %s: %s", parsed.paper_id, e)

            # ---- Step 4: Argumentation Chain ----
            logger.info("[4/6] Extracting argumentation for %s", parsed.paper_id)
            arg_graph = await self.argumentation_extractor.extract(parsed)

            # Store in Neo4j
            await self.neo4j.store_argumentation(parsed.paper_id, arg_graph)
            result.claims_added = len(arg_graph.claims)
            result.evidences_added = len(arg_graph.evidences)

            # Build and store method evolution links (best effort)
            result.relations_added += await self._build_method_evolution_links(
                paper_id=parsed.paper_id,
                paper_year=parsed.year,
                entity_result=entity_result,
            )

            # Store claim and evidence embeddings in Qdrant
            await self._store_argumentation_embeddings(arg_graph)

            # ---- Step 5: Peer Review Extraction ----
            if paper_source.related_notes_raw:
                logger.info("[5/6] Extracting peer reviews for %s", parsed.paper_id)
                try:
                    reviews = self.peer_review_extractor.extract(
                        paper_source.related_notes_raw
                    )
                    if reviews:
                        # Store decision as metadata
                        decision = self.peer_review_extractor.get_decision(reviews)
                        if decision:
                            logger.info(
                                "Paper %s decision: %s", parsed.paper_id, decision
                            )

                        # Convert reviews to claims/evidences
                        pr_claims, pr_evidences = (
                            self.peer_review_extractor.to_claims_and_evidences(
                                reviews, parsed.paper_id
                            )
                        )
                        # Merge into argumentation graph
                        arg_graph.claims.extend(pr_claims)
                        arg_graph.evidences.extend(pr_evidences)
                        result.claims_added += len(pr_claims)
                        result.evidences_added += len(pr_evidences)
                        logger.info(
                            "Added %d peer-review claims, %d evidences for %s",
                            len(pr_claims),
                            len(pr_evidences),
                            parsed.paper_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Peer review extraction failed for %s: %s",
                        parsed.paper_id,
                        e,
                    )

            # ---- Step 6: Cross-Layer Linking ----
            logger.info("[6/6] Cross-layer linking for %s", parsed.paper_id)
            await self._cross_layer_link(parsed.paper_id, entity_result, arg_graph)

            # Store section embeddings
            await self._store_section_embeddings(parsed)

        except Exception as e:
            logger.error("Failed to process paper %s: %s", paper_source.paper_id, e)
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start_time
        logger.info(
            "Paper %s processed in %.1fs: %d entities, %d relations, "
            "%d citations, %d claims, %d evidences, %d errors",
            paper_source.paper_id,
            result.duration_seconds,
            result.entities_added,
            result.relations_added,
            result.citations_added,
            result.claims_added,
            result.evidences_added,
            len(result.errors),
        )
        return result

    async def add_papers_batch(
        self,
        papers: Sequence[PaperSource],
        batch_size: int = 5,
    ) -> BatchBuildResult:
        """
        Batch-add papers with concurrency control.

        Processes `batch_size` papers concurrently, sharing the
        deduplication cache across all papers in the batch.
        """
        start_time = time.time()
        batch_result = BatchBuildResult(total_papers=len(papers))

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            logger.info(
                "Processing batch %d-%d of %d papers",
                i + 1, min(i + batch_size, len(papers)), len(papers),
            )

            # Process batch concurrently
            tasks = [self.add_paper(p) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    batch_result.failed += 1
                    batch_result.results.append(
                        BuildResult(errors=[str(r)])
                    )
                else:
                    if r.errors:
                        batch_result.failed += 1
                    else:
                        batch_result.successful += 1
                    batch_result.results.append(r)

        batch_result.total_duration_seconds = time.time() - start_time
        logger.info(
            "Batch complete: %d/%d successful in %.1fs",
            batch_result.successful,
            batch_result.total_papers,
            batch_result.total_duration_seconds,
        )
        return batch_result

    # ========================================================================
    # Embedding Storage Helpers
    # ========================================================================

    async def _store_entity_embeddings(self, entities: list) -> None:
        """Store entity embeddings in Qdrant."""
        if not entities:
            return
        texts = [
            f"{e.entity_type.value}: {e.name}. {e.description}"
            for e in entities
        ]
        embeddings = await self.embedding.embed_batch(texts)
        ids = [e.entity_id for e in entities]
        payloads = [
            make_vector_payload(
                doc_kind=DOC_KIND_ENTITY,
                kg_layer=KG_LAYER_L1,
                base={
                    "entity_type": e.entity_type.value,
                    "name": e.name,
                    "paper_id": e.source_paper_id,
                },
            )
            for e in entities
        ]
        await self.qdrant.upsert_embeddings_batch(
            ENTITY_COLLECTION, ids, embeddings, payloads
        )

    async def _store_citation_embeddings(self, citations: list) -> None:
        """Store citation context embeddings in Qdrant."""
        if not citations:
            return
        valid = [c for c in citations if c.context and len(c.context) > CITATION_CONTEXT_MIN_LENGTH]
        if not valid:
            return
        texts = [c.context for c in valid]
        embeddings = await self.embedding.embed_batch(texts)
        ids = [f"cite_{c.citing_paper_id}_{i}" for i, c in enumerate(valid)]
        payloads = [
            make_vector_payload(
                doc_kind=DOC_KIND_CITATION,
                kg_layer=KG_LAYER_L2,
                base={
                    "citing_paper_id": c.citing_paper_id,
                    "paper_id": c.citing_paper_id,
                    "cited_paper_id": c.cited_paper_id,
                    "intent": c.intent.value,
                    "section": c.section,
                },
            )
            for c in valid
        ]
        await self.qdrant.upsert_embeddings_batch(
            CLAIM_COLLECTION, ids, embeddings, payloads
        )

    async def _store_argumentation_embeddings(self, arg_graph) -> None:
        """Store claim and evidence embeddings in Qdrant."""
        # Claims
        if arg_graph.claims:
            texts = [c.text for c in arg_graph.claims]
            embeddings = await self.embedding.embed_batch(texts)
            ids = [c.claim_id for c in arg_graph.claims]
            payloads = [
                make_vector_payload(
                    doc_kind=DOC_KIND_CLAIM,
                    kg_layer=KG_LAYER_L3,
                    base={
                        "paper_id": c.source_paper_id,
                        "claim_type": c.claim_type.value,
                        "severity": c.severity.value,
                    },
                )
                for c in arg_graph.claims
            ]
            await self.qdrant.upsert_embeddings_batch(
                CLAIM_COLLECTION, ids, embeddings, payloads
            )

        # Evidence
        if arg_graph.evidences:
            texts = [e.result_summary for e in arg_graph.evidences if e.result_summary]
            valid_evidences = [e for e in arg_graph.evidences if e.result_summary]
            if texts:
                embeddings = await self.embedding.embed_batch(texts)
                ids = [e.evidence_id for e in valid_evidences]
                payloads = [
                    make_vector_payload(
                        doc_kind=DOC_KIND_EVIDENCE,
                        kg_layer=KG_LAYER_L3,
                        base={
                            "paper_id": e.source_paper_id,
                            "evidence_type": e.evidence_type.value,
                        },
                    )
                    for e in valid_evidences
                ]
                await self.qdrant.upsert_embeddings_batch(
                    CLAIM_COLLECTION, ids, embeddings, payloads
                )

    async def _store_section_embeddings(self, parsed_paper) -> None:
        """Store paper section embeddings in Qdrant."""
        sections = [
            (name, text)
            for name, text in parsed_paper.sections.items()
            if text and len(text.strip()) > SECTION_MIN_LENGTH
        ]
        if not sections:
            return

        texts = [text[:SECTION_EMBED_MAX_CHARS] for _, text in sections]  # Truncate long sections
        embeddings = await self.embedding.embed_batch(texts)
        ids = [f"sec_{parsed_paper.paper_id}_{name}" for name, _ in sections]
        payloads = [
            make_vector_payload(
                doc_kind=DOC_KIND_SECTION,
                kg_layer=KG_LAYER_L1,
                base={
                    "paper_id": parsed_paper.paper_id,
                    "section": name,
                    "title": parsed_paper.title,
                },
            )
            for name, _ in sections
        ]
        await self.qdrant.upsert_embeddings_batch(
            ENTITY_COLLECTION, ids, embeddings, payloads
        )

    # ========================================================================
    # Cross-Layer Linking
    # ========================================================================

    def _paper_entity_relation_type(self, entity_type_value: str) -> str:
        """Pick a readable Paper->Entity relation type."""
        if entity_type_value == "METHOD":
            return "PROPOSES"
        if entity_type_value == "DATASET":
            return "INTRODUCES"
        return "USES"

    async def _build_method_evolution_links(
        self,
        paper_id: str,
        paper_year: int | None,
        entity_result,
    ) -> int:
        """Build and persist EVOLVES_FROM links using current + historical methods."""
        current_method_ids = {
            e.entity_id
            for e in entity_result.entities
            if e.entity_type.value == "METHOD"
        }
        if not current_method_ids:
            return 0

        try:
            methods_with_papers = await self._call_neo4j_optional(
                "get_related_methods", paper_id, limit=60
            )
            if not methods_with_papers or len(methods_with_papers) < 2:
                return 0

            chains = await self.citation_builder.build_evolution_chains(methods_with_papers)
            edges_added = 0

            for chain in chains:
                if len(chain.steps) < 2:
                    continue

                from_step = chain.steps[0]
                to_step = chain.steps[1]
                if not from_step.method_id or not to_step.method_id:
                    continue
                if from_step.method_id == to_step.method_id:
                    continue

                # Keep links centered around methods from the current paper.
                if (
                    from_step.method_id not in current_method_ids
                    and to_step.method_id not in current_method_ids
                ):
                    continue

                linked = await self._call_neo4j_optional(
                    "add_method_evolution",
                    from_method_id=from_step.method_id,
                    to_method_id=to_step.method_id,
                    source_paper_id=paper_id,
                    delta_description=to_step.delta_description,
                    year=to_step.year or paper_year,
                    confidence=1.0,
                )
                if linked is None:
                    return 0
                if linked:
                    edges_added += 1

            if edges_added:
                logger.info("Stored %d EVOLVES_FROM links for %s", edges_added, paper_id)
            return edges_added
        except Exception as e:
            logger.warning("Evolution linking failed for %s: %s", paper_id, e)
            return 0

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize text for loose lexical matching."""
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _keyword_tokens(self, value: str) -> set[str]:
        """Extract lightweight keywords for overlap checks."""
        stopwords = {
            "the", "a", "an", "and", "or", "to", "of", "for", "in", "on",
            "with", "from", "using", "based", "via", "our", "we", "is", "are",
        }
        tokens = set(self._normalize_text(value).split())
        return {token for token in tokens if len(token) > METHOD_CLAIM_KEYWORD_MIN_LENGTH and token not in stopwords}

    def _method_claim_confidence(self, method_name: str, claim_text: str) -> float:
        """Heuristic confidence for METHOD -> CLAIM links."""
        method_norm = self._normalize_text(method_name)
        claim_norm = self._normalize_text(claim_text)
        if not method_norm or not claim_norm:
            return 0.0

        if method_norm in claim_norm:
            return 0.95

        method_tokens = self._keyword_tokens(method_name)
        claim_tokens = self._keyword_tokens(claim_text)
        if not method_tokens or not claim_tokens:
            return 0.0

        overlap = method_tokens & claim_tokens
        if not overlap:
            return 0.0

        overlap_ratio = len(overlap) / len(method_tokens)
        if overlap_ratio >= 0.66:
            return 0.85
        if overlap_ratio >= 0.40:
            return 0.72
        if any(len(token) >= 6 for token in overlap):
            return 0.62
        return 0.0

    def _dataset_evidence_confidence(self, dataset_name: str, evidence_datasets: list[str]) -> float:
        """Heuristic confidence for DATASET -> EVIDENCE links."""
        dataset_norm = self._normalize_text(dataset_name)
        if not dataset_norm or not evidence_datasets:
            return 0.0

        dataset_tokens = self._keyword_tokens(dataset_name)
        best_conf = 0.0

        for raw_name in evidence_datasets:
            candidate_norm = self._normalize_text(raw_name)
            if not candidate_norm:
                continue

            if candidate_norm == dataset_norm:
                return 0.95

            if dataset_norm in candidate_norm or candidate_norm in dataset_norm:
                best_conf = max(best_conf, 0.88)
                continue

            candidate_tokens = self._keyword_tokens(raw_name)
            if not dataset_tokens or not candidate_tokens:
                continue

            overlap = dataset_tokens & candidate_tokens
            if not overlap:
                continue

            overlap_ratio = len(overlap) / len(dataset_tokens)
            if overlap_ratio >= 0.66:
                best_conf = max(best_conf, 0.78)
            elif overlap_ratio >= 0.40:
                best_conf = max(best_conf, 0.68)

        return best_conf

    async def _cross_layer_link(self, paper_id: str, entity_result, arg_graph) -> None:
        """
        Create cross-layer links:
        - METHOD -> CLAIM via lexical overlap in claim text.
        - DATASET -> EVIDENCE via evidence.datasets name matching.
        """
        method_links = 0
        dataset_links = 0

        methods = [
            entity
            for entity in entity_result.entities
            if entity.entity_type.value == "METHOD"
        ]
        datasets = [
            entity
            for entity in entity_result.entities
            if entity.entity_type.value == "DATASET"
        ]

        for method in methods:
            for claim in arg_graph.claims:
                confidence = self._method_claim_confidence(method.name, claim.text)
                if confidence <= 0:
                    continue
                try:
                    metadata = self._relation_metadata(
                        source_rule="heuristic.method_claim_overlap",
                        confidence_source="builder.heuristic",
                    )
                    linked = await self._call_neo4j_optional(
                        "link_method_claim",
                        method_id=method.entity_id,
                        claim_id=claim.claim_id,
                        source_paper_id=paper_id,
                        confidence=confidence,
                        source_rule=metadata.source_rule,
                        confidence_source=metadata.confidence_source,
                        metadata=metadata,
                    )
                    if linked:
                        method_links += 1
                except Exception as e:
                    logger.debug(
                        "Failed METHOD->CLAIM link %s -> %s: %s",
                        method.entity_id,
                        claim.claim_id,
                        e,
                    )

        for dataset in datasets:
            for evidence in arg_graph.evidences:
                confidence = self._dataset_evidence_confidence(dataset.name, evidence.datasets)
                if confidence <= 0:
                    continue
                try:
                    metadata = self._relation_metadata(
                        source_rule="heuristic.dataset_evidence_match",
                        confidence_source="builder.heuristic",
                        evidence_span=", ".join(evidence.datasets) if evidence.datasets else None,
                    )
                    linked = await self._call_neo4j_optional(
                        "link_dataset_evidence",
                        dataset_id=dataset.entity_id,
                        evidence_id=evidence.evidence_id,
                        source_paper_id=paper_id,
                        confidence=confidence,
                        source_rule=metadata.source_rule,
                        confidence_source=metadata.confidence_source,
                        evidence_span=metadata.evidence_span,
                        metadata=metadata,
                    )
                    if linked:
                        dataset_links += 1
                except Exception as e:
                    logger.debug(
                        "Failed DATASET->EVIDENCE link %s -> %s: %s",
                        dataset.entity_id,
                        evidence.evidence_id,
                        e,
                    )

        logger.info(
            "Cross-layer linking for %s: %d METHOD->CLAIM, %d DATASET->EVIDENCE",
            paper_id,
            method_links,
            dataset_links,
        )

        # --- CITATION → GAP linking via embedding similarity ---
        await self._link_citations_to_gaps(paper_id, arg_graph)

    async def _link_citations_to_gaps(self, paper_id: str, arg_graph) -> None:
        """Link citation contexts to Gap nodes via embedding similarity.

        Finds citations whose context semantically overlaps with gap
        failure_mode/constraint descriptions and creates SUPPORTS_GAP edges.
        """
        if not arg_graph.gaps:
            return

        gap_texts = [
            f"Gap: {g.failure_mode}. Constraint: {g.constraint}"
            for g in arg_graph.gaps
        ]
        if not any(t.strip() for t in gap_texts):
            return

        try:
            gap_embeddings = await self.embedding.embed_batch(gap_texts)

            for gap, gap_emb in zip(arg_graph.gaps, gap_embeddings):
                citation_hits = await self.qdrant.search_similar(
                    CLAIM_COLLECTION,
                    gap_emb,
                    k=5,
                    filters={"doc_kind": DOC_KIND_CITATION, "paper_id": paper_id},
                )
                for hit in citation_hits:
                    if hit.get("score", 0) >= CROSS_LAYER_GAP_THRESHOLD:
                        await self._call_neo4j_optional(
                            "link_citation_gap",
                            citing_paper_id=paper_id,
                            cited_paper_id=hit.get("payload", {}).get("cited_paper_id", ""),
                            gap_id=gap.gap_id,
                            confidence=float(hit["score"]),
                            source_rule="heuristic.citation_gap_embedding",
                        )
        except Exception as e:
            logger.debug("CITATION→GAP linking failed for %s: %s", paper_id, e)
