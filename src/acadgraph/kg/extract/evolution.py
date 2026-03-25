"""
Citation Relations & Technology Evolution Chains.

Builds citation edges with 6 intent types and constructs
temporal evolution chains for methods.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from acadgraph.kg.prompts.loader import load_prompt, render_prompt
from acadgraph.kg.schema import (
    CitationEdge,
    CitationIntent,
    EvolutionChain,
    EvolutionStep,
    ParsedPaper,
    Reference,
    generate_id,
)
from acadgraph.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Load prompts from Markdown files
SYSTEM_PROMPT = load_prompt("citation", "citation_system")
CITATION_CLASSIFICATION_PROMPT = load_prompt("citation", "citation_classification")
BATCH_CITATION_PROMPT = load_prompt("citation", "citation_batch")
EVOLUTION_DETECTION_PROMPT = load_prompt("citation", "evolution_detection")

# Map string intents to enum
_INTENT_MAP: dict[str, CitationIntent] = {
    "CITES_FOR_PROBLEM": CitationIntent.CITES_FOR_PROBLEM,
    "CITES_AS_BASELINE": CitationIntent.CITES_AS_BASELINE,
    "CITES_FOR_FOUNDATION": CitationIntent.CITES_FOR_FOUNDATION,
    "CITES_AS_COMPARISON": CitationIntent.CITES_AS_COMPARISON,
    "CITES_FOR_THEORY": CitationIntent.CITES_FOR_THEORY,
    "EVOLVES_FROM": CitationIntent.EVOLVES_FROM,
}

# Section → likely default intent mapping (used as prior)
_SECTION_INTENT_PRIOR: dict[str, CitationIntent] = {
    "introduction": CitationIntent.CITES_FOR_PROBLEM,
    "related_work": CitationIntent.CITES_AS_COMPARISON,
    "method": CitationIntent.CITES_FOR_FOUNDATION,
    "experiments": CitationIntent.CITES_AS_BASELINE,
}

# Minimum length and pattern for a valid cited paper ID
_MIN_CITED_ID_LEN = 5


def _is_valid_cited_id(cited_id: str) -> bool:
    """Check if a cited paper ID looks like a real paper reference.

    Rejects: empty strings, single words, short fragments, purely numeric.
    Accepts: paper IDs (alphanumeric hashes >=8 chars), full titles (>=5 chars with spaces).
    """
    if not cited_id or not cited_id.strip():
        return False
    cited_id = cited_id.strip()
    if len(cited_id) < _MIN_CITED_ID_LEN:
        return False
    # Must contain at least one letter
    if not any(c.isalpha() for c in cited_id):
        return False
    # Single word without spaces is suspicious unless it's a hash-like ID (>=8 chars)
    if ' ' not in cited_id and len(cited_id) < 8:
        return False
    return True


class CitationEvolutionBuilder:
    """Citation relation + technology evolution chain builder."""

    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def classify_citations(self, parsed_paper: ParsedPaper) -> list[CitationEdge]:
        """
        Classify all citations in a paper by intent.

        Steps:
        1. Extract citation contexts from each section
        2. Batch-classify citation intents using LLM
        3. Generate typed citation edges
        """
        # Step 1: Extract citation contexts per section
        citation_contexts = self._extract_citation_contexts(parsed_paper)

        if not citation_contexts:
            logger.info("No citations found in paper %s", parsed_paper.paper_id)
            return []

        # Step 2: Batch classify
        edges: list[CitationEdge] = []

        # Process in batches of ~10 citations per LLM call
        batch_size = 10
        for i in range(0, len(citation_contexts), batch_size):
            batch = citation_contexts[i : i + batch_size]
            batch_edges = await self._classify_batch(batch, parsed_paper.paper_id)
            edges.extend(batch_edges)

        logger.info(
            "Classified %d citations for paper %s",
            len(edges), parsed_paper.paper_id,
        )
        return edges

    def _extract_citation_contexts(
        self, parsed_paper: ParsedPaper
    ) -> list[dict[str, Any]]:
        """Extract citation contexts from all sections."""
        contexts: list[dict[str, Any]] = []

        # Citation patterns: [1], [1,2], (Author et al., 2023), Author et al. (2023)
        citation_pattern = re.compile(
            r"(?:\[(\d+(?:,\s*\d+)*)\])"  # Numbered: [1], [1,2,3]
            r"|(?:\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,?\s*\d{4}(?:;\s*[A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,?\s*\d{4})*)\))"  # Author (Year)
        )

        for section_name, section_text in parsed_paper.sections.items():
            if not section_text:
                continue

            sentences = self._split_sentences(section_text)
            for sentence in sentences:
                matches = citation_pattern.finditer(sentence)
                for match in matches:
                    ref_key = match.group(0)
                    # Find the referenced paper info
                    ref_info = self._find_reference(ref_key, parsed_paper.references)

                    contexts.append({
                        "ref_key": ref_key,
                        "section": section_name,
                        "context": sentence.strip(),
                        "cited_title": ref_info.title if ref_info else "",
                        "cited_year": ref_info.year if ref_info else None,
                    })

        return contexts

    async def _classify_batch(
        self, batch: list[dict[str, Any]], citing_paper_id: str
    ) -> list[CitationEdge]:
        """Classify a batch of citation contexts using LLM."""
        edges: list[CitationEdge] = []

        # Format citations for the batch prompt
        citations_for_prompt = []
        for ctx in batch:
            citations_for_prompt.append({
                "ref_key": ctx["ref_key"],
                "section": ctx["section"],
                "context": ctx["context"][:300],  # Truncate long contexts
                "cited_title": ctx.get("cited_title", ""),
            })

        prompt = render_prompt(
            BATCH_CITATION_PROMPT,
            citations_json=json.dumps(citations_for_prompt, indent=2, ensure_ascii=False)
        )

        try:
            result = await self._llm.complete_json(prompt, system_prompt=SYSTEM_PROMPT)
            classifications = result.get("classifications", [])

            for i, cls in enumerate(classifications):
                if i >= len(batch):
                    break

                intent_str = cls.get("intent", "CITES_AS_COMPARISON").upper()
                intent = _INTENT_MAP.get(intent_str, CitationIntent.CITES_AS_COMPARISON)
                confidence = float(cls.get("confidence", 0.7))

                # Generate a cited_paper_id from ref_key or title
                cited_id = batch[i].get("cited_title", batch[i]["ref_key"])

                if not _is_valid_cited_id(cited_id):
                    logger.debug("Skipping invalid cited_id: %r", cited_id)
                    continue

                edges.append(CitationEdge(
                    citing_paper_id=citing_paper_id,
                    cited_paper_id=cited_id,
                    intent=intent,
                    context=batch[i]["context"],
                    section=batch[i]["section"],
                    confidence=confidence,
                ))

        except Exception as e:
            logger.warning("Batch citation classification failed: %s", e)
            # Fallback: use section-based heuristic
            for ctx in batch:
                section = ctx["section"]
                cited_id = ctx.get("cited_title", ctx["ref_key"])
                if not _is_valid_cited_id(cited_id):
                    continue
                intent = _SECTION_INTENT_PRIOR.get(section, CitationIntent.CITES_AS_COMPARISON)
                edges.append(CitationEdge(
                    citing_paper_id=citing_paper_id,
                    cited_paper_id=cited_id,
                    intent=intent,
                    context=ctx["context"],
                    section=section,
                    confidence=0.5,
                ))

        return edges

    async def build_evolution_chains(
        self, methods_with_papers: list[dict[str, Any]]
    ) -> list[EvolutionChain]:
        """
        Build technology evolution chains from method data.

        Args:
            methods_with_papers: List of dicts with keys:
                - method_id, method_name, paper_id, year, description
        """
        if len(methods_with_papers) < 2:
            return []

        # Sort by year
        sorted_methods = sorted(methods_with_papers, key=lambda m: m.get("year", 0))

        # Use LLM to detect evolution links
        methods_json = json.dumps([
            {
                "method_name": m["method_name"],
                "year": m.get("year", "unknown"),
                "description": m.get("description", "")[:200],
            }
            for m in sorted_methods
        ], indent=2, ensure_ascii=False)

        prompt = render_prompt(EVOLUTION_DETECTION_PROMPT, methods_json=methods_json)

        try:
            result = await self._llm.complete_json(prompt, system_prompt=SYSTEM_PROMPT)
            links = result.get("evolution_links", [])
        except Exception as e:
            logger.warning("Evolution detection failed: %s", e)
            return []

        # Build chains from links — with cycle detection
        chains: list[EvolutionChain] = []
        name_to_method = {m["method_name"].lower(): m for m in sorted_methods}
        # Track directed edges to detect cycles
        edge_set: set[tuple[str, str]] = set()

        for link in links:
            from_name = link.get("from_method", "").lower()
            to_name = link.get("to_method", "").lower()

            # Skip self-loops
            if from_name == to_name:
                logger.warning("Skipping self-loop evolution link: %s", from_name)
                continue

            # Skip duplicate / reverse edges (cycle detection)
            if (to_name, from_name) in edge_set:
                logger.warning(
                    "Skipping evolution link %s → %s: would create a cycle",
                    from_name, to_name,
                )
                continue

            from_method = name_to_method.get(from_name)
            to_method = name_to_method.get(to_name)

            if from_method and to_method:
                chain = EvolutionChain(
                    root_method=from_method.get("method_name", ""),
                    steps=[
                        EvolutionStep(
                            method_id=from_method.get("method_id", ""),
                            method_name=from_method.get("method_name", ""),
                            paper_id=from_method.get("paper_id", ""),
                            year=from_method.get("year", 0),
                        ),
                        EvolutionStep(
                            method_id=to_method.get("method_id", ""),
                            method_name=to_method.get("method_name", ""),
                            paper_id=to_method.get("paper_id", ""),
                            year=to_method.get("year", 0),
                            delta_description=link.get("delta_description", ""),
                        ),
                    ],
                )
                chains.append(chain)
                edge_set.add((from_name, to_name))

        return chains

    async def enrich_from_semantic_scholar(self, paper_id: str) -> list[CitationEdge]:
        """
        Supplement citation data from Semantic Scholar API.

        This fetches the paper's references and citations from the API
        to fill gaps in our full-text extraction.
        """
        import httpx

        edges: list[CitationEdge] = []
        base_url = "https://api.semanticscholar.org/graph/v1"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get references
                resp = await client.get(
                    f"{base_url}/paper/{paper_id}/references",
                    params={"fields": "title,year,citationCount"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for ref in data.get("data", []):
                        cited = ref.get("citedPaper", {})
                        if cited.get("title"):
                            edges.append(CitationEdge(
                                citing_paper_id=paper_id,
                                cited_paper_id=cited.get("paperId", cited["title"]),
                                intent=CitationIntent.CITES_AS_COMPARISON,
                                context="(from Semantic Scholar API)",
                                confidence=0.6,  # Lower confidence; intent not classified
                            ))

        except Exception as e:
            logger.warning("Semantic Scholar API call failed for %s: %s", paper_id, e)

        return edges

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _find_reference(ref_key: str, references: list[Reference]) -> Reference | None:
        """Find a Reference object matching a citation key."""
        for ref in references:
            if ref.ref_key == ref_key:
                return ref
        # Try matching by number
        num_match = re.search(r"\d+", ref_key)
        if num_match:
            idx = int(num_match.group()) - 1
            if 0 <= idx < len(references):
                return references[idx]
        return None
