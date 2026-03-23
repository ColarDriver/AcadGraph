"""
Argumentation Chain Extraction (Core Innovation).

Three-pass pipeline:
  Pass 1: Argumentative Zoning (classify paragraph rhetorical roles)
  Pass 2: Schema-Constrained Extraction (PROBLEM, GAP, CORE_IDEA, CLAIMs)
  Pass 3: Evidence-Claim Linking (EVIDENCE nodes + support strength assessment)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from acadgraph.kg.prompts.loader import load_prompt, render_prompt
from acadgraph.kg.schema import (
    ArgumentationGraph,
    Baseline,
    Claim,
    ClaimEvidenceLink,
    ClaimSeverity,
    ClaimType,
    CoreIdea,
    Evidence,
    EvidenceType,
    Gap,
    Limitation,
    NoveltyType,
    ParsedPaper,
    Problem,
    RhetoricalRole,
    SupportStrength,
    ZonedPaper,
    ZonedParagraph,
    generate_id,
)
from acadgraph.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Load prompts from Markdown files
EVIDENCE_LINKING_PROMPT = load_prompt("argumentation", "evidence_linking")
EVIDENCE_SYSTEM = load_prompt("argumentation", "evidence_system")
SCHEMA_EXTRACTION_PROMPT = load_prompt("argumentation", "argumentation_schema")
SCHEMA_SYSTEM = load_prompt("argumentation", "argumentation_schema_system")
ZONING_PROMPT = load_prompt("argumentation", "zoning")
ZONING_SYSTEM = load_prompt("argumentation", "zoning_system")

_CLAIM_TYPE_MAP: dict[str, ClaimType] = {
    "NOVELTY": ClaimType.NOVELTY,
    "PERFORMANCE": ClaimType.PERFORMANCE,
    "ROBUSTNESS": ClaimType.ROBUSTNESS,
    "EFFICIENCY": ClaimType.EFFICIENCY,
    "THEORY": ClaimType.THEORY,
    "GENERALITY": ClaimType.GENERALITY,
}

_SEVERITY_MAP: dict[str, ClaimSeverity] = {
    "P0": ClaimSeverity.P0,
    "P1": ClaimSeverity.P1,
    "P2": ClaimSeverity.P2,
}

_NOVELTY_MAP: dict[str, NoveltyType] = {
    "NEW_MECHANISM": NoveltyType.NEW_MECHANISM,
    "NEW_FORMULATION": NoveltyType.NEW_FORMULATION,
    "NEW_COMBINATION": NoveltyType.NEW_COMBINATION,
    "NEW_APPLICATION": NoveltyType.NEW_APPLICATION,
    "EFFICIENCY": NoveltyType.EFFICIENCY,
    "THEORETICAL": NoveltyType.THEORETICAL,
}

_EVIDENCE_TYPE_MAP: dict[str, EvidenceType] = {
    "EXPERIMENT": EvidenceType.EXPERIMENT,
    "ABLATION": EvidenceType.ABLATION,
    "THEOREM": EvidenceType.THEOREM,
    "CASE_STUDY": EvidenceType.CASE_STUDY,
    "USER_STUDY": EvidenceType.USER_STUDY,
    "ANALYSIS": EvidenceType.ANALYSIS,
}

_STRENGTH_MAP: dict[str, SupportStrength] = {
    "FULL": SupportStrength.FULL,
    "PARTIAL": SupportStrength.PARTIAL,
    "REFUTED": SupportStrength.REFUTED,
    "UNVERIFIABLE": SupportStrength.UNVERIFIABLE,
}

_ROLE_MAP: dict[str, RhetoricalRole] = {
    "MOTIVATION": RhetoricalRole.MOTIVATION,
    "BACKGROUND": RhetoricalRole.BACKGROUND,
    "CONTRIBUTION": RhetoricalRole.CONTRIBUTION,
    "METHOD_DESC": RhetoricalRole.METHOD_DESC,
    "RESULT": RhetoricalRole.RESULT,
    "COMPARISON": RhetoricalRole.COMPARISON,
    "LIMITATION": RhetoricalRole.LIMITATION,
    "FUTURE": RhetoricalRole.FUTURE,
}


# ========================================================================
# Pydantic Response Validation Models
# ========================================================================


class _ProblemResponse(BaseModel):
    description: str = ""
    scope: str = ""
    importance_signal: str = ""
    gap_ids: list[int] = Field(default_factory=list, description="Indices into the gaps list this problem maps to")


class _GapResponse(BaseModel):
    failure_mode: str = ""
    constraint: str = ""
    prior_methods_failing: list[str] = Field(default_factory=list)
    core_idea_ids: list[int] = Field(default_factory=list, description="Indices into the core_ideas list")


class _CoreIdeaResponse(BaseModel):
    mechanism: str = ""
    novelty_type: str = "NEW_MECHANISM"
    key_innovation: str = ""
    claim_ids: list[int] = Field(default_factory=list, description="Indices into the claims list")


class _ClaimResponse(BaseModel):
    text: str = ""
    type: str = "PERFORMANCE"
    severity: str = "P1"
    source_section: str = ""


class _SchemaExtractionResponse(BaseModel):
    """Validated LLM response for Pass 2 schema extraction."""
    problems: list[_ProblemResponse] = Field(default_factory=list)
    problem: _ProblemResponse | None = None  # backward compat: single problem
    gaps: list[_GapResponse] = Field(default_factory=list)
    gap: _GapResponse | None = None  # backward compat: single gap
    core_ideas: list[_CoreIdeaResponse] = Field(default_factory=list)
    core_idea: _CoreIdeaResponse | None = None  # backward compat: single core_idea
    claims: list[_ClaimResponse] = Field(default_factory=list)


class _EvidenceLinkResponse(BaseModel):
    claim_index: int = -1
    evidences: list[dict[str, Any]] = Field(default_factory=list)


class _NumericConsistencyIssue(BaseModel):
    claim_text: str = ""
    claimed_value: str | None = None
    table_value: str | None = None
    consistent: bool = True


class _EvidenceLinkingResponse(BaseModel):
    """Validated LLM response for Pass 3 evidence linking."""
    evidence_links: list[_EvidenceLinkResponse] = Field(default_factory=list)
    baselines: list[dict[str, Any]] = Field(default_factory=list)
    numeric_consistency_issues: list[_NumericConsistencyIssue] = Field(default_factory=list)


class ArgumentationExtractor:
    """Argumentation chain extractor — 3-Pass Pipeline."""

    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def extract(self, parsed_paper: ParsedPaper) -> ArgumentationGraph:
        """
        Full 3-pass argumentation extraction pipeline.

        Pass 1: Rhetorical zoning
        Pass 2: Schema-constrained extraction (PROBLEM, GAP, CORE_IDEA, CLAIMs)
        Pass 3: Evidence-Claim linking
        """
        paper_id = parsed_paper.paper_id

        # Pass 1: Argumentative Zoning
        logger.info("[Pass 1] Argumentative zoning for paper %s", paper_id)
        zoned = await self.pass1_argumentative_zoning(parsed_paper)

        # Pass 2: Schema-Constrained Extraction
        logger.info("[Pass 2] Schema extraction for paper %s", paper_id)
        arg_graph = await self.pass2_schema_extraction(zoned, parsed_paper)

        # Pass 3: Evidence-Claim Linking
        logger.info("[Pass 3] Evidence linking for paper %s", paper_id)
        arg_graph = await self.pass3_evidence_linking(arg_graph, parsed_paper)

        logger.info(
            "Argumentation extraction complete for %s: "
            "%d problems, %d gaps, %d claims, %d evidences",
            paper_id,
            len(arg_graph.problems),
            len(arg_graph.gaps),
            len(arg_graph.claims),
            len(arg_graph.evidences),
        )

        return arg_graph

    # ========================================================================
    # Pass 1: Argumentative Zoning
    # ========================================================================

    async def pass1_argumentative_zoning(self, parsed_paper: ParsedPaper) -> ZonedPaper:
        """
        Pass 1: Classify each paragraph's rhetorical role.

        Roles: MOTIVATION, BACKGROUND, CONTRIBUTION, METHOD_DESC,
               RESULT, COMPARISON, LIMITATION, FUTURE
        """
        paragraphs: list[ZonedParagraph] = []

        for section_name, section_text in parsed_paper.sections.items():
            if not section_text or len(section_text.strip()) < 30:
                continue

            # Split into paragraphs
            raw_paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]

            # Use LLM to classify roles
            # Truncate to avoid token limits
            text_for_prompt = "\n\n".join(raw_paragraphs[:20])
            if len(text_for_prompt) > 6000:
                text_for_prompt = text_for_prompt[:6000]

            prompt = render_prompt(ZONING_PROMPT, text=text_for_prompt)

            try:
                result = await self._llm.complete_json(prompt, system_prompt=ZONING_SYSTEM)
                classified = result.get("paragraphs", [])

                for i, raw_p in enumerate(raw_paragraphs):
                    role = RhetoricalRole.BACKGROUND  # default
                    if i < len(classified):
                        role_str = classified[i].get("role", "BACKGROUND").upper()
                        role = _ROLE_MAP.get(role_str, RhetoricalRole.BACKGROUND)

                    paragraphs.append(ZonedParagraph(
                        text=raw_p,
                        section=section_name,
                        role=role,
                        paragraph_idx=len(paragraphs),
                    ))

            except Exception as e:
                logger.warning("Zoning failed for section '%s': %s", section_name, e)
                # Fallback: assign role based on section name
                default_role = self._section_to_default_role(section_name)
                for raw_p in raw_paragraphs:
                    paragraphs.append(ZonedParagraph(
                        text=raw_p,
                        section=section_name,
                        role=default_role,
                        paragraph_idx=len(paragraphs),
                    ))

        return ZonedPaper(paper_id=parsed_paper.paper_id, paragraphs=paragraphs)

    # ========================================================================
    # Pass 2: Schema-Constrained Extraction
    # ========================================================================

    async def pass2_schema_extraction(
        self, zoned: ZonedPaper, parsed_paper: ParsedPaper
    ) -> ArgumentationGraph:
        """
        Pass 2: Extract PROBLEM, GAP, CORE_IDEA, and atomic CLAIMs.

        Uses zoned paragraphs to focus extraction on the right parts:
        - MOTIVATION/BACKGROUND paragraphs → PROBLEM
        - CONTRIBUTION paragraphs → GAP, CORE_IDEA
        - CONTRIBUTION/RESULT paragraphs → CLAIMs
        """
        paper_id = parsed_paper.paper_id

        # Gather section summaries for the prompt
        abstract = parsed_paper.sections.get("abstract", "")
        introduction = parsed_paper.sections.get("introduction", "")
        method = parsed_paper.sections.get("method", "")
        conclusion = parsed_paper.sections.get("conclusion", "")

        # Truncate long sections
        def trunc(text: str, max_len: int = 3000) -> str:
            return text[:max_len] if len(text) > max_len else text

        prompt = render_prompt(
            SCHEMA_EXTRACTION_PROMPT,
            title=parsed_paper.title,
            abstract=trunc(abstract, 1500),
            introduction=trunc(introduction, 3000),
            method_summary=trunc(method, 2000),
            conclusion=trunc(conclusion, 1500),
        )

        try:
            raw_result = await self._llm.complete_json(prompt, system_prompt=SCHEMA_SYSTEM)
        except Exception as e:
            logger.error("Schema extraction failed for %s: %s", paper_id, e)
            return ArgumentationGraph(paper_id=paper_id)

        # Validate with Pydantic
        try:
            result = _SchemaExtractionResponse.model_validate(raw_result)
        except ValidationError as e:
            logger.warning("Schema response validation failed for %s, using raw: %s", paper_id, e)
            result = _SchemaExtractionResponse()  # fallback to empty

        # Parse PROBLEMs — support both single and multiple
        problems: list[Problem] = []
        all_problems = list(result.problems)
        if result.problem and not all_problems:
            all_problems.append(result.problem)
        for prob_data in all_problems:
            problems.append(Problem(
                description=prob_data.description,
                scope=prob_data.scope,
                importance_signal=prob_data.importance_signal,
                source_paper_id=paper_id,
            ))

        # Parse GAPs — support both single and multiple
        gaps: list[Gap] = []
        all_gaps = list(result.gaps)
        if result.gap and not all_gaps:
            all_gaps.append(result.gap)
        for gap_data in all_gaps:
            gaps.append(Gap(
                failure_mode=gap_data.failure_mode,
                constraint=gap_data.constraint,
                prior_methods_failing=gap_data.prior_methods_failing,
                source_paper_id=paper_id,
            ))

        # Parse CORE_IDEAs — support both single and multiple
        core_ideas: list[CoreIdea] = []
        all_ideas = list(result.core_ideas)
        if result.core_idea and not all_ideas:
            all_ideas.append(result.core_idea)
        for idea_data in all_ideas:
            novelty_str = idea_data.novelty_type.upper()
            core_ideas.append(CoreIdea(
                mechanism=idea_data.mechanism,
                novelty_type=_NOVELTY_MAP.get(novelty_str, NoveltyType.NEW_MECHANISM),
                key_innovation=idea_data.key_innovation,
                source_paper_id=paper_id,
            ))

        # Parse CLAIMs
        claims: list[Claim] = []
        for raw_claim in result.claims:
            claim_type_str = raw_claim.type.upper()
            severity_str = raw_claim.severity.upper()
            claims.append(Claim(
                text=raw_claim.text,
                claim_type=_CLAIM_TYPE_MAP.get(claim_type_str, ClaimType.PERFORMANCE),
                severity=_SEVERITY_MAP.get(severity_str, ClaimSeverity.P1),
                source_section=raw_claim.source_section,
                source_paper_id=paper_id,
            ))

        # Build precise mappings (Problem→Gap, Gap→CoreIdea, CoreIdea→Claim)
        problem_gap_map: dict[int, list[int]] = {}
        for prob_idx, prob_data in enumerate(all_problems):
            if prob_data.gap_ids:
                problem_gap_map[prob_idx] = [
                    gid for gid in prob_data.gap_ids if 0 <= gid < len(gaps)
                ]

        gap_idea_map: dict[int, list[int]] = {}
        for gap_idx, gap_data in enumerate(all_gaps):
            if gap_data.core_idea_ids:
                gap_idea_map[gap_idx] = [
                    iid for iid in gap_data.core_idea_ids if 0 <= iid < len(core_ideas)
                ]

        idea_claim_map: dict[int, list[int]] = {}
        for idea_idx, idea_data in enumerate(all_ideas):
            if idea_data.claim_ids:
                idea_claim_map[idea_idx] = [
                    cid for cid in idea_data.claim_ids if 0 <= cid < len(claims)
                ]

        # Extract limitations from zoned paragraphs
        limitations: list[Limitation] = []
        for para in zoned.paragraphs:
            if para.role == RhetoricalRole.LIMITATION and len(para.text) > 30:
                limitations.append(Limitation(
                    text=para.text[:500],
                    scope=para.section,
                    acknowledged_by_author=True,
                    source_paper_id=paper_id,
                ))

        arg_graph = ArgumentationGraph(
            paper_id=paper_id,
            problems=problems,
            gaps=gaps,
            core_ideas=core_ideas,
            claims=claims,
            limitations=limitations,
        )

        # Attach precise mapping metadata for storage layer
        arg_graph._problem_gap_map = problem_gap_map  # type: ignore[attr-defined]
        arg_graph._gap_idea_map = gap_idea_map  # type: ignore[attr-defined]
        arg_graph._idea_claim_map = idea_claim_map  # type: ignore[attr-defined]

        return arg_graph

    # ========================================================================
    # Pass 3: Evidence-Claim Linking
    # ========================================================================

    async def pass3_evidence_linking(
        self,
        argumentation: ArgumentationGraph,
        parsed_paper: ParsedPaper,
    ) -> ArgumentationGraph:
        """
        Pass 3: Link evidence to claims.

        From the Experiments section, extract EVIDENCE nodes and assess
        how well they support each CLAIM (FULL/PARTIAL/REFUTED/UNVERIFIABLE).
        """
        if not argumentation.claims:
            logger.info("No claims to link evidence for paper %s", argumentation.paper_id)
            return argumentation

        experiments_text = parsed_paper.sections.get("experiments", "")
        if not experiments_text:
            # Try alternative section names
            experiments_text = parsed_paper.sections.get("results", "")
            if not experiments_text:
                experiments_text = parsed_paper.sections.get("evaluation", "")

        if not experiments_text:
            logger.warning("No experiments section found for paper %s", argumentation.paper_id)
            return argumentation

        # Format tables for the prompt
        tables_text = ""
        for table in parsed_paper.tables:
            tables_text += f"\n### {table.caption}\n"
            if table.headers:
                tables_text += "| " + " | ".join(table.headers) + " |\n"
                tables_text += "| " + " | ".join(["---"] * len(table.headers)) + " |\n"
                for row in table.rows[:10]:  # Limit rows
                    tables_text += "| " + " | ".join(row) + " |\n"

        # Format claims for the prompt
        claims_json = json.dumps([
            {
                "index": i,
                "text": c.text,
                "type": c.claim_type.value,
                "severity": c.severity.value,
            }
            for i, c in enumerate(argumentation.claims)
        ], indent=2, ensure_ascii=False)

        prompt = render_prompt(
            EVIDENCE_LINKING_PROMPT,
            title=parsed_paper.title,
            claims_json=claims_json,
            experiments_text=experiments_text[:5000],
            tables_text=tables_text[:3000] if tables_text else "(no tables extracted)",
        )

        try:
            raw_result = await self._llm.complete_json(prompt, system_prompt=EVIDENCE_SYSTEM)
        except Exception as e:
            logger.error("Evidence linking failed for %s: %s", argumentation.paper_id, e)
            return argumentation

        # Validate with Pydantic
        try:
            result = _EvidenceLinkingResponse.model_validate(raw_result)
        except ValidationError as e:
            logger.warning("Evidence linking response validation failed for %s: %s", argumentation.paper_id, e)
            result = _EvidenceLinkingResponse()

        # Parse evidence links
        paper_id = argumentation.paper_id
        evidences: list[Evidence] = []
        links: list[ClaimEvidenceLink] = []

        for ev_link in result.evidence_links:
            claim_idx = ev_link.claim_index
            if claim_idx < 0 or claim_idx >= len(argumentation.claims):
                continue

            claim = argumentation.claims[claim_idx]

            for raw_ev in ev_link.evidences:
                ev_type_str = raw_ev.get("evidence_type", "EXPERIMENT").upper()
                strength_str = raw_ev.get("support_strength", "UNVERIFIABLE").upper()

                evidence = Evidence(
                    evidence_type=_EVIDENCE_TYPE_MAP.get(ev_type_str, EvidenceType.EXPERIMENT),
                    result_summary=raw_ev.get("result_summary", ""),
                    datasets=raw_ev.get("datasets", []),
                    metrics=raw_ev.get("metrics", []),
                    tables=raw_ev.get("tables", []),
                    figures=raw_ev.get("figures", []),
                    numeric_results=raw_ev.get("numeric_results", {}),
                    source_paper_id=paper_id,
                )
                evidences.append(evidence)

                links.append(ClaimEvidenceLink(
                    claim_id=claim.claim_id,
                    evidence_id=evidence.evidence_id,
                    strength=_STRENGTH_MAP.get(strength_str, SupportStrength.UNVERIFIABLE),
                    explanation=raw_ev.get("explanation", ""),
                ))

        # Parse baselines
        baselines: list[Baseline] = []
        for raw_bl in result.baselines:
            baselines.append(Baseline(
                method_name=raw_bl.get("method_name", ""),
                paper_ref=raw_bl.get("paper_ref", ""),
                performance=raw_bl.get("performance", {}),
                source_paper_id=paper_id,
            ))

        # Update argumentation graph
        argumentation.evidences = evidences
        argumentation.claim_evidence_links = links
        argumentation.baselines = baselines

        # Collect numeric consistency issues and attach to Evidence nodes
        consistency_issues: list[dict[str, Any]] = []
        for issue in result.numeric_consistency_issues:
            if not issue.consistent:
                logger.warning(
                    "Numeric inconsistency in paper %s: claim='%s', "
                    "claimed=%s, table=%s",
                    paper_id,
                    issue.claim_text[:60],
                    issue.claimed_value,
                    issue.table_value,
                )
                consistency_issues.append({
                    "claim_text": issue.claim_text,
                    "claimed_value": issue.claimed_value,
                    "table_value": issue.table_value,
                })

        # Attach consistency issues to evidences for persistence
        if consistency_issues:
            argumentation.numeric_consistency_issues = consistency_issues  # type: ignore[attr-defined]

        return argumentation

    # ========================================================================
    # Helpers
    # ========================================================================

    @staticmethod
    def _section_to_default_role(section_name: str) -> RhetoricalRole:
        """Map section name to a default rhetorical role."""
        mapping = {
            "abstract": RhetoricalRole.CONTRIBUTION,
            "introduction": RhetoricalRole.MOTIVATION,
            "related_work": RhetoricalRole.BACKGROUND,
            "method": RhetoricalRole.METHOD_DESC,
            "experiments": RhetoricalRole.RESULT,
            "limitation": RhetoricalRole.LIMITATION,
            "conclusion": RhetoricalRole.FUTURE,
        }
        return mapping.get(section_name, RhetoricalRole.BACKGROUND)
