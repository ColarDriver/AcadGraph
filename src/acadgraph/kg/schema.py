"""
AcadGraph Schema — All data structures for the Three-Layer Evidence-Chain KG.

Semantic Entities: METHOD, DATASET, METRIC, TASK, MODEL, FRAMEWORK, CONCEPT
Citation Relations & Evolution Chains
Argumentation Chains: PROBLEM → GAP → CORE_IDEA → CLAIM → EVIDENCE
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ============================================================================
# Common
# ============================================================================


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}" if prefix else uid


def hash_text(text: str) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for deterministic deduplication.

    Ensures that trivially equivalent names produce the same entity_id:
      'Hilbert–Schmidt' == 'Hilbert-Schmidt'
      'F1-score'        == 'F1 score'
      'Earth Mover's'   == 'Earth Mover'
    """
    s = name.strip().lower()
    # Normalize unicode dashes (en-dash, em-dash) to ASCII hyphen
    s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u2012', '-')
    # Remove apostrophes and backticks
    s = s.replace("'", "").replace("\u2019", "").replace("`", "")
    # Normalize hyphens to spaces
    s = s.replace("-", " ")
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # Simple plural normalization (but not for short words or double-s)
    if len(s) > 3 and s.endswith('s') and not s.endswith('ss'):
        s = s[:-1]
    return s


# ============================================================================
# Paper Parsing Output
# ============================================================================


@dataclass
class TableData:
    """Structured table extracted from a paper."""
    caption: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    section: str = ""  # Which section this table belongs to


@dataclass
class FigureRef:
    """Figure reference from a paper."""
    caption: str = ""
    ref_id: str = ""  # e.g., "Figure 1"
    section: str = ""


@dataclass
class Reference:
    """A parsed bibliography entry."""
    ref_key: str = ""  # In-text citation key, e.g., "[1]" or "(Smith et al., 2023)"
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    # Which sections cite this reference
    cited_in_sections: list[str] = field(default_factory=list)
    # The sentences surrounding each citation
    citation_contexts: list[str] = field(default_factory=list)


@dataclass
class SectionTreeNode:
    """Hierarchical section node (inspired by PageIndex tree structure)."""
    title: str = ""
    heading_level: int | None = None
    page_id: int | None = None
    children: list["SectionTreeNode"] = field(default_factory=list)
    # --- PageIndex integration fields ---
    node_id: str = ""                # Unique ID, e.g. "sec_001"
    content: str = ""                # Section text content
    summary: str = ""                # LLM-generated summary
    start_page: int | None = None    # Start page number
    end_page: int | None = None      # End page number
    section_key: str = ""            # Corresponding key in sections dict
    claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.node_id:
            self.node_id = generate_id("sec")


@dataclass
class ParsedPaper:
    """Output of the full-text paper parser."""
    paper_id: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    # Standard section keys: abstract, introduction, related_work,
    #   method, experiments, limitation, conclusion
    tables: list[TableData] = field(default_factory=list)
    figures: list[FigureRef] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    raw_text: str = ""
    source_path: str = ""  # Original PDF / text path
    section_tree: list[SectionTreeNode] = field(default_factory=list)
    openreview_id: str = ""
    abstract_raw: str = ""  # Original abstract (before section split)


# ============================================================================
# Semantic Entities
# ============================================================================


class EntityType(str, Enum):
    """7 types of academic entities."""
    METHOD = "METHOD"
    DATASET = "DATASET"
    METRIC = "METRIC"
    TASK = "TASK"
    MODEL = "MODEL"
    FRAMEWORK = "FRAMEWORK"
    CONCEPT = "CONCEPT"


@dataclass
class Entity:
    """A semantic entity extracted from academic papers."""
    entity_id: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    name: str = ""
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    # e.g., for METHOD: {"category": "optimization", "components": [...]}
    # e.g., for DATASET: {"domain": "NLP", "size": "10k", "task_type": "classification"}
    # e.g., for METRIC: {"higher_is_better": True, "domain": "NLP"}
    source_paper_id: str = ""  # Paper where this entity was first extracted
    source_section: str = ""  # Section where it was found
    confidence: float = 1.0

    def __post_init__(self):
        if not self.entity_id:
            # Deterministic ID: normalized name + type → same entity_id → MERGE dedup
            canonical = f"{self.entity_type.value.lower()}:{normalize_entity_name(self.name)}"
            self.entity_id = f"{self.entity_type.value.lower()}_{hash_text(canonical)}"


class RelationType(str, Enum):
    """Relations between semantic entities."""
    APPLIED_ON = "APPLIED_ON"          # Method → Task
    EVALUATED_ON = "EVALUATED_ON"      # Method → Dataset
    MEASURED_BY = "MEASURED_BY"        # Method → Metric
    USES = "USES"                      # Method → Framework
    OUTPERFORMS = "OUTPERFORMS"        # Method → Method
    PROPOSES = "PROPOSES"              # Paper → Method
    INTRODUCES = "INTRODUCES"          # Paper → Dataset
    BELONGS_TO = "BELONGS_TO"          # Entity → Domain/Category
    EXTENDS = "EXTENDS"                # Method → Method (is extension of)
    COMPONENT_OF = "COMPONENT_OF"      # Model/Framework → Model/Framework


@dataclass
class EntityRelation:
    """A directed relation between two entities."""
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.USES
    properties: dict[str, Any] = field(default_factory=dict)
    source_paper_id: str = ""
    confidence: float = 1.0


@dataclass
class EntityExtractionResult:
    """Output of entity extraction for one paper."""
    paper_id: str = ""
    entities: list[Entity] = field(default_factory=list)
    relations: list[EntityRelation] = field(default_factory=list)


# ============================================================================
# Citation Relations & Evolution Chains
# ============================================================================


class CitationIntent(str, Enum):
    """6 types of citation intent."""
    CITES_FOR_PROBLEM = "CITES_FOR_PROBLEM"        # Background/problem motivation
    CITES_AS_BASELINE = "CITES_AS_BASELINE"        # As experimental baseline
    CITES_FOR_FOUNDATION = "CITES_FOR_FOUNDATION"  # Built on top of this work
    CITES_AS_COMPARISON = "CITES_AS_COMPARISON"     # Related work comparison
    CITES_FOR_THEORY = "CITES_FOR_THEORY"          # Theoretical basis
    EVOLVES_FROM = "EVOLVES_FROM"                  # Explicit evolution


@dataclass
class CitationEdge:
    """A citation relationship with intent."""
    citing_paper_id: str = ""
    cited_paper_id: str = ""
    cited_title: str = ""  # Title of the cited work
    intent: CitationIntent = CitationIntent.CITES_AS_COMPARISON
    context: str = ""  # The sentence(s) around the citation
    section: str = ""  # Which section the citation occurs in
    confidence: float = 1.0


@dataclass
class EvolutionStep:
    """A single step in a method evolution chain."""
    method_id: str = ""
    method_name: str = ""
    paper_id: str = ""
    year: int = 0
    delta_description: str = ""  # What changed from the previous step


@dataclass
class EvolutionChain:
    """A chain showing how a method evolved over time."""
    chain_id: str = ""
    root_method: str = ""
    steps: list[EvolutionStep] = field(default_factory=list)

    def __post_init__(self):
        if not self.chain_id:
            self.chain_id = generate_id("evo")


# ============================================================================
# Argumentation Chains
# ============================================================================


class RhetoricalRole(str, Enum):
    """Rhetorical role for argumentative zoning."""
    MOTIVATION = "MOTIVATION"
    BACKGROUND = "BACKGROUND"
    CONTRIBUTION = "CONTRIBUTION"
    METHOD_DESC = "METHOD_DESC"
    RESULT = "RESULT"
    COMPARISON = "COMPARISON"
    LIMITATION = "LIMITATION"
    FUTURE = "FUTURE"


@dataclass
class ZonedParagraph:
    """A paragraph with its rhetorical role assigned."""
    text: str = ""
    section: str = ""
    role: RhetoricalRole = RhetoricalRole.BACKGROUND
    paragraph_idx: int = 0


@dataclass
class ZonedPaper:
    """Output of rhetorical zoning."""
    paper_id: str = ""
    paragraphs: list[ZonedParagraph] = field(default_factory=list)


@dataclass
class Problem:
    """Research problem node."""
    problem_id: str = ""
    description: str = ""
    scope: str = ""
    importance_signal: str = ""  # Why this problem matters
    source_paper_id: str = ""

    def __post_init__(self):
        if not self.problem_id:
            self.problem_id = generate_id("prob")


@dataclass
class Gap:
    """Research gap node."""
    gap_id: str = ""
    failure_mode: str = ""  # What specifically fails in current approaches
    constraint: str = ""  # Under what conditions it fails
    prior_methods_failing: list[str] = field(default_factory=list)
    source_paper_id: str = ""

    def __post_init__(self):
        if not self.gap_id:
            self.gap_id = generate_id("gap")


class NoveltyType(str, Enum):
    """Types of novelty for a core idea."""
    NEW_MECHANISM = "NEW_MECHANISM"
    NEW_FORMULATION = "NEW_FORMULATION"
    NEW_COMBINATION = "NEW_COMBINATION"
    NEW_APPLICATION = "NEW_APPLICATION"
    EFFICIENCY = "EFFICIENCY"
    THEORETICAL = "THEORETICAL"


@dataclass
class CoreIdea:
    """Core idea / key innovation of a paper."""
    idea_id: str = ""
    mechanism: str = ""  # What the core mechanism/idea is
    novelty_type: NoveltyType = NoveltyType.NEW_MECHANISM
    key_innovation: str = ""  # One-sentence description of what's new
    source_paper_id: str = ""

    def __post_init__(self):
        if not self.idea_id:
            self.idea_id = generate_id("idea")


class ClaimType(str, Enum):
    """Types of claims made in a paper."""
    NOVELTY = "NOVELTY"
    PERFORMANCE = "PERFORMANCE"
    ROBUSTNESS = "ROBUSTNESS"
    EFFICIENCY = "EFFICIENCY"
    THEORY = "THEORY"
    GENERALITY = "GENERALITY"


class ClaimSeverity(str, Enum):
    """Severity levels for claims — from a reviewer's perspective."""
    P0 = "P0"  # Critical claim; paper is invalid without the evidence
    P1 = "P1"  # Important claim; significantly weakens the paper if unverified
    P2 = "P2"  # Supporting claim; nice to have evidence


@dataclass
class Claim:
    """An atomic claim made by a paper."""
    claim_id: str = ""
    text: str = ""
    claim_type: ClaimType = ClaimType.PERFORMANCE
    severity: ClaimSeverity = ClaimSeverity.P1
    source_section: str = ""
    source_paper_id: str = ""
    claim_hash: str = ""  # For deduplication

    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = generate_id("claim")
        if not self.claim_hash and self.text:
            self.claim_hash = hash_text(self.text)


class EvidenceType(str, Enum):
    """Types of evidence in a paper."""
    EXPERIMENT = "EXPERIMENT"
    ABLATION = "ABLATION"
    THEOREM = "THEOREM"
    CASE_STUDY = "CASE_STUDY"
    USER_STUDY = "USER_STUDY"
    ANALYSIS = "ANALYSIS"


class SupportStrength(str, Enum):
    """How well evidence supports a claim."""
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    REFUTED = "REFUTED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass
class Evidence:
    """Evidence node — an experimental result, theorem, ablation, etc."""
    evidence_id: str = ""
    evidence_type: EvidenceType = EvidenceType.EXPERIMENT
    result_summary: str = ""
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)  # Table references
    figures: list[str] = field(default_factory=list)  # Figure references
    numeric_results: dict[str, Any] = field(default_factory=dict)
    source_paper_id: str = ""

    def __post_init__(self):
        if not self.evidence_id:
            self.evidence_id = generate_id("evid")


@dataclass
class Baseline:
    """Baseline comparison in experiments."""
    baseline_id: str = ""
    method_name: str = ""
    paper_ref: str = ""  # Reference to the baseline's paper
    performance: dict[str, Any] = field(default_factory=dict)
    source_paper_id: str = ""

    def __post_init__(self):
        if not self.baseline_id:
            self.baseline_id = generate_id("base")


@dataclass
class Limitation:
    """Limitation node."""
    limitation_id: str = ""
    text: str = ""
    scope: str = ""
    acknowledged_by_author: bool = True
    source_paper_id: str = ""

    def __post_init__(self):
        if not self.limitation_id:
            self.limitation_id = generate_id("lim")


@dataclass
class ClaimEvidenceLink:
    """Link between a Claim and its Evidence."""
    claim_id: str = ""
    evidence_id: str = ""
    strength: SupportStrength = SupportStrength.FULL
    explanation: str = ""


@dataclass
class ArgumentationGraph:
    """Full argumentation graph for one paper."""
    paper_id: str = ""
    problems: list[Problem] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    core_ideas: list[CoreIdea] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    baselines: list[Baseline] = field(default_factory=list)
    limitations: list[Limitation] = field(default_factory=list)
    claim_evidence_links: list[ClaimEvidenceLink] = field(default_factory=list)


# ============================================================================
# Query Result Types
# ============================================================================


@dataclass
class ClaimEvidenceLedger:
    """A structured view of a paper's claims and their evidence support."""
    paper_id: str = ""
    entries: list[ClaimEvidenceLedgerEntry] = field(default_factory=list)

    @property
    def unsupported_p0(self) -> list[ClaimEvidenceLedgerEntry]:
        """Return P0 claims without full support — reviewer red flags."""
        return [
            e for e in self.entries
            if _coerce_claim_severity(e.severity) == ClaimSeverity.P0
            and _coerce_support_strength(e.support_status) != SupportStrength.FULL
        ]


@dataclass
class ClaimEvidenceLedgerEntry:
    """One row in the Claim-Evidence Ledger."""
    claim_text: str = ""
    claim_type: ClaimType = ClaimType.PERFORMANCE
    severity: ClaimSeverity = ClaimSeverity.P1
    required_evidence: str = ""  # What evidence is needed
    actual_evidence: list[str] = field(default_factory=list)  # Evidence found
    support_status: SupportStrength = SupportStrength.UNVERIFIABLE

    def __post_init__(self):
        # Normalize legacy string values loaded from storage into enum values.
        self.claim_type = _coerce_claim_type(self.claim_type)
        self.severity = _coerce_claim_severity(self.severity)
        self.support_status = _coerce_support_strength(self.support_status)


def _coerce_claim_type(value: ClaimType | str) -> ClaimType:
    if isinstance(value, ClaimType):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        try:
            return ClaimType(normalized)
        except ValueError:
            return ClaimType.PERFORMANCE
    return ClaimType.PERFORMANCE


def _coerce_claim_severity(value: ClaimSeverity | str) -> ClaimSeverity:
    if isinstance(value, ClaimSeverity):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        try:
            return ClaimSeverity(normalized)
        except ValueError:
            return ClaimSeverity.P1
    return ClaimSeverity.P1


def _coerce_support_strength(value: SupportStrength | str) -> SupportStrength:
    if isinstance(value, SupportStrength):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        try:
            return SupportStrength(normalized)
        except ValueError:
            return SupportStrength.UNVERIFIABLE
    return SupportStrength.UNVERIFIABLE


@dataclass
class CompetitionSpace:
    """The competitive space around an idea."""
    query_idea: str = ""
    nearest_papers: list[dict[str, Any]] = field(default_factory=list)
    overlap_dimensions: list[str] = field(default_factory=list)


@dataclass
class NoveltyMap:
    """Structural alignment showing novelty of an idea vs existing work."""
    dimensions: list[str] = field(default_factory=list)
    # Each dim: Problem, Setting, Assumption, Mechanism, Supervision, Metric, FailureMode
    idea_projection: dict[str, str] = field(default_factory=dict)
    competitor_projections: list[dict[str, str]] = field(default_factory=list)
    unique_dimensions: list[str] = field(default_factory=list)  # Where the idea is novel


@dataclass
class GapStatement:
    """A falsifiable gap statement generated by the system."""
    statement: str = ""
    # "现有方法在 [S] 下可以解决 [P]，但在 [F] 下仍然失败,
    #  因为它们缺少 [M]；据检索，尚无方法同时满足 [A, B, C]。"
    problem: str = ""
    setting: str = ""
    failure_constraint: str = ""
    missing_mechanism: str = ""
    novelty_checklist: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)


@dataclass
class EvolutionTimeline:
    """Timeline showing how a method evolved."""
    method_name: str = ""
    steps: list[EvolutionStep] = field(default_factory=list)


@dataclass
class InnovationPath:
    """A cross-method innovation path from Method→Gap→CoreIdea."""
    source_methods: list[str] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    addressing_ideas: list[dict[str, Any]] = field(default_factory=list)
    unaddressed_gaps: list[dict[str, Any]] = field(default_factory=list)
    suggested_combination: str = ""
    evidence_support: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CrossDomainBridge:
    """Cross-domain method bridge discovery result."""
    method_a: str = ""
    method_b: str = ""
    shared_concepts: list[dict[str, Any]] = field(default_factory=list)
    bridge_papers: list[dict[str, Any]] = field(default_factory=list)
    combination_novelty: str = ""


@dataclass
class ComponentEvidence:
    """Evidence strength analysis for one method component."""
    method_name: str = ""
    paper_count: int = 0
    claim_count: int = 0
    evidence_count: int = 0
    avg_support_strength: float = 0.0
    unsupported_claims: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildResult:
    """Result of adding one paper to the KG."""
    paper_id: str = ""
    entities_added: int = 0
    relations_added: int = 0
    citations_added: int = 0
    claims_added: int = 0
    evidences_added: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class BatchBuildResult:
    """Result of batch paper addition."""
    total_papers: int = 0
    successful: int = 0
    failed: int = 0
    results: list[BuildResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0


# ============================================================================
# Input Types
# ============================================================================


@dataclass
class PaperSource:
    """Input specification for a paper to be processed."""
    paper_id: str = ""
    pdf_path: str | None = None
    text: str | None = None  # Pre-extracted text (e.g., from OpenReview HTML)
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    content_meta: dict | None = None  # Structured ToC from JSONL
    abstract: str = ""  # Pre-extracted abstract
    related_notes_raw: str = ""  # Raw peer review data

    def __post_init__(self):
        if not self.paper_id:
            if self.title:
                self.paper_id = hash_text(self.title)
            else:
                self.paper_id = generate_id("paper")
