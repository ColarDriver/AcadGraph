"""Tests for schema.py — ensure all data structures work correctly."""

import pytest

from acadgraph.kg.schema import (
    ArgumentationGraph,
    Claim,
    ClaimEvidenceLedger,
    ClaimEvidenceLedgerEntry,
    ClaimSeverity,
    ClaimType,
    CoreIdea,
    Entity,
    EntityType,
    Evidence,
    EvidenceType,
    Gap,
    NoveltyType,
    ParsedPaper,
    Problem,
    SupportStrength,
    generate_id,
    hash_text,
)


def test_generate_id():
    """IDs should be unique and include prefix."""
    id1 = generate_id("test")
    id2 = generate_id("test")
    assert id1 != id2
    assert id1.startswith("test_")
    assert len(id1) > 5


def test_generate_id_no_prefix():
    """IDs without prefix should still work."""
    id1 = generate_id()
    assert "_" not in id1
    assert len(id1) == 12


def test_hash_text():
    """Same text should produce same hash, case insensitive."""
    h1 = hash_text("Hello World")
    h2 = hash_text("hello world")
    h3 = hash_text("  hello world  ")
    assert h1 == h2 == h3


def test_entity_auto_id():
    """Entity should auto-generate ID if not provided."""
    e = Entity(name="BERT", entity_type=EntityType.MODEL)
    assert e.entity_id.startswith("model_")
    assert e.name == "BERT"


def test_entity_with_id():
    """Entity should keep provided ID."""
    e = Entity(entity_id="custom_id", name="GPT-4", entity_type=EntityType.MODEL)
    assert e.entity_id == "custom_id"


def test_claim_hash():
    """Claim should auto-compute hash from text."""
    c = Claim(text="Our method achieves SOTA", claim_type=ClaimType.PERFORMANCE)
    assert c.claim_hash != ""
    assert c.claim_id.startswith("claim_")


def test_parsed_paper():
    """ParsedPaper should hold sections."""
    paper = ParsedPaper(
        paper_id="test_001",
        title="Test Paper",
        sections={"abstract": "This paper...", "method": "We propose..."},
    )
    assert len(paper.sections) == 2
    assert "abstract" in paper.sections


def test_argumentation_graph():
    """ArgumentationGraph should hold all Layer 3 components."""
    graph = ArgumentationGraph(
        paper_id="test_001",
        problems=[Problem(description="Problem X")],
        gaps=[Gap(failure_mode="Methods fail at Y")],
        core_ideas=[CoreIdea(mechanism="Novel approach Z", novelty_type=NoveltyType.NEW_MECHANISM)],
        claims=[
            Claim(text="We achieve SOTA", claim_type=ClaimType.PERFORMANCE, severity=ClaimSeverity.P0),
            Claim(text="Our method is efficient", claim_type=ClaimType.EFFICIENCY, severity=ClaimSeverity.P1),
        ],
        evidences=[
            Evidence(evidence_type=EvidenceType.EXPERIMENT, result_summary="95% accuracy on ImageNet"),
        ],
    )
    assert len(graph.problems) == 1
    assert len(graph.claims) == 2
    assert len(graph.evidences) == 1


def test_claim_evidence_ledger():
    """Ledger should correctly identify unsupported P0 claims."""
    ledger = ClaimEvidenceLedger(
        paper_id="test_001",
        entries=[
            ClaimEvidenceLedgerEntry(
                claim_text="SOTA performance",
                severity=ClaimSeverity.P0,
                support_status=SupportStrength.FULL,
            ),
            ClaimEvidenceLedgerEntry(
                claim_text="Robust under distribution shift",
                severity=ClaimSeverity.P0,
                support_status=SupportStrength.PARTIAL,
            ),
            ClaimEvidenceLedgerEntry(
                claim_text="Efficient training",
                severity=ClaimSeverity.P2,
                support_status=SupportStrength.UNVERIFIABLE,
            ),
        ],
    )
    unsupported = ledger.unsupported_p0
    assert len(unsupported) == 1
    assert "Robust" in unsupported[0].claim_text


def test_entity_type_values():
    """Ensure all 7 entity types exist."""
    assert len(EntityType) == 7
    assert EntityType.METHOD.value == "METHOD"
    assert EntityType.DATASET.value == "DATASET"


def test_claim_severity_ordering():
    """P0 is more critical than P1 than P2."""
    assert ClaimSeverity.P0.value == "P0"
    assert ClaimSeverity.P1.value == "P1"
    assert ClaimSeverity.P2.value == "P2"



def test_claim_evidence_ledger_normalizes_string_enums_for_unsupported_p0():
    """unsupported_p0 should be stable even when strings/enums are mixed."""
    ledger = ClaimEvidenceLedger(
        paper_id="mix_001",
        entries=[
            ClaimEvidenceLedgerEntry(
                claim_text="Critical claim without full evidence",
                severity="P0",
                support_status="PARTIAL",
            ),
            ClaimEvidenceLedgerEntry(
                claim_text="Critical claim with full evidence",
                severity=ClaimSeverity.P0,
                support_status=SupportStrength.FULL,
            ),
            ClaimEvidenceLedgerEntry(
                claim_text="Secondary claim",
                severity="P2",
                support_status="UNVERIFIABLE",
            ),
        ],
    )

    unsupported = ledger.unsupported_p0

    assert len(unsupported) == 1
    assert unsupported[0].claim_text == "Critical claim without full evidence"
    assert unsupported[0].severity == ClaimSeverity.P0
    assert unsupported[0].support_status == SupportStrength.PARTIAL
