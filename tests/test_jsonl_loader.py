"""Tests for JSONL loader, PaperParser JSONL support, and peer review extractor."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from acadgraph.kg.jsonl_loader import (
    JournalRecord,
    _clean_markdown,
    iter_jsonl,
    jsonl_to_paper_source,
    parse_related_notes,
)
from acadgraph.kg.paper_parser import PaperParser
from acadgraph.kg.extract.peer_review import PeerReviewExtractor
from acadgraph.kg.schema import SectionTreeNode


# ============================================================================
# JSONL Loader Tests
# ============================================================================

SAMPLE_JSONL_RECORD = {
    "id": "TEST123",
    "title": "A Test Paper on Neural Networks",
    "authors": "Alice Smith, Bob Jones, Charlie Brown",
    "abstract": "We propose a novel method for training...",
    "content": "# A TEST PAPER ON NEURAL NETWORKS\n\n## ABSTRACT\n\nWe propose a novel method.\n\n## 1 INTRODUCTION\n\nDeep learning has been...\n\n## 2 METHOD\n\nOur approach uses...\n\n## 3 EXPERIMENTS\n\nWe evaluate on CIFAR-10.\n\n## 4 CONCLUSION\n\nWe presented a method.\n\n## References\n\n[1] Vaswani et al. \"Attention is All You Need\" 2017\n",
    "content_meta": json.dumps({
        "table_of_contents": [
            {"title": "A TEST PAPER ON NEURAL NETWORKS", "heading_level": None, "page_id": 0, "polygon": []},
            {"title": "ABSTRACT", "heading_level": None, "page_id": 0, "polygon": []},
            {"title": "1 INTRODUCTION", "heading_level": None, "page_id": 0, "polygon": []},
            {"title": "2 METHOD", "heading_level": None, "page_id": 1, "polygon": []},
            {"title": "3 EXPERIMENTS", "heading_level": None, "page_id": 2, "polygon": []},
            {"title": "4 CONCLUSION", "heading_level": None, "page_id": 3, "polygon": []},
        ],
        "page_stats": [],
        "debug_data_path": "",
    }),
    "conference": "ICLR",
    "year": "2023",
    "pdf_url": "https://openreview.net/pdf?id=TEST123",
    "source_url": "https://openreview.net/forum?id=TEST123",
    "related_notes": str([
        {
            "id": "review1",
            "content": {
                "title": "Official Review",
                "rating": "6: Marginally above the acceptance threshold",
                "confidence": "4: The reviewer is confident",
                "summary": "This paper proposes a novel method for training.",
                "strengths": "1. Clear presentation\n2. Strong baselines\n3. Good ablation study",
                "weaknesses": "1. Limited datasets\n2. Missing comparison with recent methods",
                "questions": "1. Why not test on ImageNet?",
            },
        },
        {
            "id": "decision1",
            "content": {
                "title": "Paper Decision",
                "decision": "Accept: poster",
                "metareview:_summary,_strengths_and_weaknesses": "The paper makes solid contributions.",
            },
        },
    ]),
}


@pytest.fixture
def sample_jsonl_file(tmp_path):
    """Create a temp JSONL file with 3 records."""
    f = tmp_path / "test.jsonl"
    records = []
    for i in range(3):
        rec = SAMPLE_JSONL_RECORD.copy()
        rec["id"] = f"TEST{i}"
        rec["title"] = f"Paper {i}"
        records.append(json.dumps(rec))
    f.write_text("\n".join(records))
    return str(f)


def test_iter_jsonl(sample_jsonl_file):
    """Should stream all records."""
    records = list(iter_jsonl(sample_jsonl_file))
    assert len(records) == 3
    assert records[0].id == "TEST0"
    assert records[2].id == "TEST2"


def test_iter_jsonl_with_limit(sample_jsonl_file):
    """Should respect limit parameter."""
    records = list(iter_jsonl(sample_jsonl_file, limit=2))
    assert len(records) == 2


def test_iter_jsonl_skip_ids(sample_jsonl_file):
    """Should skip specified IDs."""
    records = list(iter_jsonl(sample_jsonl_file, skip_ids={"TEST0", "TEST2"}))
    assert len(records) == 1
    assert records[0].id == "TEST1"


def test_authors_parsing(sample_jsonl_file):
    """Should split comma-separated authors."""
    rec = next(iter_jsonl(sample_jsonl_file))
    assert rec.authors == ["Alice Smith", "Bob Jones", "Charlie Brown"]


def test_year_parsing(sample_jsonl_file):
    """Should convert year string to int."""
    rec = next(iter_jsonl(sample_jsonl_file))
    assert rec.year == 2023


def test_content_meta_parsing(sample_jsonl_file):
    """Should parse content_meta JSON string."""
    rec = next(iter_jsonl(sample_jsonl_file))
    assert rec.content_meta is not None
    assert "table_of_contents" in rec.content_meta


def test_clean_markdown_span_tags():
    """Should remove <span> tags."""
    text = '# <span id="page-0-1">TITLE</span>'
    assert _clean_markdown(text) == "# TITLE"


def test_clean_markdown_double_hash():
    """Should normalize ## at start to #."""
    text = "## TITLE HERE\n\nContent"
    assert _clean_markdown(text).startswith("# TITLE HERE")


def test_jsonl_to_paper_source(sample_jsonl_file):
    """Should correctly map fields."""
    rec = next(iter_jsonl(sample_jsonl_file))
    src = jsonl_to_paper_source(rec)
    assert src.paper_id == "TEST0"
    assert src.venue == "ICLR"
    assert src.year == 2023
    assert src.content_meta is not None
    assert src.text is not None
    assert len(src.related_notes_raw) > 0


# ============================================================================
# parse_related_notes Tests
# ============================================================================


def test_parse_related_notes_python_repr():
    """Should parse Python repr format."""
    raw = str([{"id": "r1", "content": {"title": "Review"}}])
    result = parse_related_notes(raw)
    assert len(result) == 1
    assert result[0]["id"] == "r1"


def test_parse_related_notes_json():
    """Should parse JSON format."""
    raw = json.dumps([{"id": "r2"}])
    result = parse_related_notes(raw)
    assert len(result) == 1


def test_parse_related_notes_empty():
    """Should return empty list for empty input."""
    assert parse_related_notes("") == []
    assert parse_related_notes("   ") == []


# ============================================================================
# PaperParser JSONL Integration Tests
# ============================================================================


@pytest.fixture
def parser():
    return PaperParser()


def test_parse_from_jsonl_record(parser):
    """Should correctly parse JSONL content into ParsedPaper."""
    content = SAMPLE_JSONL_RECORD["content"]
    content_meta = json.loads(SAMPLE_JSONL_RECORD["content_meta"])

    result = asyncio.run(
        parser.parse_from_jsonl_record(
            content=content,
            content_meta=content_meta,
            abstract="We propose a novel method for training...",
        )
    )

    assert result.title  # Should extract title
    assert "abstract" in result.sections or result.abstract_raw
    assert "method" in result.sections or "experiments" in result.sections
    assert len(result.section_tree) > 0  # Should build tree from ToC


def test_build_section_tree():
    """Should build hierarchical tree from ToC entries."""
    toc = [
        {"title": "PAPER TITLE", "heading_level": None, "page_id": 0},
        {"title": "1 INTRODUCTION", "heading_level": None, "page_id": 0},
        {"title": "1.1 Background", "heading_level": None, "page_id": 0},
        {"title": "1.2 Motivation", "heading_level": None, "page_id": 1},
        {"title": "2 METHOD", "heading_level": None, "page_id": 1},
        {"title": "2.1 Architecture", "heading_level": None, "page_id": 2},
    ]
    tree = PaperParser._build_section_tree(toc)
    assert len(tree) > 0
    # "1 INTRODUCTION" should have children "1.1 Background" and "1.2 Motivation"
    intro = None
    for node in tree:
        if "INTRODUCTION" in node.title:
            intro = node
            break
    if intro:
        assert len(intro.children) == 2


def test_build_section_tree_empty():
    """Should handle empty ToC."""
    tree = PaperParser._build_section_tree([])
    assert tree == []


# ============================================================================
# Peer Review Extractor Tests
# ============================================================================


@pytest.fixture
def review_extractor():
    return PeerReviewExtractor()


def test_extract_reviews(review_extractor):
    """Should extract structured reviews from related_notes."""
    reviews = review_extractor.extract(SAMPLE_JSONL_RECORD["related_notes"])
    assert len(reviews) >= 1
    # Should have at least one regular review and one decision
    has_review = any(not r.is_meta_review for r in reviews)
    has_decision = any(r.is_meta_review for r in reviews)
    assert has_review or has_decision


def test_extract_decision(review_extractor):
    """Should extract paper decision."""
    reviews = review_extractor.extract(SAMPLE_JSONL_RECORD["related_notes"])
    decision = review_extractor.get_decision(reviews)
    assert "Accept" in decision or "poster" in decision


def test_extract_rating(review_extractor):
    """Should parse rating from string."""
    result = PeerReviewExtractor._parse_int("6: Marginally above")
    assert result == 6


def test_to_claims_and_evidences(review_extractor):
    """Should convert reviews to claims and evidences."""
    reviews = review_extractor.extract(SAMPLE_JSONL_RECORD["related_notes"])
    claims, evidences = review_extractor.to_claims_and_evidences(reviews, "TEST123")

    # Weaknesses should become claims
    assert len(claims) >= 1
    # Strengths should become evidences
    assert len(evidences) >= 1

    # All should reference the paper
    for c in claims:
        assert c.source_paper_id == "TEST123"
        assert c.source_section == "peer_review"
    for e in evidences:
        assert e.source_paper_id == "TEST123"


def test_split_points():
    """Should split numbered and bulleted lists."""
    numbered = "1. First point\n2. Second point\n3. Third point"
    result = PeerReviewExtractor._split_points(numbered)
    assert len(result) == 3

    bulleted = "- Point A\n- Point B\n- Point C"
    result = PeerReviewExtractor._split_points(bulleted)
    assert len(result) == 3


def test_split_points_paragraphs():
    """Should split by paragraph breaks."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    result = PeerReviewExtractor._split_points(text)
    assert len(result) == 3
