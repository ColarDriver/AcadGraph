"""Tests for enhanced section tree building (Phase 1-3 of Graph × PageIndex integration)."""

import asyncio

import pytest

from acadgraph.kg.paper_parser import PaperParser
from acadgraph.kg.schema import Claim, ClaimType, ClaimSeverity, SectionTreeNode


# ============================================================
# _infer_heading_levels_from_markdown
# ============================================================


def test_infer_heading_levels_basic():
    text = "# Title\n## Introduction\nSome text.\n### Background\n## Method\n"
    entries = PaperParser._infer_heading_levels_from_markdown(text)
    assert len(entries) == 4
    assert entries[0]["title"] == "Title"
    assert entries[0]["heading_level"] == 1
    assert entries[1]["title"] == "Introduction"
    assert entries[1]["heading_level"] == 2
    assert entries[2]["title"] == "Background"
    assert entries[2]["heading_level"] == 3
    assert entries[3]["title"] == "Method"
    assert entries[3]["heading_level"] == 2


def test_infer_heading_levels_empty():
    entries = PaperParser._infer_heading_levels_from_markdown("No headings here.")
    assert entries == []


def test_infer_heading_levels_deep():
    text = "# L1\n## L2\n### L3\n#### L4\n##### L5\n###### L6\n"
    entries = PaperParser._infer_heading_levels_from_markdown(text)
    assert len(entries) == 6
    for i, e in enumerate(entries, start=1):
        assert e["heading_level"] == i


# ============================================================
# _build_section_tree
# ============================================================


def test_build_section_tree_from_markdown_entries():
    entries = [
        {"title": "Title", "heading_level": 1},
        {"title": "Introduction", "heading_level": 2},
        {"title": "Background", "heading_level": 3},
        {"title": "Method", "heading_level": 2},
        {"title": "Conclusion", "heading_level": 2},
    ]
    tree = PaperParser._build_section_tree(entries)
    # "Title" is L1, everything else is L2/L3 → children of Title
    assert len(tree) == 1
    root = tree[0]
    assert root.title == "Title"
    assert root.heading_level == 1
    assert len(root.children) == 3  # Introduction, Method, Conclusion
    assert root.children[0].title == "Introduction"
    assert len(root.children[0].children) == 1  # Background
    assert root.children[0].children[0].title == "Background"


def test_build_section_tree_number_inference():
    entries = [
        {"title": "1 Introduction"},
        {"title": "1.1 Background"},
        {"title": "1.2 Motivation"},
        {"title": "2 Method"},
        {"title": "3 Conclusion"},
    ]
    tree = PaperParser._build_section_tree(entries)
    assert len(tree) == 3  # Three L1 sections
    assert tree[0].title == "1 Introduction"
    assert tree[0].heading_level == 1
    assert len(tree[0].children) == 2  # 1.1 and 1.2
    assert tree[0].children[0].heading_level == 2


def test_build_section_tree_empty():
    tree = PaperParser._build_section_tree([])
    assert tree == []


def test_build_section_tree_all_caps_heuristic():
    """ALL CAPS unnumbered titles should be inferred as L1."""
    entries = [
        {"title": "ABSTRACT"},
        {"title": "INTRODUCTION"},
        {"title": "1.1 Subsection"},  # numbered → L2
    ]
    tree = PaperParser._build_section_tree(entries)
    assert len(tree) == 2  # ABSTRACT and INTRODUCTION are both L1
    assert tree[0].heading_level == 1
    assert tree[1].heading_level == 1
    assert len(tree[1].children) == 1  # 1.1 is L2 child


def test_section_tree_node_auto_id():
    node = SectionTreeNode(title="Test")
    assert node.node_id.startswith("sec_")


# ============================================================
# _enrich_toc_with_markdown
# ============================================================


def test_enrich_toc_with_markdown():
    toc = [
        {"title": "Introduction", "heading_level": None},
        {"title": "Method", "heading_level": None},
    ]
    md = [
        {"title": "Introduction", "heading_level": 2},
        {"title": "Method", "heading_level": 2},
    ]
    enriched = PaperParser._enrich_toc_with_markdown(toc, md)
    assert enriched[0]["heading_level"] == 2
    assert enriched[1]["heading_level"] == 2


def test_enrich_toc_preserves_existing_levels():
    toc = [
        {"title": "Intro", "heading_level": 3},
    ]
    md = [{"title": "Intro", "heading_level": 1}]
    enriched = PaperParser._enrich_toc_with_markdown(toc, md)
    # Should NOT overwrite existing heading_level
    assert enriched[0]["heading_level"] == 3


# ============================================================
# _attach_content_to_tree
# ============================================================


def test_attach_content_to_tree():
    text = "# Introduction\nThis is the introduction.\n# Method\nThis is the method.\n# Results\nSome results.\n"
    entries = PaperParser._infer_heading_levels_from_markdown(text)
    tree = PaperParser._build_section_tree(entries)
    PaperParser._attach_content_to_tree(tree, text)

    assert len(tree) == 3
    assert tree[0].title == "Introduction"
    assert "introduction" in tree[0].content.lower()
    assert tree[1].title == "Method"
    assert "method" in tree[1].content.lower()
    assert tree[2].title == "Results"
    assert "results" in tree[2].content.lower()


def test_attach_content_hierarchical():
    text = "# Paper\n## Intro\nIntro text.\n### Background\nBackground text.\n## Method\nMethod text.\n"
    entries = PaperParser._infer_heading_levels_from_markdown(text)
    tree = PaperParser._build_section_tree(entries)
    PaperParser._attach_content_to_tree(tree, text)

    root = tree[0]
    assert root.title == "Paper"
    # L2 children: Intro and Method
    assert len(root.children) == 2
    intro = root.children[0]
    assert "intro text" in intro.content.lower()
    method = root.children[1]
    assert "method text" in method.content.lower()


# ============================================================
# _assign_section_keys
# ============================================================


def test_assign_section_keys():
    tree = [
        SectionTreeNode(title="Abstract"),
        SectionTreeNode(title="1 Introduction"),
        SectionTreeNode(title="2 Method"),
        SectionTreeNode(title="Conclusion"),
    ]
    PaperParser._assign_section_keys(tree)
    assert tree[0].section_key == "abstract"
    assert tree[1].section_key == "introduction"
    assert tree[2].section_key == "method"
    assert tree[3].section_key == "conclusion"


# ============================================================
# parse_from_jsonl_record (enhanced)
# ============================================================


def test_parse_from_jsonl_record_builds_tree():
    text = "# Title\n## Abstract\nAbstract text.\n## Introduction\nIntro text.\n## Method\nMethod text.\n"
    parser = PaperParser()
    paper = asyncio.run(parser.parse_from_jsonl_record(text))

    assert len(paper.section_tree) > 0
    # Check that content is attached
    has_content = any(n.content for n in paper.section_tree)
    assert has_content, "At least some tree nodes should have content attached"


# ============================================================
# _link_claims_to_sections (from IncrementalKGBuilder)
# ============================================================


def test_link_claims_to_sections():
    from acadgraph.kg.incremental_builder import IncrementalKGBuilder

    tree = [
        SectionTreeNode(title="Abstract", section_key="abstract"),
        SectionTreeNode(title="Introduction", section_key="introduction"),
        SectionTreeNode(title="Experiments", section_key="experiments"),
    ]
    claims = [
        Claim(text="Claim 1", source_section="abstract"),
        Claim(text="Claim 2", source_section="introduction"),
        Claim(text="Claim 3", source_section="experiments"),
        Claim(text="Claim 4", source_section="unknown_section"),
    ]

    IncrementalKGBuilder._link_claims_to_sections(claims, tree)

    assert claims[0].claim_id in tree[0].claim_ids
    assert claims[1].claim_id in tree[1].claim_ids
    assert claims[2].claim_id in tree[2].claim_ids
    # Unknown section should not be linked
    assert all(claims[3].claim_id not in n.claim_ids for n in tree)


def test_link_claims_by_title_similarity():
    from acadgraph.kg.incremental_builder import IncrementalKGBuilder

    tree = [
        SectionTreeNode(title="4.2 Experimental Results"),
    ]
    claims = [
        Claim(text="We outperform baselines", source_section="Experimental Results"),
    ]

    IncrementalKGBuilder._link_claims_to_sections(claims, tree)
    assert claims[0].claim_id in tree[0].claim_ids


# ============================================================
# Phase 4: Tree Search helpers (KGQueryEngine)
# ============================================================


def test_format_tree_for_llm():
    from acadgraph.kg.query_engine import KGQueryEngine

    tree_data = [
        {"node_id": "s1", "title": "Introduction", "heading_level": 1, "section_key": "introduction"},
        {"node_id": "s2", "title": "Background", "heading_level": 2, "section_key": ""},
        {"node_id": "s3", "title": "Method", "heading_level": 1, "section_key": "method"},
    ]
    result = KGQueryEngine._format_tree_for_llm(tree_data)
    assert "[s1] Introduction [introduction]" in result
    assert "  - [s2] Background" in result  # indented for L2
    assert "[s3] Method [method]" in result


def test_format_tree_for_llm_empty():
    from acadgraph.kg.query_engine import KGQueryEngine
    assert KGQueryEngine._format_tree_for_llm([]) == ""


def test_parse_tree_search_response_valid():
    from acadgraph.kg.query_engine import KGQueryEngine

    response = '[{"node_id": "s1", "title": "Intro", "relevance": 0.9, "reason": "relevant"}]'
    result = KGQueryEngine._parse_tree_search_response(response)
    assert len(result) == 1
    assert result[0]["node_id"] == "s1"
    assert result[0]["relevance"] == 0.9


def test_parse_tree_search_response_markdown():
    from acadgraph.kg.query_engine import KGQueryEngine

    response = '```json\n[{"node_id": "s2", "title": "Method", "relevance": 0.8}]\n```'
    result = KGQueryEngine._parse_tree_search_response(response)
    assert len(result) == 1
    assert result[0]["node_id"] == "s2"


def test_parse_tree_search_response_invalid():
    from acadgraph.kg.query_engine import KGQueryEngine

    assert KGQueryEngine._parse_tree_search_response("") == []
    assert KGQueryEngine._parse_tree_search_response("not json") == []
    assert KGQueryEngine._parse_tree_search_response('{"no_list": true}') == []


def test_cross_path_rerank_parse_indices():
    """Test that _cross_path_rerank correctly parses LLM index arrays."""
    import json as _json

    # Valid index array
    text = "[2, 0, 1]"
    indices = _json.loads(text)
    assert isinstance(indices, list)
    assert all(isinstance(i, int) for i in indices)
    assert indices == [2, 0, 1]


def test_cross_path_rerank_handles_markdown():
    """Test markdown code block extraction for rerank response."""
    import re

    text = "```json\n[1, 0, 2]\n```"
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    assert match is not None
    inner = match.group(1).strip()
    import json as _json
    indices = _json.loads(inner)
    assert indices == [1, 0, 2]


