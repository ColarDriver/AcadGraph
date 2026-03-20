"""Tests for paper_parser.py — section splitting and text extraction."""

import asyncio

import pytest

from acadgraph.kg.paper_parser import PaperParser
from acadgraph.kg.schema import ParsedPaper


@pytest.fixture
def parser():
    return PaperParser()


SAMPLE_PAPER_TEXT = """# Efficient Attention Mechanisms for Long Sequences

## Abstract

We present a novel sparse attention mechanism that achieves linear complexity
while maintaining competitive performance on long-range benchmarks.

## Introduction

Transformer models have revolutionized NLP and computer vision [1].
However, the quadratic complexity of self-attention limits their applicability
to long sequences [2, 3]. Several works have attempted to address this issue.

## Related Work

Linformer [4] approximates attention with low-rank matrices.
Performer [5] uses random feature maps. Flash Attention [6] optimizes
memory usage but does not reduce computational complexity.

## Method

We propose SparseFlow, a method that combines local attention windows
with learned sparse global tokens. Our architecture consists of three components:
1. Local windowed attention (window size w)
2. Global sparse attention via top-k selection
3. Cross-layer information flow

## Experiments

We evaluate on Long Range Arena (LRA) benchmark and document classification.

| Model | LRA Score | Time (ms) |
|-------|-----------|-----------|
| Transformer | 58.3 | 1240 |
| Linformer | 52.1 | 320 |
| SparseFlow (ours) | 61.2 | 280 |

## Conclusion

We presented SparseFlow, achieving SOTA on LRA with 4x speedup.

## References

[1] Vaswani et al. "Attention is All You Need" NeurIPS 2017
[2] Tay et al. "Long Range Arena" ICLR 2021
[3] Katharopoulos et al. "Transformers are RNNs" ICML 2020
"""


def test_parse_from_text(parser):
    """Should correctly split text into sections."""
    result = asyncio.run(parser.parse_from_text(SAMPLE_PAPER_TEXT))

    assert isinstance(result, ParsedPaper)
    assert "abstract" in result.sections
    assert "introduction" in result.sections
    assert "method" in result.sections
    assert "experiments" in result.sections
    assert "conclusion" in result.sections

    # Check content
    assert "sparse attention" in result.sections["abstract"].lower()
    assert "sparseflow" in result.sections["method"].lower()


def test_extract_tables(parser):
    """Should extract markdown tables."""
    result = asyncio.run(parser.parse_from_text(SAMPLE_PAPER_TEXT))

    assert len(result.tables) >= 1
    table = result.tables[0]
    assert len(table.headers) > 0


def test_extract_title(parser):
    """Should extract the paper title."""
    result = asyncio.run(parser.parse_from_text(SAMPLE_PAPER_TEXT))
    assert "Efficient Attention" in result.title or "Long Sequences" in result.title


def test_extract_references(parser):
    """Should extract bibliography entries."""
    result = asyncio.run(parser.parse_from_text(SAMPLE_PAPER_TEXT))
    assert len(result.references) >= 1


def test_empty_text(parser):
    """Should handle empty text gracefully."""
    result = asyncio.run(parser.parse_from_text(""))
    assert isinstance(result, ParsedPaper)


def test_no_sections_text(parser):
    """Should fall back to 'full_text' key if no sections found."""
    plain_text = "This is just a plain text without any section headers at all."
    result = asyncio.run(parser.parse_from_text(plain_text))
    assert "full_text" in result.sections


class _StubLLM:
    def __init__(self, payload):
        self._payload = payload

    async def complete_json(self, _prompt):
        return self._payload


def test_llm_section_split_parses_line_ranges():
    """LLM line ranges should be parsed into concrete section text."""
    text = "\n".join(
        [
            "A Study on Test-Time Adaptation",
            "Abstract heading",
            "Abstract line 1",
            "Abstract line 2",
            "Introduction heading",
            "Intro line 1",
            "Intro line 2",
            "Method heading",
            "Method line 1",
            "Method line 2",
            "Method line 3",
        ]
    )
    parser = PaperParser(llm_client=_StubLLM({
        "abstract": "line 3-4",
        "introduction": "line 6-7",
        "method": "line 9-11",
    }))

    sections = asyncio.run(parser._llm_section_split(text))

    assert set(sections.keys()) == {"abstract", "introduction", "method"}
    assert sections["abstract"] == "Abstract line 1\nAbstract line 2"
    assert sections["introduction"] == "Intro line 1\nIntro line 2"
    assert sections["method"].startswith("Method line 1")


def test_llm_section_split_fallback_when_unparseable():
    """Unparseable LLM output should fallback to full_text."""
    parser = PaperParser(llm_client=_StubLLM({"abstract": "unknown"}))
    text = "line one\nline two"

    sections = asyncio.run(parser._llm_section_split(text))

    assert sections == {"full_text": text}
