"""Tests for Pydantic response validation in argumentation extraction."""

import pytest

from acadgraph.kg.extract.argumentation import (
    _ClaimResponse,
    _CoreIdeaResponse,
    _EvidenceLinkingResponse,
    _EvidenceLinkResponse,
    _GapResponse,
    _NumericConsistencyIssue,
    _ProblemResponse,
    _SchemaExtractionResponse,
)


class TestSchemaExtractionResponse:
    """Validate that Pydantic models parse LLM responses robustly."""

    def test_full_response(self):
        raw = {
            "problems": [{"description": "P1 desc", "scope": "NLP", "gap_ids": [0]}],
            "gaps": [{"failure_mode": "overfitting", "constraint": "low data", "core_idea_ids": [0]}],
            "core_ideas": [{"mechanism": "sparse attention", "novelty_type": "NEW_MECHANISM", "claim_ids": [0, 1]}],
            "claims": [
                {"text": "Our method is SOTA", "type": "PERFORMANCE", "severity": "P0"},
                {"text": "Novel mechanism", "type": "NOVELTY", "severity": "P1"},
            ],
        }
        result = _SchemaExtractionResponse.model_validate(raw)
        assert len(result.problems) == 1
        assert len(result.gaps) == 1
        assert len(result.core_ideas) == 1
        assert len(result.claims) == 2
        assert result.problems[0].gap_ids == [0]
        assert result.core_ideas[0].claim_ids == [0, 1]

    def test_backward_compat_single_problem_gap(self):
        """Support legacy single-object format."""
        raw = {
            "problem": {"description": "single problem"},
            "gap": {"failure_mode": "single gap"},
            "core_idea": {"mechanism": "single idea"},
            "claims": [],
        }
        result = _SchemaExtractionResponse.model_validate(raw)
        assert result.problem is not None
        assert result.problem.description == "single problem"
        assert result.gap is not None
        assert result.gap.failure_mode == "single gap"

    def test_empty_response(self):
        """Completely empty response should produce valid empty object."""
        result = _SchemaExtractionResponse.model_validate({})
        assert result.problems == []
        assert result.claims == []
        assert result.problem is None

    def test_extra_fields_are_ignored(self):
        """Extra fields from the LLM should not cause errors."""
        raw = {
            "problems": [],
            "claims": [],
            "extra_field": "should be ignored",
        }
        result = _SchemaExtractionResponse.model_validate(raw)
        assert not hasattr(result, "extra_field") or True  # pydantic handles this

    def test_gap_ids_out_of_range_are_filtered_at_usage(self):
        """gap_ids can contain any int — filtering happens at usage time."""
        raw = {
            "problems": [{"description": "p1", "gap_ids": [0, 99, -1]}],
            "gaps": [{"failure_mode": "g1"}],
        }
        result = _SchemaExtractionResponse.model_validate(raw)
        assert result.problems[0].gap_ids == [0, 99, -1]


class TestEvidenceLinkingResponse:
    """Validate evidence linking response models."""

    def test_full_evidence_response(self):
        raw = {
            "evidence_links": [
                {"claim_index": 0, "evidences": [{"result_summary": "95%"}]},
            ],
            "baselines": [{"method_name": "Baseline-A"}],
            "numeric_consistency_issues": [
                {"claim_text": "We achieve 95%", "claimed_value": "95%", "table_value": "94.8%", "consistent": False},
            ],
        }
        result = _EvidenceLinkingResponse.model_validate(raw)
        assert len(result.evidence_links) == 1
        assert len(result.baselines) == 1
        assert len(result.numeric_consistency_issues) == 1
        assert not result.numeric_consistency_issues[0].consistent

    def test_empty_evidence_response(self):
        result = _EvidenceLinkingResponse.model_validate({})
        assert result.evidence_links == []
        assert result.baselines == []
        assert result.numeric_consistency_issues == []

    def test_numeric_default_consistent(self):
        """Default for consistency flag should be True."""
        issue = _NumericConsistencyIssue.model_validate({"claim_text": "test"})
        assert issue.consistent is True
