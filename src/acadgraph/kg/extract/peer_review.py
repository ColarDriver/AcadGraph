"""
Peer Review Extractor — Converts OpenReview related_notes into structured
review data and augments the argumentation graph.

Reviewer weaknesses → Limitation/Gap nodes
Reviewer strengths → supporting Evidence nodes
Decision → Paper node property
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from acadgraph.kg.jsonl_loader import parse_related_notes
from acadgraph.kg.schema import (
    Claim,
    ClaimSeverity,
    ClaimType,
    Evidence,
    EvidenceType,
    generate_id,
)

logger = logging.getLogger(__name__)


@dataclass
class PeerReview:
    """A single peer review from OpenReview."""

    reviewer_id: str = ""
    rating: int | None = None
    confidence: int | None = None
    decision: str = ""  # "Accept: poster" / "Reject"
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    is_meta_review: bool = False


class PeerReviewExtractor:
    """Extracts structured peer reviews from OpenReview related_notes."""

    def extract(self, related_notes_raw: str) -> list[PeerReview]:
        """Parse related_notes string into structured PeerReview list."""
        notes = parse_related_notes(related_notes_raw)
        if not notes:
            return []

        reviews: list[PeerReview] = []
        for note in notes:
            review = self._parse_note(note)
            if review:
                reviews.append(review)

        return reviews

    def get_decision(self, reviews: list[PeerReview]) -> str:
        """Extract the paper decision from reviews."""
        for r in reviews:
            if r.decision:
                return r.decision
        return ""

    def to_claims_and_evidences(
        self,
        reviews: list[PeerReview],
        paper_id: str,
    ) -> tuple[list[Claim], list[Evidence]]:
        """Convert peer reviews into Claim/Evidence nodes for the argumentation layer.

        - Weaknesses → Claims of type LIMITATION with source="peer_review"
        - Strengths → Evidence with evidence_type="peer_review"
        """
        claims: list[Claim] = []
        evidences: list[Evidence] = []

        for review in reviews:
            if review.is_meta_review:
                # Meta-review summary as a high-level claim
                if review.summary:
                    claim = Claim(
                        claim_id=generate_id("claim_pr"),
                        text=f"[Meta-review] {review.summary}",
                        claim_type=ClaimType.PERFORMANCE,
                        severity=ClaimSeverity.P1,
                        source_section="peer_review",
                        source_paper_id=paper_id,
                    )
                    claims.append(claim)
                continue

            # Weaknesses → Limitation claims
            for i, weakness in enumerate(review.weaknesses):
                if len(weakness.strip()) < 10:
                    continue
                claim = Claim(
                    claim_id=generate_id("claim_pr"),
                    text=weakness.strip(),
                    claim_type=ClaimType.ROBUSTNESS,
                    severity=ClaimSeverity.P2,
                    source_section="peer_review",
                    source_paper_id=paper_id,
                )
                claims.append(claim)

            # Strengths → Supporting evidence
            for i, strength in enumerate(review.strengths):
                if len(strength.strip()) < 10:
                    continue
                evidence = Evidence(
                    evidence_id=generate_id("evi_pr"),
                    evidence_type=EvidenceType.ANALYSIS,
                    result_summary=strength.strip(),
                    source_paper_id=paper_id,
                )
                evidences.append(evidence)

        return claims, evidences

    def _parse_note(self, note: dict[str, Any]) -> PeerReview | None:
        """Parse a single OpenReview note dict into a PeerReview."""
        content = note.get("content", {})
        if not isinstance(content, dict):
            return None

        # Skip discussion/response notes (only have 'comment' + 'title')
        content_keys = set(content.keys())
        if content_keys <= {"comment", "title"}:
            return None

        # Skip original submission metadata
        if "paperhash" in content or "authorids" in content:
            return None

        # Detect meta-review / decision notes
        decision = self._extract_field(content, "decision")
        is_meta = bool(decision)

        # Extract summary from various field name patterns
        summary = self._extract_field(
            content,
            "metareview:_summary,_strengths_and_weaknesses",
            "metareview",
            "summary_of_the_review",
            "summary_of_the_paper",
            "main_review",
            "review",
            "summary",
        )

        # Extract rating (ICLR uses 'recommendation', others use 'rating')
        rating = self._parse_int(
            self._extract_field(content, "rating", "recommendation")
        )

        # Extract confidence
        confidence = self._parse_int(
            self._extract_field(content, "confidence")
        )

        # Extract strengths and weaknesses
        # ICLR format: combined 'strength_and_weaknesses' field
        strengths_raw = self._extract_field(
            content, "strengths", "strength"
        )
        weaknesses_raw = self._extract_field(
            content, "weaknesses", "weakness", "limitations"
        )

        # Handle combined 'strength_and_weaknesses' field (common in ICLR)
        combined_raw = self._extract_field(
            content,
            "strength_and_weaknesses",
            "strengths_and_weaknesses",
            "strength_and_weakness",
        )
        if combined_raw and not strengths_raw and not weaknesses_raw:
            strengths_raw, weaknesses_raw = self._split_strengths_weaknesses(
                combined_raw
            )

        questions_raw = self._extract_field(
            content, "questions", "questions_for_authors"
        )

        strengths = self._split_points(strengths_raw) if strengths_raw else []
        weaknesses = self._split_points(weaknesses_raw) if weaknesses_raw else []
        questions = self._split_points(questions_raw) if questions_raw else []

        # Skip notes with no useful content
        if not summary and not strengths and not weaknesses and not decision:
            return None

        return PeerReview(
            reviewer_id=str(note.get("id", "")),
            rating=rating,
            confidence=confidence,
            decision=decision,
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            questions=questions,
            is_meta_review=is_meta,
        )

    @staticmethod
    def _extract_field(content: dict, *keys: str) -> str:
        """Try multiple field names, return first non-empty value."""
        for key in keys:
            val = content.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    @staticmethod
    def _parse_int(value: str) -> int | None:
        """Extract integer from rating strings like '6: Marginally above'."""
        if not value:
            return None
        match = re.match(r"(\d+)", value.strip())
        return int(match.group(1)) if match else None

    @staticmethod
    def _split_points(text: str) -> list[str]:
        """Split a block of text into individual bullet points.

        Handles: numbered lists, bullet lists, paragraph breaks.
        """
        if not text:
            return []

        # Try splitting by numbered list pattern: "1. ...", "2. ..."
        numbered = re.split(r"\n\s*\d+[\.\)]\s+", "\n" + text)
        if len(numbered) > 2:
            return [p.strip() for p in numbered if p.strip()]

        # Try splitting by bullet points: "- ...", "* ..."
        bulleted = re.split(r"\n\s*[-\*•]\s+", "\n" + text)
        if len(bulleted) > 2:
            return [p.strip() for p in bulleted if p.strip()]

        # Try splitting by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1:
            return [p.strip() for p in paragraphs if p.strip()]

        # Return as single item
        return [text.strip()] if text.strip() else []

    @staticmethod
    def _split_strengths_weaknesses(combined: str) -> tuple[str, str]:
        """Split a combined strengths/weaknesses text block.

        Common ICLR format:
            Strengths:
            - point 1
            Weaknesses:
            - point 2
        """
        patterns = [
            r"(?i)\n\s*weakness(?:es)?[\s:]*\n",
            r"(?i)\n\s*limitation(?:s)?[\s:]*\n",
            r"(?i)\n\s*cons?[\s:]*\n",
            r"(?i)\bweakness(?:es)?\s*:",
            r"(?i)\blimitation(?:s)?\s*:",
        ]
        for pattern in patterns:
            match = re.search(pattern, combined)
            if match:
                return (
                    combined[: match.start()].strip(),
                    combined[match.end() :].strip(),
                )
        return combined, ""
