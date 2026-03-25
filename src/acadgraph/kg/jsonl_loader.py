"""
JSONL Data Loader — Streams paper_reviews_dataset.jsonl into PaperSource objects.

Memory-efficient: reads one line at a time, never loads the full 2GB file.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

from acadgraph.kg.schema import PaperSource

logger = logging.getLogger(__name__)

# HTML tag pattern for cleaning Markdown content
_SPAN_TAG_RE = re.compile(r"<span[^>]*>|</span>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class JournalRecord:
    """One record from paper_reviews_dataset.jsonl."""

    id: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    content: str = ""  # Full Markdown text
    content_meta: dict | None = None  # Parsed JSON with table_of_contents
    conference: str = ""
    year: int | None = None
    pdf_url: str = ""
    source_url: str = ""
    related_notes_raw: str = ""  # Raw string, parsed later


def iter_jsonl(
    path: str,
    *,
    limit: int | None = None,
    skip_ids: set[str] | None = None,
) -> Iterator[JournalRecord]:
    """Stream JSONL records one by one. Memory-efficient.

    Args:
        path: Path to the JSONL file.
        limit: Maximum number of records to yield.
        skip_ids: Set of paper IDs to skip (for resume support).
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON at line %d: %s", line_num, e)
                continue

            paper_id = obj.get("id", "")
            if skip_ids and paper_id in skip_ids:
                continue

            record = _parse_record(obj)

            count += 1
            yield record

            if limit is not None and count >= limit:
                return


def _parse_record(obj: dict) -> JournalRecord:
    """Parse a raw JSON dict into a JournalRecord."""
    # Parse authors: may be "Alice, Bob" (str) or ["Alice", "Bob"] (list)
    authors_raw = obj.get("authors", "")
    if isinstance(authors_raw, list):
        authors = [str(a).strip() for a in authors_raw if a]
    else:
        authors = [a.strip() for a in str(authors_raw).split(",") if a.strip()]

    # Parse year
    year_raw = obj.get("year", "")
    try:
        year = int(year_raw) if year_raw else None
    except (ValueError, TypeError):
        year = None

    # Parse content_meta (JSON string -> dict)
    content_meta = None
    meta_raw = obj.get("content_meta", "")
    if meta_raw:
        try:
            content_meta = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse content_meta for %s", obj.get("id", "?"))

    # Clean content — support both 'content' (format A) and 'context' (format B)
    raw_content = obj.get("content", "") or obj.get("context", "")
    content = _clean_markdown(raw_content)

    return JournalRecord(
        id=obj.get("id", ""),
        title=obj.get("title", ""),
        authors=authors,
        abstract=obj.get("abstract", ""),
        content=content,
        content_meta=content_meta,
        conference=obj.get("conference", ""),
        year=year,
        pdf_url=obj.get("pdf_url", ""),
        source_url=obj.get("source_url", ""),
        related_notes_raw=obj.get("related_notes", ""),
    )


def _clean_markdown(text: str) -> str:
    """Clean HTML artifacts from Markdown content.

    Handles: <span id="page-0-1"> tags, stray HTML tags, etc.
    """
    if not text:
        return text
    # Remove <span> tags (common in this dataset)
    text = _SPAN_TAG_RE.sub("", text)
    # Normalize ## title at start to # title
    if text.startswith("## "):
        text = "#" + text[2:]
    return text


def jsonl_to_paper_source(record: JournalRecord) -> PaperSource:
    """Convert a JournalRecord to a PaperSource for the existing pipeline."""
    return PaperSource(
        paper_id=record.id,
        text=record.content,
        title=record.title,
        authors=record.authors,
        year=record.year,
        venue=record.conference,
        metadata={
            "pdf_url": record.pdf_url,
            "source_url": record.source_url,
        },
        content_meta=record.content_meta,
        abstract=record.abstract,
        related_notes_raw=record.related_notes_raw,
    )


def parse_related_notes(raw: str) -> list[dict]:
    """Parse the related_notes string into a list of note dicts.

    The field can be:
    - A Python dict repr (single note)
    - A Python list repr (multiple notes)
    - A JSON string
    """
    if not raw or not raw.strip():
        return []

    raw = raw.strip()

    # Try JSON first
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return [result]
        if isinstance(result, (list, tuple)):
            return [r for r in result if isinstance(r, dict)]
    except json.JSONDecodeError:
        pass

    # Try Python literal eval (the dataset uses Python repr format)
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, dict):
            return [result]
        if isinstance(result, (list, tuple)):
            return [r for r in result if isinstance(r, dict)]
    except (ValueError, SyntaxError):
        pass

    logger.warning("Failed to parse related_notes (len=%d)", len(raw))
    return []
