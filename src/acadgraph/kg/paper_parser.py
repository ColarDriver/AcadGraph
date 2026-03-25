"""
Paper Parser — PDF/Text → Structured ParsedPaper.

Uses `marker` for PDF conversion and regex + LLM for section zoning.
Fallback to `pymupdf4llm` if marker fails.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from acadgraph.kg.schema import FigureRef, ParsedPaper, Reference, SectionTreeNode, TableData
from acadgraph.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Standard academic section names and their variants
SECTION_PATTERNS: dict[str, list[str]] = {
    "abstract": [r"abstract"],
    "introduction": [r"introduction", r"1\.\s*introduction"],
    "related_work": [
        r"related\s*work",
        r"background",
        r"literature\s*review",
        r"prior\s*work",
        r"2\.\s*related\s*work",
        r"2\.\s*background",
    ],
    "method": [
        r"method(?:ology)?",
        r"approach",
        r"proposed\s*method",
        r"our\s*(?:method|approach)",
        r"model",
        r"framework",
        r"3\.\s*method",
    ],
    "experiments": [
        r"experiment(?:s|al)?(?:\s*(?:results|setup|and\s*results))?",
        r"evaluation",
        r"results",
        r"empirical\s*(?:study|evaluation|results)",
        r"4\.\s*experiment",
    ],
    "limitation": [
        r"limitation(?:s)?",
        r"discussion",
        r"broader\s*impact",
        r"ethics",
    ],
    "conclusion": [
        r"conclusion(?:s)?",
        r"summary",
        r"conclusion(?:s)?\s*(?:and\s*future\s*work)?",
        r"future\s*work",
    ],
}

# Compile patterns
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    key: [re.compile(p, re.IGNORECASE) for p in patterns]
    for key, patterns in SECTION_PATTERNS.items()
}


class PaperParser:
    """Full-text paper parser with section-level zoning."""

    def __init__(self, llm_client: LLMClient | None = None):
        self._llm = llm_client

    async def parse(self, pdf_path: str) -> ParsedPaper:
        """Parse a PDF file into a structured ParsedPaper."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text = await self._pdf_to_text(pdf_path)
        paper = await self.parse_from_text(text)
        paper.source_path = pdf_path
        return paper

    async def parse_from_text(self, full_text: str) -> ParsedPaper:
        """Parse from pre-extracted text (e.g., OpenReview HTML, arXiv)."""
        paper = ParsedPaper(raw_text=full_text)

        # Step 1: Section splitting
        paper.sections = self._split_sections(full_text)

        # Step 2: Extract tables
        paper.tables = self._extract_tables(full_text)

        # Step 3: Extract figure references
        paper.figures = self._extract_figures(full_text)

        # Step 4: Extract references
        paper.references = self._extract_references(full_text)

        # Step 5: Extract title from first line or abstract
        paper.title = self._extract_title(full_text)

        # Step 6: If section split failed, use LLM-assisted splitting
        if len(paper.sections) <= 1 and self._llm:
            paper.sections = await self._llm_section_split(full_text)

        return paper

    async def parse_from_jsonl_record(
        self,
        content: str,
        content_meta: dict | None = None,
        abstract: str = "",
    ) -> ParsedPaper:
        """Parse from pre-extracted JSONL data. No PDF/OCR needed.

        Args:
            content: Full Markdown text of the paper.
            content_meta: Structured metadata with table_of_contents.
            abstract: Pre-extracted abstract text.
        """
        paper = await self.parse_from_text(content)

        # Override abstract with the clean version if provided
        if abstract:
            paper.abstract_raw = abstract
            if "abstract" not in paper.sections or not paper.sections["abstract"]:
                paper.sections["abstract"] = abstract

        # Build enhanced section tree:
        # Priority: Markdown headings (most reliable) → content_meta ToC → number inference
        md_entries = self._infer_heading_levels_from_markdown(content)

        if md_entries and len(md_entries) > 2:
            # Use Markdown heading levels (most accurate)
            paper.section_tree = self._build_section_tree(md_entries)
        elif content_meta and "table_of_contents" in content_meta:
            # Fallback to content_meta ToC, enriched with markdown info
            toc = content_meta["table_of_contents"]
            enriched = self._enrich_toc_with_markdown(toc, md_entries)
            paper.section_tree = self._build_section_tree(enriched)
        else:
            paper.section_tree = []

        # Attach section content to tree nodes
        if paper.section_tree:
            self._attach_content_to_tree(paper.section_tree, content)
            # Map tree nodes to standard section keys
            self._assign_section_keys(paper.section_tree)

        return paper

    @staticmethod
    def _infer_heading_levels_from_markdown(text: str) -> list[dict]:
        """Infer heading levels from Markdown # markers.

        # Title → level 1
        ## Section → level 2
        ### Subsection → level 3

        Returns list of {title, heading_level, line_pos} dicts.
        """
        entries = []
        for i, line in enumerate(text.split("\n")):
            stripped = line.strip()
            m = re.match(r"^(#{1,6})\s+(.+)", stripped)
            if m:
                entries.append({
                    "title": m.group(2).strip(),
                    "heading_level": len(m.group(1)),
                    "line_num": i,
                })
        return entries

    @staticmethod
    def _enrich_toc_with_markdown(
        toc_entries: list[dict],
        md_entries: list[dict],
    ) -> list[dict]:
        """Enrich content_meta ToC with heading levels from Markdown.

        Matches by title similarity and assigns markdown-inferred levels
        to ToC entries that have heading_level=None.
        """
        if not md_entries:
            return toc_entries

        # Build a lookup: normalized title → heading level
        md_levels: dict[str, int] = {}
        for entry in md_entries:
            key = re.sub(r"\s+", " ", entry["title"]).strip().lower()
            md_levels[key] = entry["heading_level"]

        enriched = []
        for entry in toc_entries:
            entry = dict(entry)  # Don't mutate original
            if entry.get("heading_level") is None:
                title_key = re.sub(
                    r"\s+", " ", entry.get("title", "")
                ).strip().lower()
                if title_key in md_levels:
                    entry["heading_level"] = md_levels[title_key]
            enriched.append(entry)

        return enriched

    @staticmethod
    def _build_section_tree(
        entries: list[dict],
    ) -> list[SectionTreeNode]:
        """Build hierarchical SectionTreeNode tree from heading entries.

        Priority chain for heading_level:
        1. Explicit heading_level (from Markdown # or content_meta)
        2. Section number inference (1→L1, 1.1→L2, 1.1.1→L3)
        3. Default: same level as previous entry
        """
        if not entries:
            return []

        nodes: list[SectionTreeNode] = []
        stack: list[tuple[int, SectionTreeNode]] = []  # (level, node)

        for entry in entries:
            title = entry.get("title", "").replace("\n", " ").strip()
            if not title:
                continue

            # Priority chain for level inference
            level = entry.get("heading_level")
            if level is None:
                # Try section number inference: "1.2.3 Title" → level 3
                num_match = re.match(r"^(\d+(?:\.\d+)*)\s", title)
                if num_match:
                    level = len(num_match.group(1).split("."))
                else:
                    # Heuristic: unnumbered ALL-CAPS titles are likely L1
                    clean = re.sub(r"[^a-zA-Z\s]", "", title)
                    if clean and clean == clean.upper() and len(clean) > 3:
                        level = 1
                    else:
                        # Same level as previous, or L1 default
                        level = stack[-1][0] if stack else 1

            node = SectionTreeNode(
                title=title,
                heading_level=level,
                page_id=entry.get("page_id"),
                start_page=entry.get("page_id"),
            )

            # Build hierarchy: walk stack backwards to find parent
            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                stack[-1][1].children.append(node)
            else:
                nodes.append(node)

            stack.append((level, node))

        return nodes

    @staticmethod
    def _attach_content_to_tree(
        tree: list[SectionTreeNode],
        raw_text: str,
    ) -> None:
        """Attach section text content to tree nodes by matching headings.

        Finds each node's title in the raw text and extracts content
        until the next heading of same or higher level.
        """
        if not raw_text or not tree:
            return

        # Collect all nodes in DFS order
        all_nodes: list[SectionTreeNode] = []

        def _collect(nodes: list[SectionTreeNode]) -> None:
            for n in nodes:
                all_nodes.append(n)
                _collect(n.children)

        _collect(tree)

        # Find heading positions in raw text using Markdown markers
        heading_positions: list[tuple[int, int, str]] = []  # (pos, level, title)
        for i, line in enumerate(raw_text.split("\n")):
            stripped = line.strip()
            m = re.match(r"^(#{1,6})\s+(.+)", stripped)
            if m:
                # Find actual character position
                pos = 0
                for j, l in enumerate(raw_text.split("\n")):
                    if j == i:
                        break
                    pos += len(l) + 1
                heading_positions.append((pos, len(m.group(1)), m.group(2).strip()))

        if not heading_positions:
            return

        # Match nodes to heading positions by title similarity
        for node in all_nodes:
            node_title_lower = re.sub(r"\s+", " ", node.title).strip().lower()

            best_idx = -1
            best_ratio = 0.0
            for idx, (pos, level, htitle) in enumerate(heading_positions):
                htitle_lower = re.sub(r"\s+", " ", htitle).strip().lower()
                # Simple containment check
                if (node_title_lower in htitle_lower
                        or htitle_lower in node_title_lower
                        or node_title_lower == htitle_lower):
                    ratio = 1.0
                else:
                    # Partial match ratio
                    common = len(set(node_title_lower.split()) & set(htitle_lower.split()))
                    total = max(len(node_title_lower.split()), len(htitle_lower.split()), 1)
                    ratio = common / total

                if ratio > best_ratio and ratio > 0.5:
                    best_ratio = ratio
                    best_idx = idx

            if best_idx >= 0:
                start_pos = heading_positions[best_idx][0]
                start_level = heading_positions[best_idx][1]

                # Find end: next heading at same or higher (lower number) level
                end_pos = len(raw_text)
                for idx in range(best_idx + 1, len(heading_positions)):
                    if heading_positions[idx][1] <= start_level:
                        end_pos = heading_positions[idx][0]
                        break

                # Extract content (skip the heading line itself)
                heading_end = raw_text.find("\n", start_pos)
                if heading_end != -1 and heading_end < end_pos:
                    content = raw_text[heading_end + 1 : end_pos].strip()
                    # Truncate very long sections to avoid memory issues
                    if len(content) > 10000:
                        content = content[:10000] + "..."
                    node.content = content

    @staticmethod
    def _assign_section_keys(tree: list[SectionTreeNode]) -> None:
        """Map tree node titles to standard section keys (abstract, method, etc.)."""
        all_nodes: list[SectionTreeNode] = []

        def _collect(nodes: list[SectionTreeNode]) -> None:
            for n in nodes:
                all_nodes.append(n)
                _collect(n.children)

        _collect(tree)

        for node in all_nodes:
            clean_title = re.sub(r"^#+\s*", "", node.title)
            clean_title = re.sub(r"^\d+\.?\s*", "", clean_title).strip()

            for section_key, patterns in _COMPILED_PATTERNS.items():
                for pattern in patterns:
                    if pattern.fullmatch(clean_title):
                        node.section_key = section_key
                        break
                if node.section_key:
                    break

    async def _pdf_to_text(self, pdf_path: str) -> str:
        """Convert PDF to text using marker, fallback to pymupdf4llm."""
        # Try marker first
        try:
            return self._parse_with_marker(pdf_path)
        except Exception as e:
            logger.warning("marker failed for %s: %s, trying pymupdf4llm", pdf_path, e)

        # Fallback to pymupdf4llm
        try:
            return self._parse_with_pymupdf(pdf_path)
        except Exception as e:
            logger.error("Both parsers failed for %s: %s", pdf_path, e)
            raise

    @staticmethod
    def _parse_with_marker(pdf_path: str) -> str:
        """Parse PDF using marker."""
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            models = create_model_dict()
            converter = PdfConverter(artifact_dict=models)
            rendered = converter(pdf_path)
            return rendered.markdown
        except ImportError:
            # marker might have different API versions
            from marker.convert import convert_single_pdf
            full_text, _, _ = convert_single_pdf(pdf_path)
            return full_text

    @staticmethod
    def _parse_with_pymupdf(pdf_path: str) -> str:
        """Parse PDF using pymupdf4llm."""
        import pymupdf4llm
        return pymupdf4llm.to_markdown(pdf_path)

    def _split_sections(self, text: str) -> dict[str, str]:
        """Split text into sections using heading patterns."""
        sections: dict[str, str] = {}

        # Find all heading positions
        heading_positions: list[tuple[int, str, str]] = []  # (pos, section_key, heading_text)

        for line_start in self._find_line_starts(text):
            line_end = text.find("\n", line_start)
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end].strip()

            # Skip empty lines
            if not line:
                continue

            # Check if this line is a section heading
            # Format: "# Heading" or "## Heading" or "1. Heading" or just "HEADING"
            clean_line = re.sub(r"^#+\s*", "", line)  # Remove markdown heading marks
            clean_line = re.sub(r"^\d+\.?\s*", "", clean_line)  # Remove numbering

            for section_key, patterns in _COMPILED_PATTERNS.items():
                for pattern in patterns:
                    if pattern.fullmatch(clean_line.strip()):
                        heading_positions.append((line_start, section_key, line))
                        break
                else:
                    continue
                break

        # Extract content between headings
        for i, (pos, key, _heading) in enumerate(heading_positions):
            # Content starts after the heading line
            content_start = text.find("\n", pos)
            if content_start == -1:
                continue
            content_start += 1

            # Content ends at the next heading or end of text
            if i + 1 < len(heading_positions):
                content_end = heading_positions[i + 1][0]
            else:
                content_end = len(text)

            content = text[content_start:content_end].strip()
            if content:
                sections[key] = content

        # If no sections found, treat entire text as a single section
        if not sections:
            sections["full_text"] = text

        return sections

    @staticmethod
    def _find_line_starts(text: str) -> list[int]:
        """Find the start positions of all lines in text."""
        starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n" and i + 1 < len(text):
                starts.append(i + 1)
        return starts

    @staticmethod
    def _extract_tables(text: str) -> list[TableData]:
        """Extract markdown tables from text."""
        tables: list[TableData] = []
        # Match markdown table blocks
        table_pattern = re.compile(
            r"(?:(?:Table\s+\d+[.:]\s*([^\n]+)\n)?\s*)"  # Optional caption
            r"(\|[^\n]+\|\n(?:\|[-:| ]+\|\n)?(?:\|[^\n]+\|\n)*)",
            re.MULTILINE,
        )

        for match in table_pattern.finditer(text):
            caption = match.group(1) or ""
            table_text = match.group(2)
            rows = []
            headers = []
            for i, line in enumerate(table_text.strip().split("\n")):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if i == 0:
                    headers = cells
                elif re.match(r"^[-:| ]+$", line):
                    continue  # separator row
                else:
                    rows.append(cells)

            tables.append(TableData(caption=caption, headers=headers, rows=rows))

        return tables

    @staticmethod
    def _extract_figures(text: str) -> list[FigureRef]:
        """Extract figure references from text."""
        figures: list[FigureRef] = []
        fig_pattern = re.compile(
            r"(?:Figure|Fig\.?)\s+(\d+)[.:]\s*([^\n]+)",
            re.IGNORECASE,
        )
        for match in fig_pattern.finditer(text):
            figures.append(FigureRef(
                ref_id=f"Figure {match.group(1)}",
                caption=match.group(2).strip(),
            ))
        return figures

    @staticmethod
    def _extract_references(text: str) -> list[Reference]:
        """Extract bibliography entries from the references section."""
        references: list[Reference] = []

        # Find the references section
        ref_section_match = re.search(
            r"(?:^|\n)(?:#+\s*)?(?:References|Bibliography)\s*\n(.*)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not ref_section_match:
            return references

        ref_text = ref_section_match.group(1)

        # Parse individual references:
        # Format 1: numbered [1] or 1.
        # Format 2: markdown list "- Author..."
        ref_entries = re.split(r"\n(?:\[\d+\]|\d+\.|-)\s+", ref_text)
        for i, entry in enumerate(ref_entries):
            # Strip HTML span tags
            entry = re.sub(r"<span[^>]*>|</span>", "", entry).strip()
            if not entry or len(entry) < 10:
                continue

            ref = Reference(ref_key=f"[{i + 1}]")
            # Try to extract title (in quotes, italics, or after first period)
            title_match = re.search(r'["""](.+?)["""]', entry)
            if not title_match:
                # Try italic markdown: *Title*
                title_match = re.search(r'\*(.+?)\*', entry)
            if not title_match:
                # Try text after first period (Author list. Title. Venue)
                parts = entry.split('. ', 2)
                if len(parts) >= 2:
                    ref.title = parts[1].strip().rstrip('.')
            if title_match:
                ref.title = title_match.group(1)

            # Try to extract year
            year_match = re.search(r"\b((?:19|20)\d{2})\b", entry)
            if year_match:
                ref.year = int(year_match.group(1))

            # Extract authors (first part before the first period)
            author_part = entry.split('.')[0].strip()
            if author_part:
                ref.authors = [a.strip() for a in author_part.split(',') if a.strip()]

            references.append(ref)

        return references

    @staticmethod
    def _extract_title(text: str) -> str:
        """Extract the paper title from the beginning of text."""
        lines = text.strip().split("\n")
        for line in lines[:10]:
            clean = line.strip().lstrip("#").strip()
            if clean and len(clean) > 10 and not clean.startswith("Abstract"):
                return clean
        return ""


    @staticmethod
    def _parse_line_range(value: Any) -> tuple[int, int] | None:
        """Parse a line-range descriptor into (start_line, end_line)."""
        if isinstance(value, dict):
            start = value.get("start") or value.get("start_line")
            end = value.get("end") or value.get("end_line")
            if isinstance(start, int) and isinstance(end, int):
                return (start, end)
            return None

        if isinstance(value, list) and len(value) == 2:
            if isinstance(value[0], int) and isinstance(value[1], int):
                return (value[0], value[1])
            return None

        if isinstance(value, int):
            return (value, value)

        if not isinstance(value, str):
            return None

        cleaned = value.strip().lower()
        cleaned = cleaned.replace("lines", "line").replace("line", "").strip()

        span_match = re.search(r"(\d+)\s*(?:-|to|~|—|–)\s*(\d+)", cleaned)
        if span_match:
            return (int(span_match.group(1)), int(span_match.group(2)))

        single_match = re.search(r"\b(\d+)\b", cleaned)
        if single_match:
            line_num = int(single_match.group(1))
            return (line_num, line_num)

        return None

    async def _llm_section_split(self, text: str) -> dict[str, str]:
        """Use LLM to identify section boundaries when regex fails."""
        if not self._llm:
            return {"full_text": text}

        # Take first ~6000 chars to identify structure
        sample = text[:6000]
        prompt = f"""Analyze this academic paper text and identify section boundaries.
Return a JSON object mapping section names to their line ranges.
Valid section keys: abstract, introduction, related_work, method, experiments, limitation, conclusion

Text:
{sample}

Return JSON like:
{{"abstract": "line 5-20", "introduction": "line 21-80", ...}}
"""

        try:
            result = await self._llm.complete_json(prompt)
            logger.info("LLM section analysis: %s", result)
        except Exception as e:
            logger.warning("LLM section split failed: %s", e)
            return {"full_text": text}

        if not isinstance(result, dict) or not result:
            return {"full_text": text}

        line_based_sections: dict[str, Any] = {}
        valid_keys = {
            "abstract",
            "introduction",
            "related_work",
            "method",
            "experiments",
            "limitation",
            "conclusion",
        }

        # Preferred format: direct mapping from section key -> line range descriptor.
        for key in valid_keys:
            if key in result:
                line_based_sections[key] = result[key]

        # Also support list-based payloads: {"sections": [{"name": ..., "range": ...}, ...]}
        if not line_based_sections and isinstance(result.get("sections"), list):
            for item in result["sections"]:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("name") or item.get("section") or "").strip().lower()
                if key not in valid_keys:
                    continue
                line_based_sections[key] = (
                    item.get("line_range")
                    or item.get("range")
                    or item.get("lines")
                    or item
                )

        if not line_based_sections:
            return {"full_text": text}

        lines = text.splitlines()
        if not lines:
            return {"full_text": text}

        sections: dict[str, str] = {}
        total_lines = len(lines)
        for key, raw_range in line_based_sections.items():
            parsed = self._parse_line_range(raw_range)
            if not parsed:
                continue

            start_line, end_line = parsed
            if start_line > end_line:
                start_line, end_line = end_line, start_line

            # LLM uses 1-based line numbers; clamp to valid bounds.
            start_idx = max(1, min(start_line, total_lines)) - 1
            end_idx = max(1, min(end_line, total_lines))
            content = "\n".join(lines[start_idx:end_idx]).strip()
            if content:
                sections[key] = content

        return sections or {"full_text": text}
