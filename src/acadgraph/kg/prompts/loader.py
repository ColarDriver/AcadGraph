"""Markdown prompt loader.

Loads prompt templates from .md files in the prompts directory.
Uses ``{{ placeholder }}`` syntax for template variables (Jinja2-style),
keeping JSON examples with plain ``{ }`` braces unescaped.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent

# Pattern: {{ variable_name }}
_PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@lru_cache(maxsize=64)
def load_prompt(category: str, name: str) -> str:
    """Load a prompt template from a Markdown file.

    Args:
        category: Subdirectory name (e.g., 'entity', 'citation', 'argumentation').
        name: Filename without extension (e.g., 'entity_extraction_system').

    Returns:
        The raw prompt template string (with ``{{ placeholders }}``).

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = _PROMPTS_DIR / category / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def render_prompt(template: str, **kwargs: str) -> str:
    """Render a prompt template by substituting ``{{ placeholder }}`` variables.

    Args:
        template: The template string loaded via :func:`load_prompt`.
        **kwargs: Variable name → value mappings.

    Returns:
        The rendered prompt string.
    """
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in kwargs:
            return kwargs[key]
        return match.group(0)  # Leave unmatched placeholders as-is

    return _PLACEHOLDER_RE.sub(_replace, template)
