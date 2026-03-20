"""Prompt templates for KG extraction.

All prompts are stored as Markdown files organized by category:
- entity/       — Entity extraction prompts
- citation/     — Citation classification & evolution detection prompts
- argumentation/ — Zoning, schema extraction, and evidence linking prompts
"""

from acadgraph.kg.prompts.loader import load_prompt

__all__ = ["load_prompt"]
