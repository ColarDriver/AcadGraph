"""
Async LLM client — wraps an OpenAI-compatible API.

Provides retry logic, concurrency limiting, and structured JSON output.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from acadgraph.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Async wrapper around an OpenAI-compatible LLM endpoint."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = AsyncOpenAI(
            base_url=self.config.api_base,
            api_key=self.config.api_key,
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a single-turn chat completion request."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content.strip()

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Send a completion request and parse the response as JSON.

        The prompt should instruct the model to respond with valid JSON.
        Attempts to extract JSON from markdown code fences if present.
        """
        raw = await self.complete(prompt, system_prompt, temperature, max_tokens)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Extract JSON from text, handling markdown code fences."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code fences
        import re

        patterns = [
            r"```json\s*\n(.*?)\n\s*```",
            r"```\s*\n(.*?)\n\s*```",
            r"\{.*\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if match.lastindex else match.group(0))
                except (json.JSONDecodeError, IndexError):
                    continue

        logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
        return {}

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
