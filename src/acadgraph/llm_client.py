"""
Async LLM client — wraps an OpenAI-compatible API.

Provides retry logic, concurrency limiting, and structured JSON output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

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
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            content = response.choices[0].message.content or ""
            # Safety net: strip thinking blocks if they still appear
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(r"^Thinking Process:.*?(?=[\[{])", "", content, flags=re.DOTALL)
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
        """Extract JSON from text, handling markdown code fences robustly."""
        import re

        if not text or not text.strip():
            return {}

        text = text.strip()

        # 0. Strip markdown code fences (including unclosed ones from truncation)
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

        # 1. Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"items": result}
        except json.JSONDecodeError:
            pass

        # 2. Extract from markdown code fences (flexible whitespace)
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            inner = fence_match.group(1).strip()
            try:
                result = json.loads(inner)
                if isinstance(result, dict):
                    return result
                if isinstance(result, list):
                    return {"items": result}
            except json.JSONDecodeError:
                pass

        # 3. Find outermost { ... } using brace counting (handles nested objects)
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

        # 4. Find outermost [ ... ] (JSON arrays)
        start = text.find("[")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            result = json.loads(candidate)
                            if isinstance(result, list):
                                return {"items": result}
                        except json.JSONDecodeError:
                            break

        # 5. Last resort: try to repair truncated JSON
        repaired = LLMClient._repair_truncated_json(text)
        if repaired:
            return repaired

        logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
        return {}

    @staticmethod
    def _repair_truncated_json(text: str) -> dict[str, Any] | None:
        """Attempt to repair JSON truncated by max_tokens.

        Finds the first { or [, then appends missing closing brackets.
        """
        # Find JSON start
        start = -1
        start_char = ""
        for i, ch in enumerate(text):
            if ch == "{":
                start = i
                start_char = "{"
                break
            elif ch == "[":
                start = i
                start_char = "["
                break

        if start == -1:
            return None

        fragment = text[start:]

        # Remove trailing incomplete values (cut mid-string, mid-number, mid-bool)
        import re
        # Remove trailing incomplete string: ,"key": "incomplete text
        fragment = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', fragment)
        # Remove trailing incomplete number: ,"key": 20
        fragment = re.sub(r',\s*"[^"]*"\s*:\s*\d*$', '', fragment)
        # Remove trailing incomplete key
        fragment = re.sub(r',\s*"[^"]*$', '', fragment)
        # Remove trailing incomplete object/array start
        fragment = re.sub(r',\s*$', '', fragment)

        # Count unclosed brackets
        stack = []
        in_string = False
        escape = False
        for ch in fragment:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()

        # Append missing closers
        closers = ""
        for opener in reversed(stack):
            closers += "]" if opener == "[" else "}"

        repaired = fragment + closers
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                logger.debug("Repaired truncated JSON (added %d closers)", len(closers))
                return result
            if isinstance(result, list):
                return {"items": result}
        except json.JSONDecodeError:
            pass

        return None

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
