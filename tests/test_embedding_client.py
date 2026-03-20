"""Tests for embedding_client.py — caching and context manager."""

import asyncio

import pytest

from acadgraph.embedding_client import EmbeddingClient


class TestEmbeddingCacheKey:
    """Verify cache key generation uses deterministic hashing."""

    def test_same_text_same_key(self):
        """Identical text should produce identical cache keys."""
        k1 = EmbeddingClient._cache_key("hello world")
        k2 = EmbeddingClient._cache_key("hello world")
        assert k1 == k2

    def test_different_text_different_key(self):
        """Different text should produce different keys."""
        k1 = EmbeddingClient._cache_key("hello")
        k2 = EmbeddingClient._cache_key("world")
        assert k1 != k2

    def test_key_is_deterministic(self):
        """Key should be based on SHA-256, not built-in hash."""
        import hashlib
        text = "test string"
        expected = hashlib.sha256(text.encode()).hexdigest()[:24]
        actual = EmbeddingClient._cache_key(text)
        assert actual == expected

    def test_key_length(self):
        """Cache key should be truncated to 24 chars."""
        key = EmbeddingClient._cache_key("any text")
        assert len(key) == 24


class TestEmbeddingClientContextManager:
    """Verify async context manager protocol."""

    def test_has_context_manager_methods(self):
        assert hasattr(EmbeddingClient, "__aenter__")
        assert hasattr(EmbeddingClient, "__aexit__")
