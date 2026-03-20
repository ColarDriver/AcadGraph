"""Tests for neo4j_store.py — validation, injection prevention, and hashing."""

import pytest

from acadgraph.kg.storage.neo4j_store import _SAFE_PROPERTY_KEY


class TestSafePropertyKeyRegex:
    """Verify the regex used to prevent Cypher injection via property keys."""

    def test_valid_keys(self):
        """Standard snake_case keys should pass."""
        valid = ["name", "source_paper_id", "confidence", "claim_type", "entity_id", "a1"]
        for key in valid:
            assert _SAFE_PROPERTY_KEY.match(key), f"Expected '{key}' to be valid"

    def test_invalid_keys_with_special_chars(self):
        """Keys with special characters should be rejected."""
        invalid = [
            "name; MATCH (n) DETACH DELETE n//",
            "foo.bar",
            "a b",
            "hello-world",
            "",
            " name",
            "name ",
            "123start",
            "a+b",
            "key`injection",
        ]
        for key in invalid:
            assert not _SAFE_PROPERTY_KEY.match(key), f"Expected '{key}' to be rejected"

    def test_underscore_prefix(self):
        """Keys starting with underscore should pass."""
        assert _SAFE_PROPERTY_KEY.match("_private_key")

    def test_all_numeric_after_letter(self):
        """Alphanumeric keys should pass if first char is letter/underscore."""
        assert _SAFE_PROPERTY_KEY.match("x123")
        assert _SAFE_PROPERTY_KEY.match("_456")


class TestQdrantIntId:
    """Verify deterministic hashing for Qdrant point IDs."""

    def test_deterministic_across_calls(self):
        """Same input should always produce the same integer ID."""
        from acadgraph.kg.storage.qdrant_store import QdrantKGStore

        id1 = QdrantKGStore._to_int_id("test_entity_abc")
        id2 = QdrantKGStore._to_int_id("test_entity_abc")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        """Different inputs should produce different IDs (no trivial collisions)."""
        from acadgraph.kg.storage.qdrant_store import QdrantKGStore

        id1 = QdrantKGStore._to_int_id("entity_001")
        id2 = QdrantKGStore._to_int_id("entity_002")
        assert id1 != id2

    def test_fits_int64_range(self):
        """Generated IDs must fit in a signed 64-bit integer."""
        from acadgraph.kg.storage.qdrant_store import QdrantKGStore

        for suffix in ["a", "b", "c", "longprefix_" * 20]:
            val = QdrantKGStore._to_int_id(suffix)
            assert 0 <= val < 2**63, f"ID {val} out of int64 range"

    def test_uses_sha256_not_builtin_hash(self):
        """The implementation should use hashlib, not built-in hash()."""
        import hashlib
        from acadgraph.kg.storage.qdrant_store import QdrantKGStore

        test_id = "verification_string"
        expected_digest = hashlib.sha256(test_id.encode()).hexdigest()
        expected = int(expected_digest[:15], 16)
        actual = QdrantKGStore._to_int_id(test_id)
        assert actual == expected
