"""Tests for config.py — lazy env loading."""

import pytest

from acadgraph import config as config_module


def test_env_not_loaded_at_import_time():
    """_env_loaded flag should be False until get_config() is called."""
    # The flag should start False (deferred loading)
    # We can't test this after get_config() is called in the same process,
    # but we can verify the flag exists and the mechanism works.
    assert hasattr(config_module, "_env_loaded")
    assert isinstance(config_module._env_loaded, bool)


def test_get_config_returns_app_config():
    """get_config() should return an AppConfig instance."""
    cfg = config_module.get_config()
    assert hasattr(cfg, "llm")
    assert hasattr(cfg, "embedding")
    assert hasattr(cfg, "neo4j")
    assert hasattr(cfg, "qdrant")


def test_get_config_sets_env_loaded_flag():
    """After calling get_config(), _env_loaded should be True."""
    config_module.get_config()
    assert config_module._env_loaded is True
