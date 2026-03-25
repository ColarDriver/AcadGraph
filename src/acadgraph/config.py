"""
AcadGraph configuration management.

Loads settings from environment variables (or .env file).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Try loading .env from project root."""
    # Walk upward to find .env
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return
    load_dotenv()  # fallback to default behaviour


_env_loaded = False


@dataclass(frozen=True)
class LLMConfig:
    api_base: str = field(default_factory=lambda: os.getenv("LLM_API_BASE", "http://localhost:8000/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "no-key"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "local-model"))
    temperature: float = 0.0
    max_tokens: int = 10240
    max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "5"))
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    api_base: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_BASE", "http://localhost:8001/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "no-key"))
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B"))
    dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "4096")))


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "acadgraph2024"))


@dataclass(frozen=True)
class QdrantConfig:
    host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


def get_config() -> AppConfig:
    """Return a fresh configuration snapshot.

    The first call triggers ``.env`` loading to avoid import-time side effects.
    """
    global _env_loaded
    if not _env_loaded:
        _load_env()
        _env_loaded = True
    return AppConfig()
