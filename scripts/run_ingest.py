#!/usr/bin/env python3
"""
Batch JSONL Ingestion CLI — Streams papers from JSONL into AcadGraph KG.

Usage:
    python scripts/run_ingest.py --input data/paper_reviews_dataset.jsonl --limit 10
    python scripts/run_ingest.py --input data/paper_reviews_dataset.jsonl --batch-size 5 --skip-existing
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import click

# Ensure src is importable when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acadgraph.config import get_config
from acadgraph.embedding_client import EmbeddingClient
from acadgraph.kg.incremental_builder import IncrementalKGBuilder
from acadgraph.kg.jsonl_loader import iter_jsonl, jsonl_to_paper_source
from acadgraph.kg.storage.neo4j_store import Neo4jKGStore
from acadgraph.kg.storage.qdrant_store import QdrantKGStore
from acadgraph.llm_client import LLMClient

logger = logging.getLogger("acadgraph.ingest")

CHECKPOINT_FILE = "ingest_checkpoint.json"


def load_checkpoint(path: str) -> set[str]:
    """Load set of already-processed paper IDs."""
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text())
        return set(data.get("processed_ids", []))
    return set()


def save_checkpoint(path: str, processed_ids: set[str]) -> None:
    """Save processed paper IDs to checkpoint file."""
    Path(path).write_text(
        json.dumps({"processed_ids": sorted(processed_ids)}, indent=2)
    )


@click.command()
@click.option("--input", "input_path", required=True, help="Path to JSONL file")
@click.option("--limit", default=None, type=int, help="Max papers to process")
@click.option("--batch-size", default=5, type=int, help="Concurrent batch size")
@click.option("--skip-existing", is_flag=True, help="Skip papers already in Neo4j")
@click.option("--checkpoint", default=CHECKPOINT_FILE, help="Checkpoint file path")
@click.option("--log-level", default="INFO", help="Logging level")
def main(
    input_path: str,
    limit: int | None,
    batch_size: int,
    skip_existing: bool,
    checkpoint: str,
    log_level: str,
) -> None:
    """Ingest papers from JSONL into AcadGraph knowledge graph."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = get_config()
    processed_ids = load_checkpoint(checkpoint)
    logger.info("Loaded %d previously processed IDs from checkpoint", len(processed_ids))

    asyncio.run(
        _run_ingestion(
            input_path=input_path,
            config=config,
            limit=limit,
            batch_size=batch_size,
            skip_existing=skip_existing,
            checkpoint_path=checkpoint,
            processed_ids=processed_ids,
        )
    )


async def _run_ingestion(
    input_path: str,
    config,
    limit: int | None,
    batch_size: int,
    skip_existing: bool,
    checkpoint_path: str,
    processed_ids: set[str],
) -> None:
    """Main async ingestion loop."""
    # Initialize services
    neo4j_store = Neo4jKGStore(config.neo4j)
    qdrant_store = QdrantKGStore(config.qdrant, embedding_dim=config.embedding.dim)
    llm = LLMClient(config.llm)
    embedding = EmbeddingClient(config.embedding)

    await neo4j_store.connect()
    await neo4j_store.init_schema()
    await qdrant_store.connect()
    await qdrant_store.init_collections()

    builder = IncrementalKGBuilder(
        neo4j_store=neo4j_store,
        qdrant_store=qdrant_store,
        llm=llm,
        embedding=embedding,
    )

    # Stream and process
    skip_ids = processed_ids if skip_existing else None
    records = list(iter_jsonl(input_path, limit=limit, skip_ids=skip_ids))
    total = len(records)
    logger.info("Loaded %d records to process", total)

    # Statistics
    stats = {
        "total": total,
        "success": 0,
        "failed": 0,
        "entities": 0,
        "claims": 0,
        "evidences": 0,
    }
    start_time = time.time()

    # Process in batches
    for i in range(0, total, batch_size):
        batch_records = records[i : i + batch_size]
        batch_sources = [jsonl_to_paper_source(r) for r in batch_records]

        logger.info(
            "Processing batch %d-%d of %d",
            i + 1,
            min(i + batch_size, total),
            total,
        )

        batch_result = await builder.add_papers_batch(batch_sources, batch_size=batch_size)

        # Update stats
        stats["success"] += batch_result.successful
        stats["failed"] += batch_result.failed
        for r in batch_result.results:
            stats["entities"] += r.entities_added
            stats["claims"] += r.claims_added
            stats["evidences"] += r.evidences_added

        # Update checkpoint
        for r in batch_result.results:
            if r.paper_id:
                processed_ids.add(r.paper_id)
        save_checkpoint(checkpoint_path, processed_ids)

        elapsed = time.time() - start_time
        processed = i + len(batch_records)
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0
        logger.info(
            "Progress: %d/%d (%.1f%%) | Rate: %.1f papers/min | ETA: %.0fs",
            processed,
            total,
            100 * processed / total,
            rate * 60,
            eta,
        )

    # Cleanup
    await neo4j_store.close()
    await qdrant_store.close()
    await llm.close()
    await embedding.close()

    elapsed = time.time() - start_time
    logger.info(
        "\n=== Ingestion Complete ===\n"
        "Total: %d | Success: %d | Failed: %d\n"
        "Entities: %d | Claims: %d | Evidences: %d\n"
        "Duration: %.1fs (%.1f papers/min)",
        stats["total"],
        stats["success"],
        stats["failed"],
        stats["entities"],
        stats["claims"],
        stats["evidences"],
        elapsed,
        stats["total"] / elapsed * 60 if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()
