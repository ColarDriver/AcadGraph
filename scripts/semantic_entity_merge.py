"""
Post-Ingest Entity Semantic Merge (Plan B) — ANN-accelerated.

Uses Qdrant ANN search instead of O(n²) brute-force:
  For each entity → Qdrant top-k search → only compare similar pairs → LLM decides

Complexity: O(n × k × log n) instead of O(n²)

Usage:
    python scripts/semantic_entity_merge.py [--dry-run] [--threshold 0.85] [--types Method,Task]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    # Manual .env loading
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from neo4j import GraphDatabase
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "acadgraph2024")

EMBEDDING_BASE = os.getenv("EMBEDDING_API_BASE", "http://127.0.0.1:8002/v1")
EMBEDDING_KEY = os.getenv("EMBEDDING_API_KEY", "no-key")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "4096"))

LLM_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8000/v1")
LLM_KEY = os.getenv("LLM_API_KEY", "no-key")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3-32B")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

ENTITY_TYPES = ["Method", "Task", "Dataset", "Metric", "Model", "Framework", "Concept"]
MERGE_COLLECTION = "entity_merge_tmp"  # Temporary collection for dedup


# ─── Data ──────────────────────────────────────────────────────────────
@dataclass
class EntityNode:
    neo_id: int
    entity_id: str
    name: str
    entity_type: str
    embedding: list[float] | None = None
    point_id: str = ""  # Qdrant point ID


# ─── Step 1: Load entities from Neo4j ──────────────────────────────────
def load_entities(driver, entity_type: str) -> list[EntityNode]:
    """Load all entities of a given type from Neo4j."""
    with driver.session() as s:
        records = list(s.run(
            f"MATCH (e:{entity_type}) RETURN id(e) AS neo_id, e.entity_id AS eid, e.name AS name"
        ))
    entities = []
    seen_names = set()
    for r in records:
        name = (r["name"] or "").strip()
        if not name or len(name) < 2:
            continue
        # Skip exact lowercase duplicates (already handled by hash dedup)
        lower = name.lower()
        if lower in seen_names:
            continue
        seen_names.add(lower)
        entities.append(EntityNode(
            neo_id=r["neo_id"],
            entity_id=r["eid"] or "",
            name=name,
            entity_type=entity_type,
            point_id=str(uuid.uuid4()),
        ))
    return entities


# ─── Step 2: Embed + index in Qdrant (parallel) ───────────────────────
async def embed_and_index(
    entities: list[EntityNode],
    qdrant: QdrantClient,
    entity_type: str,
    batch_size: int = 64,
    max_concurrent: int = 16,
) -> None:
    """Embed entity names and upsert to Qdrant. Runs concurrent batches."""
    client = AsyncOpenAI(base_url=EMBEDDING_BASE, api_key=EMBEDDING_KEY)
    sem = asyncio.Semaphore(max_concurrent)
    progress = {"done": 0}
    total = len(entities)
    qdrant_lock = asyncio.Lock()

    async def process_batch(batch_idx: int, batch: list[EntityNode]):
        async with sem:
            texts = [e.name for e in batch]
            try:
                resp = await client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
                points = []
                for j, emb_data in enumerate(resp.data):
                    batch[j].embedding = emb_data.embedding
                    points.append(PointStruct(
                        id=batch[j].point_id,
                        vector=emb_data.embedding,
                        payload={
                            "name": batch[j].name,
                            "neo_id": batch[j].neo_id,
                            "entity_id": batch[j].entity_id,
                            "entity_type": entity_type,
                        },
                    ))
                async with qdrant_lock:
                    qdrant.upsert(collection_name=MERGE_COLLECTION, points=points)
            except Exception as e:
                logger.error("Embedding/index failed at batch %d: %s", batch_idx, e)

            progress["done"] += len(batch)
            if progress["done"] % 2000 < batch_size:
                logger.info("  Embedded & indexed %d/%d entities", progress["done"], total)

    # Create all tasks
    tasks = []
    for i in range(0, total, batch_size):
        batch = entities[i:i + batch_size]
        tasks.append(process_batch(i, batch))

    # Run all concurrently (bounded by semaphore)
    await asyncio.gather(*tasks)
    logger.info("  Embedded & indexed %d/%d entities (done)", total, total)

    await client.close()


# ─── Step 3: ANN search for similar pairs ──────────────────────────────
def find_similar_pairs_ann(
    entities: list[EntityNode],
    qdrant: QdrantClient,
    entity_type: str,
    threshold: float = 0.85,
    top_k: int = 10,
) -> list[tuple[EntityNode, EntityNode, float]]:
    """Use Qdrant ANN search to find similar entity pairs. O(n × k × log n)."""
    # Build lookup: point_id → entity
    id_to_entity = {e.point_id: e for e in entities}
    # Track seen pairs to avoid duplicates
    seen_pairs: set[tuple[str, str]] = set()
    pairs: list[tuple[EntityNode, EntityNode, float]] = []

    for entity in entities:
        if entity.embedding is None:
            continue

        try:
            hits = qdrant.query_points(
                collection_name=MERGE_COLLECTION,
                query=entity.embedding,
                limit=top_k + 1,  # +1 because self is included
                query_filter=Filter(
                    must=[FieldCondition(key="entity_type", match=MatchValue(value=entity_type))]
                ),
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning("ANN search failed for '%s': %s", entity.name, e)
            continue

        for hit in hits:
            hit_id = hit.id
            score = hit.score

            # Skip self
            if hit_id == entity.point_id:
                continue
            # Skip below threshold
            if score < threshold:
                continue
            # Skip already-seen pair (canonical ordering)
            pair_key = tuple(sorted([entity.point_id, hit_id]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Skip exact name match (already deduped)
            hit_name = hit.payload.get("name", "")
            if entity.name.lower() == hit_name.lower():
                continue

            other = id_to_entity.get(hit_id)
            if other:
                pairs.append((entity, other, score))

    # Sort by similarity descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ─── Step 4: LLM batch decision ───────────────────────────────────────
MERGE_PROMPT = """You are an academic entity deduplication expert.
For each pair of entity names below, decide if they refer to the SAME concept/method/task.

Rules:
- "SGD" and "Stochastic Gradient Descent" → SAME (acronym vs full name)
- "ResNet" and "ResNet-50" → DIFFERENT (specific variant)
- "Image Classification" and "Image Recognition" → SAME (synonyms)
- "GAN" and "Generative Adversarial Network" → SAME
- "BERT" and "RoBERTa" → DIFFERENT (different models)
- "Object Detection" and "Object Detection Task" → SAME (redundant suffix)

Return JSON only:
{"decisions": [{"pair_id": 0, "same": true, "canonical_name": "preferred name"}, ...]}

Pairs to evaluate:
"""


async def llm_decide_merges(
    pairs: list[tuple[EntityNode, EntityNode, float]],
    batch_size: int = 20,
    max_concurrent: int = 8,
) -> list[tuple[EntityNode, EntityNode, str]]:
    """Use LLM to decide which pairs should be merged. Runs concurrent batches."""
    client = AsyncOpenAI(base_url=LLM_BASE, api_key=LLM_KEY)
    merges: list[tuple[EntityNode, EntityNode, str]] = []
    sem = asyncio.Semaphore(max_concurrent)

    async def process_batch(batch_idx: int, batch: list):
        async with sem:
            pair_text = "\n".join([
                f'{j}. "{a.name}" vs "{b.name}" (sim: {sim:.3f})'
                for j, (a, b, sim) in enumerate(batch)
            ])
            prompt = MERGE_PROMPT + pair_text

            try:
                resp = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2048,
                    extra_body={"enable_thinking": False},
                )
                content = resp.choices[0].message.content or ""

                # Clean response
                # 1. Strip thinking tags
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                # 2. Strip code fences (including unclosed)
                content = re.sub(r'^```(?:json)?\s*\n?', '', content.strip())
                content = re.sub(r'\n?```\s*$', '', content)
                content = content.strip()

                # 3. Try parsing
                result = None
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    raw = json_match.group()
                    try:
                        result = json.loads(raw)
                    except json.JSONDecodeError:
                        # Try repair: strip trailing incomplete fields
                        repaired = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', raw)
                        repaired = re.sub(r',\s*"[^"]*"\s*:\s*\d*$', '', repaired)
                        repaired = re.sub(r',\s*"[^"]*$', '', repaired)
                        repaired = re.sub(r',\s*$', '', repaired)
                        # Close unclosed brackets
                        stack = []
                        in_str = False
                        esc = False
                        for ch in repaired:
                            if esc: esc = False; continue
                            if ch == '\\': esc = True; continue
                            if ch == '"': in_str = not in_str; continue
                            if in_str: continue
                            if ch in '{[': stack.append(ch)
                            elif ch == '}' and stack and stack[-1] == '{': stack.pop()
                            elif ch == ']' and stack and stack[-1] == '[': stack.pop()
                        for opener in reversed(stack):
                            repaired += ']' if opener == '[' else '}'
                        try:
                            result = json.loads(repaired)
                        except json.JSONDecodeError:
                            logger.debug("JSON repair failed for batch %d: %s", batch_idx, repaired[:200])

                if result:
                    for dec in result.get("decisions", []):
                        pid = dec.get("pair_id", -1)
                        if 0 <= pid < len(batch) and dec.get("same", False):
                            a, b, _ = batch[pid]
                            canonical = dec.get("canonical_name", a.name)
                            merges.append((a, b, canonical))
                else:
                    logger.debug("No JSON found in LLM batch %d response: %s", batch_idx, content[:150])
            except Exception as e:
                logger.warning("LLM batch %d failed: %s", batch_idx, e)

    # Create all tasks
    tasks = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        tasks.append(process_batch(i, batch))

    # Run with progress logging
    total_batches = len(tasks)
    logger.info("  Dispatching %d LLM batches (%d concurrent)...", total_batches, max_concurrent)
    await asyncio.gather(*tasks)
    logger.info("  LLM approved %d merges from %d pairs", len(merges), len(pairs))

    await client.close()
    return merges


# ─── Step 5: Execute merges in Neo4j ──────────────────────────────────
def execute_merges(
    driver,
    merges: list[tuple[EntityNode, EntityNode, str]],
    dry_run: bool = False,
) -> int:
    """Merge entity pairs in Neo4j. Uses Union-Find to handle transitive merges."""
    if not merges:
        return 0

    # Union-Find to handle chains: if A≈B and B≈C, merge all three
    parent: dict[int, int] = {}
    canonical: dict[int, str] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int, name: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            canonical[ra] = name

    for keeper, dupe, canon in merges:
        union(keeper.neo_id, dupe.neo_id, canon)

    # Group by root
    groups: dict[int, list[int]] = {}
    all_ids = set()
    for keeper, dupe, _ in merges:
        all_ids.add(keeper.neo_id)
        all_ids.add(dupe.neo_id)
    for nid in all_ids:
        root = find(nid)
        groups.setdefault(root, []).append(nid)

    merged = 0
    with driver.session() as s:
        for root_id, members in groups.items():
            dupes = [m for m in members if m != root_id]
            if not dupes:
                continue

            canon_name = canonical.get(root_id, "")

            if dry_run:
                logger.info("  [DRY-RUN] Merge group: keeper=%d, dupes=%s → '%s'",
                            root_id, dupes, canon_name)
                merged += len(dupes)
                continue

            for dupe_id in dupes:
                # Move incoming rels
                rels_in = list(s.run("""
                    MATCH (dupe) WHERE id(dupe) = $did
                    MATCH (dupe)<-[r]-(src)
                    WHERE id(src) <> $kid
                    RETURN type(r) AS rtype, id(src) AS src_id
                """, did=dupe_id, kid=root_id))

                for rel in rels_in:
                    s.run(f"""
                        MATCH (src) WHERE id(src) = $sid
                        MATCH (keeper) WHERE id(keeper) = $kid
                        MERGE (src)-[:{rel['rtype']}]->(keeper)
                    """, sid=rel["src_id"], kid=root_id)

                # Move outgoing rels
                rels_out = list(s.run("""
                    MATCH (dupe) WHERE id(dupe) = $did
                    MATCH (dupe)-[r]->(dst)
                    WHERE id(dst) <> $kid
                    RETURN type(r) AS rtype, id(dst) AS dst_id
                """, did=dupe_id, kid=root_id))

                for rel in rels_out:
                    s.run(f"""
                        MATCH (keeper) WHERE id(keeper) = $kid
                        MATCH (dst) WHERE id(dst) = $did2
                        MERGE (keeper)-[:{rel['rtype']}]->(dst)
                    """, kid=root_id, did2=rel["dst_id"])

                # Delete dupe
                s.run("MATCH (n) WHERE id(n) = $did DETACH DELETE n", did=dupe_id)
                merged += 1

            # Update keeper name
            if canon_name:
                s.run("MATCH (n) WHERE id(n) = $kid SET n.name = $name",
                      kid=root_id, name=canon_name)

    return merged


# ─── Phase 0: Exact + Normalized string match merge ───────────────────
def _normalize_name(name: str) -> str:
    """Normalize entity name for fuzzy exact matching.
    
    Handles: en-dash/em-dash, apostrophes, hyphens, plurals, etc.
    """
    s = name.strip().lower()
    # Normalize unicode dashes to ASCII hyphen
    s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u2012', '-')
    # Remove apostrophes and backticks
    s = s.replace("'", "").replace("'", "").replace("`", "")
    # Normalize hyphens: "trade-off" → "tradeoff", "f1-score" → "f1 score"
    s = s.replace("-", " ")
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # Remove trailing "s" for simple plural normalization (but not if 2 chars or less)
    if len(s) > 3 and s.endswith('s') and not s.endswith('ss'):
        s = s[:-1]
    return s


def exact_merge(driver, entity_type: str, dry_run: bool = False) -> tuple[int, list[str]]:
    """Merge entities with identical normalized names. No LLM needed.
    Returns (merged_count, deleted_entity_ids)."""
    merged = 0
    deleted_entity_ids: list[str] = []
    with driver.session() as s:
        # Load all entities
        records = list(s.run(f"""
            MATCH (e:{entity_type})
            WHERE e.name IS NOT NULL
            RETURN id(e) AS neo_id, e.name AS name, e.entity_id AS eid
        """))

        # Group by normalized name
        from collections import defaultdict
        groups: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for r in records:
            name = r["name"]
            norm = _normalize_name(name)
            groups[norm].append((r["neo_id"], name, r["eid"] or ""))

        # Filter to groups with duplicates
        dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
        logger.info("  Phase 0: Found %d groups of normalized duplicates (%d entities to merge)",
                     len(dup_groups), sum(len(v) - 1 for v in dup_groups.values()))

        for norm_name, members in dup_groups.items():
            # Pick the "best" name as keeper (prefer shortest)
            members.sort(key=lambda x: len(x[1]))
            keeper_id, keeper_name, keeper_eid = members[0]
            dupes = members[1:]

            if dry_run:
                if len(dupes) <= 3:
                    names = [n for _, n, _ in dupes]
                    logger.debug("  [DRY-RUN] Merge: '%s' ← %s", keeper_name, names)
                merged += len(dupes)
                deleted_entity_ids.extend(eid for _, _, eid in dupes if eid)
                continue

            for dupe_id, dupe_name, dupe_eid in dupes:
                # Move incoming relationships
                rels_in = list(s.run("""
                    MATCH (dupe) WHERE id(dupe) = $did
                    MATCH (dupe)<-[r]-(src)
                    WHERE id(src) <> $kid
                    RETURN type(r) AS rtype, id(src) AS src_id
                """, did=dupe_id, kid=keeper_id))

                for rel in rels_in:
                    s.run(f"""
                        MATCH (src) WHERE id(src) = $sid
                        MATCH (keeper) WHERE id(keeper) = $kid
                        MERGE (src)-[:{rel['rtype']}]->(keeper)
                    """, sid=rel["src_id"], kid=keeper_id)

                # Move outgoing relationships
                rels_out = list(s.run("""
                    MATCH (dupe) WHERE id(dupe) = $did
                    MATCH (dupe)-[r]->(dst)
                    WHERE id(dst) <> $kid
                    RETURN type(r) AS rtype, id(dst) AS dst_id
                """, did=dupe_id, kid=keeper_id))

                for rel in rels_out:
                    s.run(f"""
                        MATCH (keeper) WHERE id(keeper) = $kid
                        MATCH (dst) WHERE id(dst) = $did2
                        MERGE (keeper)-[:{rel['rtype']}]->(dst)
                    """, kid=keeper_id, did2=rel["dst_id"])

                # Delete duplicate
                s.run("MATCH (n) WHERE id(n) = $did DETACH DELETE n", did=dupe_id)
                if dupe_eid:
                    deleted_entity_ids.append(dupe_eid)
                merged += 1

            if merged % 1000 == 0 and merged > 0:
                logger.info("  Phase 0: Merged %d duplicates so far...", merged)

    return merged, deleted_entity_ids


# ─── Main ──────────────────────────────────────────────────────────────
async def process_entity_type(
    driver, qdrant: QdrantClient, entity_type: str, threshold: float, dry_run: bool
) -> dict:
    """Process a single entity type end-to-end."""
    logger.info("=" * 60)
    logger.info("Processing: %s", entity_type)

    # Phase 0: Exact string match merge (fast, no LLM)
    logger.info("  Phase 0: Exact string match merge...")
    exact_merged, deleted_eids = exact_merge(driver, entity_type, dry_run=dry_run)
    logger.info("  Phase 0: Merged %d exact duplicates", exact_merged)

    # Clean up Qdrant: delete embeddings for merged entities
    if deleted_eids and not dry_run:
        QDRANT_ENTITY_COLLECTION = "entities"
        batch_size = 500
        for i in range(0, len(deleted_eids), batch_size):
            batch = deleted_eids[i:i + batch_size]
            try:
                qdrant.delete(
                    collection_name=QDRANT_ENTITY_COLLECTION,
                    points_selector=batch,
                )
            except Exception as e:
                logger.debug("Qdrant cleanup skipped (batch %d): %s", i, e)
        logger.info("  Cleaned %d embeddings from Qdrant", len(deleted_eids))

    # Step 1: Load (after exact merge, count is reduced)
    entities = load_entities(driver, entity_type)
    logger.info("  Loaded %d unique entities (after exact merge)", len(entities))

    if len(entities) < 2:
        return {"type": entity_type, "total": len(entities), "pairs": 0,
                "exact_merged": exact_merged, "semantic_merged": 0}

    # Step 2: Embed & index in Qdrant
    logger.info("  Embedding & indexing in Qdrant...")
    await embed_and_index(entities, qdrant, entity_type)

    # Step 3: ANN search for similar pairs
    logger.info("  ANN searching for similar pairs (threshold=%.2f)...", threshold)
    pairs = find_similar_pairs_ann(entities, qdrant, entity_type, threshold)
    logger.info("  Found %d candidate pairs", len(pairs))

    if not pairs:
        return {"type": entity_type, "total": len(entities), "pairs": 0,
                "exact_merged": exact_merged, "semantic_merged": 0}

    # Log top pairs
    for a, b, sim in pairs[:15]:
        logger.info("    %.3f: '%s' <-> '%s'", sim, a.name, b.name)
    if len(pairs) > 15:
        logger.info("    ... and %d more pairs", len(pairs) - 15)

    # Step 4: LLM decision
    logger.info("  Sending %d pairs to LLM for merge decision...", len(pairs))
    merges = await llm_decide_merges(pairs)

    # Step 5: Execute
    prefix = "[DRY-RUN] " if dry_run else ""
    logger.info("  %sExecuting merges...", prefix)
    n = execute_merges(driver, merges, dry_run=dry_run)
    logger.info("  %sMerged %d entities", prefix, n)

    return {"type": entity_type, "total": len(entities), "pairs": len(pairs),
            "exact_merged": exact_merged, "semantic_merged": n}


async def main():
    parser = argparse.ArgumentParser(description="Semantic entity merge (ANN-accelerated)")
    parser.add_argument("--dry-run", action="store_true", help="Print merges without executing")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold")
    parser.add_argument("--types", type=str, default=",".join(ENTITY_TYPES),
                        help="Comma-separated entity types to process")
    parser.add_argument("--top-k", type=int, default=10, help="Number of ANN neighbors to check")
    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",")]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Create/recreate temporary merge collection
    try:
        qdrant.delete_collection(MERGE_COLLECTION)
    except Exception:
        pass
    qdrant.create_collection(
        MERGE_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.info("Created temporary Qdrant collection: %s", MERGE_COLLECTION)

    logger.info("Semantic Entity Merge (ANN-accelerated)")
    logger.info("  Threshold: %.2f, Top-k: %d", args.threshold, args.top_k)
    logger.info("  Types: %s", types)
    logger.info("  Dry run: %s", args.dry_run)

    results = []
    for entity_type in types:
        # Clear collection between types
        qdrant.delete_collection(MERGE_COLLECTION)
        qdrant.create_collection(
            MERGE_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        result = await process_entity_type(driver, qdrant, entity_type, args.threshold, args.dry_run)
        results.append(result)

    # Cleanup
    try:
        qdrant.delete_collection(MERGE_COLLECTION)
    except Exception:
        pass

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("%-12s %8s %8s %10s %10s", "Type", "After P0", "Pairs", "ExactMerge", "LLM Merge")
    logger.info("-" * 55)
    for r in results:
        logger.info("%-12s %8d %8d %10d %10d",
                    r["type"], r["total"], r.get("pairs", 0),
                    r.get("exact_merged", 0), r.get("semantic_merged", 0))

    total_exact = sum(r.get("exact_merged", 0) for r in results)
    total_semantic = sum(r.get("semantic_merged", 0) for r in results)
    logger.info("-" * 55)
    logger.info("Total: exact=%d  semantic=%d  combined=%d", total_exact, total_semantic, total_exact + total_semantic)

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())
