"""
Post-Ingest Entity Semantic Merge (Plan B).

After all papers are ingested, this script:
1. Loads all entities of each type from Neo4j
2. Embeds entity names using the same embedding model
3. Finds similar entity pairs via cosine similarity clustering
4. Uses LLM to batch-decide which pairs should be merged
5. Executes merges in Neo4j (rewires all relationships)

Usage:
    python scripts/semantic_entity_merge.py [--dry-run] [--threshold 0.85] [--types METHOD,TASK]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
from neo4j import GraphDatabase
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "acadgraph2024")

EMBEDDING_BASE = os.getenv("EMBEDDING_API_BASE", "http://127.0.0.1:8002/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")

LLM_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3-32B")

ENTITY_TYPES = ["Method", "Task", "Dataset", "Metric", "Model", "Framework", "Concept"]


# ─── Data ──────────────────────────────────────────────────────────────
@dataclass
class EntityNode:
    neo_id: int           # Neo4j internal ID
    entity_id: str        # Our entity_id property
    name: str
    entity_type: str
    embedding: list[float] | None = None


# ─── Step 1: Load entities from Neo4j ──────────────────────────────────
def load_entities(driver, entity_type: str) -> list[EntityNode]:
    """Load all entities of a given type from Neo4j."""
    with driver.session() as s:
        records = list(s.run(
            f"MATCH (e:{entity_type}) RETURN id(e) AS neo_id, e.entity_id AS eid, e.name AS name"
        ))
    entities = []
    for r in records:
        name = r["name"] or ""
        if not name.strip():
            continue
        entities.append(EntityNode(
            neo_id=r["neo_id"],
            entity_id=r["eid"] or "",
            name=name.strip(),
            entity_type=entity_type,
        ))
    return entities


# ─── Step 2: Embed entity names ───────────────────────────────────────
async def embed_entities(entities: list[EntityNode], batch_size: int = 64) -> None:
    """Embed all entity names using the embedding API."""
    client = AsyncOpenAI(base_url=EMBEDDING_BASE, api_key="no-key")
    
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        texts = [e.name for e in batch]
        try:
            resp = await client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
            for j, emb_data in enumerate(resp.data):
                batch[j].embedding = emb_data.embedding
        except Exception as e:
            logger.error("Embedding failed for batch %d: %s", i, e)
            for ent in batch:
                ent.embedding = None
        
        if (i // batch_size) % 10 == 0:
            logger.info("  Embedded %d/%d entities", min(i + batch_size, len(entities)), len(entities))
    
    await client.close()


# ─── Step 3: Find similar pairs via cosine similarity ──────────────────
def find_similar_pairs(
    entities: list[EntityNode], threshold: float = 0.85
) -> list[tuple[EntityNode, EntityNode, float]]:
    """Find entity pairs with cosine similarity >= threshold."""
    # Filter entities with valid embeddings
    valid = [e for e in entities if e.embedding is not None]
    if len(valid) < 2:
        return []
    
    # Build matrix
    names = [e.name.lower() for e in valid]
    matrix = np.array([e.embedding for e in valid])
    
    # Normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix_norm = matrix / norms
    
    # Compute similarity (block-wise to save memory)
    pairs: list[tuple[EntityNode, EntityNode, float]] = []
    block_size = 500
    
    for i in range(0, len(valid), block_size):
        end_i = min(i + block_size, len(valid))
        block = matrix_norm[i:end_i]
        
        # Compare with all entities after this block
        for j in range(i, len(valid), block_size):
            end_j = min(j + block_size, len(valid))
            target = matrix_norm[j:end_j]
            
            sim = block @ target.T  # (block_size, block_size)
            
            for bi in range(sim.shape[0]):
                for bj in range(sim.shape[1]):
                    abs_i = i + bi
                    abs_j = j + bj
                    if abs_i >= abs_j:  # Skip self and already-seen pairs
                        continue
                    if names[abs_i] == names[abs_j]:  # Already exact match deduped
                        continue
                    if sim[bi, bj] >= threshold:
                        pairs.append((valid[abs_i], valid[abs_j], float(sim[bi, bj])))
    
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
- "Object Detection" and "Object Detection Task" → SAME (redundant word)

Return JSON:
```json
{
  "decisions": [
    {"pair_id": 0, "same": true, "canonical_name": "preferred name"},
    {"pair_id": 1, "same": false}
  ]
}
```

Pairs to evaluate:
"""


async def llm_decide_merges(
    pairs: list[tuple[EntityNode, EntityNode, float]],
    batch_size: int = 20,
) -> list[tuple[EntityNode, EntityNode, str]]:
    """Use LLM to decide which pairs should be merged."""
    client = AsyncOpenAI(base_url=LLM_BASE, api_key="no-key")
    merges: list[tuple[EntityNode, EntityNode, str]] = []
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        
        pair_text = "\n".join([
            f'{j}. "{a.name}" vs "{b.name}" (similarity: {sim:.3f})'
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
            # Extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                for dec in result.get("decisions", []):
                    pid = dec.get("pair_id", -1)
                    if 0 <= pid < len(batch) and dec.get("same", False):
                        a, b, _ = batch[pid]
                        canonical = dec.get("canonical_name", a.name)
                        merges.append((a, b, canonical))
        except Exception as e:
            logger.warning("LLM decision failed for batch %d: %s", i, e)
        
        if (i // batch_size) % 5 == 0:
            logger.info("  LLM decided %d/%d pairs, %d merges so far",
                        min(i + batch_size, len(pairs)), len(pairs), len(merges))
    
    await client.close()
    return merges


# ─── Step 5: Execute merges in Neo4j ──────────────────────────────────
def execute_merges(
    driver,
    merges: list[tuple[EntityNode, EntityNode, str]],
    dry_run: bool = False,
) -> int:
    """Merge entity pairs in Neo4j."""
    merged = 0
    
    with driver.session() as s:
        for keeper, dupe, canonical_name in merges:
            if dry_run:
                logger.info("  [DRY-RUN] Would merge '%s' ← '%s' → '%s'",
                            keeper.name, dupe.name, canonical_name)
                merged += 1
                continue
            
            try:
                # Move all incoming relationships from dupe to keeper
                s.run("""
                    MATCH (dupe) WHERE id(dupe) = $did
                    MATCH (dupe)<-[r]-(src)
                    WHERE id(src) <> $kid
                    WITH src, type(r) AS rtype, properties(r) AS props, dupe
                    MATCH (keeper) WHERE id(keeper) = $kid
                    CALL apoc.create.relationship(src, rtype, props, keeper) YIELD rel
                    RETURN count(rel)
                """, did=dupe.neo_id, kid=keeper.neo_id)
            except Exception:
                # Fallback without APOC
                rels_in = list(s.run("""
                    MATCH (dupe) WHERE id(dupe) = $did
                    MATCH (dupe)<-[r]-(src)
                    WHERE id(src) <> $kid
                    RETURN type(r) AS rtype, id(src) AS src_id
                """, did=dupe.neo_id, kid=keeper.neo_id))
                
                for rel in rels_in:
                    s.run(f"""
                        MATCH (src) WHERE id(src) = $sid
                        MATCH (keeper) WHERE id(keeper) = $kid
                        MERGE (src)-[:{rel['rtype']}]->(keeper)
                    """, sid=rel["src_id"], kid=keeper.neo_id)
            
            try:
                # Move all outgoing relationships from dupe to keeper
                s.run("""
                    MATCH (dupe) WHERE id(dupe) = $did
                    MATCH (dupe)-[r]->(dst)
                    WHERE id(dst) <> $kid
                    WITH dst, type(r) AS rtype, properties(r) AS props, dupe
                    MATCH (keeper) WHERE id(keeper) = $kid
                    CALL apoc.create.relationship(keeper, rtype, props, dst) YIELD rel
                    RETURN count(rel)
                """, did=dupe.neo_id, kid=keeper.neo_id)
            except Exception:
                rels_out = list(s.run("""
                    MATCH (dupe) WHERE id(dupe) = $did
                    MATCH (dupe)-[r]->(dst)
                    WHERE id(dst) <> $kid
                    RETURN type(r) AS rtype, id(dst) AS dst_id
                """, did=dupe.neo_id, kid=keeper.neo_id))
                
                for rel in rels_out:
                    s.run(f"""
                        MATCH (keeper) WHERE id(keeper) = $kid
                        MATCH (dst) WHERE id(dst) = $did
                        MERGE (keeper)-[:{rel['rtype']}]->(dst)
                    """, kid=keeper.neo_id, did=rel["dst_id"])
            
            # Update keeper name to canonical name and delete dupe
            s.run("""
                MATCH (keeper) WHERE id(keeper) = $kid
                SET keeper.name = $name
            """, kid=keeper.neo_id, name=canonical_name)
            
            s.run("MATCH (n) WHERE id(n) = $did DETACH DELETE n", did=dupe.neo_id)
            merged += 1
    
    return merged


# ─── Main ──────────────────────────────────────────────────────────────
async def process_entity_type(
    driver, entity_type: str, threshold: float, dry_run: bool
) -> dict:
    """Process a single entity type end-to-end."""
    logger.info("=" * 60)
    logger.info("Processing: %s", entity_type)
    
    # Step 1: Load
    entities = load_entities(driver, entity_type)
    logger.info("  Loaded %d entities", len(entities))
    
    if len(entities) < 2:
        return {"type": entity_type, "total": len(entities), "pairs": 0, "merged": 0}
    
    # Step 2: Embed
    logger.info("  Embedding entity names...")
    await embed_entities(entities)
    
    # Step 3: Find similar pairs
    logger.info("  Finding similar pairs (threshold=%.2f)...", threshold)
    pairs = find_similar_pairs(entities, threshold)
    logger.info("  Found %d candidate pairs", len(pairs))
    
    if not pairs:
        return {"type": entity_type, "total": len(entities), "pairs": 0, "merged": 0}
    
    # Log top pairs for debugging
    for a, b, sim in pairs[:10]:
        logger.info("    %.3f: '%s' <-> '%s'", sim, a.name, b.name)
    
    # Step 4: LLM decision
    logger.info("  Sending %d pairs to LLM for merge decision...", len(pairs))
    merges = await llm_decide_merges(pairs)
    logger.info("  LLM approved %d merges", len(merges))
    
    # Step 5: Execute
    prefix = "[DRY-RUN] " if dry_run else ""
    logger.info("  %sExecuting merges...", prefix)
    n = execute_merges(driver, merges, dry_run=dry_run)
    logger.info("  %sMerged %d entities", prefix, n)
    
    return {"type": entity_type, "total": len(entities), "pairs": len(pairs), "merged": n}


async def main():
    parser = argparse.ArgumentParser(description="Semantic entity merge (post-ingest)")
    parser.add_argument("--dry-run", action="store_true", help="Print merges without executing")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold")
    parser.add_argument("--types", type=str, default=",".join(ENTITY_TYPES),
                        help="Comma-separated entity types to process")
    args = parser.parse_args()
    
    types = [t.strip() for t in args.types.split(",")]
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    
    logger.info("Semantic Entity Merge")
    logger.info("  Threshold: %.2f", args.threshold)
    logger.info("  Types: %s", types)
    logger.info("  Dry run: %s", args.dry_run)
    
    results = []
    for entity_type in types:
        result = await process_entity_type(driver, entity_type, args.threshold, args.dry_run)
        results.append(result)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("%-12s %8s %8s %8s", "Type", "Total", "Pairs", "Merged")
    logger.info("-" * 40)
    for r in results:
        logger.info("%-12s %8d %8d %8d", r["type"], r["total"], r["pairs"], r["merged"])
    
    total_merged = sum(r["merged"] for r in results)
    logger.info("-" * 40)
    logger.info("Total merged: %d", total_merged)
    
    driver.close()


if __name__ == "__main__":
    asyncio.run(main())
