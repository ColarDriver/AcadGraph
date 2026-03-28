"""
Post-Ingest Entity Semantic Merge (Plan B) — ANN-accelerated.

Uses Qdrant ANN search instead of O(n²) brute-force:
  For each entity → Qdrant top-k search → only compare similar pairs → LLM decides

Complexity: O(n × k × log n) instead of O(n²)

Usage:
    python scripts/semantic_entity_merge.py [--dry-run] [--threshold 0.85] [--types Method,Task]
"""
# type: ignore

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

# ─── Logging setup (console + file) ────────────────────────────────────
_log_dir = Path(__file__).resolve().parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"semantic_merge_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("semantic_merge")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")

_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

_fh = logging.FileHandler(_log_file, encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

logger.info("Log file: %s", _log_file)

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
    seen_pairs: set[tuple[str, ...]] = set()
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


# ─── Step 3b: Build clusters from ANN pairs ───────────────────────────
def build_clusters_from_pairs(
    pairs: list[tuple[EntityNode, EntityNode, float]],
) -> list[list[EntityNode]]:
    """Build connected components from ANN candidate pairs using Union-Find.

    Instead of sending individual pairs to LLM, we group all entities
    connected by ANN similarity into clusters for batch LLM evaluation.
    """
    parent: dict[str, str] = {}  # point_id → parent point_id

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Build Union-Find from pairs
    entity_map: dict[str, EntityNode] = {}
    for a, b, _ in pairs:
        union(a.point_id, b.point_id)
        entity_map[a.point_id] = a
        entity_map[b.point_id] = b

    # Group by root
    groups: dict[str, list[EntityNode]] = {}
    for pid, entity in entity_map.items():
        root = find(pid)
        groups.setdefault(root, []).append(entity)

    # Only return clusters with 2+ entities
    clusters = [members for members in groups.values() if len(members) >= 2]
    # Sort by size descending
    clusters.sort(key=len, reverse=True)
    return clusters


# ─── Step 4: LLM cluster-based grouping ───────────────────────────────
CLUSTER_PROMPT = """You are an academic entity deduplication expert.
Below is a list of entity names that may refer to the same or different concepts.
Group them: entities referring to the SAME concept/method/task/dataset should be in the same group.

Rules:
- "SGD" and "Stochastic Gradient Descent" → SAME (acronym vs full name)
- "ResNet" and "ResNet-50" → DIFFERENT (specific variant)
- "Image Classification" and "Image Recognition" → SAME (synonyms)
- "GAN" and "Generative Adversarial Network" → SAME
- "BERT" and "RoBERTa" → DIFFERENT (different models)
- "Object Detection" and "Object Detection Task" → SAME (redundant suffix)
- "Few-shot Learning" and "Few-Shot Learning" and "few shot learning" → SAME (casing/formatting)

Return ONLY JSON. Each group contains the IDs of entities that are the same, plus the best canonical name.
Omit singletons (entities that don't match any other).

{"groups": [{"ids": [0, 2, 5], "canonical_name": "best name"}, {"ids": [1, 3], "canonical_name": "best name"}]}

Entities to group:
"""


def _parse_llm_json(content: str) -> dict | None:
    """Extract and parse JSON from LLM response, with repair for truncated output."""
    # 1. Strip thinking tags
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    # 2. Strip code fences
    content = re.sub(r'^```(?:json)?\s*\n?', '', content.strip())
    content = re.sub(r'\n?```\s*$', '', content)
    content = content.strip()

    # 3. Find JSON object
    json_match = re.search(r'\{[\s\S]*\}', content)
    if not json_match:
        return None

    raw = json_match.group()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 4. Try repair: strip trailing incomplete fields
    repaired = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', raw)
    repaired = re.sub(r',\s*"[^"]*"\s*:\s*\d*$', '', repaired)
    repaired = re.sub(r',\s*"[^"]*$', '', repaired)
    repaired = re.sub(r',\s*$', '', repaired)
    # Close unclosed brackets
    stack = []
    in_str = False
    esc = False
    for ch in repaired:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
    for opener in reversed(stack):
        repaired += ']' if opener == '[' else '}'
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


async def llm_cluster_merges(
    clusters: list[list[EntityNode]],
    max_cluster_size: int = 30,
    max_concurrent: int = 32,
) -> list[tuple[EntityNode, EntityNode, str]]:
    """Send entity clusters to LLM for group-based dedup decision.

    Instead of asking "is A the same as B?" for each pair, we send the
    full cluster and ask: "which of these are the same entity? group them."

    Returns merge pairs compatible with execute_merges().
    """
    client = AsyncOpenAI(base_url=LLM_BASE, api_key=LLM_KEY)
    merges: list[tuple[EntityNode, EntityNode, str]] = []
    merge_lock = asyncio.Lock()
    sem = asyncio.Semaphore(max_concurrent)

    # Diagnostic counters
    diag = {"api_errors": 0, "json_parse_failures": 0, "no_json_found": 0,
            "empty_groups": 0, "has_groups": 0, "sample_logged": False,
            "total_merges": 0}

    # Split large clusters into manageable chunks
    tasks_input: list[list[EntityNode]] = []
    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            tasks_input.append(cluster)
        else:
            # Split large clusters into overlapping chunks
            for i in range(0, len(cluster), max_cluster_size - 2):
                chunk = cluster[i:i + max_cluster_size]
                if len(chunk) >= 2:
                    tasks_input.append(chunk)

    async def process_cluster(task_idx: int, cluster: list[EntityNode]):
        async with sem:
            entity_text = "\n".join([
                f'{i}. "{e.name}"' for i, e in enumerate(cluster)
            ])
            prompt = CLUSTER_PROMPT + entity_text

            try:
                resp = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                    extra_body={"enable_thinking": False},
                )
                content = resp.choices[0].message.content or ""

                # Log first raw response as sample
                if not diag["sample_logged"]:
                    logger.info("  [DIAG] Sample LLM raw response (first 500 chars): %s",
                                content[:500])
                    diag["sample_logged"] = True

                result = _parse_llm_json(content)

                if result is None:
                    diag["no_json_found"] += 1
                    logger.info("  [DIAG] No JSON in cluster %d (first 300 chars): %s",
                                task_idx, content[:300])
                    return

                groups = result.get("groups", [])
                local_merges: list[tuple[EntityNode, EntityNode, str]] = []

                # Log LLM clustering result for this cluster
                entity_names = [e.name for e in cluster]
                merged_ids: set[int] = set()
                logger.info("  ── Cluster %d (%d entities) ──", task_idx, len(cluster))
                for gi, group in enumerate(groups):
                    ids = group.get("ids", [])
                    canon = group.get("canonical_name", "")

                    # Validate: need >=2 valid IDs
                    valid_ids = [i for i in ids
                                 if isinstance(i, int) and 0 <= i < len(cluster)]
                    if len(valid_ids) < 2:
                        continue

                    merged_ids.update(valid_ids)
                    group_names = [entity_names[i] for i in valid_ids]
                    logger.info("    Group %d → '%s': %s", gi, canon, group_names)

                    # First valid entity is the keeper; rest are dupes
                    keeper = cluster[valid_ids[0]]
                    if not canon:
                        canon = keeper.name
                    for idx in valid_ids[1:]:
                        local_merges.append((keeper, cluster[idx], canon))

                # Log entities that remain independent
                independent = [entity_names[i] for i in range(len(cluster)) if i not in merged_ids]
                if independent:
                    logger.info("    Independent (%d): %s", len(independent),
                                independent[:10] if len(independent) > 10
                                else independent)

                async with merge_lock:
                    merges.extend(local_merges)
                    diag["total_merges"] += len(local_merges)

                if local_merges:
                    diag["has_groups"] += 1
                else:
                    diag["empty_groups"] += 1

                progress["done"] += 1
                if progress["done"] % 100 == 0 or progress["done"] == total_tasks:
                    logger.info("  Progress: %d/%d clusters done, %d merges so far",
                                progress["done"], total_tasks, diag["total_merges"])

            except Exception as e:
                diag["api_errors"] += 1
                logger.warning("Cluster %d failed: %s", task_idx, e)

    # Create all tasks
    progress = {"done": 0}
    tasks = []
    for i, cluster in enumerate(tasks_input):
        tasks.append(process_cluster(i, cluster))

    total_tasks = len(tasks)
    logger.info("  Dispatching %d cluster tasks (%d concurrent, max_size=%d)...",
                total_tasks, max_concurrent, max_cluster_size)
    await asyncio.gather(*tasks)

    logger.info("  LLM approved %d merges from %d clusters (%d tasks after splitting)",
                len(merges), len(clusters), total_tasks)
    logger.info("  [DIAG] api_errors=%d  json_failures=%d  no_json=%d  "
                "empty_groups=%d  has_groups=%d",
                diag["api_errors"], diag["json_parse_failures"], diag["no_json_found"],
                diag["empty_groups"], diag["has_groups"])

    await client.close()
    return merges


# ─── Step 5: Execute merges in Neo4j ──────────────────────────────────
def execute_merges(
    driver,
    merges: list[tuple[EntityNode, EntityNode, str]],
    dry_run: bool = False,
) -> tuple[int, list[str]]:
    """Merge entity pairs in Neo4j. Uses Union-Find to handle transitive merges.
    Returns (merged_count, deleted_entity_ids)."""
    if not merges:
        return 0, []

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

    # Build entity_id lookup from merge pairs
    neo_id_to_entity_id: dict[int, str] = {}
    for keeper, dupe, canon in merges:
        union(keeper.neo_id, dupe.neo_id, canon)
        neo_id_to_entity_id[keeper.neo_id] = keeper.entity_id
        neo_id_to_entity_id[dupe.neo_id] = dupe.entity_id

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
    deleted_entity_ids: list[str] = []
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
                deleted_entity_ids.extend(
                    neo_id_to_entity_id[d] for d in dupes if d in neo_id_to_entity_id
                )
                continue

            # Batch merge using UNWIND
            # 1. Move incoming rels
            rel_types_in = list(s.run("""
                UNWIND $dids AS did
                MATCH (dupe) WHERE id(dupe) = did
                MATCH (dupe)<-[r]-(src)
                WHERE id(src) <> $kid
                RETURN DISTINCT type(r) AS rtype
            """, dids=dupes, kid=root_id))

            for rt in rel_types_in:
                rtype = rt["rtype"]
                s.run(f"""
                    UNWIND $dids AS did
                    MATCH (dupe) WHERE id(dupe) = did
                    MATCH (dupe)<-[r:{rtype}]-(src)
                    WHERE id(src) <> $kid
                    MATCH (keeper) WHERE id(keeper) = $kid
                    MERGE (src)-[:{rtype}]->(keeper)
                """, dids=dupes, kid=root_id)

            # 2. Move outgoing rels
            rel_types_out = list(s.run("""
                UNWIND $dids AS did
                MATCH (dupe) WHERE id(dupe) = did
                MATCH (dupe)-[r]->(dst)
                WHERE id(dst) <> $kid
                RETURN DISTINCT type(r) AS rtype
            """, dids=dupes, kid=root_id))

            for rt in rel_types_out:
                rtype = rt["rtype"]
                s.run(f"""
                    UNWIND $dids AS did
                    MATCH (dupe) WHERE id(dupe) = did
                    MATCH (dupe)-[r:{rtype}]->(dst)
                    WHERE id(dst) <> $kid
                    MATCH (keeper) WHERE id(keeper) = $kid
                    MERGE (keeper)-[:{rtype}]->(dst)
                """, dids=dupes, kid=root_id)

            # 3. Delete all dupes
            s.run("""
                UNWIND $dids AS did
                MATCH (n) WHERE id(n) = did
                DETACH DELETE n
            """, dids=dupes)

            deleted_entity_ids.extend(
                neo_id_to_entity_id[d] for d in dupes if d in neo_id_to_entity_id
            )
            merged += len(dupes)

            # Update keeper name
            if canon_name:
                s.run("MATCH (n) WHERE id(n) = $kid SET n.name = $name",
                      kid=root_id, name=canon_name)

    return merged, deleted_entity_ids


# ─── Phase 0: Exact + Normalized string match merge ───────────────────
def _normalize_name(name: str) -> str:
    """Normalize entity name for fuzzy exact matching.
    
    Handles: en-dash/em-dash, apostrophes, hyphens, plurals, etc.
    """
    s = name.strip().lower()
    # Normalize unicode dashes to ASCII hyphen
    s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u2012', '-')
    # Remove apostrophes and backticks
    s = s.replace("'", "").replace("\u2019", "").replace("`", "")
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
    merged: int = 0
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
                merged += len(dupes)  # type: ignore[operator]
                deleted_entity_ids.extend(eid for _, _, eid in dupes if eid)
                continue

            # Batch merge: collect all dupe_ids then use batch Cypher
            dupe_neo_ids = [did for did, _, _ in dupes]

            # 1. Move ALL incoming rels from ALL dupes to keeper in one query per rel type
            rel_types_in = list(s.run("""
                UNWIND $dids AS did
                MATCH (dupe) WHERE id(dupe) = did
                MATCH (dupe)<-[r]-(src)
                WHERE id(src) <> $kid
                RETURN DISTINCT type(r) AS rtype
            """, dids=dupe_neo_ids, kid=keeper_id))

            for rt in rel_types_in:
                rtype = rt["rtype"]
                s.run(f"""
                    UNWIND $dids AS did
                    MATCH (dupe) WHERE id(dupe) = did
                    MATCH (dupe)<-[r:{rtype}]-(src)
                    WHERE id(src) <> $kid
                    MATCH (keeper) WHERE id(keeper) = $kid
                    MERGE (src)-[:{rtype}]->(keeper)
                """, dids=dupe_neo_ids, kid=keeper_id)

            # 2. Move ALL outgoing rels
            rel_types_out = list(s.run("""
                UNWIND $dids AS did
                MATCH (dupe) WHERE id(dupe) = did
                MATCH (dupe)-[r]->(dst)
                WHERE id(dst) <> $kid
                RETURN DISTINCT type(r) AS rtype
            """, dids=dupe_neo_ids, kid=keeper_id))

            for rt in rel_types_out:
                rtype = rt["rtype"]
                s.run(f"""
                    UNWIND $dids AS did
                    MATCH (dupe) WHERE id(dupe) = did
                    MATCH (dupe)-[r:{rtype}]->(dst)
                    WHERE id(dst) <> $kid
                    MATCH (keeper) WHERE id(keeper) = $kid
                    MERGE (keeper)-[:{rtype}]->(dst)
                """, dids=dupe_neo_ids, kid=keeper_id)

            # 3. Delete all dupes in one query
            s.run("""
                UNWIND $dids AS did
                MATCH (n) WHERE id(n) = did
                DETACH DELETE n
            """, dids=dupe_neo_ids)

            deleted_entity_ids.extend(eid for _, _, eid in dupes if eid)
            merged += len(dupes)  # type: ignore[operator]

            if merged % 1000 == 0 and merged > 0:
                logger.info("  Phase 0: Merged %d duplicates so far...", merged)

    return merged, deleted_entity_ids


# ─── Qdrant cleanup helper ────────────────────────────────────────────
def _eid_to_qdrant_id(entity_id: str) -> int:
    """Convert entity_id string to Qdrant integer point ID (same logic as qdrant_store.py)."""
    import hashlib as _hl
    return int(_hl.sha256(entity_id.encode()).hexdigest()[:15], 16)


def cleanup_qdrant_embeddings(qdrant: QdrantClient, deleted_eids: list[str], dry_run: bool = False):
    """Delete merged entity embeddings from Qdrant 'entities' collection."""
    if not deleted_eids or dry_run:
        return
    QDRANT_ENTITY_COLLECTION = "entities"
    # Convert string entity_ids to Qdrant integer point IDs
    int_ids = [_eid_to_qdrant_id(eid) for eid in deleted_eids]
    batch_size = 500
    for i in range(0, len(int_ids), batch_size):
        batch = int_ids[i:i + batch_size]
        try:
            qdrant.delete(
                collection_name=QDRANT_ENTITY_COLLECTION,
                points_selector=batch,
            )
        except Exception as e:
            logger.debug("Qdrant cleanup skipped (batch %d): %s", i, e)
    logger.info("  Cleaned %d embeddings from Qdrant", len(deleted_eids))


# ─── Main ──────────────────────────────────────────────────────────────
async def process_entity_type(
    driver, qdrant: QdrantClient, entity_type: str, threshold: float, dry_run: bool
) -> dict:
    """Process a single entity type end-to-end."""
    logger.info("=" * 60)
    logger.info("Processing: %s", entity_type)

    # Phase 0: Exact string match merge (fast, no LLM)
    logger.info("  Phase 0: Exact string match merge...")
    exact_merged, p0_deleted_eids = exact_merge(driver, entity_type, dry_run=dry_run)
    logger.info("  Phase 0: Merged %d exact duplicates", exact_merged)
    cleanup_qdrant_embeddings(qdrant, p0_deleted_eids, dry_run=dry_run)

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

    # Step 3b: Build clusters from ANN pairs
    clusters = build_clusters_from_pairs(pairs)
    sizes = [len(c) for c in clusters]
    logger.info("  Built %d clusters from %d pairs (max=%d, median=%d, total_entities=%d)",
                len(clusters), len(pairs),
                max(sizes) if sizes else 0,
                sorted(sizes)[len(sizes) // 2] if sizes else 0,
                sum(sizes))

    # Log top pairs for reference
    for a, b, sim in pairs[:10]:
        logger.info("    %.3f: '%s' <-> '%s'", sim, a.name, b.name)
    if len(pairs) > 10:
        logger.info("    ... and %d more pairs", len(pairs) - 10)

    # Step 4: LLM cluster-based grouping
    logger.info("  Sending %d clusters to LLM for grouping...", len(clusters))
    merges = await llm_cluster_merges(clusters)

    # Step 5: Execute
    prefix = "[DRY-RUN] " if dry_run else ""
    logger.info("  %sExecuting merges...", prefix)
    n, p1_deleted_eids = execute_merges(driver, merges, dry_run=dry_run)
    logger.info("  %sMerged %d entities", prefix, n)

    # Step 6: Qdrant cleanup for Phase 1
    cleanup_qdrant_embeddings(qdrant, p1_deleted_eids, dry_run=dry_run)

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
