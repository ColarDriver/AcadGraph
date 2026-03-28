# AcadGraph

**Three-Layer Evidence-Chain Knowledge Graph for Academic Literature**

AcadGraph constructs a structured knowledge graph from academic papers through three interconnected layers:

| Layer | Nodes | Purpose |
|-------|-------|---------|
| **Semantic Entity** | Method · Dataset · Metric · Task · Model · Framework · Concept | Canonicalized research vocabulary with ANN-accelerated deduplication |
| **Citation & Evolution** | Citation edges with 6 intent types, evolution chains | Method lineage tracking and bibliographic coupling |
| **Argumentation** | Problem → Gap → CoreIdea → Claim → Evidence | Per-paper reasoning structure with P0/P1/P2 severity grading |

---

## Architecture

```
PDF / Text ──▶ Parser ──▶ LLM Extraction ──▶ Neo4j (graph) + Qdrant (vectors)
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
            Entity & Relation     Argumentation Chain
              Extraction            (Claim-Evidence)
```

**Storage**: Neo4j (graph structure) + Qdrant (embedding index for ANN retrieval)

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n acadgraph python=3.13 -y
conda activate acadgraph
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
```

Required variables in `.env`:

```
LLM_API_BASE, LLM_API_KEY, LLM_MODEL
EMBEDDING_API_BASE, EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
QDRANT_HOST, QDRANT_PORT
```

### 3. Launch Services

```bash
docker-compose up -d   # Neo4j + Qdrant
acadgraph init         # Initialize schema & indexes
```

---

## Build: Ingesting Papers

```bash
# Single paper
acadgraph add ./paper.pdf --title "Paper Title" --year 2024 --venue "ICLR"

# Batch ingestion
acadgraph batch ./papers/ --batch-size 5 --ext .pdf
```

**Pipeline per paper**: PDF parsing → section tree construction → entity extraction (7 types) → relation extraction → argumentation chain extraction (Problem/Gap/CoreIdea/Claim/Evidence) → embedding indexing → deterministic dedup (name normalization) → semantic dedup (ANN + LLM cluster merge).

Post-ingestion entity deduplication:

```bash
# Cluster-based semantic merge (ANN candidates → connected components → LLM grouping)
python scripts/semantic_entity_merge.py --types Framework --dry-run
python scripts/semantic_entity_merge.py --threshold 0.85
```

---

## Query: Knowledge-Driven Research Reasoning

AcadGraph provides five query modes, each exploiting different graph traversal strategies.

### Competition Space

Retrieve the competitive landscape around a research idea via three-path fusion: semantic similarity (Qdrant ANN) → citation network expansion (co-citation + bibliographic coupling) → merge & rank.

```bash
acadgraph query "attention mechanism for long sequences" --mode competition -k 10
```

### Gap Mining

Generate a falsifiable gap statement grounded in the KG's `Problem → Gap → CoreIdea` structure. Uses structural alignment along 7 dimensions (Problem, Setting, Assumption, Mechanism, Supervision, Metric, FailureMode) to identify where the proposed idea departs from existing work.

```bash
acadgraph query "sparse attention with linear complexity" --mode gap
```

**Output template**: *"Existing methods can solve [P] under [S], but fail under [F], because they lack [M]; no existing method simultaneously satisfies [A, B, C]."*

### Innovation Path

Cross-method innovation mining: Method → known Gaps → existing CoreIdeas → unaddressed Gaps → LLM-synthesized combination suggestion. Requires specifying the methods to combine.

```bash
acadgraph query "combine DPO with tree search for GUI agents" \
  --mode innovation --methods "DPO,MCTS,GUI grounding"
```

### Cross-Domain Bridge

Discover shared concepts, bridge papers, and integration points between two method families. Evaluates combination novelty based on existing cross-domain literature density.

```bash
acadgraph query "visual grounding meets reinforcement learning" \
  --mode bridge --method-a "visual grounding" --method-b "GRPO"
```

### Enhanced Recall

Five-path multi-signal retrieval with LLM re-ranking:

| Path | Signal | Source |
|------|--------|--------|
| 1 | Entity similarity | Qdrant ANN |
| 2 | Claim similarity | Qdrant ANN |
| 3 | Evolution chain expansion | Neo4j graph |
| 4 | Section similarity | Qdrant ANN |
| 5 | LLM-guided section tree reasoning | Neo4j + LLM |

Cross-path diversity bonus + LLM holistic re-ranking for final ordering.

```bash
acadgraph query "efficient transformer" --mode recall -k 10
```

---

## Utilities

```bash
acadgraph audit <paper_id>   # Claim-Evidence ledger with P0/P1/P2 severity
acadgraph stats               # Graph statistics (nodes, edges, embeddings)
acadgraph --help
```

---

## License

MIT
