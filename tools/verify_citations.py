#!/usr/bin/env python3
"""Verify all citations by querying from knowledge graph."""

import asyncio
import sys
sys.path.insert(0, 'src')

# Papers to verify
PAPERS_TO_VERIFY = [
    # Classic RL/Entropy papers
    ("SAC", "Soft Actor-Critic maximum entropy deep reinforcement learning"),
    ("REINFORCE", "policy gradient REINFORCE Williams"),
    ("MPO", "Maximum a Posteriori Policy Optimisation"),
    ("Adaptive Entropy", "adaptive entropy regularization reinforcement learning"),
    ("Unlikelihood", "Neural Text Generation Unlikelihood Training"),
    ("QUARK", "QUARK Controllable Text Generation Reinforced Unlearning"),

    # Reward shaping and constrained RL
    ("Reward Shaping", "Policy Invariance Reward Transformations Shaping Ng"),
    ("CPO", "Constrained Policy Optimization Achiam"),
    ("RCPO", "Reward Constrained Policy Optimization Tessler"),

    # Neuro-symbolic
    ("NS-CL", "Neuro-Symbolic Concept Learner"),
    ("Constitutional AI", "Constitutional AI Harmlessness Feedback Bai"),

    # Core papers from original draft
    ("Grounding-DINO", "Grounding DINO Marrying DINO Grounded Pre-Training"),
    ("Set-of-Mark", "Set-of-Mark Visual Prompting GPT-4V"),
    ("DeepSeek-Math", "DeepSeekMath Pushing Limits Mathematical Reasoning"),
    ("DPO", "Direct Preference Optimization Language Model Reward"),
    ("RLHF", "Training Language Models Follow Instructions Human Feedback Ouyang"),
    ("CogAgent", "CogAgent Visual Language Model GUI Agents"),
    ("GPT-4V", "GPT-4V Generalist Web Agent Grounded"),
    ("AppAgent", "AppAgent Multimodal Agents Smartphone Users"),
    ("MiniWob", "Reinforcement Learning Web Interfaces Workflow-Guided"),
    ("AndroidWorld", "AndroidWorld Dynamic Benchmarking Environment Autonomous Agents"),
    ("WebArena", "WebArena Realistic Web Environment Building Autonomous Agents"),
]

async def search_paper(query_text):
    """Search for a paper using enhanced recall."""
    from acadgraph.config import get_config
    from acadgraph.kg.storage.neo4j_store import Neo4jKGStore
    from acadgraph.kg.storage.qdrant_store import QdrantKGStore
    from acadgraph.kg.query_engine import KGQueryEngine
    from acadgraph.llm_client import LLMClient
    from acadgraph.embedding_client import EmbeddingClient

    config = get_config()
    neo4j = Neo4jKGStore(config.neo4j)
    qdrant = QdrantKGStore(config.qdrant, embedding_dim=config.embedding.dim)
    llm = LLMClient(config.llm)
    embedding = EmbeddingClient(config.embedding)

    await neo4j.connect()
    await qdrant.connect()

    try:
        engine = KGQueryEngine(neo4j, qdrant, llm, embedding)
        results = await engine.enhanced_recall(query_text, k=3)

        if results:
            # Get full metadata for top result
            paper_id = results[0]['paper_id']
            async with neo4j.driver.session() as session:
                result = await session.run('''
                    MATCH (p:Paper {paper_id: $id})
                    RETURN p.paper_id AS id, p.title AS title,
                           p.authors AS authors, p.year AS year,
                           p.venue AS venue
                ''', {'id': paper_id})
                rows = await result.data()
                if rows:
                    return rows[0]
        return None
    finally:
        await neo4j.close()
        await qdrant.close()
        await llm.close()
        await embedding.close()

async def main():
    for name, query in PAPERS_TO_VERIFY:
        print(f"\n{'='*80}")
        print(f"Searching: {name}")
        print(f"Query: {query}")
        print(f"{'='*80}")

        result = await search_paper(query)

        if result:
            print(f"✓ FOUND")
            print(f"  ID:     {result['id']}")
            print(f"  Title:  {result['title']}")
            print(f"  Year:   {result['year']}")
            print(f"  Venue:  {result['venue']}")
            authors = result['authors']
            if isinstance(authors, list) and authors:
                print(f"  Authors: {', '.join(authors[:3])}")
        else:
            print(f"✗ NOT FOUND in knowledge graph")

if __name__ == "__main__":
    asyncio.run(main())
