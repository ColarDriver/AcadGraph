#!/usr/bin/env python3
"""Query paper metadata from Neo4j."""

import asyncio
from acadgraph.config import get_config
from acadgraph.kg.storage.neo4j_store import Neo4jKGStore

async def query_paper_metadata(paper_ids):
    """Query metadata for given paper IDs."""
    config = get_config()
    neo4j = Neo4jKGStore(config.neo4j)
    await neo4j.connect()

    try:
        query = """
        MATCH (p:Paper)
        WHERE p.paper_id IN $paper_ids
        RETURN p.paper_id AS paper_id,
               p.title AS title,
               p.authors AS authors,
               p.year AS year,
               p.venue AS venue,
               p.abstract AS abstract
        """

        result = await neo4j.execute_read(query, {"paper_ids": paper_ids})

        for record in result:
            print(f"\nPaper ID: {record['paper_id']}")
            print(f"Title: {record['title']}")
            print(f"Authors: {record['authors']}")
            print(f"Year: {record['year']}")
            print(f"Venue: {record['venue']}")
            print("-" * 80)

    finally:
        await neo4j.close()

if __name__ == "__main__":
    # Paper IDs from the recall query
    paper_ids = ["zrH2A1upAo", "1XLjrmKZ4p", "9WiPZy3Kro", "sZ0DsaRsd4", "dsQHm7YX9c"]
    asyncio.run(query_paper_metadata(paper_ids))
