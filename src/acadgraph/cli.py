"""
AcadGraph CLI — Command-line interface for the Three-Layer KG system.

Commands:
- init: Initialize databases (create schemas, collections)
- add: Add a paper to the KG
- batch: Batch-add papers from a directory or JSON file
- query: Query the KG (competition space, gap, evidence ledger)
- stats: Show KG statistics
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure rich logging."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


async def _get_components():
    """Initialize all components."""
    from acadgraph.config import get_config
    from acadgraph.embedding_client import EmbeddingClient
    from acadgraph.kg.storage.neo4j_store import Neo4jKGStore
    from acadgraph.kg.storage.qdrant_store import QdrantKGStore
    from acadgraph.llm_client import LLMClient

    config = get_config()

    neo4j = Neo4jKGStore(config.neo4j)
    await neo4j.connect()

    qdrant = QdrantKGStore(config.qdrant, embedding_dim=config.embedding.dim)
    await qdrant.connect()

    llm = LLMClient(config.llm)
    embedding = EmbeddingClient(config.embedding)

    return neo4j, qdrant, llm, embedding, config


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def main(log_level: str) -> None:
    """AcadGraph — Three-Layer Evidence-Chain Knowledge Graph."""
    setup_logging(log_level)


@main.command()
def init() -> None:
    """Initialize Neo4j schema and Qdrant collections."""

    async def _init():
        neo4j, qdrant, llm, embedding, _ = await _get_components()
        try:
            console.print("[bold green]Initializing Neo4j schema...[/]")
            await neo4j.init_schema()
            console.print("[bold green]Initializing Qdrant collections...[/]")
            await qdrant.init_collections()
            console.print("[bold green]✓ Initialization complete![/]")
        finally:
            await neo4j.close()
            await qdrant.close()
            await llm.close()
            await embedding.close()

    asyncio.run(_init())


@main.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--paper-id", default=None, help="Paper ID (generated from title if not given)")
@click.option("--title", default="", help="Paper title")
@click.option("--year", default=None, type=int, help="Publication year")
@click.option("--venue", default="", help="Publication venue")
def add(source: str, paper_id: str | None, title: str, year: int | None, venue: str) -> None:
    """Add a single paper to the KG from PDF or text file."""

    async def _add():
        from acadgraph.kg.incremental_builder import IncrementalKGBuilder
        from acadgraph.kg.schema import PaperSource

        neo4j, qdrant, llm, embedding, _ = await _get_components()
        try:
            builder = IncrementalKGBuilder(neo4j, qdrant, llm, embedding)

            path = Path(source)
            paper = PaperSource(
                paper_id=paper_id or "",
                title=title,
                year=year,
                venue=venue,
            )

            if path.suffix.lower() == ".pdf":
                paper.pdf_path = str(path)
            else:
                paper.text = path.read_text(encoding="utf-8")

            console.print(f"[bold]Adding paper: {source}[/]")
            result = await builder.add_paper(paper)

            if result.errors:
                console.print(f"[bold red]Errors: {result.errors}[/]")
            else:
                console.print(f"[bold green]✓ Paper added in {result.duration_seconds:.1f}s[/]")
                console.print(
                    f"  Entities: {result.entities_added}, Citations: {result.citations_added}, "
                    f"Claims: {result.claims_added}, Evidence: {result.evidences_added}"
                )
        finally:
            await neo4j.close()
            await qdrant.close()
            await llm.close()
            await embedding.close()

    asyncio.run(_add())


@main.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--batch-size", default=5, help="Concurrent papers per batch")
@click.option("--ext", default=".pdf", help="File extension to look for")
def batch(directory: str, batch_size: int, ext: str) -> None:
    """Batch-add papers from a directory."""

    async def _batch():
        from acadgraph.kg.incremental_builder import IncrementalKGBuilder
        from acadgraph.kg.schema import PaperSource

        neo4j, qdrant, llm, embedding, _ = await _get_components()
        try:
            builder = IncrementalKGBuilder(neo4j, qdrant, llm, embedding)

            dir_path = Path(directory)
            files = sorted(dir_path.glob(f"*{ext}"))
            console.print(f"[bold]Found {len(files)} {ext} files[/]")

            papers = []
            for f in files:
                paper = PaperSource(title=f.stem)
                if ext == ".pdf":
                    paper.pdf_path = str(f)
                else:
                    paper.text = f.read_text(encoding="utf-8")
                papers.append(paper)

            result = await builder.add_papers_batch(papers, batch_size=batch_size)

            console.print(
                f"\n[bold green]Batch complete: "
                f"{result.successful}/{result.total_papers} successful "
                f"in {result.total_duration_seconds:.1f}s[/]"
            )
        finally:
            await neo4j.close()
            await qdrant.close()
            await llm.close()
            await embedding.close()

    asyncio.run(_batch())


@main.command()
@click.argument("idea")
@click.option("--mode", type=click.Choice(["competition", "gap", "recall"]), default="competition")
@click.option("-k", default=10, help="Number of results")
def query(idea: str, mode: str, k: int) -> None:
    """Query the KG with a research idea."""

    async def _query():
        from acadgraph.kg.query_engine import KGQueryEngine

        neo4j, qdrant, llm, embedding, _ = await _get_components()
        try:
            engine = KGQueryEngine(neo4j, qdrant, llm, embedding)

            if mode == "competition":
                console.print(f"[bold]Finding competition space for:[/] {idea}")
                space = await engine.find_competition_space(idea, k=k)
                table = Table(title="Competition Space")
                table.add_column("Paper ID", style="cyan")
                table.add_column("Score", style="green")
                for p in space.nearest_papers:
                    table.add_row(p["paper_id"][:40], f"{p['score']:.3f}")
                console.print(table)

            elif mode == "gap":
                console.print(f"[bold]Generating gap statement for:[/] {idea}")
                gap = await engine.generate_gap_statement(idea)
                console.print(f"\n[bold green]Gap Statement:[/]\n{gap.statement}")
                console.print(f"\n[bold]Problem:[/] {gap.problem}")
                console.print(f"[bold]Failure:[/] {gap.failure_constraint}")
                console.print(f"[bold]Missing:[/] {gap.missing_mechanism}")

            elif mode == "recall":
                console.print(f"[bold]Enhanced recall for:[/] {idea}")
                results = await engine.enhanced_recall(idea, k=k)
                table = Table(title="Recall Results")
                table.add_column("Paper ID", style="cyan")
                table.add_column("Score", style="green")
                table.add_column("Path", style="yellow")
                for r in results:
                    table.add_row(
                        r["paper_id"][:40],
                        f"{r['score']:.3f}",
                        r.get("recall_path", ""),
                    )
                console.print(table)

        finally:
            await neo4j.close()
            await qdrant.close()
            await llm.close()
            await embedding.close()

    asyncio.run(_query())


@main.command()
@click.argument("paper_id")
def audit(paper_id: str) -> None:
    """Audit a paper's claim-evidence support."""

    async def _audit():
        from acadgraph.kg.query_engine import KGQueryEngine

        neo4j, qdrant, llm, embedding, _ = await _get_components()
        try:
            engine = KGQueryEngine(neo4j, qdrant, llm, embedding)
            ledger = await engine.get_claim_evidence_ledger(paper_id)

            table = Table(title=f"Claim-Evidence Ledger: {paper_id}")
            table.add_column("Claim", style="white", max_width=50)
            table.add_column("Type", style="cyan")
            table.add_column("Severity", style="yellow")
            table.add_column("Support", style="green")
            table.add_column("Evidence Count", style="blue")

            for entry in ledger.entries:
                severity_style = "bold red" if entry.severity == "P0" else "yellow"
                support_style = (
                    "bold green" if entry.support_status.value == "FULL"
                    else "bold red" if entry.support_status.value in ("REFUTED", "UNVERIFIABLE")
                    else "yellow"
                )
                table.add_row(
                    entry.claim_text[:50],
                    str(entry.claim_type),
                    f"[{severity_style}]{entry.severity}[/]",
                    f"[{support_style}]{entry.support_status.value}[/]",
                    str(len(entry.actual_evidence)),
                )

            console.print(table)

            # Highlight red flags
            unsupported = ledger.unsupported_p0
            if unsupported:
                console.print(f"\n[bold red]⚠ {len(unsupported)} P0 claims without full support![/]")
                for entry in unsupported:
                    console.print(f"  [red]• {entry.claim_text[:80]}[/]")
            else:
                console.print("\n[bold green]✓ All P0 claims are fully supported.[/]")

        finally:
            await neo4j.close()
            await qdrant.close()
            await llm.close()
            await embedding.close()

    asyncio.run(_audit())


@main.command()
def stats() -> None:
    """Show KG statistics."""

    async def _stats():
        from acadgraph.kg.query_engine import KGQueryEngine

        neo4j, qdrant, llm, embedding, _ = await _get_components()
        try:
            engine = KGQueryEngine(neo4j, qdrant, llm, embedding)
            stats_data = await engine.get_stats()

            # Neo4j stats
            table = Table(title="Neo4j Node Counts")
            table.add_column("Node Type", style="cyan")
            table.add_column("Count", style="green", justify="right")
            for label, count in stats_data.get("neo4j", {}).items():
                if count > 0:
                    table.add_row(label, str(count))
            console.print(table)

            # Qdrant stats
            table2 = Table(title="Qdrant Collection Sizes")
            table2.add_column("Collection", style="cyan")
            table2.add_column("Points", style="green", justify="right")
            for name, count in stats_data.get("qdrant", {}).items():
                table2.add_row(name, str(count))
            console.print(table2)

        finally:
            await neo4j.close()
            await qdrant.close()
            await llm.close()
            await embedding.close()

    asyncio.run(_stats())


if __name__ == "__main__":
    main()
