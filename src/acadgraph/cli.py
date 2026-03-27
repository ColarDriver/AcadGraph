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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acadgraph.config import AppConfig
    from acadgraph.embedding_client import EmbeddingClient
    from acadgraph.kg.storage.neo4j_store import Neo4jKGStore
    from acadgraph.kg.storage.qdrant_store import QdrantKGStore
    from acadgraph.llm_client import LLMClient

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()


def _trunc(value: object, maxlen: int) -> str:
    """Convert *value* to str and truncate to *maxlen* characters."""
    s = str(value)
    if len(s) <= maxlen:
        return s
    # Use join to avoid slice syntax (Pyre2 false-positive with PEP 563).
    return "".join(s[i] for i in range(maxlen))


def setup_logging(level: str = "INFO") -> None:
    """Configure rich logging."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


async def _get_components() -> tuple[
    "Neo4jKGStore", "QdrantKGStore", "LLMClient", "EmbeddingClient", "AppConfig"
]:
    """Initialize all components."""
    from acadgraph.config import AppConfig, get_config
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


from contextlib import asynccontextmanager


@asynccontextmanager
async def managed_components():
    """Yield all components and ensure clean shutdown."""
    neo4j, qdrant, llm, embedding, config = await _get_components()
    try:
        yield neo4j, qdrant, llm, embedding, config
    finally:
        await neo4j.close()
        await qdrant.close()
        await llm.close()
        await embedding.close()


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def main(log_level: str) -> None:
    """AcadGraph — Three-Layer Evidence-Chain Knowledge Graph."""
    setup_logging(log_level)


@main.command()
def init() -> None:
    """Initialize Neo4j schema and Qdrant collections."""

    async def _init():
        async with managed_components() as (neo4j, qdrant, llm, embedding, _):
            console.print("[bold green]Initializing Neo4j schema...[/]")
            await neo4j.init_schema()
            console.print("[bold green]Initializing Qdrant collections...[/]")
            await qdrant.init_collections()
            console.print("[bold green]✓ Initialization complete![/]")

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

        async with managed_components() as (neo4j, qdrant, llm, embedding, _):
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

        async with managed_components() as (neo4j, qdrant, llm, embedding, _):
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

    asyncio.run(_batch())


@main.command()
@click.argument("jsonl_path", type=click.Path(exists=True))
@click.option("--limit", default=None, type=int, help="Max papers to process")
@click.option("--batch-size", default=3, help="Concurrent papers per batch")
@click.option("--skip-existing/--no-skip-existing", default=True, help="Skip papers already in KG")
def ingest(jsonl_path: str, limit: int | None, batch_size: int, skip_existing: bool) -> None:
    """Ingest papers from a JSONL file (e.g. paper_reviews_dataset.jsonl).

    Example:
        acadgraph ingest data/paper_reviews_dataset.jsonl --limit 10
    """

    async def _ingest():
        import time

        from acadgraph.kg.incremental_builder import IncrementalKGBuilder
        from acadgraph.kg.jsonl_loader import iter_jsonl, jsonl_to_paper_source

        async with managed_components() as (neo4j, qdrant, llm, embedding, _):
            builder = IncrementalKGBuilder(neo4j, qdrant, llm, embedding)

            # Get existing paper IDs if skip_existing
            skip_ids: set[str] | None = None
            if skip_existing:
                try:
                    skip_ids = await neo4j.get_existing_paper_ids()
                    if skip_ids:
                        console.print(
                            f"[dim]Skipping {len(skip_ids)} existing papers[/]"
                        )
                except Exception:
                    skip_ids = None

            console.print(f"[bold]Ingesting from:[/] {jsonl_path}")
            console.print(f"[dim]Batch size: {batch_size} (parallel)[/]")
            if limit:
                console.print(f"[dim]Limit: {limit} papers[/]")

            # Collect all PaperSource objects
            papers = []
            for record in iter_jsonl(jsonl_path, limit=limit, skip_ids=skip_ids):
                papers.append(jsonl_to_paper_source(record))

            total = len(papers)
            console.print(f"[bold]Loaded {total} papers, processing...[/]\n")

            counts = {"ok": 0, "fail": 0}
            t0 = time.time()

            # Sliding window: always keep `batch_size` papers in flight
            sem = asyncio.Semaphore(batch_size)
            lock = asyncio.Lock()

            async def _process_one(paper, idx):
                async with sem:
                    pid = paper.paper_id or paper.title[:30]
                    try:
                        r = await builder.add_paper(paper)
                        async with lock:
                            if r.errors:
                                counts["fail"] += 1
                                done = counts["ok"] + counts["fail"]
                                console.print(
                                    f"[yellow][{done}/{total}] ⚠ {pid}: "
                                    f"{r.errors[0][:60]}[/]"
                                )
                            else:
                                counts["ok"] += 1
                                done = counts["ok"] + counts["fail"]
                                console.print(
                                    f"[green][{done}/{total}] ✓ {pid}[/] "
                                    f"({r.entities_added}E "
                                    f"{r.claims_added}C "
                                    f"{r.evidences_added}Ev "
                                    f"{r.duration_seconds:.1f}s)"
                                )
                    except Exception as e:
                        async with lock:
                            counts["fail"] += 1
                            done = counts["ok"] + counts["fail"]
                            console.print(
                                f"[red][{done}/{total}] ✗ {pid}: {e}[/]"
                            )

            # Launch all tasks; semaphore controls concurrency
            tasks = [
                asyncio.create_task(_process_one(p, i))
                for i, p in enumerate(papers)
            ]
            await asyncio.gather(*tasks)

            elapsed = time.time() - t0
            console.print(
                f"\n[bold green]Done:[/] {counts['ok']}/{total} succeeded, "
                f"{counts['fail']} errors, {elapsed:.1f}s total"
            )

    asyncio.run(_ingest())


@main.command()
@click.argument("idea")
@click.option("--mode", type=click.Choice(["competition", "gap", "recall", "innovation", "bridge"]), default="competition")
@click.option("-k", default=10, help="Number of results")
@click.option("--methods", default="", help="Comma-separated method names (for innovation mode)")
@click.option("--method-a", default="", help="First method family (for bridge mode)")
@click.option("--method-b", default="", help="Second method family (for bridge mode)")
def query(idea: str, mode: str, k: int, methods: str, method_a: str, method_b: str) -> None:
    """Query the KG with a research idea."""

    async def _query():
        from acadgraph.kg.query_engine import KGQueryEngine

        async with managed_components() as (neo4j, qdrant, llm, embedding, _):
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
                method_list = [m.strip() for m in methods.split(",") if m.strip()]
                if method_list:
                    gap = await engine.gap_grounded_statement(idea, method_list)
                    console.print("[dim]Using graph-grounded gap generation[/dim]")
                else:
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

            elif mode == "innovation":
                method_list = [m.strip() for m in methods.split(",") if m.strip()]
                if not method_list:
                    console.print("[red]Error: --methods is required for innovation mode[/]")
                    return
                console.print(f"[bold]Finding innovation paths for:[/] {idea}")
                console.print(f"[dim]Methods: {method_list}[/dim]")
                path = await engine.find_innovation_paths(idea, method_list, k=k)

                console.print(f"\n[bold green]Suggested Innovation:[/]\n{path.suggested_combination}")

                if path.gaps:
                    table = Table(title=f"Known Gaps ({len(path.gaps)} found)")
                    table.add_column("Method", style="cyan")
                    table.add_column("Problem", style="white", max_width=40)
                    table.add_column("Failure Mode", style="yellow", max_width=30)
                    for g in path.gaps[:10]:
                        table.add_row(
                            str(g.get("method", "")),
                            _trunc(g.get("problem", ""), 40),
                            _trunc(g.get("failure_mode", ""), 30),
                        )
                    console.print(table)

                if path.unaddressed_gaps:
                    table = Table(title=f"Open Gaps ({len(path.unaddressed_gaps)} unresolved)")
                    table.add_column("Problem", style="white", max_width=40)
                    table.add_column("Constraint", style="red", max_width=30)
                    for g in path.unaddressed_gaps[:10]:
                        table.add_row(
                            _trunc(g.get("problem", ""), 40),
                            _trunc(g.get("constraint", ""), 30),
                        )
                    console.print(table)

                if path.addressing_ideas:
                    table = Table(title=f"Existing Solutions ({len(path.addressing_ideas)})")
                    table.add_column("Mechanism", style="green", max_width=40)
                    table.add_column("Innovation", style="cyan", max_width=30)
                    for ci in path.addressing_ideas[:10]:
                        table.add_row(
                            _trunc(ci.get("mechanism", ""), 40),
                            _trunc(ci.get("key_innovation", ""), 30),
                        )
                    console.print(table)

            elif mode == "bridge":
                a_names = [m.strip() for m in method_a.split(",") if m.strip()]
                b_names = [m.strip() for m in method_b.split(",") if m.strip()]
                if not a_names or not b_names:
                    console.print("[red]Error: --method-a and --method-b required for bridge mode[/]")
                    return
                console.print(f"[bold]Finding bridges:[/] {a_names} ↔ {b_names}")
                # Use list() + range() to slice — avoids PEP 563 / Pyre2
                # false-positive on list.__getitem__(slice).
                a_variants = [a_names[i] for i in range(1, len(a_names))]
                b_variants = [b_names[i] for i in range(1, len(b_names))]
                bridge = await engine.find_cross_domain_bridges(
                    a_names[0], b_names[0],
                    method_a_variants=a_variants,
                    method_b_variants=b_variants,
                )
                console.print(f"\n[bold green]Novelty Assessment:[/]\n{bridge.combination_novelty}")

                if bridge.shared_concepts:
                    table = Table(title=f"Shared Concepts ({len(bridge.shared_concepts)})")
                    table.add_column("Type", style="cyan")
                    table.add_column("Name", style="white")
                    table.add_column("Papers A", style="green")
                    for sc in bridge.shared_concepts[:15]:
                        table.add_row(
                            str(sc.get("entity_type", "")),
                            str(sc.get("name", "")),
                            str(sc.get("papers_a_count", 0)),
                        )
                    console.print(table)

                if bridge.bridge_papers:
                    table = Table(title=f"Bridge Papers ({len(bridge.bridge_papers)})")
                    table.add_column("Paper ID", style="cyan")
                    table.add_column("Title", style="white", max_width=50)
                    table.add_column("Year", style="yellow")
                    for bp in bridge.bridge_papers[:10]:
                        table.add_row(
                            _trunc(bp.get("paper_id", ""), 30),
                            _trunc(bp.get("title", ""), 50),
                            str(bp.get("year", "")),
                        )
                    console.print(table)

    asyncio.run(_query())


@main.command()
@click.argument("paper_id")
def audit(paper_id: str) -> None:
    """Audit a paper's claim-evidence support."""

    async def _audit():
        from acadgraph.kg.query_engine import KGQueryEngine

        async with managed_components() as (neo4j, qdrant, llm, embedding, _):
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

    asyncio.run(_audit())


@main.command()
def stats() -> None:
    """Show KG statistics."""

    async def _stats():
        from acadgraph.kg.query_engine import KGQueryEngine

        async with managed_components() as (neo4j, qdrant, llm, embedding, _):
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

    asyncio.run(_stats())


if __name__ == "__main__":
    main()
