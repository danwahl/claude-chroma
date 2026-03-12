"""CLI interface for claude-chroma."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from claude_chroma.ingest import ingest as run_ingest

app = typer.Typer(help="Ingest Claude conversation exports into ChromaDB.")
console = Console()

DEFAULT_DATA_DIR = Path("./data")
DEFAULT_CHROMA_DIR = Path("./chroma_data")
DEFAULT_COLLECTION = "claude_conversations"


@app.command()
def ingest(
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, help="Directory containing JSON exports"
    ),
    chroma_dir: Path = typer.Option(
        DEFAULT_CHROMA_DIR, help="ChromaDB storage directory"
    ),
) -> None:
    """Ingest all conversation exports into ChromaDB."""
    logging.basicConfig(level=logging.INFO)
    stats = run_ingest(data_dir=data_dir, chroma_dir=chroma_dir)

    console.print()
    console.print("[bold green]Ingestion complete![/]")
    console.print(f"  Files processed:          {stats.files_processed}")
    console.print(f"  Conversations processed:  {stats.conversations_processed}")
    console.print(f"  Conversations skipped:    {stats.conversations_skipped}")
    console.print(f"  Chunks created:           {stats.chunks_created}")
    console.print(f"  Time elapsed:             {stats.elapsed_seconds:.1f}s")
    if stats.errors:
        console.print(f"  [red]Errors: {len(stats.errors)}[/]")
        for err in stats.errors:
            console.print(f"    [red]{err}[/]")


@app.command()
def stats(
    chroma_dir: Path = typer.Option(
        DEFAULT_CHROMA_DIR, help="ChromaDB storage directory"
    ),
) -> None:
    """Show database statistics."""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(name=DEFAULT_COLLECTION)
    except Exception:
        console.print("[red]No collection found. Run 'ingest' first.[/]")
        raise typer.Exit(1)

    count = collection.count()
    console.print(f"[bold]Collection:[/] {DEFAULT_COLLECTION}")
    console.print(f"[bold]Total chunks:[/] {count}")

    if count == 0:
        return

    # Get all metadata to compute stats
    result = collection.get(include=["metadatas"])
    metadatas = result["metadatas"] or []

    conv_ids: set[str] = set()
    conv_chunks: dict[str, int] = {}
    conv_names: dict[str, str] = {}
    dates: list[str] = []

    for meta in metadatas:
        cid = str(meta.get("conversation_id", ""))
        conv_ids.add(cid)
        conv_chunks[cid] = conv_chunks.get(cid, 0) + 1
        name = str(meta.get("conversation_name", ""))
        if name:
            conv_names[cid] = name
        created = str(meta.get("conversation_created_at", ""))
        if created:
            dates.append(created)

    console.print(f"[bold]Total conversations:[/] {len(conv_ids)}")
    if dates:
        console.print(f"[bold]Date range:[/] {min(dates)[:10]} — {max(dates)[:10]}")

    # Top 10 most-chunked conversations
    top = sorted(conv_chunks.items(), key=lambda x: x[1], reverse=True)[:10]
    if top:
        table = Table(title="Top 10 Most-Chunked Conversations")
        table.add_column("Conversation", style="cyan")
        table.add_column("Chunks", justify="right")
        for cid, n in top:
            name = conv_names.get(cid, cid[:12] + "…")
            table.add_row(name or "(untitled)", str(n))
        console.print(table)


@app.command()
def search(
    query: str = typer.Argument(help="Search query text"),
    n_results: int = typer.Option(5, "--n-results", "-n", help="Number of results"),
    chroma_dir: Path = typer.Option(
        DEFAULT_CHROMA_DIR, help="ChromaDB storage directory"
    ),
) -> None:
    """Test semantic search against the conversation database."""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(name=DEFAULT_COLLECTION)
    except Exception:
        console.print("[red]No collection found. Run 'ingest' first.[/]")
        raise typer.Exit(1)

    results = collection.query(query_texts=[query], n_results=n_results)

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    if not documents:
        console.print("[yellow]No results found.[/]")
        return

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        score = 1 - dist  # cosine distance → similarity
        name = meta.get("conversation_name", "(untitled)") or "(untitled)"
        date = str(meta.get("created_at", ""))[:10]
        human = meta.get("human_message", "")

        console.print(f"\n[bold cyan]Result {i + 1}[/] — score: {score:.3f}")
        console.print(f"  [dim]Conversation:[/] {name}  [dim]Date:[/] {date}")
        if human:
            console.print(f"  [dim]Question:[/] {str(human)[:150]}")
        console.print(f"  [dim]Text:[/] {doc[:300]}{'…' if len(doc) > 300 else ''}")
