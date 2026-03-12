"""Orchestration: parse → chunk → embed → store in ChromaDB."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from rich.progress import Progress

from claude_chroma.chunk import Chunk, chunk_conversation
from claude_chroma.parse import parse_export

logger = logging.getLogger(__name__)

BATCH_SIZE = 200


@dataclass
class IngestStats:
    """Statistics from an ingestion run."""

    files_processed: int = 0
    conversations_processed: int = 0
    conversations_skipped: int = 0
    chunks_created: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def _get_collection(
    chroma_dir: Path, collection_name: str
) -> tuple[Any, Any]:
    """Get or create a ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _existing_updated_at(collection: Any, conversation_id: str) -> str | None:
    """Check if conversation already exists and return its updated_at."""
    results = collection.get(
        where={"conversation_id": conversation_id},
        limit=1,
        include=["metadatas"],
    )
    if results["ids"]:
        metadatas = results["metadatas"]
        if metadatas:
            return str(metadatas[0].get("conversation_updated_at", ""))
    return None


def _upsert_chunks(collection: Any, chunks: list[Chunk]) -> None:
    """Upsert chunks into ChromaDB in batches."""
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )


def ingest(
    data_dir: Path = Path("./data"),
    chroma_dir: Path = Path("./chroma_data"),
    collection_name: str = "claude_conversations",
) -> IngestStats:
    """Ingest all conversation exports from data_dir into ChromaDB."""
    start = time.monotonic()
    stats = IngestStats()

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", data_dir)
        return stats

    _, collection = _get_collection(chroma_dir, collection_name)

    with Progress() as progress:
        for json_file in json_files:
            task = progress.add_task(f"[cyan]{json_file.name}", total=None)
            stats.files_processed += 1

            try:
                for conversation in parse_export(json_file):
                    # Check deduplication
                    existing = _existing_updated_at(collection, conversation.uuid)
                    if existing and existing >= conversation.updated_at:
                        stats.conversations_skipped += 1
                        continue

                    chunks = chunk_conversation(conversation)
                    if chunks:
                        _upsert_chunks(collection, chunks)
                        stats.chunks_created += len(chunks)
                    stats.conversations_processed += 1
            except Exception as e:
                msg = f"Error processing {json_file.name}: {e}"
                logger.error(msg)
                stats.errors.append(msg)

            progress.update(task, completed=True)

    stats.elapsed_seconds = time.monotonic() - start
    return stats
