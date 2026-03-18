"""Orchestration: parse → chunk → embed → store in ChromaDB."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
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
) -> tuple[ClientAPI, Collection]:
    """Get or create a ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _load_existing_timestamps(
    collection: Collection,
) -> dict[str, str]:
    """Load all conversation updated_at timestamps from the collection.

    Returns a dict mapping conversation_id to its latest
    conversation_updated_at value. On an empty collection this
    returns an empty dict without issuing any query.
    """
    if collection.count() == 0:
        return {}

    results = collection.get(include=["metadatas"])
    metadatas = results["metadatas"] or []

    timestamps: dict[str, str] = {}
    for meta in metadatas:
        cid = str(meta.get("conversation_id", ""))
        updated = str(meta.get("conversation_updated_at", ""))
        if cid and updated:
            # Keep the latest timestamp seen for each conversation
            existing = timestamps.get(cid, "")
            if updated > existing:
                timestamps[cid] = updated

    return timestamps


def _upsert_chunks(collection: Collection, chunks: list[Chunk]) -> None:
    """Upsert chunks into ChromaDB in batches."""
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )


def ingest(
    claude_dir: Path = Path("./claude_data"),
    chroma_dir: Path = Path("./chroma_data"),
    collection_name: str = "claude_conversations",
) -> IngestStats:
    """Ingest all conversation exports from claude_dir into ChromaDB."""
    start = time.monotonic()
    stats = IngestStats()

    json_files = sorted(claude_dir.glob("**/*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", claude_dir)
        return stats

    _, collection = _get_collection(chroma_dir, collection_name)

    # Load existing timestamps once upfront instead of querying
    # per-conversation, which is slow at scale.
    existing_ts = _load_existing_timestamps(collection)

    with Progress() as progress:
        for json_file in json_files:
            task = progress.add_task(f"[cyan]{json_file.name}", total=1)
            stats.files_processed += 1

            try:
                for conversation in parse_export(json_file):
                    existing = existing_ts.get(conversation.uuid)
                    if existing and existing >= conversation.updated_at:
                        stats.conversations_skipped += 1
                        continue

                    chunks = chunk_conversation(conversation)
                    if chunks:
                        _upsert_chunks(collection, chunks)
                        stats.chunks_created += len(chunks)
                        # Update cache so later files in the same
                        # run see this conversation as ingested.
                        existing_ts[conversation.uuid] = conversation.updated_at
                    stats.conversations_processed += 1
            except Exception as e:
                msg = f"Error processing {json_file.name}: {e}"
                logger.error(msg)
                stats.errors.append(msg)

            progress.update(task, completed=1)

    stats.elapsed_seconds = time.monotonic() - start
    return stats
