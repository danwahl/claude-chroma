"""Tests for ingest module (integration tests with ephemeral ChromaDB)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from claude_chroma.ingest import IngestStats, ingest

_counter = 0


def _ingest_with_ephemeral(
    claude_dir: Path,
    client: ClientAPI,
    collection_name: str | None = None,
) -> tuple[IngestStats, Collection]:
    """Run ingest using an ephemeral client."""
    if collection_name is None:
        global _counter  # noqa: PLW0603
        _counter += 1
        collection_name = f"test_conv_{_counter}"
    with patch("claude_chroma.ingest._get_collection") as mock_gc:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        mock_gc.return_value = (client, collection)
        stats = ingest(claude_dir=claude_dir, collection_name=collection_name)
    return stats, collection


def test_ingest_single_file(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: ClientAPI,
) -> None:
    claude_dir = tmp_path / "claude_data"
    claude_dir.mkdir()
    (claude_dir / "conversations.json").write_text(json.dumps([sample_conversation]))

    stats, collection = _ingest_with_ephemeral(claude_dir, chroma_client)

    assert stats.files_processed == 1
    assert stats.conversations_processed == 1
    assert stats.chunks_created == 3  # 3 exchanges
    assert collection.count() == 3


def test_ingest_subdirectory(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: ClientAPI,
) -> None:
    """JSON files in subdirectories should also be discovered."""
    # Use a unique UUID so dedup doesn't skip it
    sample_conversation = dict(sample_conversation, uuid="conv-subdir")
    claude_dir = tmp_path / "claude_data"
    subdir = claude_dir / "2024"
    subdir.mkdir(parents=True)
    (subdir / "conversations.json").write_text(json.dumps([sample_conversation]))

    stats, collection = _ingest_with_ephemeral(claude_dir, chroma_client)

    assert stats.files_processed == 1
    assert stats.conversations_processed == 1
    assert collection.count() == 3


def test_ingest_ignores_other_export_files(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: ClientAPI,
) -> None:
    """Manifests, projects, and other export JSON should not be parsed."""
    sample_conversation = dict(sample_conversation, uuid="conv-other-files")
    claude_dir = tmp_path / "claude_data"
    (claude_dir / "projects").mkdir(parents=True)
    (claude_dir / "conversations.json").write_text(json.dumps([sample_conversation]))
    (claude_dir / "manifest-abc.json").write_text(json.dumps({"data_files": []}))
    (claude_dir / "users.json").write_text(json.dumps([{"uuid": "u1"}]))
    (claude_dir / "projects" / "p1.json").write_text(json.dumps({"uuid": "p1"}))

    stats, collection = _ingest_with_ephemeral(claude_dir, chroma_client)

    assert stats.files_processed == 1
    assert stats.conversations_processed == 1
    assert collection.count() == 3


def test_upsert_idempotency(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: ClientAPI,
) -> None:
    claude_dir = tmp_path / "claude_data"
    claude_dir.mkdir()
    (claude_dir / "conversations.json").write_text(json.dumps([sample_conversation]))

    # Ingest twice with same collection
    name = "test_idempotency"
    _ingest_with_ephemeral(claude_dir, chroma_client, name)
    stats2, collection = _ingest_with_ephemeral(claude_dir, chroma_client, name)

    # Second run should skip the conversation
    assert stats2.conversations_skipped == 1
    assert stats2.conversations_processed == 0
    assert collection.count() == 3  # No duplicates


def test_updated_export_replaces(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: ClientAPI,
) -> None:
    claude_dir = tmp_path / "claude_data"
    claude_dir.mkdir()
    (claude_dir / "conversations.json").write_text(json.dumps([sample_conversation]))

    name = "test_updated"
    _ingest_with_ephemeral(claude_dir, chroma_client, name)

    # Update the conversation with a later timestamp
    sample_conversation["updated_at"] = "2024-12-01T00:00:00Z"
    (claude_dir / "conversations.json").write_text(json.dumps([sample_conversation]))

    stats2, collection = _ingest_with_ephemeral(claude_dir, chroma_client, name)
    assert stats2.conversations_processed == 1
    assert stats2.conversations_skipped == 0


def test_stats_reported(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: ClientAPI,
) -> None:
    claude_dir = tmp_path / "claude_data"
    claude_dir.mkdir()
    (claude_dir / "conversations.json").write_text(json.dumps([sample_conversation]))

    stats, _ = _ingest_with_ephemeral(claude_dir, chroma_client)
    assert stats.elapsed_seconds > 0
    assert stats.errors == []


def test_search_returns_results(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: ClientAPI,
) -> None:
    claude_dir = tmp_path / "claude_data"
    claude_dir.mkdir()
    (claude_dir / "conversations.json").write_text(json.dumps([sample_conversation]))

    _, collection = _ingest_with_ephemeral(claude_dir, chroma_client)

    results = collection.query(query_texts=["Python programming"], n_results=2)
    assert len(results["ids"][0]) > 0
