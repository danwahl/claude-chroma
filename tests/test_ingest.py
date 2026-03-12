"""Tests for ingest module (integration tests with ephemeral ChromaDB)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import chromadb

from claude_chroma.ingest import ingest


def _ingest_with_ephemeral(data_dir: Path, client: Any) -> Any:
    """Run ingest using an ephemeral client."""
    collection_name = "test_conversations"
    with patch("claude_chroma.ingest._get_collection") as mock_gc:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        mock_gc.return_value = (client, collection)
        stats = ingest(data_dir=data_dir, collection_name=collection_name)
    return stats, collection


def test_ingest_single_file(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: Any,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "conv.json").write_text(json.dumps([sample_conversation]))

    stats, collection = _ingest_with_ephemeral(data_dir, chroma_client)

    assert stats.files_processed == 1
    assert stats.conversations_processed == 1
    assert stats.chunks_created == 3  # 3 exchanges
    assert collection.count() == 3


def test_upsert_idempotency(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: Any,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "conv.json").write_text(json.dumps([sample_conversation]))

    # Ingest twice
    _ingest_with_ephemeral(data_dir, chroma_client)
    stats2, collection = _ingest_with_ephemeral(data_dir, chroma_client)

    # Second run should skip the conversation
    assert stats2.conversations_skipped == 1
    assert stats2.conversations_processed == 0
    assert collection.count() == 3  # No duplicates


def test_updated_export_replaces(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: Any,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "conv.json").write_text(json.dumps([sample_conversation]))

    _ingest_with_ephemeral(data_dir, chroma_client)

    # Update the conversation with a later timestamp
    sample_conversation["updated_at"] = "2024-12-01T00:00:00Z"
    (data_dir / "conv.json").write_text(json.dumps([sample_conversation]))

    stats2, collection = _ingest_with_ephemeral(data_dir, chroma_client)
    assert stats2.conversations_processed == 1
    assert stats2.conversations_skipped == 0


def test_stats_reported(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: Any,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "conv.json").write_text(json.dumps([sample_conversation]))

    stats, _ = _ingest_with_ephemeral(data_dir, chroma_client)
    assert stats.elapsed_seconds > 0
    assert stats.errors == []


def test_search_returns_results(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    chroma_client: Any,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "conv.json").write_text(json.dumps([sample_conversation]))

    _, collection = _ingest_with_ephemeral(data_dir, chroma_client)

    results = collection.query(query_texts=["Python programming"], n_results=2)
    assert len(results["ids"][0]) > 0
