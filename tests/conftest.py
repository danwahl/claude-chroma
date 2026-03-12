"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import pytest


def _make_message(
    uuid: str,
    sender: str,
    text: str,
    created_at: str = "2024-03-08T01:00:00Z",
    content_type: str = "text",
) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "text": text,
        "content": [
            {
                "type": content_type,
                "text": text,
                "start_timestamp": created_at,
                "stop_timestamp": created_at,
            }
        ],
        "sender": sender,
        "created_at": created_at,
        "updated_at": created_at,
        "attachments": [],
        "files": [],
    }


@pytest.fixture
def sample_conversation() -> dict[str, Any]:
    """A conversation with 3 exchanges (6 messages)."""
    return {
        "uuid": "conv-001",
        "name": "Test Conversation",
        "summary": "",
        "created_at": "2024-03-08T01:00:00Z",
        "updated_at": "2024-03-08T02:00:00Z",
        "account": {"uuid": "acct-001"},
        "chat_messages": [
            _make_message("m1", "human", "What is Python?", "2024-03-08T01:00:00Z"),
            _make_message(
                "m2",
                "assistant",
                "Python is a programming language.",
                "2024-03-08T01:01:00Z",
            ),
            _make_message("m3", "human", "Tell me more.", "2024-03-08T01:02:00Z"),
            _make_message(
                "m4",
                "assistant",
                "It was created by Guido van Rossum.",
                "2024-03-08T01:03:00Z",
            ),
            _make_message("m5", "human", "Thanks!", "2024-03-08T01:04:00Z"),
            _make_message("m6", "assistant", "You're welcome!", "2024-03-08T01:05:00Z"),
        ],
    }


@pytest.fixture
def long_response_conversation() -> dict[str, Any]:
    """A conversation with a very long assistant response (>2000 chars)."""
    long_text = "This is a detailed response. " * 150  # ~4500 chars
    return {
        "uuid": "conv-002",
        "name": "Long Response",
        "summary": "",
        "created_at": "2024-03-08T01:00:00Z",
        "updated_at": "2024-03-08T02:00:00Z",
        "account": {"uuid": "acct-001"},
        "chat_messages": [
            _make_message("m1", "human", "Explain everything.", "2024-03-08T01:00:00Z"),
            _make_message("m2", "assistant", long_text, "2024-03-08T01:01:00Z"),
        ],
    }


@pytest.fixture
def empty_conversation() -> dict[str, Any]:
    """A conversation with no messages."""
    return {
        "uuid": "conv-003",
        "name": "Empty",
        "summary": "",
        "created_at": "2024-03-08T01:00:00Z",
        "updated_at": "2024-03-08T02:00:00Z",
        "account": {"uuid": "acct-001"},
        "chat_messages": [],
    }


@pytest.fixture
def non_text_conversation() -> dict[str, Any]:
    """A conversation with non-text content blocks."""
    return {
        "uuid": "conv-004",
        "name": "Non-text",
        "summary": "",
        "created_at": "2024-03-08T01:00:00Z",
        "updated_at": "2024-03-08T02:00:00Z",
        "account": {"uuid": "acct-001"},
        "chat_messages": [
            _make_message("m1", "human", "Show me an image.", "2024-03-08T01:00:00Z"),
            {
                "uuid": "m2",
                "text": "",
                "content": [
                    {"type": "image", "text": "", "url": "http://example.com/img.png"}
                ],
                "sender": "assistant",
                "created_at": "2024-03-08T01:01:00Z",
                "updated_at": "2024-03-08T01:01:00Z",
                "attachments": [],
                "files": [],
            },
        ],
    }


@pytest.fixture
def chroma_client() -> Any:
    """Ephemeral in-memory ChromaDB client."""
    return chromadb.EphemeralClient()


@pytest.fixture
def sample_export_file(tmp_path: Path, sample_conversation: dict[str, Any]) -> Path:
    """Write a sample export JSON file."""
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps([sample_conversation]))
    return path


@pytest.fixture
def multi_export_file(
    tmp_path: Path,
    sample_conversation: dict[str, Any],
    empty_conversation: dict[str, Any],
    non_text_conversation: dict[str, Any],
) -> Path:
    """Write an export with multiple conversations including edge cases."""
    path = tmp_path / "conversations.json"
    path.write_text(
        json.dumps([sample_conversation, empty_conversation, non_text_conversation])
    )
    return path
