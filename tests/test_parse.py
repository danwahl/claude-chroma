"""Tests for parse module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claude_chroma.parse import parse_export


def test_parse_valid_conversation(
    tmp_path: Path, sample_conversation: dict[str, Any]
) -> None:
    path = tmp_path / "conv.json"
    path.write_text(json.dumps([sample_conversation]))
    convs = list(parse_export(path))
    assert len(convs) == 1
    assert convs[0].uuid == "conv-001"
    assert convs[0].name == "Test Conversation"
    assert len(convs[0].messages) == 6


def test_skip_empty_chat_messages(
    tmp_path: Path, empty_conversation: dict[str, Any]
) -> None:
    path = tmp_path / "conv.json"
    path.write_text(json.dumps([empty_conversation]))
    convs = list(parse_export(path))
    assert len(convs) == 0


def test_extract_text_from_content_blocks(
    tmp_path: Path, sample_conversation: dict[str, Any]
) -> None:
    path = tmp_path / "conv.json"
    path.write_text(json.dumps([sample_conversation]))
    convs = list(parse_export(path))
    assert convs[0].messages[0].text == "What is Python?"


def test_fallback_to_toplevel_text(tmp_path: Path) -> None:
    conv = {
        "uuid": "conv-fb",
        "name": "Fallback",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "chat_messages": [
            {
                "uuid": "m1",
                "text": "Hello from top-level",
                "content": [],
                "sender": "human",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        ],
    }
    path = tmp_path / "conv.json"
    path.write_text(json.dumps([conv]))
    convs = list(parse_export(path))
    assert convs[0].messages[0].text == "Hello from top-level"


def test_filter_non_text_content(
    tmp_path: Path, non_text_conversation: dict[str, Any]
) -> None:
    path = tmp_path / "conv.json"
    path.write_text(json.dumps([non_text_conversation]))
    convs = list(parse_export(path))
    # The assistant message with only image content should be skipped
    # Only the human message remains
    assert len(convs) == 1
    assert len(convs[0].messages) == 1
    assert convs[0].messages[0].sender == "human"


def test_empty_name_handled(tmp_path: Path) -> None:
    conv = {
        "uuid": "conv-noname",
        "name": "",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "chat_messages": [
            {
                "uuid": "m1",
                "text": "Hi",
                "content": [{"type": "text", "text": "Hi"}],
                "sender": "human",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        ],
    }
    path = tmp_path / "conv.json"
    path.write_text(json.dumps([conv]))
    convs = list(parse_export(path))
    assert convs[0].name == ""
