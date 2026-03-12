"""Parse Claude conversation export JSON files."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single chat message."""

    uuid: str
    sender: str  # "human" or "assistant"
    text: str
    created_at: str  # ISO 8601


@dataclass
class Conversation:
    """A parsed conversation with its messages."""

    uuid: str
    name: str
    created_at: str
    updated_at: str
    messages: list[Message]


def _extract_text(msg: dict[str, Any]) -> str:
    """Extract text from a message, preferring content blocks."""
    content = msg.get("content")
    if content and isinstance(content, list):
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and block.get("text")
        ]
        if texts:
            return "\n\n".join(texts)
    # Fall back to top-level text
    return str(msg.get("text", ""))


def _parse_message(msg: dict[str, Any]) -> Message | None:
    """Parse a single message dict into a Message, or None if invalid."""
    uuid = msg.get("uuid", "")
    sender = msg.get("sender", "")
    if sender not in ("human", "assistant"):
        logger.warning("Skipping message %s with unknown sender: %s", uuid, sender)
        return None
    text = _extract_text(msg)
    if not text:
        logger.debug("Skipping message %s with empty text", uuid)
        return None
    created_at = msg.get("created_at", "")
    return Message(uuid=uuid, sender=sender, text=text, created_at=created_at)


def parse_export(path: Path) -> Iterator[Conversation]:
    """Parse a conversations.json export file, yielding Conversation objects.

    Skips conversations with no messages. Extracts text from content blocks
    (type=="text" only), falling back to top-level text field.
    """
    logger.info("Parsing export file: %s", path)
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Expected a JSON array in %s", path)
        return

    for conv_dict in data:
        if not isinstance(conv_dict, dict):
            logger.warning("Skipping non-dict conversation entry")
            continue

        raw_messages = conv_dict.get("chat_messages", [])
        if not raw_messages:
            logger.debug(
                "Skipping conversation %s with no messages",
                conv_dict.get("uuid", "unknown"),
            )
            continue

        messages: list[Message] = []
        for msg in raw_messages:
            parsed = _parse_message(msg)
            if parsed is not None:
                messages.append(parsed)

        if not messages:
            continue

        yield Conversation(
            uuid=conv_dict.get("uuid", ""),
            name=conv_dict.get("name", ""),
            created_at=conv_dict.get("created_at", ""),
            updated_at=conv_dict.get("updated_at", ""),
            messages=messages,
        )
