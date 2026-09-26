"""Tests for chunk module."""

from __future__ import annotations

from claude_chroma.chunk import chunk_conversation
from claude_chroma.parse import Conversation, Message


def _conv(messages: list[Message], uuid: str = "conv-test") -> Conversation:
    return Conversation(
        uuid=uuid,
        name="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=messages,
    )


def _msg(sender: str, text: str, uuid: str = "m") -> Message:
    return Message(
        uuid=uuid,
        sender=sender,
        text=text,
        created_at="2024-01-01T00:00:00Z",
    )


def test_basic_exchange_pairing() -> None:
    conv = _conv([_msg("human", "Hi"), _msg("assistant", "Hello!")])
    chunks = chunk_conversation(conv)
    assert len(chunks) == 1
    assert chunks[0].text == "Human: Hi\n\nAssistant: Hello!"
    assert chunks[0].metadata["sender"] == "exchange"


def test_solo_human_message() -> None:
    conv = _conv(
        [
            _msg("human", "Q1"),
            _msg("assistant", "A1"),
            _msg("human", "Q2"),  # no assistant reply
        ]
    )
    chunks = chunk_conversation(conv)
    assert len(chunks) == 2
    assert chunks[1].metadata["sender"] == "human_solo"
    assert chunks[1].text == "Human: Q2"


def test_long_response_splitting() -> None:
    long_text = "Word " * 600  # ~3000 chars
    conv = _conv([_msg("human", "Explain"), _msg("assistant", long_text)])
    chunks = chunk_conversation(conv)
    assert len(chunks) > 1
    assert all(c.metadata["chunk_type"] == "split_exchange" for c in chunks)
    # Only the first chunk carries the human message text
    assert chunks[0].text.startswith("Human: Explain\n\nAssistant: Word")
    assert not any("Human:" in c.text for c in chunks[1:])


def test_split_chunks_carry_human_message_metadata() -> None:
    long_text = "Paragraph one. " * 200
    conv = _conv([_msg("human", "My question"), _msg("assistant", long_text)])
    chunks = chunk_conversation(conv)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["human_message"] == "My question"


def test_long_human_message_split_in_place() -> None:
    """A long human message is split like any other text, not repeated."""
    long_question = "Why is this so? " * 200  # ~3200 chars
    conv = _conv([_msg("human", long_question), _msg("assistant", "Because.")])
    chunks = chunk_conversation(conv)
    assert len(chunks) > 1
    assert sum(c.text.count("Human:") for c in chunks) == 1
    assert chunks[-1].text.endswith("Assistant: Because.")


def test_deterministic_chunk_ids() -> None:
    conv = _conv(
        [_msg("human", "Hi"), _msg("assistant", "Hello!")],
        uuid="abc-123",
    )
    chunks = chunk_conversation(conv)
    assert chunks[0].id == "abc-123:0"


def test_split_chunk_ids() -> None:
    long_text = "Word " * 600
    conv = _conv(
        [_msg("human", "Q"), _msg("assistant", long_text)],
        uuid="abc-123",
    )
    chunks = chunk_conversation(conv)
    assert chunks[0].id == "abc-123:0:0"
    assert chunks[1].id == "abc-123:0:1"


def test_metadata_populated() -> None:
    conv = _conv([_msg("human", "Hi"), _msg("assistant", "Hello!")])
    chunks = chunk_conversation(conv)
    meta = chunks[0].metadata
    assert meta["conversation_id"] == "conv-test"
    assert meta["conversation_name"] == "Test"
    assert meta["turn_index"] == 0
    assert meta["human_message"] == "Hi"
    assert meta["chunk_type"] == "full_exchange"


def test_chunk_text_format() -> None:
    conv = _conv([_msg("human", "Q"), _msg("assistant", "A")])
    chunks = chunk_conversation(conv)
    assert chunks[0].text == "Human: Q\n\nAssistant: A"


def test_long_solo_message_is_split() -> None:
    conv = _conv([_msg("human", "Unanswered. " * 300)])  # ~3600 chars
    chunks = chunk_conversation(conv)
    assert len(chunks) > 1
    assert all(len(c.text) <= 2000 for c in chunks)
    assert all(c.metadata["sender"] == "human_solo" for c in chunks)
