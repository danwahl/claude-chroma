"""Exchange-level chunking of conversations."""

from __future__ import annotations

from dataclasses import dataclass

from claude_chroma.parse import Conversation, Message

MAX_CHUNK_CHARS = 2000
OVERLAP_CHARS = 200
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    """A chunk of conversation text ready for embedding."""

    id: str
    text: str
    metadata: dict[str, str | int]


def _recursive_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text recursively using progressively finer separators."""
    if len(text) <= max_chars:
        return [text]

    for sep in SEPARATORS:
        parts = text.split(sep)
        if len(parts) == 1:
            continue

        chunks: list[str] = []
        current = parts[0]
        for part in parts[1:]:
            candidate = current + sep + part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                # Add overlap from end of previous chunk
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + sep + part
                else:
                    current = part
        chunks.append(current)
        return chunks

    # Last resort: hard split by character
    chunks = []
    for i in range(0, len(text), max_chars - overlap):
        chunks.append(text[i : i + max_chars])
    return chunks


def chunk_conversation(conversation: Conversation) -> list[Chunk]:
    """Convert a Conversation into a list of Chunks ready for embedding."""
    chunks: list[Chunk] = []
    messages = conversation.messages
    turn_index = 0
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg.sender == "human":
            human_msg = msg
            # Collect following assistant messages
            assistant_texts: list[str] = []
            j = i + 1
            while j < len(messages) and messages[j].sender == "assistant":
                assistant_texts.append(messages[j].text)
                j += 1

            if assistant_texts:
                assistant_text = "\n\n".join(assistant_texts)
                _make_exchange_chunks(
                    chunks, conversation, human_msg, assistant_text, turn_index
                )
            else:
                # Solo human message
                chunks.append(
                    _make_chunk(
                        conversation,
                        turn_index,
                        None,
                        f"Human: {human_msg.text}",
                        human_msg,
                        "human_solo",
                        "full_exchange",
                    )
                )
            i = j
        else:
            # Solo assistant message (no preceding human)
            chunks.append(
                _make_chunk(
                    conversation,
                    turn_index,
                    None,
                    f"Assistant: {msg.text}",
                    msg,
                    "assistant_solo",
                    "full_exchange",
                )
            )
            i += 1

        turn_index += 1

    return chunks


def _make_exchange_chunks(
    chunks: list[Chunk],
    conv: Conversation,
    human_msg: Message,
    assistant_text: str,
    turn_index: int,
) -> None:
    """Create chunks for a human+assistant exchange, splitting if needed."""
    full_text = f"Human: {human_msg.text}\n\nAssistant: {assistant_text}"

    if len(full_text) <= MAX_CHUNK_CHARS:
        chunks.append(
            _make_chunk(
                conv,
                turn_index,
                None,
                full_text,
                human_msg,
                "exchange",
                "full_exchange",
            )
        )
    else:
        # Split the assistant text, prepending human context to each
        prefix = f"Human: {human_msg.text}\n\nAssistant: "
        available = MAX_CHUNK_CHARS - len(prefix)
        if available < 100:
            available = 100  # Ensure minimum chunk size
        parts = _recursive_split(assistant_text, available, OVERLAP_CHARS)
        for sub_index, part in enumerate(parts):
            chunk_text = prefix + part
            chunks.append(
                _make_chunk(
                    conv,
                    turn_index,
                    sub_index,
                    chunk_text,
                    human_msg,
                    "exchange",
                    "split_exchange",
                )
            )


def _make_chunk(
    conv: Conversation,
    turn_index: int,
    sub_index: int | None,
    text: str,
    first_msg: Message,
    sender: str,
    chunk_type: str,
) -> Chunk:
    """Create a Chunk with proper ID and metadata."""
    chunk_id = f"{conv.uuid}:{turn_index}"
    if sub_index is not None:
        chunk_id += f":{sub_index}"

    return Chunk(
        id=chunk_id,
        text=text,
        metadata={
            "conversation_id": conv.uuid,
            "conversation_name": conv.name,
            "conversation_created_at": conv.created_at,
            "conversation_updated_at": conv.updated_at,
            "turn_index": turn_index,
            "human_message": (
                first_msg.text[:500] if first_msg.sender == "human" else ""
            ),
            "sender": sender,
            "created_at": first_msg.created_at,
            "chunk_type": chunk_type,
        },
    )
