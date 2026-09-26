"""Exchange-level chunking of conversations."""

from __future__ import annotations

from dataclasses import dataclass

from claude_chroma.parse import Conversation, Message

MAX_CHUNK_CHARS = 2000
OVERLAP_CHARS = 200
SEPARATORS = ["\n\n", "\n", ". ", " "]
# Truncation limit for the human_message metadata field. Kept short to
# avoid bloating ChromaDB metadata; the full text is in the first chunk(s)
# of the exchange.
MAX_HUMAN_MESSAGE_META = 500


@dataclass
class Chunk:
    """A chunk of conversation text ready for embedding."""

    id: str
    text: str
    metadata: dict[str, str | int]


def _split_pieces(text: str, max_chars: int, separators: list[str]) -> list[str]:
    """Break text into pieces of at most max_chars, using the coarsest
    separator that works. Separators stay attached, so the pieces
    concatenate back to the original text."""
    if len(text) <= max_chars:
        return [text]
    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        parts = text.split(sep)
        pieces = [part + sep for part in parts[:-1]] + [parts[-1]]
        return [
            sub
            for piece in pieces
            if piece
            for sub in _split_pieces(piece, max_chars, separators[i + 1 :])
        ]
    # Last resort: hard split by character
    return [text[j : j + max_chars] for j in range(0, len(text), max_chars)]


def _recursive_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into chunks of at most max_chars, preferring natural
    boundaries and repeating up to `overlap` chars between chunks."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for piece in _split_pieces(text, max_chars, SEPARATORS):
        if current and len(current) + len(piece) > max_chars:
            chunks.append(current)
            # Carry overlap from the end of the previous chunk if it fits
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail if len(tail) + len(piece) <= max_chars else ""
        current += piece
    chunks.append(current)
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
                _make_text_chunks(
                    chunks,
                    conversation,
                    human_msg,
                    f"Human: {human_msg.text}\n\nAssistant: {assistant_text}",
                    turn_index,
                    "exchange",
                )
            else:
                # Solo human message
                _make_text_chunks(
                    chunks,
                    conversation,
                    human_msg,
                    f"Human: {human_msg.text}",
                    turn_index,
                    "human_solo",
                )
            i = j
        else:
            # Solo assistant message (no preceding human)
            _make_text_chunks(
                chunks,
                conversation,
                msg,
                f"Assistant: {msg.text}",
                turn_index,
                "assistant_solo",
            )
            i += 1

        turn_index += 1

    return chunks


def _make_text_chunks(
    chunks: list[Chunk],
    conv: Conversation,
    first_msg: Message,
    text: str,
    turn_index: int,
    sender: str,
) -> None:
    """Append chunks for one turn's text, splitting it if needed.

    Split pieces are cut from the text as one continuous stream; the human
    message is carried in each chunk's human_message metadata for context.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        chunks.append(
            _make_chunk(
                conv, turn_index, None, text, first_msg, sender, "full_exchange"
            )
        )
        return
    parts = _recursive_split(text, MAX_CHUNK_CHARS, OVERLAP_CHARS)
    for sub_index, part in enumerate(parts):
        chunks.append(
            _make_chunk(
                conv, turn_index, sub_index, part, first_msg, sender, "split_exchange"
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
                first_msg.text[:MAX_HUMAN_MESSAGE_META]
                if first_msg.sender == "human"
                else ""
            ),
            "sender": sender,
            "created_at": first_msg.created_at,
            "chunk_type": chunk_type,
        },
    )
