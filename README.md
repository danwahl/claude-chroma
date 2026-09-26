# claude-chroma

A Python CLI tool that ingests Claude AI conversation exports into a local ChromaDB vector database, making your entire conversation history semantically searchable. Pair it with the official `chroma-mcp` server to give Claude Desktop or Claude Code access to everything you've ever discussed.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Quickstart

```bash
# Clone and install
git clone https://github.com/danwahl/claude-chroma.git
cd claude-chroma
uv sync

# Drop your Claude export(s) into claude_data/
cp ~/Downloads/conversations.json claude_data/

# Ingest into ChromaDB
uv run claude-chroma ingest

# Verify
uv run claude-chroma stats

# Test a search
uv run claude-chroma search "free will compatibilism"
```

## Connecting to Claude Desktop / Claude Code

Add the official `chroma-mcp` server to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "chroma": {
      "command": "uvx",
      "args": [
        "chroma-mcp",
        "--client-type", "persistent",
        "--data-dir", "/absolute/path/to/claude-chroma/chroma_data"
      ]
    }
  }
}
```

Claude will then be able to query your full conversation history via the `chroma` MCP tool.

## CLI Reference

### `claude-chroma ingest`

Ingest every `conversations.json` in the Claude data directory (including subdirectories) into ChromaDB. Other export files (manifests, projects, artifacts) are ignored.

```
Options:
  --claude-dir PATH  Directory containing JSON exports [default: ./claude_data]
  --chroma-dir PATH  ChromaDB storage directory [default: ./chroma_data]
```

### `claude-chroma stats`

Show database statistics: total conversations, chunks, date range, and top 10 most-chunked conversations.

```
Options:
  --chroma-dir PATH  ChromaDB storage directory [default: ./chroma_data]
```

### `claude-chroma search`

Run a semantic similarity search for development and debugging.

```
Arguments:
  QUERY              Search query text

Options:
  -n, --n-results    Number of results [default: 5]
  --chroma-dir PATH  ChromaDB storage directory [default: ./chroma_data]
```

## Chunking Strategy

Conversations are chunked at the **exchange level** — each human message is paired with the subsequent assistant response. This preserves the question + answer arc as a single semantic unit. Long exchanges (>2000 characters) are split with overlap as one continuous text, so the human message appears once at the start; every sub-chunk carries the human message (first 500 characters) in its `human_message` metadata for context.

Chunk IDs are deterministic (`{conversation_uuid}:{turn_index}`), so re-ingesting the same export is a no-op, and updated exports overwrite stale data.

## License

MIT
