# SkillSearch MCP Server

MCP server that discovers [Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) from a directory, embeds `SKILL.md` content into PostgreSQL pgvector via an OpenAI-compatible `/embeddings` endpoint, and provides semantic vector cosine similarity search — same pattern as mcphub's `search_tools`.

**Auto-sync**: watches the skills directory with `fs.watch` and automatically indexes new/changed skills and removes deleted ones. Falls back to periodic polling if native file watching is unavailable.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DB_URL` | **Yes** | — | PostgreSQL connection string (must have pgvector) |
| `SKILLSEARCH_SKILLS_DIR` | **Yes** | — | Root directory; each subdirectory = one skill |
| `API_BASE_URL` | No | — | Embedding API base URL (e.g. `https://xxx/v1`) |
| `API_KEY` | No | — | API key for the embedding provider |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Model name |
| `TRANSPORT` | No | `stdio` | `stdio` or `http` |
| `PORT` | No | `3000` | HTTP port (only when `TRANSPORT=http`) |
| `WATCH_POLL_INTERVAL` | No | `30000` | Polling interval in ms (fallback mode only) |

## Transport Modes

### stdio (default)

```bash
DB_URL=postgresql://user:pass@host:5432/db \
SKILLSEARCH_SKILLS_DIR=/path/to/skills \
API_BASE_URL=https://your-api/v1 \
API_KEY=sk-... \
EMBEDDING_MODEL=gemini-embedding-001 \
  npx github:ptbsare/skillsearch-mcp-server
```

### HTTP (Streamable HTTP)

```bash
DB_URL=postgresql://user:pass@host:5432/db \
SKILLSEARCH_SKILLS_DIR=/path/to/skills \
API_BASE_URL=https://your-api/v1 \
API_KEY=sk-... \
EMBEDDING_MODEL=gemini-embedding-001 \
TRANSPORT=http \
PORT=3000 \
  node dist/index.js
```

Endpoint: `POST /mcp` (and `POST /`)

## MCP Tools

### `skill_search`

Semantic vector search for Agent Skills. Returns matching skills with frontmatter metadata, SKILL.md absolute path, and **all** file paths under the skill directory.

Parameters:
- `query` (string, required) — Natural language search query
- `limit` (integer, optional) — Max results (default: 5, max: 50)
- `threshold` (number, optional) — Min similarity 0.0–1.0 (default: 0.3)

### `skill_list`

List all indexed skills with names, descriptions, paths, and file counts.

## Auto-Sync (File Watching)

The server automatically monitors the skills directory for changes:

| Event | Action |
|---|---|
| New skill directory created | Indexed automatically |
| `SKILL.md` modified | Re-indexed automatically |
| Skill directory removed | Removed from index automatically |
| Other files changed | Detected, content-hash skip avoids unnecessary re-index |

**Watch strategy**:
1. **Primary**: `fs.watch` on the root skills directory (detects added/removed skill dirs) + recursive `fs.watch` on each skill sub-directory (detects SKILL.md changes)
2. **Fallback**: If `fs.watch` is unavailable or errors out, automatically switches to periodic full-scan polling (`WATCH_POLL_INTERVAL`, default 30s)
3. **Debounce**: Rapid filesystem events are coalesced (500ms) into a single sync pass

No manual `skill_reindex` tool needed — just add, modify, or remove skill directories on disk.

## Skill Directory Structure

```
skills/
├── my-skill/
│   ├── SKILL.md              # Required, with YAML frontmatter
│   ├── references/           # Any files (recursively collected)
│   ├── scripts/              # Any files (recursively collected)
│   ├── assets/               # Any files (recursively collected)
│   └── templates/
│       └── template.yml      # Also collected
└── another-skill/
    └── SKILL.md
```

**All files** under the skill directory (except `SKILL.md` itself) are recursively collected and returned in search results so the LLM can read them on demand.

### SKILL.md Format

```markdown
---
name: my-skill
description: Short description of what this skill does
version: "1.0"
---

# My Skill

Detailed instructions here...
```

## Embedding Models

**Vector dimensions are auto-detected from the API response**, not guessed from the model name. At startup the server generates one sample embedding via `/embeddings` and reads the actual `embedding.length` returned by the provider — the same pattern used by mcphub.

This means:
- **Model-agnostic**: works with any OpenAI-compatible embedding provider (OpenAI, Azure OpenAI, Gemini, BGE, Jina, SiliconFlow, etc.) without hardcoded dimension tables
- **Auto-adapts**: if you change `EMBEDDING_MODEL` to a model with different dimensions, the server detects the mismatch on next startup, resizes the pgvector column, clears stale data, and re-indexes everything
- **pgvector type selection**: dimensions ≤ 2000 use `vector` type; dimensions > 2000 use `halfvec` type (pgvector limit is 2000 for `vector`, 4000 for `halfvec`)

Common models and their typical dimensions:

| Model | Dimensions | pgvector type |
|---|---|---|
| `gemini-embedding-001` | 3072 | halfvec |
| `text-embedding-3-large` | 3072 | halfvec |
| `jina-embeddings-v3` | 1024 | vector |
| `text-embedding-3-small` | 1536 | vector |
| `text-embedding-ada-002` | 1536 | vector |
| `bge-m3` | 1024 | vector |
| `bge-large-zh` | 1024 | vector |
| `bge-base-zh` | 768 | vector |

> **Note**: The table above is for reference only. The actual dimension is always determined by the API response, not by this table. Any model supported by your `API_BASE_URL` provider will work.

### Dimension Mismatch Handling

When the detected dimension differs from the existing database column:

1. Existing HNSW index is dropped
2. All rows in the `skills` table are cleared (stale embeddings with wrong dimensions)
3. The `embedding` column type is resized (e.g. `vector(1536)` → `halfvec(3072)`)
4. HNSW index is recreated
5. All skills are re-indexed from disk

This ensures the database always matches the provider's actual output, even after model changes.

## Database Setup

Requires PostgreSQL with pgvector:

```sql
CREATE EXTENSION vector;
```

Table and HNSW index are created automatically on first run.

## Claude Desktop Configuration

```json
{
  "mcpServers": {
    "skillsearch": {
      "command": "npx",
      "args": ["github:ptbsare/skillsearch-mcp-server"],
      "env": {
        "DB_URL": "postgresql://user:pass@localhost:5432/db",
        "SKILLSEARCH_SKILLS_DIR": "/path/to/skills",
        "API_BASE_URL": "https://your-api/v1",
        "API_KEY": "sk-...",
        "EMBEDDING_MODEL": "gemini-embedding-001"
      }
    }
  }
}
```

## License

GPLv3.0
