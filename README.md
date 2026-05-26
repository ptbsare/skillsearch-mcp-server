# SkillSearch MCP Server

MCP server that discovers [Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) from a directory, embeds `SKILL.md` content into PostgreSQL pgvector via an OpenAI-compatible `/embeddings` endpoint, and provides semantic vector cosine similarity search — same pattern as mcphub's `search_tools`.

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

Semantic vector search for Agent Skills. Returns matching skills with frontmatter metadata, SKILL.md absolute path, and **all** file paths under the skill directory (references, scripts, assets, templates, etc.).

Parameters:
- `query` (string, required) — Natural language search query
- `limit` (integer, optional) — Max results (default: 5, max: 50)
- `threshold` (number, optional) — Min similarity 0.0–1.0 (default: 0.3)

### `skill_list`

List all indexed skills with names, descriptions, paths, and file counts.

### `skill_reindex`

Re-scan the skills directory and re-index all skills. Use after adding or modifying skills.

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

Auto-detected dimensions:

| Model | Dimensions | pgvector type |
|---|---|---|
| `gemini-embedding-001` | 3072 | halfvec |
| `text-embedding-3-large` | 3072 | halfvec |
| `text-embedding-3-small` | 1536 | vector |
| `text-embedding-ada-002` | 1536 | vector |
| `bge-m3` | 1024 | vector |

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

GPL-3.0-or-later
