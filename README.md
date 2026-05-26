# SkillSearch MCP Server

MCP server that discovers [Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) from a directory, embeds `SKILL.md` content into PostgreSQL pgvector, and provides semantic vector similarity search.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DB_URL` | **Yes** | PostgreSQL connection string (must have pgvector) |
| `SKILLSEARCH_SKILLS_DIR` | **Yes** | Root directory; each subdirectory = one skill |
| `API_BASE_URL` | No | Embedding API base URL (e.g. `https://xxx/v1`) |
| `API_KEY` | No | API key for the embedding provider |
| `EMBEDDING_MODEL` | No | Model name (default: `text-embedding-3-small`) |

## MCP Tools

- **`skill_search`** — Semantic vector search. Returns frontmatter, SKILL.md path, and ALL file paths under the skill directory. Params: `query`, `limit` (default 5), `threshold` (default 0.3).
- **`skill_list`** — List all indexed skills.
- **`skill_reindex`** — Re-scan and re-index.

## Skill Directory Structure

```
skills/
├── my-skill/
│   ├── SKILL.md          # Required, with YAML frontmatter
│   ├── references/       # Any files (recursively collected)
│   ├── scripts/          # Any files (recursively collected)
│   └── assets/           # Any other files/dirs (recursively collected)
└── another-skill/
    └── SKILL.md
```

All files under the skill directory (except `SKILL.md` itself) are recursively collected and returned in search results so the LLM can read them on demand.

## Run

```bash
npx github:ptbsare/skillsearch-mcp-server
```

## Claude Desktop

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

## Database

Requires PostgreSQL with pgvector:

```sql
CREATE EXTENSION vector;
```

Table and HNSW index are created automatically on first run.

## License

GPL-3.0-or-later
