# SkillSearch MCP Server

A standalone MCP server that discovers [Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) from a directory, indexes their `SKILL.md` content into PostgreSQL with pgvector, and provides semantic search with reranking for AI agents.

## Features

- **Auto-discovery**: Scans a directory for skill folders (each containing `SKILL.md`)
- **Vector search**: Embeds `SKILL.md` content into pgvector for semantic similarity search
- **Reranking**: Supports OpenAI-compatible rerank APIs for improved result quality
- **Full-text fallback**: Works without pgvector via PostgreSQL full-text search
- **Frontmatter parsing**: Extracts YAML frontmatter from `SKILL.md` files
- **Resource discovery**: Tracks `references/` and `scripts/` directories for each skill

## Quick Start

### Via npx

```bash
npx github:ptbsare/skillsearch-mcp-server
```

### Via local clone

```bash
git clone https://github.com/ptbsare/skillsearch-mcp-server.git
cd skillsearch-mcp-server
npm install
npm run build
SKILLSEARCH_DB_URL=postgresql://user:pass@localhost:5432/db \
SKILLSEARCH_SKILLS_DIR=/path/to/skills \
  node dist/index.js
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SKILLSEARCH_DB_URL` | **Yes** | PostgreSQL connection string (must have pgvector extension) |
| `SKILLSEARCH_SKILLS_DIR` | **Yes** | Root directory containing skill subdirectories |
| `SKILLSEARCH_RERANK_BASE_URL` | No | Rerank/embedding API base URL (e.g. `https://api.openai.com/v1`) |
| `SKILLSEARCH_RERANK_API_KEY` | No | API key for the rerank/embedding provider |
| `SKILLSEARCH_RERANK_MODEL` | No | Model name (default: `rerank-english-v3.0`) |

## MCP Tools

### `skill_search`

Search for skills by semantic query. Returns matching skills with frontmatter, paths, and similarity scores.

Parameters:
- `query` (string, required): Natural language search query
- `limit` (integer, optional): Max results (default: 5, max: 50)
- `threshold` (number, optional): Min similarity 0.0-1.0 (default: 0.3)

### `skill_list`

List all indexed skills with names, descriptions, and directory paths.

### `skill_reindex`

Re-scan the skills directory and re-index all skills. Use after adding or modifying skills.

## Skill Directory Structure

```
skills/
├── my-skill/
│   ├── SKILL.md          # Main skill file with YAML frontmatter
│   ├── references/       # Optional reference files
│   │   └── guide.md
│   └── scripts/          # Optional scripts
│       └── run.sh
└── another-skill/
    └── SKILL.md
```

### SKILL.md Format

```markdown
---
name: my-skill
description: A skill for doing something useful
version: "1.0"
---

# My Skill

Detailed instructions here...
```

## Claude Desktop Configuration

```json
{
  "mcpServers": {
    "skillsearch": {
      "command": "npx",
      "args": ["github:ptbsare/skillsearch-mcp-server"],
      "env": {
        "SKILLSEARCH_DB_URL": "postgresql://user:pass@localhost:5432/skills_db",
        "SKILLSEARCH_SKILLS_DIR": "/path/to/skills",
        "SKILLSEARCH_RERANK_BASE_URL": "https://api.openai.com/v1",
        "SKILLSEARCH_RERANK_API_KEY": "sk-...",
        "SKILLSEARCH_RERANK_MODEL": "rerank-english-v3.0"
      }
    }
  }
}
```

## Database Setup

Requires PostgreSQL with the `pgvector` extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The server auto-creates the `skills` table and HNSW index on first run.

## License

MIT
