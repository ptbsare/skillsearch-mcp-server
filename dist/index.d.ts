/**
 * SkillSearch MCP Server
 *
 * A standalone MCP server that discovers Agent Skills from a directory,
 * embeds their SKILL.md content into pgvector, and provides semantic search
 * with reranking for AI agents.
 *
 * Environment variables:
 *   SKILLSEARCH_DB_URL          - PostgreSQL connection string with pgvector
 *   SKILLSEARCH_RERANK_BASE_URL - Rerank model base URL (OpenAI-compatible, e.g. https://api.openai.com/v1)
 *   SKILLSEARCH_RERANK_API_KEY  - Rerank model API key
 *   SKILLSEARCH_RERANK_MODEL    - Rerank model name (e.g. "rerank-english-v3.0")
 *   SKILLSEARCH_SKILLS_DIR      - Root directory to scan for skills (each subdirectory = one skill)
 */
export {};
//# sourceMappingURL=index.d.ts.map