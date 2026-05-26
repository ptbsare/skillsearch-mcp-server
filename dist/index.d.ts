/**
 * SkillSearch MCP Server
 *
 * A standalone MCP server that discovers Agent Skills from a directory,
 * embeds their SKILL.md content into pgvector, and provides semantic search
 * with reranking for AI agents.
 *
 * Environment variables:
 *   SKILLSEARCH_DB_URL / DB_URL   - PostgreSQL connection string with pgvector
 *   SKILLSEARCH_RERANK_BASE_URL / API_BASE_URL
 *                                   - Embedding/Rerank model base URL (OpenAI-compatible)
 *   SKILLSEARCH_RERANK_API_KEY / API_KEY
 *                                   - API key for the rerank/embedding provider
 *   SKILLSEARCH_RERANK_MODEL / EMBEDDING_MODEL
 *                                   - Model name (e.g. "gemini-embedding-001", "text-embedding-3-small")
 *   SKILLSEARCH_SKILLS_DIR         - Root directory to scan for skills (each subdirectory = one skill)
 */
export {};
//# sourceMappingURL=index.d.ts.map