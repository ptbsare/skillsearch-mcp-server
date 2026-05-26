/**
 * SkillSearch MCP Server
 *
 * Discovers Agent Skills from a directory, embeds SKILL.md content into
 * pgvector via an OpenAI-compatible /embeddings endpoint, and provides
 * semantic vector similarity search.
 *
 * Environment variables (single naming, no aliases):
 *   DB_URL                 - PostgreSQL connection string with pgvector
 *   API_BASE_URL           - Embedding API base URL (e.g. https://xxx/v1)
 *   API_KEY                - API key for the embedding provider
 *   EMBEDDING_MODEL        - Model name (e.g. "gemini-embedding-001")
 *   SKILLSEARCH_SKILLS_DIR - Root directory; each subdirectory = one skill
 */
export {};
//# sourceMappingURL=index.d.ts.map