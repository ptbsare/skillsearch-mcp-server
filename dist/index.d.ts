/**
 * SkillSearch MCP Server
 *
 * Discovers Agent Skills from a directory, embeds SKILL.md content into
 * pgvector via an OpenAI-compatible /embeddings endpoint, and provides
 * semantic vector similarity search — same pattern as mcphub search_tools.
 *
 * Environment variables (single naming):
 *   DB_URL                 - PostgreSQL connection string with pgvector
 *   API_BASE_URL           - Embedding API base URL (e.g. https://xxx/v1)
 *   API_KEY                - API key
 *   EMBEDDING_MODEL        - Model name (default: text-embedding-3-small)
 *   SKILLSEARCH_SKILLS_DIR - Root directory; each subdirectory = one skill
 *   TRANSPORT              - "stdio" (default) or "http"
 *   PORT                   - HTTP port (default: 3000), only used when TRANSPORT=http
 */
export {};
//# sourceMappingURL=index.d.ts.map