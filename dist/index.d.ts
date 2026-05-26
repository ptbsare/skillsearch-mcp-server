/**
 * SkillSearch MCP Server
 *
 * Discovers Agent Skills from a directory, embeds SKILL.md content into
 * pgvector via an OpenAI-compatible /embeddings endpoint, and provides
 * semantic vector cosine similarity search — same pattern as mcphub.
 *
 * Auto-sync: watches the skills directory for changes (fs.watch) and
 * falls back to periodic polling if native watching is unavailable.
 *
 * Environment variables (single naming):
 *   DB_URL                 - PostgreSQL connection string with pgvector
 *   API_BASE_URL           - Embedding API base URL (e.g. https://xxx/v1)
 *   API_KEY                - API key
 *   EMBEDDING_MODEL        - Model name (default: text-embedding-3-small)
 *   SKILLSEARCH_SKILLS_DIR - Root directory; each subdirectory = one skill
 *   TRANSPORT              - "stdio" (default) or "http"
 *   PORT                   - HTTP port (default: 3000)
 *   WATCH_POLL_INTERVAL    - Polling interval ms (default: 30000), fallback only
 */
export {};
//# sourceMappingURL=index.d.ts.map