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
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { ListToolsRequestSchema, CallToolRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
import pg from "pg";
const { Pool } = pg;
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
// ---------------------------------------------------------------------------
// Configuration from environment
// Support both SKILLSEARCH_ prefixed and plain env var names.
// Plain names (DB_URL, API_BASE_URL, API_KEY, EMBEDDING_MODEL) take priority.
// ---------------------------------------------------------------------------
const DB_URL = process.env.DB_URL ?? process.env.SKILLSEARCH_DB_URL ?? "";
const RERANK_BASE_URL = process.env.API_BASE_URL ?? process.env.SKILLSEARCH_RERANK_BASE_URL ?? "";
const RERANK_API_KEY = process.env.API_KEY ?? process.env.SKILLSEARCH_RERANK_API_KEY ?? "";
const RERANK_MODEL = process.env.EMBEDDING_MODEL ??
    process.env.SKILLSEARCH_RERANK_MODEL ??
    "text-embedding-3-small";
const SKILLS_DIR = process.env.SKILLSEARCH_SKILLS_DIR ?? "";
// ---------------------------------------------------------------------------
// Database layer
// ---------------------------------------------------------------------------
let pool = null;
function getPool() {
    if (!pool) {
        if (!DB_URL) {
            throw new Error("DB_URL (or SKILLSEARCH_DB_URL) environment variable is not set");
        }
        pool = new Pool({ connectionString: DB_URL });
    }
    return pool;
}
async function ensureSchema(db) {
    // Create table without unique constraint on skill_name — we use
    // (skill_name, skill_dir) as the upsert key, or just use ON CONFLICT
    // on a unique index we create separately.
    await db.query(`
    CREATE TABLE IF NOT EXISTS skills (
      id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      skill_name        TEXT NOT NULL,
      skill_dir         TEXT NOT NULL,
      skill_md_path     TEXT NOT NULL,
      frontmatter       JSONB NOT NULL DEFAULT '{}'::jsonb,
      content           TEXT NOT NULL,
      content_hash      TEXT NOT NULL,
      all_files         TEXT[] NOT NULL DEFAULT '{}',
      created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
  `);
    // Unique index for upsert — allows same skill name in different dirs
    await db.query(`
    CREATE UNIQUE INDEX IF NOT EXISTS idx_skills_name_dir
    ON skills (skill_name, skill_dir);
  `);
    await db.query(`
    CREATE INDEX IF NOT EXISTS idx_skills_content_hash ON skills (content_hash);
  `);
    // Try to add embedding column; ignore error if it already exists or
    // the vector extension is not available.
    try {
        await db.query(`CREATE EXTENSION IF NOT EXISTS vector;`);
        // Detect embedding dimension from model name
        const embDims = getEmbeddingDimensions(RERANK_MODEL);
        console.error(`[skillsearch] Embedding dimensions for "${RERANK_MODEL}": ${embDims}`);
        const col = await db.query(`
      SELECT 1 FROM information_schema.columns
      WHERE table_name = 'skills' AND column_name = 'embedding';
    `);
        if (col.rowCount === 0) {
            // Use halfvec for dimensions > 2000
            const vecType = embDims <= 2000 ? "vector" : "halfvec";
            await db.query(`
        ALTER TABLE skills ADD COLUMN embedding ${vecType}(${embDims});
      `);
            console.error(`[skillsearch] Created embedding column: ${vecType}(${embDims})`);
        }
        else {
            // Column exists — try to resize if needed
            try {
                const existing = await db.query(`
          SELECT atttypmod FROM pg_attribute
          WHERE attrelid = 'skills'::regclass AND attname = 'embedding'
        `);
                if (existing.rowCount && existing.rows[0].atttypmod !== embDims) {
                    console.error(`[skillsearch] Resizing embedding column from ${existing.rows[0].atttypmod} to ${embDims}`);
                    // Drop index first if exists
                    await db.query(`DROP INDEX IF EXISTS idx_skills_embedding;`);
                    const vecType = embDims <= 2000 ? "vector" : "halfvec";
                    await db.query(`
            ALTER TABLE skills ALTER COLUMN embedding TYPE ${vecType}(${embDims});
          `);
                }
            }
            catch (resizeErr) {
                console.warn(`[skillsearch] Could not resize embedding column: ${resizeErr}`);
            }
        }
        // Create HNSW index
        if (embDims <= 2000) {
            await db.query(`
        CREATE INDEX IF NOT EXISTS idx_skills_embedding
        ON skills USING hnsw (embedding vector_cosine_ops);
      `);
        }
        else {
            await db.query(`
        CREATE INDEX IF NOT EXISTS idx_skills_embedding
        ON skills USING hnsw ((embedding::halfvec(${embDims})) halfvec_cosine_ops);
      `);
        }
        console.error("[skillsearch] HNSW index ensured");
    }
    catch (err) {
        console.warn(`[skillsearch] pgvector not available or error: ${err}. ` +
            `Run 'CREATE EXTENSION vector;' manually if you want vector search.`);
    }
}
function getEmbeddingDimensions(model) {
    const m = model.toLowerCase();
    if (m.includes("text-embedding-3-large") || m.includes("gemini-embedding-001"))
        return 3072;
    if (m.includes("text-embedding-3-small"))
        return 1536;
    if (m.includes("text-embedding-ada-002"))
        return 1536;
    if (m.includes("bge-m3"))
        return 1024;
    if (m.includes("bge-large"))
        return 1024;
    if (m.includes("bge-small"))
        return 512;
    if (m.includes("bge-base"))
        return 768;
    // Default to 1536 (most common)
    return 1536;
}
// ---------------------------------------------------------------------------
// Skill discovery
// ---------------------------------------------------------------------------
function discoverSkills(rootDir) {
    const skills = [];
    if (!fs.existsSync(rootDir)) {
        console.warn(`[skillsearch] Skills directory not found: ${rootDir}`);
        return skills;
    }
    const entries = fs.readdirSync(rootDir, { withFileTypes: true });
    for (const entry of entries) {
        if (!entry.isDirectory())
            continue;
        const skillDir = path.join(rootDir, entry.name);
        const skillMdPath = path.join(skillDir, "SKILL.md");
        if (!fs.existsSync(skillMdPath)) {
            console.warn(`[skillsearch] No SKILL.md in ${skillDir}, skipping`);
            continue;
        }
        const raw = fs.readFileSync(skillMdPath, "utf-8");
        const { frontmatter } = parseFrontmatter(raw);
        const name = frontmatter["name"] ?? entry.name;
        const description = frontmatter["description"] ?? "";
        // Collect ALL files recursively under skillDir, excluding SKILL.md itself
        const allFiles = collectAllFilesInDir(skillDir).filter((f) => path.resolve(f) !== path.resolve(skillMdPath));
        skills.push({
            name,
            description,
            frontmatter,
            skillMdPath: path.resolve(skillMdPath),
            skillDir: path.resolve(skillDir),
            content: raw,
            allFiles: allFiles.map((f) => path.resolve(f)),
            skillMdPathAbsolute: path.resolve(skillMdPath),
        });
        console.error(`[skillsearch] Discovered skill "${name}": ${allFiles.length} files in ${skillDir}`);
    }
    return skills;
}
function parseFrontmatter(raw) {
    const frontmatter = {};
    let body = raw;
    // Support YAML frontmatter between --- delimiters
    const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
    if (match) {
        const yamlBlock = match[1];
        body = match[2];
        for (const line of yamlBlock.split(/\r?\n/)) {
            // Skip empty lines and comments
            if (!line.trim() || line.trim().startsWith("#"))
                continue;
            const kv = line.match(/^([a-zA-Z0-9_-]+)\s*:\s*(.*)$/);
            if (kv) {
                frontmatter[kv[1]] = kv[2].trim();
            }
        }
    }
    return { frontmatter, body };
}
/** Recursively collect ALL files in a directory */
function collectAllFilesInDir(dir) {
    if (!fs.existsSync(dir))
        return [];
    const results = [];
    const stack = [dir];
    while (stack.length > 0) {
        const current = stack.pop();
        let entries;
        try {
            entries = fs.readdirSync(current, { withFileTypes: true });
        }
        catch {
            continue;
        }
        for (const e of entries) {
            const full = path.join(current, e.name);
            if (e.isDirectory()) {
                stack.push(full);
            }
            else {
                results.push(full);
            }
        }
    }
    return results;
}
function contentHash(content) {
    return crypto.createHash("sha256").update(content).digest("hex");
}
// ---------------------------------------------------------------------------
// Embedding generation via OpenAI-compatible /embeddings endpoint
// ---------------------------------------------------------------------------
async function generateEmbedding(text) {
    if (!RERANK_BASE_URL || !RERANK_API_KEY) {
        throw new Error("Embedding provider not configured (API_BASE_URL / API_KEY missing)");
    }
    const url = `${RERANK_BASE_URL.replace(/\/+$/, "")}/embeddings`;
    console.error(`[skillsearch] Generating embedding via ${url}, model=${RERANK_MODEL}`);
    // Truncate to avoid token limits — conservative 60000 chars (~20000 tokens)
    const truncated = text.length > 60000 ? text.substring(0, 60000) : text;
    const res = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${RERANK_API_KEY}`,
        },
        body: JSON.stringify({
            model: RERANK_MODEL,
            input: truncated,
        }),
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`Embedding API error ${res.status}: ${body}`);
    }
    const data = (await res.json());
    const embedding = data.data?.[0]?.embedding;
    if (!embedding || !Array.isArray(embedding)) {
        throw new Error(`Unexpected embedding response format: ${JSON.stringify(data).substring(0, 200)}`);
    }
    console.error(`[skillsearch] Embedding generated: ${embedding.length} dimensions`);
    return embedding;
}
// ---------------------------------------------------------------------------
// Indexing: upsert skills into DB, generate embeddings
// ---------------------------------------------------------------------------
async function indexSkills(db, skills) {
    let indexed = 0;
    for (const skill of skills) {
        const hash = contentHash(skill.content);
        // Check if already up-to-date
        const existing = await db.query(`SELECT content_hash FROM skills WHERE skill_name = $1 AND skill_dir = $2`, [skill.name, skill.skillDir]);
        if (existing.rowCount && existing.rows[0].content_hash === hash) {
            console.error(`[skillsearch] "${skill.name}" unchanged, skipping`);
            continue;
        }
        // Generate embedding if provider is configured
        let embedding = null;
        if (RERANK_BASE_URL && RERANK_API_KEY) {
            try {
                embedding = await generateEmbedding(skill.content);
            }
            catch (err) {
                console.warn(`[skillsearch] Embedding failed for "${skill.name}": ${err}`);
            }
        }
        // Upsert using (skill_name, skill_dir) unique index
        if (embedding) {
            await db.query(`INSERT INTO skills
           (skill_name, skill_dir, skill_md_path, frontmatter, content, content_hash,
            all_files, embedding, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8::vector,NOW())
         ON CONFLICT (skill_name, skill_dir) DO UPDATE SET
           skill_md_path   = EXCLUDED.skill_md_path,
           frontmatter     = EXCLUDED.frontmatter,
           content         = EXCLUDED.content,
           content_hash    = EXCLUDED.content_hash,
           all_files       = EXCLUDED.all_files,
           embedding       = EXCLUDED.embedding,
           updated_at      = NOW()`, [
                skill.name,
                skill.skillDir,
                skill.skillMdPath,
                JSON.stringify(skill.frontmatter),
                skill.content,
                hash,
                skill.allFiles,
                `[${embedding.join(",")}]`,
            ]);
        }
        else {
            await db.query(`INSERT INTO skills
           (skill_name, skill_dir, skill_md_path, frontmatter, content, content_hash,
            all_files, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
         ON CONFLICT (skill_name, skill_dir) DO UPDATE SET
           skill_md_path   = EXCLUDED.skill_md_path,
           frontmatter     = EXCLUDED.frontmatter,
           content         = EXCLUDED.content,
           content_hash    = EXCLUDED.content_hash,
           all_files       = EXCLUDED.all_files,
           updated_at      = NOW()`, [
                skill.name,
                skill.skillDir,
                skill.skillMdPath,
                JSON.stringify(skill.frontmatter),
                skill.content,
                hash,
                skill.allFiles,
            ]);
        }
        indexed++;
        console.error(`[skillsearch] Indexed "${skill.name}" (${skill.allFiles.length} files)`);
    }
    return indexed;
}
// ---------------------------------------------------------------------------
// Search: vector similarity + optional rerank
// ---------------------------------------------------------------------------
async function searchSkills(db, query, limit, threshold) {
    // Check if embedding column exists and is populated
    const hasEmbedding = await db.query(`
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'skills' AND column_name = 'embedding'
  `);
    let rows;
    if (hasEmbedding.rowCount &&
        hasEmbedding.rowCount > 0 &&
        RERANK_BASE_URL &&
        RERANK_API_KEY) {
        // Vector search
        const queryEmbedding = await generateEmbedding(query);
        const vecStr = `[${queryEmbedding.join(",")}]`;
        // Fetch extra candidates for rerank
        const fetchLimit = Math.max(limit * 3, 30);
        const result = await db.query(`SELECT id, skill_name, skill_dir, skill_md_path, frontmatter, content,
              all_files,
              1 - (embedding <=> $1::vector) AS similarity
       FROM skills
       WHERE embedding IS NOT NULL
         AND 1 - (embedding <=> $1::vector) >= $2
       ORDER BY embedding <=> $1::vector
       LIMIT $3`, [vecStr, threshold, fetchLimit]);
        rows = result.rows.map((r) => ({
            id: r.id,
            skill_name: r.skill_name,
            skill_dir: r.skill_dir,
            skill_md_path: r.skill_md_path,
            frontmatter: r.frontmatter,
            content: r.content,
            all_files: r.all_files ?? [],
            similarity: parseFloat(r.similarity),
        }));
        console.error(`[skillsearch] Vector search returned ${rows.length} results (threshold=${threshold})`);
    }
    else {
        // Fallback: full-text search
        console.error("[skillsearch] Using full-text search fallback");
        const result = await db.query(`SELECT id, skill_name, skill_dir, skill_md_path, frontmatter, content,
              all_files,
              ts_rank(
                to_tsvector('english', content || ' ' || skill_name || ' ' || COALESCE(frontmatter->>'description', '')),
                plainto_tsquery('english', $1)
              ) AS similarity
       FROM skills
       WHERE to_tsvector('english', content || ' ' || skill_name || ' ' || COALESCE(frontmatter->>'description', ''))
             @@ plainto_tsquery('english', $1)
       ORDER BY similarity DESC
       LIMIT $2`, [query, limit * 3]);
        rows = result.rows.map((r) => ({
            id: r.id,
            skill_name: r.skill_name,
            skill_dir: r.skill_dir,
            skill_md_path: r.skill_md_path,
            frontmatter: r.frontmatter,
            content: r.content,
            all_files: r.all_files ?? [],
            similarity: parseFloat(r.similarity),
        }));
    }
    // Rerank if provider is available and we have results
    if (RERANK_BASE_URL && RERANK_API_KEY && rows.length > 0) {
        try {
            rows = await rerankResults(query, rows, limit);
        }
        catch (err) {
            console.warn(`[skillsearch] Rerank failed, using vector scores: ${err}`);
            rows = rows.slice(0, limit);
        }
    }
    else {
        rows = rows.slice(0, limit);
    }
    return rows;
}
// ---------------------------------------------------------------------------
// Rerank via /rerank or /reranking endpoint
// ---------------------------------------------------------------------------
async function rerankResults(query, candidates, limit) {
    const endpoints = ["/rerank", "/reranking"];
    const documents = candidates.map((c) => `${c.skill_name}: ${c.frontmatter["description"] ?? ""}`.trim());
    for (const ep of endpoints) {
        const url = `${RERANK_BASE_URL.replace(/\/+$/, "")}${ep}`;
        console.error(`[skillsearch] Trying rerank endpoint: ${url}`);
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${RERANK_API_KEY}`,
                },
                body: JSON.stringify({
                    model: RERANK_MODEL,
                    query,
                    documents,
                }),
            });
            if (!res.ok) {
                const body = await res.text();
                console.warn(`[skillsearch] Rerank ${ep} → ${res.status}: ${body.substring(0, 200)}`);
                continue;
            }
            const data = (await res.json());
            const scored = [];
            if (data.results) {
                for (const r of data.results) {
                    scored.push({ index: r.index, score: r.relevance_score ?? r.score ?? 0 });
                }
            }
            else if (data.data) {
                for (const r of data.data) {
                    scored.push({ index: r.index, score: r.score ?? r.relevance_score ?? 0 });
                }
            }
            if (scored.length === 0) {
                console.warn(`[skillsearch] Rerank ${ep} returned no scored results`);
                continue;
            }
            scored.sort((a, b) => b.score - a.score);
            const reranked = [];
            for (const s of scored) {
                if (s.index >= 0 && s.index < candidates.length) {
                    reranked.push({ ...candidates[s.index], similarity: s.score });
                }
            }
            console.error(`[skillsearch] Reranked ${reranked.length} results via ${ep}`);
            return reranked.slice(0, limit);
        }
        catch (err) {
            console.warn(`[skillsearch] Rerank ${ep} error: ${err}`);
            continue;
        }
    }
    console.warn("[skillsearch] All rerank endpoints failed, using original scores");
    return candidates.slice(0, limit);
}
// ---------------------------------------------------------------------------
// MCP Server setup
// ---------------------------------------------------------------------------
function createServer() {
    const server = new Server({ name: "skillsearch-mcp-server", version: "1.0.0" }, { capabilities: { tools: {} } });
    // ---- ListTools ----
    server.setRequestHandler(ListToolsRequestSchema, async () => {
        const tools = [
            {
                name: "skill_search",
                description: "Search for Agent Skills by semantic query. Returns matching skills with frontmatter metadata, SKILL.md absolute path, and ALL file paths under the skill directory (references, scripts, assets, etc.). Use this to discover available skills before reading their full content.",
                inputSchema: {
                    type: "object",
                    properties: {
                        query: {
                            type: "string",
                            description: "Natural language search query describing the skill you need. Be specific about the task or domain.",
                        },
                        limit: {
                            type: "integer",
                            description: "Maximum number of results to return (default: 5).",
                            default: 5,
                        },
                        threshold: {
                            type: "number",
                            description: "Minimum similarity threshold (0.0 to 1.0). Higher values return fewer but more relevant results. Default: 0.3",
                            default: 0.3,
                        },
                    },
                    required: ["query"],
                },
            },
            {
                name: "skill_list",
                description: "List all indexed skills with their names, descriptions, directory paths, and file counts.",
                inputSchema: { type: "object", properties: {}, required: [] },
            },
            {
                name: "skill_reindex",
                description: "Re-scan the skills directory and re-index all skills into the database. Use this after adding or modifying skills.",
                inputSchema: { type: "object", properties: {}, required: [] },
            },
        ];
        return { tools };
    });
    // ---- CallTool ----
    server.setRequestHandler(CallToolRequestSchema, async (request) => {
        const db = getPool();
        switch (request.params.name) {
            case "skill_search": {
                const args = (request.params.arguments ?? {});
                const query = String(args.query ?? "");
                const limit = Math.min(Math.max(Number(args.limit ?? 5), 1), 50);
                const threshold = Math.min(Math.max(Number(args.threshold ?? 0.3), 0), 1);
                if (!query) {
                    return {
                        content: [{ type: "text", text: "Error: 'query' is required" }],
                        isError: true,
                    };
                }
                const results = await searchSkills(db, query, limit, threshold);
                if (results.length === 0) {
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify({
                                    query,
                                    threshold,
                                    totalResults: 0,
                                    message: "No matching skills found. Try a different query or lower the threshold.",
                                    skills: [],
                                }, null, 2),
                            },
                        ],
                    };
                }
                const response = {
                    query,
                    threshold,
                    totalResults: results.length,
                    skills: results.map((r) => ({
                        name: r.skill_name,
                        description: r.frontmatter["description"] ?? "",
                        frontmatter: r.frontmatter,
                        skillMdPath: r.skill_md_path,
                        skillDir: r.skill_dir,
                        similarity: Math.round(r.similarity * 10000) / 10000,
                        allFiles: r.all_files,
                    })),
                };
                return {
                    content: [{ type: "text", text: JSON.stringify(response, null, 2) }],
                };
            }
            case "skill_list": {
                const result = await db.query(`SELECT skill_name, skill_dir, skill_md_path, frontmatter, all_files
           FROM skills
           ORDER BY skill_name`);
                const skills = result.rows.map((r) => ({
                    name: r.skill_name,
                    description: r.frontmatter?.description ?? "",
                    skillDir: r.skill_dir,
                    skillMdPath: r.skill_md_path,
                    fileCount: r.all_files?.length ?? 0,
                }));
                return {
                    content: [
                        { type: "text", text: JSON.stringify({ total: skills.length, skills }, null, 2) },
                    ],
                };
            }
            case "skill_reindex": {
                if (!SKILLS_DIR) {
                    return {
                        content: [
                            { type: "text", text: "Error: SKILLSEARCH_SKILLS_DIR is not set" },
                        ],
                        isError: true,
                    };
                }
                const skills = discoverSkills(SKILLS_DIR);
                const indexed = await indexSkills(db, skills);
                return {
                    content: [
                        {
                            type: "text",
                            text: JSON.stringify({
                                message: "Reindex complete",
                                skillsFound: skills.length,
                                skillsIndexed: indexed,
                                skillsDir: path.resolve(SKILLS_DIR),
                            }, null, 2),
                        },
                    ],
                };
            }
            default:
                return {
                    content: [{ type: "text", text: `Unknown tool: ${request.params.name}` }],
                    isError: true,
                };
        }
    });
    return server;
}
// ---------------------------------------------------------------------------
// Main: discover → index → start MCP server
// ---------------------------------------------------------------------------
async function main() {
    console.error("[skillsearch] Starting SkillSearch MCP Server...");
    console.error(`[skillsearch] DB_URL: ${DB_URL ? "set" : "MISSING"}`);
    console.error(`[skillsearch] API_BASE_URL: ${RERANK_BASE_URL || "not set"}`);
    console.error(`[skillsearch] EMBEDDING_MODEL: ${RERANK_MODEL}`);
    console.error(`[skillsearch] SKILLS_DIR: ${SKILLS_DIR || "MISSING"}`);
    if (!DB_URL) {
        console.error("[skillsearch] FATAL: DB_URL (or SKILLSEARCH_DB_URL) is not set");
        process.exit(1);
    }
    if (!SKILLS_DIR) {
        console.error("[skillsearch] FATAL: SKILLSEARCH_SKILLS_DIR is not set");
        process.exit(1);
    }
    // Connect to DB
    const db = getPool();
    try {
        await db.query("SELECT 1");
        console.error("[skillsearch] Database connected");
    }
    catch (err) {
        console.error(`[skillsearch] FATAL: Cannot connect to database: ${err.message}`);
        process.exit(1);
    }
    // Ensure schema
    await ensureSchema(db);
    console.error("[skillsearch] Database schema ensured");
    // Discover & index skills
    const skills = discoverSkills(SKILLS_DIR);
    console.error(`[skillsearch] Discovered ${skills.length} skills in ${SKILLS_DIR}`);
    if (skills.length > 0) {
        const indexed = await indexSkills(db, skills);
        console.error(`[skillsearch] Indexed ${indexed} skills (${skills.length - indexed} unchanged)`);
    }
    // Start MCP server (stdio transport)
    const server = createServer();
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("[skillsearch] MCP server running on stdio");
}
main().catch((err) => {
    console.error(`[skillsearch] Fatal error: ${err}`);
    process.exit(1);
});
//# sourceMappingURL=index.js.map