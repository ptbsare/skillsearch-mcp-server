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
 *   ENABLE_SKILL_LIST      - "true" to enable skill_list tool (default: false)
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { ListToolsRequestSchema, CallToolRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
import pg from "pg";
const { Pool } = pg;
import http from "node:http";
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const DB_URL = process.env.DB_URL ?? "";
const API_BASE_URL = process.env.API_BASE_URL ?? "";
const API_KEY = process.env.API_KEY ?? "";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL ?? "text-embedding-3-small";
const SKILLS_DIR = process.env.SKILLSEARCH_SKILLS_DIR ?? "";
const TRANSPORT = (process.env.TRANSPORT ?? "stdio").toLowerCase();
const PORT = parseInt(process.env.PORT ?? "3000", 10);
const POLL_INTERVAL = parseInt(process.env.WATCH_POLL_INTERVAL ?? "30000", 10);
const ENABLE_SKILL_LIST = (process.env.ENABLE_SKILL_LIST ?? "").toLowerCase() === "true";
const ENABLE_SKILL_SEARCH_HINT = (process.env.ENABLE_SKILL_SEARCH_HINT ?? "true").toLowerCase() !== "false";
// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------
let pool = null;
function getPool() {
    if (!pool) {
        if (!DB_URL)
            throw new Error("DB_URL environment variable is not set");
        pool = new Pool({ connectionString: DB_URL });
    }
    return pool;
}
async function ensureBaseTable(db) {
    await db.query(`
    CREATE TABLE IF NOT EXISTS skills (
      id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      skill_name      TEXT NOT NULL,
      skill_dir       TEXT NOT NULL,
      skill_md_path   TEXT NOT NULL,
      frontmatter     JSONB NOT NULL DEFAULT '{}'::jsonb,
      content         TEXT NOT NULL,
      content_hash    TEXT NOT NULL,
      all_files       TEXT[] NOT NULL DEFAULT '{}',
      created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
  `);
    await db.query(`
    CREATE UNIQUE INDEX IF NOT EXISTS idx_skills_name_dir
    ON skills (skill_name, skill_dir);
  `);
    await db.query(`
    CREATE INDEX IF NOT EXISTS idx_skills_hash ON skills (content_hash);
  `);
    try {
        await db.query(`CREATE EXTENSION IF NOT EXISTS vector;`);
        console.error("[skillsearch] pgvector extension ensured");
    }
    catch (err) {
        console.warn(`[skillsearch] pgvector not available: ${err.message}`);
    }
}
/**
 * Ensure the embedding column matches the actual dimension from the API.
 * mcphub pattern: detect dims from embedding.length, not from model name.
 * Creates column if missing, resizes if mismatch, drops + recreates HNSW index.
 * Returns true if dimensions changed (caller should resync all embeddings).
 */
async function ensureVectorColumn(db, dimsNeeded) {
    // Get current column dimension from pg_attribute
    let currentDims = 0;
    try {
        const result = await db.query(`
      SELECT atttypmod AS dims FROM pg_attribute
      WHERE attrelid = 'skills'::regclass AND attname = 'embedding';
    `);
        if (result.rowCount && result.rows[0].dims > 0) {
            currentDims = result.rows[0].dims;
        }
    }
    catch {
        // table might not have column yet
    }
    const vecType = dimsNeeded <= 2000 ? "vector" : "halfvec";
    // Column does not exist → create it
    if (currentDims === 0) {
        await db.query(`ALTER TABLE skills ADD COLUMN embedding ${vecType}(${dimsNeeded});`);
        console.error(`[skillsearch] created embedding column ${vecType}(${dimsNeeded})`);
    }
    else if (currentDims !== dimsNeeded) {
        // Dimension mismatch → drop index, clear all data, resize column
        console.error(`[skillsearch] dimension mismatch: DB=${currentDims}, API=${dimsNeeded}. Resizing...`);
        await db.query(`DROP INDEX IF EXISTS idx_skills_embedding;`);
        await db.query(`DELETE FROM skills;`);
        await db.query(`ALTER TABLE skills ALTER COLUMN embedding TYPE ${vecType}(${dimsNeeded});`);
        console.error(`[skillsearch] resized embedding column to ${vecType}(${dimsNeeded})`);
    }
    else {
        console.error(`[skillsearch] embedding column already correct: ${vecType}(${dimsNeeded})`);
    }
    // Create HNSW index
    try {
        if (dimsNeeded <= 2000) {
            await db.query(`
        CREATE INDEX IF NOT EXISTS idx_skills_embedding
        ON skills USING hnsw (embedding vector_cosine_ops);
      `);
        }
        else {
            await db.query(`
        CREATE INDEX IF NOT EXISTS idx_skills_embedding
        ON skills USING hnsw ((embedding::halfvec(${dimsNeeded})) halfvec_cosine_ops);
      `);
        }
        console.error("[skillsearch] HNSW index ensured");
    }
    catch (err) {
        console.warn(`[skillsearch] HNSW index failed: ${err.message}`);
    }
    return currentDims !== 0 && currentDims !== dimsNeeded;
}
// ---------------------------------------------------------------------------
// Skill discovery — collect ALL files under skill dir (except SKILL.md)
// ---------------------------------------------------------------------------
function discoverSkills(rootDir) {
    const skills = [];
    if (!fs.existsSync(rootDir)) {
        console.warn(`[skillsearch] skills dir not found: ${rootDir}`);
        return skills;
    }
    // Recursively find all directories containing SKILL.md
    // Supports both flat (skills/my-skill/SKILL.md) and
    // category-grouped (skills/category/my-skill/SKILL.md) layouts
    const stack = [rootDir];
    while (stack.length) {
        const current = stack.pop();
        let entries;
        try {
            entries = fs.readdirSync(current, { withFileTypes: true });
        }
        catch {
            continue;
        }
        let hasSkillMd = false;
        const subdirs = [];
        for (const entry of entries) {
            if (entry.isDirectory()) {
                subdirs.push(path.join(current, entry.name));
            }
            else if (entry.name === "SKILL.md") {
                hasSkillMd = true;
            }
        }
        if (hasSkillMd) {
            // This directory is a skill root — parse it
            const skillDir = path.resolve(current);
            const skillMdPath = path.join(skillDir, "SKILL.md");
            const raw = fs.readFileSync(skillMdPath, "utf-8");
            const { frontmatter } = parseFrontmatter(raw);
            // For category-grouped paths like skills/apple/findmy/SKILL.md,
            // use the parent dir name as a display prefix if not overridden
            const dirName = path.basename(skillDir);
            const name = frontmatter["name"] ?? dirName;
            const description = frontmatter["description"] ?? "";
            const absMd = path.resolve(skillMdPath);
            const allFiles = walkFiles(skillDir).filter((f) => path.resolve(f) !== absMd);
            skills.push({
                name, description, frontmatter,
                skillMdPath: absMd,
                skillDir,
                content: raw,
                allFiles: allFiles.map((f) => path.resolve(f)),
            });
            console.error(`[skillsearch] discovered "${name}" in ${skillDir}`);
            // Don't recurse deeper — this IS the skill directory
        }
        else {
            // No SKILL.md here — recurse into subdirectories (category folders)
            for (const sd of subdirs)
                stack.push(sd);
        }
    }
    return skills;
}
/** Scan a single skill dir and return entry, or null if invalid */
function scanOneSkill(skillDir) {
    skillDir = path.resolve(skillDir);
    const entry = path.basename(skillDir);
    const skillMdPath = path.join(skillDir, "SKILL.md");
    if (!fs.existsSync(skillMdPath))
        return null;
    const raw = fs.readFileSync(skillMdPath, "utf-8");
    const { frontmatter } = parseFrontmatter(raw);
    const name = frontmatter["name"] ?? entry;
    const absMd = path.resolve(skillMdPath);
    const allFiles = walkFiles(skillDir).filter((f) => path.resolve(f) !== absMd);
    return {
        name, description: frontmatter["description"] ?? "", frontmatter,
        skillMdPath: absMd, skillDir, content: raw,
        allFiles: allFiles.map((f) => path.resolve(f)),
    };
}
function parseFrontmatter(raw) {
    const frontmatter = {};
    let body = raw;
    const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
    if (match) {
        body = match[2];
        for (const line of match[1].split(/\r?\n/)) {
            if (!line.trim() || line.trim().startsWith("#"))
                continue;
            const kv = line.match(/^([a-zA-Z0-9_-]+)\s*:\s*(.*)$/);
            if (kv)
                frontmatter[kv[1]] = kv[2].trim();
        }
    }
    return { frontmatter, body };
}
function walkFiles(dir) {
    if (!fs.existsSync(dir))
        return [];
    const results = [];
    const stack = [dir];
    while (stack.length) {
        const cur = stack.pop();
        let entries;
        try {
            entries = fs.readdirSync(cur, { withFileTypes: true });
        }
        catch {
            continue;
        }
        for (const e of entries) {
            const full = path.join(cur, e.name);
            e.isDirectory() ? stack.push(full) : results.push(full);
        }
    }
    return results;
}
function sha256(text) {
    return crypto.createHash("sha256").update(text).digest("hex");
}
// ---------------------------------------------------------------------------
// Embedding — single /embeddings call, same as mcphub generateEmbedding()
// ---------------------------------------------------------------------------
async function embed(text) {
    if (!API_BASE_URL || !API_KEY)
        throw new Error("API_BASE_URL / API_KEY not set");
    const input = text.length > 60000 ? text.slice(0, 60000) : text;
    const url = `${API_BASE_URL.replace(/\/+$/, "")}/embeddings`;
    console.error(`[skillsearch] embed → ${url} model=${EMBEDDING_MODEL}`);
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` },
        body: JSON.stringify({ model: EMBEDDING_MODEL, input }),
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`embeddings ${res.status}: ${body}`);
    }
    const data = (await res.json());
    const vec = data.data?.[0]?.embedding;
    if (!Array.isArray(vec))
        throw new Error(`unexpected response: ${JSON.stringify(data).slice(0, 200)}`);
    console.error(`[skillsearch] embedding ok: ${vec.length}d`);
    return vec;
}
// ---------------------------------------------------------------------------
// Indexing helpers
// ---------------------------------------------------------------------------
async function upsertSkill(db, skill) {
    const hash = sha256(skill.content);
    const existing = await db.query(`SELECT content_hash FROM skills WHERE skill_name = $1 AND skill_dir = $2`, [skill.name, skill.skillDir]);
    if (existing.rowCount && existing.rows[0].content_hash === hash) {
        console.error(`[skillsearch] "${skill.name}" unchanged, skip`);
        return;
    }
    let embedding = null;
    if (API_BASE_URL && API_KEY) {
        try {
            embedding = await embed(skill.content);
        }
        catch (err) {
            console.warn(`[skillsearch] embed failed "${skill.name}": ${err.message}`);
        }
    }
    if (embedding) {
        await db.query(`INSERT INTO skills
         (skill_name, skill_dir, skill_md_path, frontmatter, content, content_hash, all_files, embedding, updated_at)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8::vector,NOW())
       ON CONFLICT (skill_name, skill_dir) DO UPDATE SET
         skill_md_path = EXCLUDED.skill_md_path,
         frontmatter   = EXCLUDED.frontmatter,
         content       = EXCLUDED.content,
         content_hash  = EXCLUDED.content_hash,
         all_files     = EXCLUDED.all_files,
         embedding     = EXCLUDED.embedding,
         updated_at    = NOW()`, [skill.name, skill.skillDir, skill.skillMdPath, JSON.stringify(skill.frontmatter),
            skill.content, hash, skill.allFiles, `[${embedding.join(",")}]`]);
    }
    else {
        await db.query(`INSERT INTO skills
         (skill_name, skill_dir, skill_md_path, frontmatter, content, content_hash, all_files, updated_at)
       VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
       ON CONFLICT (skill_name, skill_dir) DO UPDATE SET
         skill_md_path = EXCLUDED.skill_md_path,
         frontmatter   = EXCLUDED.frontmatter,
         content       = EXCLUDED.content,
         content_hash  = EXCLUDED.content_hash,
         all_files     = EXCLUDED.all_files,
         updated_at    = NOW()`, [skill.name, skill.skillDir, skill.skillMdPath, JSON.stringify(skill.frontmatter),
            skill.content, hash, skill.allFiles]);
    }
    console.error(`[skillsearch] indexed "${skill.name}" (${skill.allFiles.length} files)`);
}
async function removeSkill(db, skillName, skillDirPath) {
    const res = await db.query(`DELETE FROM skills WHERE skill_name = $1 AND skill_dir = $2`, [skillName, skillDirPath]);
    if (res.rowCount && res.rowCount > 0) {
        console.error(`[skillsearch] removed "${skillName}" from index`);
    }
}
/** Full scan: index new/changed, remove deleted */
async function syncAll(db) {
    const skills = discoverSkills(SKILLS_DIR);
    const currentKeys = new Set();
    for (const skill of skills) {
        currentKeys.add(`${skill.name}\0${skill.skillDir}`);
        await upsertSkill(db, skill);
    }
    // Remove skills no longer on disk
    const indexed = await db.query(`SELECT skill_name, skill_dir FROM skills`);
    for (const row of indexed.rows) {
        const key = `${row.skill_name}\0${row.skill_dir}`;
        if (!currentKeys.has(key)) {
            await removeSkill(db, row.skill_name, row.skill_dir);
        }
    }
}
// ---------------------------------------------------------------------------
// File system watcher — auto sync on changes, with polling fallback
// ---------------------------------------------------------------------------
function startWatcher(db) {
    // Try native fs.watch first
    let usePolling = false;
    let watcher = null;
    // Debounce: coalesce rapid events into one sync
    let debounceTimer = null;
    const scheduleSync = () => {
        if (debounceTimer)
            clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            try {
                console.error("[skillsearch] change detected, syncing...");
                await syncAll(db);
                console.error("[skillsearch] sync complete");
            }
            catch (err) {
                console.error(`[skillsearch] sync error: ${err.message}`);
            }
        }, 500);
    };
    // Collect all directories that contain SKILL.md (recursively)
    // This handles both flat and category-grouped layouts
    const collectSkillDirs = (dir) => {
        const result = [];
        if (!fs.existsSync(dir))
            return result;
        let entries;
        try {
            entries = fs.readdirSync(dir, { withFileTypes: true });
        }
        catch {
            return result;
        }
        let hasSkillMd = false;
        const subdirs = [];
        for (const entry of entries) {
            if (entry.name === "SKILL.md")
                hasSkillMd = true;
            else if (entry.isDirectory())
                subdirs.push(path.join(dir, entry.name));
        }
        if (hasSkillMd) {
            result.push(path.resolve(dir));
        }
        else {
            for (const sd of subdirs) {
                result.push(...collectSkillDirs(sd));
            }
        }
        return result;
    };
    // Watch a directory recursively (covers skill dirs and category dirs)
    const watchedDirs = new Set();
    const watchDir = (dir) => {
        if (watchedDirs.has(dir))
            return;
        watchedDirs.add(dir);
        try {
            const w = fs.watch(dir, { persistent: true }, () => scheduleSync());
            w.on("error", () => { watchedDirs.delete(dir); });
        }
        catch {
            // ignore
        }
    };
    try {
        // Watch the root
        watcher = fs.watch(SKILLS_DIR, { persistent: true }, () => scheduleSync());
        watcher.on("error", (err) => {
            console.error(`[skillsearch] fs.watch error: ${err.message}, falling back to polling`);
            usePolling = true;
            watcher?.close();
            startPolling();
        });
        watchedDirs.add(path.resolve(SKILLS_DIR));
        // Watch all skill dirs and their category parent dirs
        if (fs.existsSync(SKILLS_DIR)) {
            for (const skillDir of collectSkillDirs(SKILLS_DIR)) {
                watchDir(skillDir);
                // Also watch the parent (category) dir so new siblings are detected
                const parent = path.dirname(skillDir);
                if (parent !== path.resolve(SKILLS_DIR))
                    watchDir(parent);
            }
        }
        console.error(`[skillsearch] fs.watch active on ${watchedDirs.size} directories`);
    }
    catch (err) {
        console.error(`[skillsearch] fs.watch unavailable: ${err.message}`);
        usePolling = true;
    }
    // Polling fallback
    let pollTimer = null;
    const startPolling = () => {
        if (pollTimer)
            return;
        console.error(`[skillsearch] polling fallback every ${POLL_INTERVAL}ms`);
        pollTimer = setInterval(async () => {
            try {
                await syncAll(db);
            }
            catch (err) {
                console.error(`[skillsearch] poll sync error: ${err.message}`);
            }
        }, POLL_INTERVAL);
    };
    if (usePolling) {
        startPolling();
    }
    // Clean up on process exit
    const cleanup = () => {
        watcher?.close();
        if (pollTimer)
            clearInterval(pollTimer);
    };
    process.on("exit", cleanup);
    process.on("SIGINT", () => { cleanup(); process.exit(0); });
    process.on("SIGTERM", () => { cleanup(); process.exit(0); });
}
// ---------------------------------------------------------------------------
// Search — pure vector cosine similarity, same as mcphub searchToolsByVector
// ---------------------------------------------------------------------------
async function searchSkills(db, query, limit, threshold) {
    const hasCol = await db.query(`
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'skills' AND column_name = 'embedding';
  `);
    if (!(hasCol.rowCount && hasCol.rowCount > 0))
        return [];
    const queryVec = await embed(query);
    const vecStr = `[${queryVec.join(",")}]`;
    const result = await db.query(`SELECT id, skill_name, skill_dir, skill_md_path, frontmatter, content, all_files,
            1 - (embedding <=> $1::vector) AS similarity
     FROM skills
     WHERE embedding IS NOT NULL
       AND 1 - (embedding <=> $1::vector) >= $2
     ORDER BY embedding <=> $1::vector
     LIMIT $3`, [vecStr, threshold, limit]);
    console.error(`[skillsearch] vector search → ${result.rowCount} rows (threshold=${threshold})`);
    return result.rows.map((r) => ({
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
// ---------------------------------------------------------------------------
// MCP Server — tools definition (no skill_reindex)
// ---------------------------------------------------------------------------
function createServer(db) {
    const server = new Server({ name: "skillsearch-mcp-server", version: "1.0.0" }, { capabilities: { tools: {} } });
    server.setRequestHandler(ListToolsRequestSchema, async () => {
        // Build the base description
        let skillSearchDescription = "Semantic vector search for Agent Skills. Returns matching skills with frontmatter metadata, SKILL.md absolute path, and ALL file paths under the skill directory (references, scripts, assets, templates, etc.). Skills are auto-indexed when added or modified — no manual reindex needed.";
        // Optionally inject available skill names as a hint for the LLM
        if (ENABLE_SKILL_SEARCH_HINT) {
            try {
                const result = await db.query(`SELECT skill_name FROM skills ORDER BY skill_name`);
                if (result.rowCount && result.rowCount > 0) {
                    const names = result.rows.map((r) => r.skill_name);
                    skillSearchDescription += `\n\nAvailable skills: ${names.join(", ")}.`;
                }
            }
            catch (err) {
                console.warn(`[skillsearch] failed to fetch skill names for hint: ${err.message}`);
            }
        }
        const tools = [
            {
                name: "skill_search",
                description: skillSearchDescription,
                inputSchema: {
                    type: "object",
                    properties: {
                        query: {
                            type: "string",
                            description: "Natural language search query describing the skill you need.",
                        },
                        limit: {
                            type: "integer",
                            description: "Maximum results to return (default: 5, max: 50).",
                            default: 5,
                        },
                        threshold: {
                            type: "number",
                            description: "Minimum similarity threshold 0.0–1.0 (default: 0.5). Higher = stricter.",
                            default: 0.5,
                        },
                    },
                    required: ["query"],
                },
            },
        ];
        if (ENABLE_SKILL_LIST) {
            tools.push({
                name: "skill_list",
                description: "List all indexed skills with names, descriptions, paths, and file counts.",
                inputSchema: { type: "object", properties: {}, required: [] },
            });
        }
        return { tools };
    });
    server.setRequestHandler(CallToolRequestSchema, async (request) => {
        const db = getPool();
        switch (request.params.name) {
            case "skill_search": {
                const args = (request.params.arguments ?? {});
                const query = String(args.query ?? "");
                const limit = Math.min(Math.max(Number(args.limit ?? 5), 1), 50);
                const threshold = Math.min(Math.max(Number(args.threshold ?? 0.5), 0), 1);
                if (!query) {
                    return { content: [{ type: "text", text: "Error: 'query' is required" }], isError: true };
                }
                const results = await searchSkills(db, query, limit, threshold);
                if (!results.length) {
                    return { content: [{ type: "text", text: JSON.stringify({ query, threshold, totalResults: 0, message: "No matching skills found.", skills: [] }, null, 2) }] };
                }
                return { content: [{ type: "text", text: JSON.stringify({
                                query, threshold, totalResults: results.length,
                                skills: results.map((r) => ({
                                    name: r.skill_name,
                                    description: r.frontmatter["description"] ?? "",
                                    skillMdPath: r.skill_md_path,
                                    skillDir: r.skill_dir,
                                    similarity: Math.round(r.similarity * 10000) / 10000,
                                    allFiles: r.all_files,
                                })),
                            }, null, 2) }] };
            }
            case "skill_list": {
                const result = await db.query(`SELECT skill_name, skill_dir, skill_md_path, frontmatter, all_files
           FROM skills ORDER BY skill_name`);
                return { content: [{ type: "text", text: JSON.stringify({
                                total: result.rowCount,
                                skills: result.rows.map((r) => ({
                                    name: r.skill_name,
                                    description: r.frontmatter?.description ?? "",
                                    skillDir: r.skill_dir,
                                    skillMdPath: r.skill_md_path,
                                    fileCount: r.all_files?.length ?? 0,
                                })),
                            }, null, 2) }] };
            }
            default:
                return { content: [{ type: "text", text: `Unknown tool: ${request.params.name}` }], isError: true };
        }
    });
    return server;
}
// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
    console.error("[skillsearch] starting...");
    console.error(`  DB_URL         = ${DB_URL ? "set" : "MISSING"}`);
    console.error(`  API_BASE_URL   = ${API_BASE_URL || "not set"}`);
    console.error(`  EMBEDDING_MODEL= ${EMBEDDING_MODEL}`);
    console.error(`  SKILLS_DIR     = ${SKILLS_DIR || "MISSING"}`);
    console.error(`  TRANSPORT      = ${TRANSPORT}`);
    if (TRANSPORT === "http")
        console.error(`  PORT           = ${PORT}`);
    console.error(`  ENABLE_SKILL_LIST= ${ENABLE_SKILL_LIST}`);
    if (!DB_URL) {
        console.error("[skillsearch] FATAL: DB_URL not set");
        process.exit(1);
    }
    if (!SKILLS_DIR) {
        console.error("[skillsearch] FATAL: SKILLSEARCH_SKILLS_DIR not set");
        process.exit(1);
    }
    const db = getPool();
    try {
        await db.query("SELECT 1");
        console.error("[skillsearch] database connected");
    }
    catch (err) {
        console.error(`[skillsearch] FATAL: db connect failed: ${err.message}`);
        process.exit(1);
    }
    await ensureBaseTable(db);
    // Discover skills first
    const skills = discoverSkills(SKILLS_DIR);
    console.error(`[skillsearch] ${skills.length} skills discovered`);
    // If we have an embedding provider and skills, generate one embedding
    // to detect the actual vector dimension (same pattern as mcphub).
    if (skills.length > 0 && API_BASE_URL && API_KEY) {
        try {
            const sampleVec = await embed(skills[0].content);
            const actualDims = sampleVec.length;
            console.error(`[skillsearch] detected actual embedding dimension: ${actualDims}`);
            const dimsChanged = await ensureVectorColumn(db, actualDims);
            if (dimsChanged) {
                console.error("[skillsearch] vector column was resized, clearing stale data for re-index");
            }
        }
        catch (err) {
            console.warn(`[skillsearch] dimension detection failed: ${err.message}`);
        }
    }
    // Index all skills
    if (skills.length > 0) {
        for (const s of skills)
            await upsertSkill(db, s);
        console.error(`[skillsearch] initial index complete`);
    }
    // Start file watcher (auto-sync)
    startWatcher(db);
    // MCP server
    const mcpServer = createServer(db);
    if (TRANSPORT === "http") {
        const httpServer = http.createServer((req, res) => {
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.setHeader("Access-Control-Allow-Headers", "Content-Type, Accept, Mcp-Session-Id");
            if (req.method === "OPTIONS") {
                res.writeHead(204);
                res.end();
                return;
            }
            const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
            if (url.pathname !== "/mcp" && url.pathname !== "/") {
                res.writeHead(404);
                res.end("Not Found");
                return;
            }
            let body = "";
            req.on("data", (chunk) => { body += chunk; });
            req.on("end", async () => {
                let parsedBody;
                if (body) {
                    try {
                        parsedBody = JSON.parse(body);
                    }
                    catch { /* ok */ }
                }
                const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
                res.on("close", () => transport.close());
                await mcpServer.connect(transport);
                await transport.handleRequest(req, res, parsedBody);
            });
        });
        httpServer.listen(PORT, () => {
            console.error(`[skillsearch] HTTP MCP server listening on :${PORT}/mcp`);
        });
    }
    else {
        const transport = new StdioServerTransport();
        await mcpServer.connect(transport);
        console.error("[skillsearch] MCP server running on stdio");
    }
}
main().catch((err) => { console.error(`[skillsearch] fatal: ${err}`); process.exit(1); });
//# sourceMappingURL=index.js.map