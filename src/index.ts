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

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import pg from "pg";
import type { Pool as PoolType } from "pg";
const { Pool } = pg;
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

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SkillEntry {
  name: string;
  description: string;
  frontmatter: Record<string, string>;
  skillMdPath: string;
  skillDir: string;
  content: string;
  allFiles: string[];   // every file under skill dir except SKILL.md itself
}

interface DbRow {
  id: string;
  skill_name: string;
  skill_dir: string;
  skill_md_path: string;
  frontmatter: Record<string, string>;
  content: string;
  all_files: string[];
  similarity: number;
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

let pool: PoolType | null = null;

function getPool(): PoolType {
  if (!pool) {
    if (!DB_URL) throw new Error("DB_URL environment variable is not set");
    pool = new Pool({ connectionString: DB_URL });
  }
  return pool;
}

async function ensureSchema(db: PoolType): Promise<void> {
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

  // pgvector
  try {
    await db.query(`CREATE EXTENSION IF NOT EXISTS vector;`);

    const dims = embeddingDimensions(EMBEDDING_MODEL);
    console.error(`[skillsearch] embedding dimensions: ${dims}`);

    const col = await db.query(`
      SELECT 1 FROM information_schema.columns
      WHERE table_name = 'skills' AND column_name = 'embedding';
    `);

    if (col.rowCount === 0) {
      const vecType = dims <= 2000 ? "vector" : "halfvec";
      await db.query(`ALTER TABLE skills ADD COLUMN embedding ${vecType}(${dims});`);
      console.error(`[skillsearch] created embedding column ${vecType}(${dims})`);
    }

    // HNSW index
    if (dims <= 2000) {
      await db.query(`
        CREATE INDEX IF NOT EXISTS idx_skills_embedding
        ON skills USING hnsw (embedding vector_cosine_ops);
      `);
    } else {
      await db.query(`
        CREATE INDEX IF NOT EXISTS idx_skills_embedding
        ON skills USING hnsw ((embedding::halfvec(${dims})) halfvec_cosine_ops);
      `);
    }
    console.error("[skillsearch] HNSW index ensured");
  } catch (err: any) {
    console.warn(`[skillsearch] pgvector not available: ${err.message}`);
  }
}

function embeddingDimensions(model: string): number {
  const m = model.toLowerCase();
  if (m.includes("text-embedding-3-large") || m.includes("gemini-embedding-001")) return 3072;
  if (m.includes("text-embedding-3-small") || m.includes("text-embedding-ada-002")) return 1536;
  if (m.includes("bge-m3") || m.includes("bge-large")) return 1024;
  if (m.includes("bge-base")) return 768;
  if (m.includes("bge-small")) return 512;
  return 1536;
}

// ---------------------------------------------------------------------------
// Skill discovery — collect ALL files under skill dir
// ---------------------------------------------------------------------------

function discoverSkills(rootDir: string): SkillEntry[] {
  const skills: SkillEntry[] = [];
  if (!fs.existsSync(rootDir)) {
    console.warn(`[skillsearch] skills dir not found: ${rootDir}`);
    return skills;
  }

  for (const entry of fs.readdirSync(rootDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;

    const skillDir = path.join(rootDir, entry.name);
    const skillMdPath = path.join(skillDir, "SKILL.md");
    if (!fs.existsSync(skillMdPath)) {
      console.warn(`[skillsearch] no SKILL.md in ${skillDir}, skipping`);
      continue;
    }

    const raw = fs.readFileSync(skillMdPath, "utf-8");
    const { frontmatter } = parseFrontmatter(raw);
    const name = frontmatter["name"] ?? entry.name;
    const description = frontmatter["description"] ?? "";

    // ALL files recursively under skillDir, excluding SKILL.md itself
    const absMd = path.resolve(skillMdPath);
    const allFiles = walkFiles(skillDir).filter((f) => path.resolve(f) !== absMd);

    skills.push({
      name,
      description,
      frontmatter,
      skillMdPath: absMd,
      skillDir: path.resolve(skillDir),
      content: raw,
      allFiles: allFiles.map((f) => path.resolve(f)),
    });

    console.error(`[skillsearch] discovered "${name}": ${allFiles.length} files in ${skillDir}`);
  }

  return skills;
}

function parseFrontmatter(raw: string): { frontmatter: Record<string, string>; body: string } {
  const frontmatter: Record<string, string> = {};
  let body = raw;
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (match) {
    body = match[2];
    for (const line of match[1].split(/\r?\n/)) {
      if (!line.trim() || line.trim().startsWith("#")) continue;
      const kv = line.match(/^([a-zA-Z0-9_-]+)\s*:\s*(.*)$/);
      if (kv) frontmatter[kv[1]] = kv[2].trim();
    }
  }
  return { frontmatter, body };
}

function walkFiles(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  const results: string[] = [];
  const stack = [dir];
  while (stack.length) {
    const cur = stack.pop()!;
    let entries: fs.Dirent[];
    try { entries = fs.readdirSync(cur, { withFileTypes: true }); } catch { continue; }
    for (const e of entries) {
      const full = path.join(cur, e.name);
      e.isDirectory() ? stack.push(full) : results.push(full);
    }
  }
  return results;
}

function sha256(text: string): string {
  return crypto.createHash("sha256").update(text).digest("hex");
}

// ---------------------------------------------------------------------------
// Embedding — single /embeddings call, same as mcphub
// ---------------------------------------------------------------------------

async function embed(text: string): Promise<number[]> {
  if (!API_BASE_URL || !API_KEY) throw new Error("API_BASE_URL / API_KEY not set");

  // Conservative truncation: 60k chars ≈ 20k tokens
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

  const data = (await res.json()) as any;
  const vec = data.data?.[0]?.embedding;
  if (!Array.isArray(vec)) throw new Error(`unexpected response: ${JSON.stringify(data).slice(0, 200)}`);

  console.error(`[skillsearch] embedding ok: ${vec.length}d`);
  return vec;
}

// ---------------------------------------------------------------------------
// Indexing — upsert with content-hash skip (same pattern as mcphub)
// ---------------------------------------------------------------------------

async function indexSkills(db: PoolType, skills: SkillEntry[]): Promise<number> {
  let indexed = 0;

  for (const skill of skills) {
    const hash = sha256(skill.content);

    // Skip check
    const existing = await db.query(
      `SELECT content_hash FROM skills WHERE skill_name = $1 AND skill_dir = $2`,
      [skill.name, skill.skillDir],
    );
    if (existing.rowCount && existing.rows[0].content_hash === hash) {
      console.error(`[skillsearch] "${skill.name}" unchanged, skip`);
      continue;
    }

    // Generate embedding
    let embedding: number[] | null = null;
    if (API_BASE_URL && API_KEY) {
      try { embedding = await embed(skill.content); }
      catch (err: any) { console.warn(`[skillsearch] embed failed "${skill.name}": ${err.message}`); }
    }

    // Upsert
    if (embedding) {
      await db.query(
        `INSERT INTO skills
           (skill_name, skill_dir, skill_md_path, frontmatter, content, content_hash, all_files, embedding, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8::vector,NOW())
         ON CONFLICT (skill_name, skill_dir) DO UPDATE SET
           skill_md_path = EXCLUDED.skill_md_path,
           frontmatter   = EXCLUDED.frontmatter,
           content       = EXCLUDED.content,
           content_hash  = EXCLUDED.content_hash,
           all_files     = EXCLUDED.all_files,
           embedding     = EXCLUDED.embedding,
           updated_at    = NOW()`,
        [skill.name, skill.skillDir, skill.skillMdPath, JSON.stringify(skill.frontmatter),
         skill.content, hash, skill.allFiles, `[${embedding.join(",")}]`],
      );
    } else {
      await db.query(
        `INSERT INTO skills
           (skill_name, skill_dir, skill_md_path, frontmatter, content, content_hash, all_files, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
         ON CONFLICT (skill_name, skill_dir) DO UPDATE SET
           skill_md_path = EXCLUDED.skill_md_path,
           frontmatter   = EXCLUDED.frontmatter,
           content       = EXCLUDED.content,
           content_hash  = EXCLUDED.content_hash,
           all_files     = EXCLUDED.all_files,
           updated_at    = NOW()`,
        [skill.name, skill.skillDir, skill.skillMdPath, JSON.stringify(skill.frontmatter),
         skill.content, hash, skill.allFiles],
      );
    }

    indexed++;
    console.error(`[skillsearch] indexed "${skill.name}" (${skill.allFiles.length} files)`);
  }

  return indexed;
}

// ---------------------------------------------------------------------------
// Search — pure vector cosine similarity, same as mcphub searchToolsByVector
// ---------------------------------------------------------------------------

async function searchSkills(
  db: PoolType,
  query: string,
  limit: number,
  threshold: number,
): Promise<DbRow[]> {
  // Check embedding column
  const hasCol = await db.query(`
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'skills' AND column_name = 'embedding';
  `);

  if (!(hasCol.rowCount && hasCol.rowCount > 0)) {
    console.error("[skillsearch] no embedding column, returning empty");
    return [];
  }

  const queryVec = await embed(query);
  const vecStr = `[${queryVec.join(",")}]`;

  const result = await db.query(
    `SELECT id, skill_name, skill_dir, skill_md_path, frontmatter, content, all_files,
            1 - (embedding <=> $1::vector) AS similarity
     FROM skills
     WHERE embedding IS NOT NULL
       AND 1 - (embedding <=> $1::vector) >= $2
     ORDER BY embedding <=> $1::vector
     LIMIT $3`,
    [vecStr, threshold, limit],
  );

  console.error(`[skillsearch] vector search → ${result.rowCount} rows (threshold=${threshold})`);

  return result.rows.map((r: any) => ({
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
// MCP Server
// ---------------------------------------------------------------------------

function createServer(): Server {
  const server = new Server(
    { name: "skillsearch-mcp-server", version: "1.0.0" },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const tools: Tool[] = [
      {
        name: "skill_search",
        description:
          "Semantic vector search for Agent Skills. Returns matching skills with frontmatter metadata, SKILL.md absolute path, and ALL file paths under the skill directory (references, scripts, assets, etc.). Use this to discover available skills before reading their full content.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Natural language search query describing the skill you need.",
            },
            limit: {
              type: "integer",
              description: "Maximum results (default: 5).",
              default: 5,
            },
            threshold: {
              type: "number",
              description: "Minimum similarity 0.0–1.0 (default: 0.3). Higher = stricter.",
              default: 0.3,
            },
          },
          required: ["query"],
        },
      },
      {
        name: "skill_list",
        description: "List all indexed skills with names, descriptions, paths, and file counts.",
        inputSchema: { type: "object", properties: {}, required: [] },
      },
      {
        name: "skill_reindex",
        description:
          "Re-scan the skills directory and re-index all skills. Use after adding or modifying skills.",
        inputSchema: { type: "object", properties: {}, required: [] },
      },
    ];
    return { tools };
  });

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const db = getPool();

    switch (request.params.name) {
      case "skill_search": {
        const args = (request.params.arguments ?? {}) as Record<string, unknown>;
        const query = String(args.query ?? "");
        const limit = Math.min(Math.max(Number(args.limit ?? 5), 1), 50);
        const threshold = Math.min(Math.max(Number(args.threshold ?? 0.3), 0), 1);

        if (!query) {
          return { content: [{ type: "text" as const, text: "Error: 'query' is required" }], isError: true };
        }

        const results = await searchSkills(db, query, limit, threshold);

        if (!results.length) {
          return {
            content: [{ type: "text" as const, text: JSON.stringify(
              { query, threshold, totalResults: 0, message: "No matching skills found.", skills: [] }, null, 2) }],
          };
        }

        return {
          content: [{ type: "text" as const, text: JSON.stringify({
            query, threshold, totalResults: results.length,
            skills: results.map((r) => ({
              name: r.skill_name,
              description: r.frontmatter["description"] ?? "",
              frontmatter: r.frontmatter,
              skillMdPath: r.skill_md_path,
              skillDir: r.skill_dir,
              similarity: Math.round(r.similarity * 10000) / 10000,
              allFiles: r.all_files,
            })),
          }, null, 2) }],
        };
      }

      case "skill_list": {
        const result = await db.query(
          `SELECT skill_name, skill_dir, skill_md_path, frontmatter, all_files
           FROM skills ORDER BY skill_name`);
        const skills = result.rows.map((r: any) => ({
          name: r.skill_name,
          description: r.frontmatter?.description ?? "",
          skillDir: r.skill_dir,
          skillMdPath: r.skill_md_path,
          fileCount: r.all_files?.length ?? 0,
        }));
        return { content: [{ type: "text" as const, text: JSON.stringify({ total: skills.length, skills }, null, 2) }] };
      }

      case "skill_reindex": {
        if (!SKILLS_DIR) {
          return { content: [{ type: "text" as const, text: "Error: SKILLSEARCH_SKILLS_DIR not set" }], isError: true };
        }
        const skills = discoverSkills(SKILLS_DIR);
        const indexed = await indexSkills(db, skills);
        return { content: [{ type: "text" as const, text: JSON.stringify(
          { message: "Reindex complete", skillsFound: skills.length, skillsIndexed: indexed, skillsDir: path.resolve(SKILLS_DIR) },
          null, 2) }] };
      }

      default:
        return { content: [{ type: "text" as const, text: `Unknown tool: ${request.params.name}` }], isError: true };
    }
  });

  return server;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.error("[skillsearch] starting...");
  console.error(`  DB_URL         = ${DB_URL ? "set" : "MISSING"}`);
  console.error(`  API_BASE_URL   = ${API_BASE_URL || "not set"}`);
  console.error(`  EMBEDDING_MODEL= ${EMBEDDING_MODEL}`);
  console.error(`  SKILLS_DIR     = ${SKILLS_DIR || "MISSING"}`);

  if (!DB_URL) { console.error("[skillsearch] FATAL: DB_URL not set"); process.exit(1); }
  if (!SKILLS_DIR) { console.error("[skillsearch] FATAL: SKILLSEARCH_SKILLS_DIR not set"); process.exit(1); }

  const db = getPool();
  try {
    await db.query("SELECT 1");
    console.error("[skillsearch] database connected");
  } catch (err: any) {
    console.error(`[skillsearch] FATAL: db connect failed: ${err.message}`);
    process.exit(1);
  }

  await ensureSchema(db);

  const skills = discoverSkills(SKILLS_DIR);
  console.error(`[skillsearch] ${skills.length} skills discovered`);

  if (skills.length > 0) {
    const n = await indexSkills(db, skills);
    console.error(`[skillsearch] indexed ${n} (${skills.length - n} unchanged)`);
  }

  const server = createServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[skillsearch] MCP server running on stdio");
}

main().catch((err) => { console.error(`[skillsearch] fatal: ${err}`); process.exit(1); });
