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
// Configuration from environment
// ---------------------------------------------------------------------------

const DB_URL = process.env.SKILLSEARCH_DB_URL ?? "";
const RERANK_BASE_URL = process.env.SKILLSEARCH_RERANK_BASE_URL ?? "";
const RERANK_API_KEY = process.env.SKILLSEARCH_RERANK_API_KEY ?? "";
const RERANK_MODEL = process.env.SKILLSEARCH_RERANK_MODEL ?? "rerank-english-v3.0";
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
  content: string;       // full SKILL.md content
  references: string[];  // absolute paths to files in references/
  scripts: string[];     // absolute paths to files in scripts/
  allExtraFiles: string[]; // references + scripts combined
}

interface DbRow {
  id: string;
  skill_name: string;
  skill_dir: string;
  skill_md_path: string;
  frontmatter: Record<string, string>;
  content: string;
  references: string[];
  scripts: string[];
  all_extra_files: string[];
  similarity: number;
}

// ---------------------------------------------------------------------------
// Database layer
// ---------------------------------------------------------------------------

let pool: PoolType | null = null;

function getPool(): PoolType {
  if (!pool) {
    if (!DB_URL) {
      throw new Error("SKILLSEARCH_DB_URL environment variable is not set");
    }
    pool = new Pool({ connectionString: DB_URL });
  }
  return pool;
}

async function ensureSchema(db: PoolType): Promise<void> {
  await db.query(`
    CREATE TABLE IF NOT EXISTS skills (
      id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      skill_name        TEXT NOT NULL,
      skill_dir         TEXT NOT NULL,
      skill_md_path     TEXT NOT NULL,
      frontmatter       JSONB NOT NULL DEFAULT '{}'::jsonb,
      content           TEXT NOT NULL,
      content_hash      TEXT NOT NULL,
      references        TEXT[] NOT NULL DEFAULT '{}',
      scripts           TEXT[] NOT NULL DEFAULT '{}',
      all_extra_files   TEXT[] NOT NULL DEFAULT '{}',
      created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_skills_skill_name ON skills (skill_name);
    CREATE INDEX IF NOT EXISTS idx_skills_content_hash ON skills (content_hash);
  `);

  // Try to add embedding column; ignore error if it already exists or
  // the vector extension is not available (user must create it manually).
  try {
    await db.query(`CREATE EXTENSION IF NOT EXISTS vector;`);
    const col = await db.query(`
      SELECT 1 FROM information_schema.columns
      WHERE table_name = 'skills' AND column_name = 'embedding';
    `);
    if (col.rowCount === 0) {
      await db.query(`
        ALTER TABLE skills ADD COLUMN embedding vector(1536);
      `);
    }
    await db.query(`
      CREATE INDEX IF NOT EXISTS idx_skills_embedding
      ON skills USING hnsw (embedding vector_cosine_ops);
    `);
  } catch {
    console.warn(
      "[skillsearch] pgvector not available; embedding column/index skipped. " +
      "Run 'CREATE EXTENSION vector;' manually if you want vector search."
    );
  }
}

// ---------------------------------------------------------------------------
// Skill discovery & indexing
// ---------------------------------------------------------------------------

function discoverSkills(rootDir: string): SkillEntry[] {
  const skills: SkillEntry[] = [];

  if (!fs.existsSync(rootDir)) {
    console.warn(`[skillsearch] Skills directory not found: ${rootDir}`);
    return skills;
  }

  const entries = fs.readdirSync(rootDir, { withFileTypes: true });

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;

    const skillDir = path.join(rootDir, entry.name);
    const skillMdPath = path.join(skillDir, "SKILL.md");

    if (!fs.existsSync(skillMdPath)) {
      console.warn(`[skillsearch] No SKILL.md in ${skillDir}, skipping`);
      continue;
    }

    const raw = fs.readFileSync(skillMdPath, "utf-8");
    const { frontmatter, body } = parseFrontmatter(raw);

    const name = frontmatter["name"] ?? entry.name;
    const description = frontmatter["description"] ?? "";

    // Discover references/
    const refDir = path.join(skillDir, "references");
    const references = collectFilesRecursive(refDir);

    // Discover scripts/
    const scriptsDir = path.join(skillDir, "scripts");
    const scripts = collectFilesRecursive(scriptsDir);

    skills.push({
      name,
      description,
      frontmatter,
      skillMdPath: path.resolve(skillMdPath),
      skillDir: path.resolve(skillDir),
      content: raw,
      references: references.map((f) => path.resolve(f)),
      scripts: scripts.map((f) => path.resolve(f)),
      allExtraFiles: [...references, ...scripts].map((f) => path.resolve(f)),
    });
  }

  return skills;
}

function parseFrontmatter(raw: string): {
  frontmatter: Record<string, string>;
  body: string;
} {
  const frontmatter: Record<string, string> = {};
  let body = raw;

  // Support YAML frontmatter between --- delimiters
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (match) {
    const yamlBlock = match[1];
    body = match[2];
    for (const line of yamlBlock.split(/\r?\n/)) {
      const kv = line.match(/^([a-zA-Z0-9_-]+)\s*:\s*(.*)$/);
      if (kv) {
        frontmatter[kv[1]] = kv[2].trim();
      }
    }
  }

  return { frontmatter, body };
}

function collectFilesRecursive(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  const results: string[] = [];
  const walk = (d: string) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) walk(full);
      else results.push(full);
    }
  };
  walk(dir);
  return results;
}

function contentHash(content: string): string {
  return crypto.createHash("sha256").update(content).digest("hex");
}

// ---------------------------------------------------------------------------
// Indexing: upsert skills into DB, generate embeddings via rerank provider
// ---------------------------------------------------------------------------

async function indexSkills(db: PoolType, skills: SkillEntry[]): Promise<number> {
  let indexed = 0;

  for (const skill of skills) {
    const hash = contentHash(skill.content);

    // Check if already up-to-date
    const existing = await db.query(
      `SELECT content_hash FROM skills WHERE skill_name = $1 AND skill_dir = $2`,
      [skill.name, skill.skillDir]
    );

    if (existing.rowCount && existing.rows[0].content_hash === hash) {
      continue; // unchanged, skip
    }

    // Generate embedding if rerank provider is configured
    let embedding: number[] | null = null;
    if (RERANK_BASE_URL && RERANK_API_KEY) {
      try {
        embedding = await generateEmbedding(skill.content);
      } catch (err) {
        console.warn(
          `[skillsearch] Embedding failed for "${skill.name}": ${err}`
        );
      }
    }

    // Upsert
    if (embedding) {
      await db.query(
        `INSERT INTO skills
           (skill_name, skill_dir, skill_md_path, frontmatter, content, content_hash,
            references, scripts, all_extra_files, embedding, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::vector,NOW())
         ON CONFLICT (skill_name) DO UPDATE SET
           skill_dir       = EXCLUDED.skill_dir,
           skill_md_path   = EXCLUDED.skill_md_path,
           frontmatter     = EXCLUDED.frontmatter,
           content         = EXCLUDED.content,
           content_hash    = EXCLUDED.content_hash,
           references      = EXCLUDED.references,
           scripts         = EXCLUDED.scripts,
           all_extra_files = EXCLUDED.all_extra_files,
           embedding       = EXCLUDED.embedding,
           updated_at      = NOW()`,
        [
          skill.name,
          skill.skillDir,
          skill.skillMdPath,
          JSON.stringify(skill.frontmatter),
          skill.content,
          hash,
          skill.references,
          skill.scripts,
          skill.allExtraFiles,
          `[${embedding.join(",")}]`,
        ]
      );
    } else {
      await db.query(
        `INSERT INTO skills
           (skill_name, skill_dir, skill_md_path, frontmatter, content, content_hash,
            references, scripts, all_extra_files, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NOW())
         ON CONFLICT (skill_name) DO UPDATE SET
           skill_dir       = EXCLUDED.skill_dir,
           skill_md_path   = EXCLUDED.skill_md_path,
           frontmatter     = EXCLUDED.frontmatter,
           content         = EXCLUDED.content,
           content_hash    = EXCLUDED.content_hash,
           references      = EXCLUDED.references,
           scripts         = EXCLUDED.scripts,
           all_extra_files = EXCLUDED.all_extra_files,
           updated_at      = NOW()`,
        [
          skill.name,
          skill.skillDir,
          skill.skillMdPath,
          JSON.stringify(skill.frontmatter),
          skill.content,
          hash,
          skill.references,
          skill.scripts,
          skill.allExtraFiles,
        ]
      );
    }

    indexed++;
  }

  return indexed;
}

// ---------------------------------------------------------------------------
// Embedding generation via OpenAI-compatible /embeddings endpoint
// ---------------------------------------------------------------------------

async function generateEmbedding(text: string): Promise<number[]> {
  if (!RERANK_BASE_URL || !RERANK_API_KEY) {
    throw new Error("Rerank/embedding provider not configured");
  }

  // Use the rerank base URL's /embeddings endpoint
  const url = `${RERANK_BASE_URL.replace(/\/+$/, "")}/embeddings`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${RERANK_API_KEY}`,
    },
    body: JSON.stringify({
      model: RERANK_MODEL,
      input: text,
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Embedding API error ${res.status}: ${body}`);
  }

  const data = (await res.json()) as any;
  return data.data[0].embedding;
}

// ---------------------------------------------------------------------------
// Search: vector similarity + optional rerank
// ---------------------------------------------------------------------------

async function searchSkills(
  db: PoolType,
  query: string,
  limit: number,
  threshold: number
): Promise<DbRow[]> {
  // Check if embedding column exists
  const hasEmbedding = await db.query(`
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'skills' AND column_name = 'embedding'
  `);

  let rows: DbRow[];

  if (hasEmbedding.rowCount && hasEmbedding.rowCount > 0 && RERANK_BASE_URL && RERANK_API_KEY) {
    // Vector search
    const queryEmbedding = await generateEmbedding(query);
    const vecStr = `[${queryEmbedding.join(",")}]`;

    const result = await db.query(
      `SELECT id, skill_name, skill_dir, skill_md_path, frontmatter, content,
              references, scripts, all_extra_files,
              1 - (embedding <=> $1::vector) AS similarity
       FROM skills
       WHERE embedding IS NOT NULL
         AND 1 - (embedding <=> $1::vector) > $2
       ORDER BY embedding <=> $1::vector
       LIMIT $3`,
      [vecStr, threshold, limit * 3] // fetch extra for rerank
    );

    rows = result.rows.map((r: any) => ({
      id: r.id,
      skill_name: r.skill_name,
      skill_dir: r.skill_dir,
      skill_md_path: r.skill_md_path,
      frontmatter: r.frontmatter,
      content: r.content,
      references: r.references ?? [],
      scripts: r.scripts ?? [],
      all_extra_files: r.all_extra_files ?? [],
      similarity: parseFloat(r.similarity),
    }));
  } else {
    // Fallback: full-text search on content
    const result = await db.query(
      `SELECT id, skill_name, skill_dir, skill_md_path, frontmatter, content,
              references, scripts, all_extra_files,
              ts_rank(
                to_tsvector('english', content || ' ' || skill_name || ' ' || COALESCE(frontmatter->>'description','')),
                plainto_tsquery('english', $1)
              ) AS similarity
       FROM skills
       WHERE to_tsvector('english', content || ' ' || skill_name || ' ' || COALESCE(frontmatter->>'description',''))
             @@ plainto_tsquery('english', $1)
       ORDER BY similarity DESC
       LIMIT $2`,
      [query, limit * 3]
    );

    rows = result.rows.map((r: any) => ({
      id: r.id,
      skill_name: r.skill_name,
      skill_dir: r.skill_dir,
      skill_md_path: r.skill_md_path,
      frontmatter: r.frontmatter,
      content: r.content,
      references: r.references ?? [],
      scripts: r.scripts ?? [],
      all_extra_files: r.all_extra_files ?? [],
      similarity: parseFloat(r.similarity),
    }));
  }

  // Rerank if provider is available
  if (RERANK_BASE_URL && RERANK_API_KEY && rows.length > 0) {
    try {
      rows = await rerankResults(query, rows, limit);
    } catch (err) {
      console.warn(`[skillsearch] Rerank failed, using vector scores: ${err}`);
      rows = rows.slice(0, limit);
    }
  } else {
    rows = rows.slice(0, limit);
  }

  return rows;
}

// ---------------------------------------------------------------------------
// Rerank via OpenAI-compatible /rerank or /reranking endpoint
// ---------------------------------------------------------------------------

async function rerankResults(
  query: string,
  candidates: DbRow[],
  limit: number
): Promise<DbRow[]> {
  // Try common rerank endpoints
  const endpoints = [
    "/rerank",
    "/reranking",
  ];

  const documents = candidates.map((c) =>
    `${c.skill_name}: ${c.frontmatter["description"] ?? ""}`.trim()
  );

  for (const ep of endpoints) {
    const url = `${RERANK_BASE_URL.replace(/\/+$/, "")}${ep}`;
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
        console.warn(`[skillsearch] Rerank endpoint ${ep} returned ${res.status}: ${body}`);
        continue;
      }

      const data = (await res.json()) as any;

      // Handle different response formats
      // Cohere-style: { results: [{ index, relevance_score }] }
      // Jina-style: { results: [{ index, relevance_score }] }
      // OpenAI-style: { data: [{ index, score }] }
      const scored: Array<{ index: number; score: number }> = [];

      if (data.results) {
        for (const r of data.results) {
          scored.push({
            index: r.index,
            score: r.relevance_score ?? r.score ?? 0,
          });
        }
      } else if (data.data) {
        for (const r of data.data) {
          scored.push({
            index: r.index,
            score: r.score ?? r.relevance_score ?? 0,
          });
        }
      }

      scored.sort((a, b) => b.score - a.score);

      const reranked: DbRow[] = [];
      for (const s of scored) {
        if (s.index >= 0 && s.index < candidates.length) {
          reranked.push({
            ...candidates[s.index],
            similarity: s.score,
          });
        }
      }

      return reranked.slice(0, limit);
    } catch (err) {
      console.warn(`[skillsearch] Rerank endpoint ${ep} error: ${err}`);
      continue;
    }
  }

  // All endpoints failed, return top by original similarity
  return candidates.slice(0, limit);
}

// ---------------------------------------------------------------------------
// MCP Server setup
// ---------------------------------------------------------------------------

function createServer(): Server {
  const server = new Server(
    {
      name: "skillsearch-mcp-server",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // ---- ListTools ----
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const tools: Tool[] = [
      {
        name: "skill_search",
        description:
          "Search for Agent Skills by semantic query. Returns matching skills with their frontmatter metadata, SKILL.md path, and paths to reference/script files. Use this to discover available skills before reading their full content.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description:
                "Natural language search query describing the skill you need. Be specific about the task or domain.",
            },
            limit: {
              type: "integer",
              description: "Maximum number of results to return (default: 5).",
              default: 5,
            },
            threshold: {
              type: "number",
              description:
                "Minimum similarity threshold (0.0 to 1.0). Higher values return fewer but more relevant results. Default: 0.3",
              default: 0.3,
            },
          },
          required: ["query"],
        },
      },
      {
        name: "skill_list",
        description:
          "List all indexed skills with their names, descriptions, and directory paths.",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "skill_reindex",
        description:
          "Re-scan the skills directory and re-index all skills into the database. Use this after adding or modifying skills.",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
    ];

    return { tools };
  });

  // ---- CallTool ----
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const db = getPool();

    switch (request.params.name) {
      case "skill_search": {
        const args = (request.params.arguments ?? {}) as Record<string, unknown>;
        const query = String(args.query ?? "");
        const limit = Math.min(Math.max(Number(args.limit ?? 5), 1), 50);
        const threshold = Math.min(Math.max(Number(args.threshold ?? 0.3), 0), 1);

        if (!query) {
          return {
            content: [{ type: "text" as const, text: "Error: 'query' is required" }],
            isError: true,
          };
        }

        const results = await searchSkills(db, query, limit, threshold);

        if (results.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: JSON.stringify(
                  {
                    query,
                    threshold,
                    totalResults: 0,
                    message: "No matching skills found. Try a different query or lower the threshold.",
                    skills: [],
                  },
                  null,
                  2
                ),
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
            references: r.references,
            scripts: r.scripts,
            allExtraFiles: r.all_extra_files,
          })),
        };

        return {
          content: [{ type: "text" as const, text: JSON.stringify(response, null, 2) }],
        };
      }

      case "skill_list": {
        const result = await db.query(
          `SELECT skill_name, skill_dir, skill_md_path, frontmatter
           FROM skills
           ORDER BY skill_name`
        );

        const skills = result.rows.map((r: any) => ({
          name: r.skill_name,
          description: r.frontmatter?.description ?? "",
          skillDir: r.skill_dir,
          skillMdPath: r.skill_md_path,
        }));

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({ total: skills.length, skills }, null, 2),
            },
          ],
        };
      }

      case "skill_reindex": {
        if (!SKILLS_DIR) {
          return {
            content: [
              {
                type: "text" as const,
                text: "Error: SKILLSEARCH_SKILLS_DIR environment variable is not set",
              },
            ],
            isError: true,
          };
        }

        const skills = discoverSkills(SKILLS_DIR);
        const indexed = await indexSkills(db, skills);

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  message: "Reindex complete",
                  skillsFound: skills.length,
                  skillsIndexed: indexed,
                  skillsDir: path.resolve(SKILLS_DIR),
                },
                null,
                2
              ),
            },
          ],
        };
      }

      default:
        return {
          content: [{ type: "text" as const, text: `Unknown tool: ${request.params.name}` }],
          isError: true,
        };
    }
  });

  return server;
}

// ---------------------------------------------------------------------------
// Main: discover → index → start MCP server
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.error("[skillsearch] Starting SkillSearch MCP Server...");

  // 1. Validate config
  if (!DB_URL) {
    console.error("[skillsearch] FATAL: SKILLSEARCH_DB_URL is not set");
    process.exit(1);
  }
  if (!SKILLS_DIR) {
    console.error("[skillsearch] FATAL: SKILLSEARCH_SKILLS_DIR is not set");
    process.exit(1);
  }

  // 2. Connect to DB
  const db = getPool();
  try {
    await db.query("SELECT 1");
    console.error("[skillsearch] Database connected");
  } catch (err) {
    console.error(`[skillsearch] FATAL: Cannot connect to database: ${err}`);
    process.exit(1);
  }

  // 3. Ensure schema
  await ensureSchema(db);
  console.error("[skillsearch] Database schema ensured");

  // 4. Discover & index skills
  const skills = discoverSkills(SKILLS_DIR);
  console.error(`[skillsearch] Discovered ${skills.length} skills in ${SKILLS_DIR}`);

  if (skills.length > 0) {
    const indexed = await indexSkills(db, skills);
    console.error(`[skillsearch] Indexed ${indexed} skills (${skills.length - indexed} unchanged)`);
  }

  // 5. Start MCP server (stdio transport)
  const server = createServer();
  const transport = new StdioServerTransport();

  await server.connect(transport);
  console.error("[skillsearch] MCP server running on stdio");
}

main().catch((err) => {
  console.error(`[skillsearch] Fatal error: ${err}`);
  process.exit(1);
});
