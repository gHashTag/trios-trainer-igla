import type { Express, Request, Response } from "express";
import type { Server } from "node:http";
import { z } from "zod";
import { db } from "./storage";
import { bpbSamples, runs, insertBpbSampleSchema } from "@shared/schema";
import { sql, eq } from "drizzle-orm";
import { bootstrap } from "./seed";

// Ingest token guard.  Set INGEST_TOKEN on Railway; scarab/cron sends it as
// `Authorization: Bearer <token>` or `X-Ingest-Token: <token>`.
function requireIngestToken(req: Request, res: Response): boolean {
  const expected = process.env.INGEST_TOKEN;
  if (!expected) {
    res.status(503).json({ error: "INGEST_TOKEN not configured on server" });
    return false;
  }
  const auth = req.header("authorization") ?? "";
  const bearer = auth.startsWith("Bearer ") ? auth.slice(7) : null;
  const headerToken = req.header("x-ingest-token");
  const provided = bearer ?? headerToken ?? "";
  if (provided !== expected) {
    res.status(401).json({ error: "invalid ingest token" });
    return false;
  }
  return true;
}

const ingestSampleSchema = insertBpbSampleSchema.extend({
  // Loosen ts: accept omission, ms, or ISO; coerce to unix seconds in handler.
  ts: z.union([z.number(), z.string()]).optional(),
});

export async function registerRoutes(httpServer: Server, app: Express): Promise<Server> {
  // Boot data on first start (idempotent).
  bootstrap();

  // ---- Health -----------------------------------------------------------
  app.get("/api/health", (_req, res) => {
    const sampleCount = (db.select({ c: sql<number>`count(*)` }).from(bpbSamples).get() as any)?.c ?? 0;
    const runCount = (db.select({ c: sql<number>`count(*)` }).from(runs).get() as any)?.c ?? 0;
    res.json({
      ok: true,
      anchor: "phi^2 + phi^-2 = 3",
      doi: "10.5281/zenodo.19227877",
      counts: { bpb_samples: sampleCount, runs: runCount },
      ingest_token_configured: Boolean(process.env.INGEST_TOKEN),
    });
  });

  // ---- Champion KPI -----------------------------------------------------
  app.get("/api/champion", (_req, res) => {
    const row = db
      .select()
      .from(bpbSamples)
      .orderBy(sql`bpb ASC`)
      .limit(1)
      .get();
    if (!row) {
      res.status(404).json({ error: "no samples yet" });
      return;
    }
    const gateTarget = 1.85;
    res.json({
      bpb: row.bpb,
      format: row.format,
      algo: row.algo,
      seed: row.seed,
      hidden: row.hidden,
      lr: row.lr,
      step: row.step,
      sha: row.sha,
      gateTarget,
      gateMargin: row.bpb - gateTarget,
      gateStatus: row.gateStatus,
      ts: row.ts,
    });
  });

  // ---- Leaderboard (best BPB per (format,algo,seed)) --------------------
  app.get("/api/leaderboard", (req, res) => {
    const limit = Math.max(1, Math.min(100, parseInt(String(req.query.limit ?? "20"), 10)));
    const rows = db.all<any>(sql`
      SELECT format, algo, seed, hidden, lr,
             MIN(bpb) AS best_bpb,
             MAX(step) AS step,
             MAX(ts)   AS ts,
             MAX(sha)  AS sha,
             MAX(gate_status) AS gate_status
      FROM bpb_samples
      GROUP BY format, algo, seed, hidden, lr
      ORDER BY best_bpb ASC
      LIMIT ${limit}
    `);
    res.json(
      rows.map((r: any, i: number) => ({
        rank: i + 1,
        format: r.format,
        algo: r.algo,
        seed: r.seed,
        hidden: r.hidden,
        lr: r.lr,
        bestBpb: r.best_bpb,
        step: r.step,
        sha: r.sha,
        gateStatus: r.gate_status,
        ts: r.ts,
      })),
    );
  });

  // ---- Format × Algo matrix ---------------------------------------------
  app.get("/api/matrix", (_req, res) => {
    const rows = db.all<any>(sql`
      SELECT format, algo,
             MIN(bpb)        AS best_bpb,
             AVG(ema_bpb)    AS ema_bpb,
             COUNT(*)        AS sample_count,
             MAX(ts)         AS last_ts,
             MAX(seed)       AS last_seed
      FROM bpb_samples
      GROUP BY format, algo
    `);
    res.json(rows.map((r: any) => ({
      format: r.format,
      algo: r.algo,
      bestBpb: r.best_bpb,
      emaBpb: r.ema_bpb,
      sampleCount: r.sample_count,
      lastTs: r.last_ts,
      lastSeed: r.last_seed,
    })));
  });

  // ---- Runs (queue listing for Drizzle Studio companion) ----------------
  app.get("/api/runs", (req, res) => {
    const status = req.query.status as string | undefined;
    const q = status
      ? db.select().from(runs).where(eq(runs.status, status))
      : db.select().from(runs);
    res.json(q.all());
  });

  // ---- Ingest (scarab/cron writes BPB samples here) ---------------------
  app.post("/api/ingest", (req, res) => {
    if (!requireIngestToken(req, res)) return;

    const parsed = ingestSampleSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "invalid payload", issues: parsed.error.issues });
      return;
    }
    const sample = parsed.data;

    let ts: number;
    if (sample.ts === undefined) {
      ts = Math.floor(Date.now() / 1000);
    } else if (typeof sample.ts === "number") {
      ts = sample.ts > 1e12 ? Math.floor(sample.ts / 1000) : sample.ts;
    } else {
      const parsedDate = Date.parse(sample.ts);
      ts = Number.isNaN(parsedDate) ? Math.floor(Date.now() / 1000) : Math.floor(parsedDate / 1000);
    }

    const inserted = db.insert(bpbSamples).values({
      runId: sample.runId ?? null,
      queueId: sample.queueId ?? null,
      format: sample.format,
      algo: sample.algo,
      seed: sample.seed,
      hidden: sample.hidden,
      lr: sample.lr,
      step: sample.step,
      bpb: sample.bpb,
      emaBpb: sample.emaBpb ?? null,
      sha: sample.sha ?? null,
      gateStatus: sample.gateStatus ?? null,
      ts,
    }).returning().get();

    res.status(201).json({ ok: true, id: inserted.id, ts });
  });

  // Bulk ingest — scarab posts an array of samples in one shot.
  app.post("/api/ingest/batch", (req, res) => {
    if (!requireIngestToken(req, res)) return;
    const arr = z.array(ingestSampleSchema).safeParse(req.body);
    if (!arr.success) {
      res.status(400).json({ error: "expected array of samples", issues: arr.error.issues });
      return;
    }
    const now = Math.floor(Date.now() / 1000);
    const values = arr.data.map((s) => {
      let ts: number;
      if (s.ts === undefined) ts = now;
      else if (typeof s.ts === "number") ts = s.ts > 1e12 ? Math.floor(s.ts / 1000) : s.ts;
      else {
        const p = Date.parse(s.ts);
        ts = Number.isNaN(p) ? now : Math.floor(p / 1000);
      }
      return {
        runId: s.runId ?? null,
        queueId: s.queueId ?? null,
        format: s.format,
        algo: s.algo,
        seed: s.seed,
        hidden: s.hidden,
        lr: s.lr,
        step: s.step,
        bpb: s.bpb,
        emaBpb: s.emaBpb ?? null,
        sha: s.sha ?? null,
        gateStatus: s.gateStatus ?? null,
        ts,
      };
    });
    if (values.length === 0) {
      res.json({ ok: true, inserted: 0 });
      return;
    }
    db.insert(bpbSamples).values(values).run();
    res.status(201).json({ ok: true, inserted: values.length });
  });

  // ---- Root: redirect to Drizzle Studio ---------------------------------
  app.get("/", (_req, res) => {
    res.type("html").send(`<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>IGLA RACE · format×algo dashboard</title>
<style>
  body{margin:0;background:#0b0d10;color:#e6e7e9;font:14px ui-monospace,Menlo,monospace;
       padding:48px;max-width:760px}
  h1{font-size:18px;margin:0 0 24px;color:#f4d03f}
  code{background:#1c1f24;padding:2px 6px;border-radius:4px;color:#f4d03f}
  a{color:#5dade2}
  .row{margin:12px 0}
  .anchor{margin-top:32px;color:#888;font-size:12px}
</style>
</head><body>
<h1>IGLA RACE — format × algo dashboard</h1>
<div class="row">UI is provided by <strong>Drizzle Studio</strong>. Run locally:</div>
<div class="row"><code>npm run db:studio</code> → opens <a href="https://local.drizzle.studio">https://local.drizzle.studio</a></div>
<div class="row">REST API for scarab/cron:</div>
<ul>
  <li><a href="/api/health">GET /api/health</a></li>
  <li><a href="/api/champion">GET /api/champion</a></li>
  <li><a href="/api/leaderboard">GET /api/leaderboard</a></li>
  <li><a href="/api/matrix">GET /api/matrix</a></li>
  <li><a href="/api/runs">GET /api/runs</a></li>
  <li>POST /api/ingest        — single sample (Bearer INGEST_TOKEN)</li>
  <li>POST /api/ingest/batch  — array of samples</li>
</ul>
<div class="anchor">φ² + φ⁻² = 3 · DOI 10.5281/zenodo.19227877</div>
</body></html>`);
  });

  return httpServer;
}
