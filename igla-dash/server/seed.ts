import { db } from "./storage";
import { formats, algos, bpbSamples, runs } from "@shared/schema";
import { sql } from "drizzle-orm";

// Numerical formats live in the IGLA race grid.
// Family: ieee | brain | golden | posit | integer | fp8.
const FORMAT_REGISTRY: { name: string; family: string; bits: number; notes?: string }[] = [
  // IEEE family — canonical short aliases used in the trios#446 PhD matrix.
  { name: "fp64", family: "ieee", bits: 64, notes: "IEEE-754 double" },
  { name: "fp32", family: "ieee", bits: 32, notes: "IEEE-754 single (alias binary32)" },
  { name: "binary32", family: "ieee", bits: 32, notes: "IEEE-754 single (alias fp32)" },
  { name: "fp16", family: "ieee", bits: 16, notes: "IEEE-754 half (alias binary16)" },
  { name: "binary16", family: "ieee", bits: 16, notes: "IEEE-754 half (alias fp16)" },
  { name: "bf16", family: "brain", bits: 16, notes: "Google Brain float (alias bfloat16)" },
  { name: "bfloat16", family: "brain", bits: 16, notes: "Google Brain float (alias bf16)" },
  { name: "tf32", family: "ieee", bits: 19, notes: "NVIDIA TensorFloat-32" },
  // Golden-float family.
  { name: "gf64", family: "golden", bits: 64, notes: "Golden-float, phi^2+phi^-2=3" },
  { name: "gf32", family: "golden", bits: 32 },
  { name: "gf24", family: "golden", bits: 24 },
  { name: "gf20", family: "golden", bits: 20 },
  { name: "gf16", family: "golden", bits: 16 },
  { name: "gf12", family: "golden", bits: 12 },
  { name: "gf8", family: "golden", bits: 8 },
  { name: "gf4", family: "golden", bits: 4 },
  // FP8 OCP.
  { name: "fp8_e4m3", family: "fp8", bits: 8, notes: "OCP fp8 E4M3" },
  { name: "fp8_e5m2", family: "fp8", bits: 8, notes: "OCP fp8 E5M2" },
  // Integer.
  { name: "int32", family: "integer", bits: 32 },
  { name: "int16", family: "integer", bits: 16 },
  { name: "int8", family: "integer", bits: 8 },
  { name: "uint8", family: "integer", bits: 8 },
];

const ALGO_REGISTRY: { name: string; family: string; notes?: string }[] = [
  { name: "adamw", family: "first_order", notes: "Champion baseline" },
  { name: "muon", family: "second_order", notes: "Newton-Schulz orthogonalised" },
  { name: "sgd", family: "first_order" },
  { name: "lion", family: "first_order" },
  { name: "lamb", family: "first_order" },
  { name: "soap", family: "second_order" },
  { name: "adafactor", family: "first_order" },
  { name: "schedulefree", family: "schedulefree" },
];

// Champion seed_results.jsonl (gHashTag/trios-trainer-igla).
// Two rows + Wave-8-HONEST manifest (40 jobs, queue 19327..19366) — synthetic placeholder
// telemetry for first boot only.  Real samples arrive via /api/ingest.
const CHAMPION_ROWS = [
  {
    seed: 43, bpb: 2.1919, step: 81000, sha: "cd91c45",
    format: "binary32", algo: "adamw", hidden: 828, lr: 0.003,
    gateStatus: "champion",
  },
  {
    seed: 44, bpb: 2.2024, step: 81000, sha: "cd91c45",
    format: "binary32", algo: "adamw", hidden: 828, lr: 0.003,
    gateStatus: "above_target",
  },
];

const WAVE8_FORMATS = ["binary32", "binary16", "bfloat16", "gf16"];
const WAVE8_ALGOS = ["adamw", "muon"];
const WAVE8_SEEDS = [1597, 2584, 4181, 6765, 10946];
const WAVE8_IMAGE_SHA = "sha256:ecce23e9e72e61c662cfa7a149292087ccd3c1d7d5be24615bae0175700d5832";

// Wave-9 ASYMLOGIT-NGRAM port — 19 formats x 2 algos x 5 seeds = 190 jobs,
// queue_id range [19367..19556].  Image pin TBD until L-AL1 lands and CI tags ghcr.
// Stack delta vs Wave-8 = AsymLogit Rescale (init 30.0) + Token-Only N-gram tilt
// (TOKEN_ORDER=8, THRESHOLD=0.80, BOOST=2.625).  Within/word/agree channels are
// hard-disabled per pre-registration (R7 forbidden values).
//
// === GATING ===
// Wave-9 dispatch is GATED on gHashTag/trios#509 P0 (cpu_train.rs::forward() does
// not call quantize/gf16_encode/from_f32 — every format degenerates to f32).  Until
// trios#515 is merged AND a quantize-hook lands in forward(), running this grid
// would produce 190 noise-floor curves and constitute an R5 violation.
// Default: WAVE9_GATED_ON_509=false (jobs NOT inserted).  Set to true ONLY after
// the unblock conditions on the gated manifest are observed in trios main.
const WAVE9_GATED_ON_509 = (process.env.WAVE9_GATED_ON_509 ?? "false").toLowerCase() === "true";
const WAVE9_FORMATS = [
  "fp32", "fp64", "fp16", "bf16", "tf32",
  "gf64", "gf32", "gf24", "gf20", "gf16", "gf12", "gf8", "gf4",
  "fp8_e4m3", "fp8_e5m2",
  "int32", "int16", "int8", "uint8",
];
const WAVE9_ALGOS = ["adamw", "muon"];
const WAVE9_SEEDS = [1597, 2584, 4181, 6765, 10946];
const WAVE9_QUEUE_START = 19367;
const WAVE9_QUEUE_END = 19556;
const WAVE9_IMAGE_SHA = "sha256:0000000000000000000000000000000000000000000000000000000000000000"; // placeholder, replaced when ONE SHOT image lands
const WAVE9_STACK_TAG = "asymlogit_ngram_v1";

export function ensureSchema() {
  // Tables created via drizzle-kit push.  Idempotent CREATE for first boot when
  // running under Railway without a migration step.
  db.run(sql`CREATE TABLE IF NOT EXISTS formats (
    name TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    bits INTEGER NOT NULL,
    notes TEXT
  )`);
  db.run(sql`CREATE TABLE IF NOT EXISTS algos (
    name TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    notes TEXT
  )`);
  db.run(sql`CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    queue_id TEXT NOT NULL UNIQUE,
    canon_name TEXT NOT NULL,
    format TEXT NOT NULL,
    algo TEXT NOT NULL,
    seed INTEGER NOT NULL,
    hidden INTEGER NOT NULL,
    lr REAL NOT NULL,
    status TEXT NOT NULL,
    image_sha TEXT,
    started_at INTEGER,
    finished_at INTEGER
  )`);
  db.run(sql`CREATE TABLE IF NOT EXISTS bpb_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    queue_id TEXT,
    format TEXT NOT NULL,
    algo TEXT NOT NULL,
    seed INTEGER NOT NULL,
    hidden INTEGER NOT NULL,
    lr REAL NOT NULL,
    step INTEGER NOT NULL,
    bpb REAL NOT NULL,
    ema_bpb REAL,
    sha TEXT,
    gate_status TEXT,
    ts INTEGER NOT NULL
  )`);
  db.run(sql`CREATE INDEX IF NOT EXISTS idx_bpb_format_algo ON bpb_samples (format, algo)`);
  db.run(sql`CREATE INDEX IF NOT EXISTS idx_bpb_ts ON bpb_samples (ts)`);
}

export function seedRegistry() {
  const fmtCount = (db.select({ c: sql<number>`count(*)` }).from(formats).get() as any)?.c ?? 0;
  if (fmtCount === 0) {
    db.insert(formats).values(FORMAT_REGISTRY).run();
  }
  const algoCount = (db.select({ c: sql<number>`count(*)` }).from(algos).get() as any)?.c ?? 0;
  if (algoCount === 0) {
    db.insert(algos).values(ALGO_REGISTRY).run();
  }
}

export function seedChampion() {
  const sampleCount = (db.select({ c: sql<number>`count(*)` }).from(bpbSamples).get() as any)?.c ?? 0;
  if (sampleCount > 0) return;

  const now = Math.floor(Date.now() / 1000);
  const rows: any[] = [];

  for (const r of CHAMPION_ROWS) {
    rows.push({
      runId: null,
      queueId: `champion-${r.seed}`,
      format: r.format,
      algo: r.algo,
      seed: r.seed,
      hidden: r.hidden,
      lr: r.lr,
      step: r.step,
      bpb: r.bpb,
      emaBpb: r.bpb,
      sha: r.sha,
      gateStatus: r.gateStatus,
      ts: now - 86400, // 1 day ago
    });
  }
  db.insert(bpbSamples).values(rows).run();

  // Wave-8-HONEST 40-job manifest as queued runs, image pinned.
  const launchTs = now - 3600 * 4; // 4h ago
  let queue = 19327;
  for (const fmt of WAVE8_FORMATS) {
    for (const algo of WAVE8_ALGOS) {
      for (const seed of WAVE8_SEEDS) {
        const queueId = String(queue++);
        db.insert(runs).values({
          queueId,
          canonName: `wave8-${fmt}-${algo}-${seed}`,
          format: fmt,
          algo,
          seed,
          hidden: 828,
          lr: 0.003,
          status: "running",
          imageSha: WAVE8_IMAGE_SHA,
          startedAt: launchTs,
          finishedAt: null,
        }).run();
      }
    }
  }

  // Wave-9 ASYMLOGIT-NGRAM port — 190 queued jobs, status "queued" until
  // dispatcher picks them up.  Idempotent: skip if Wave-9 head already present.
  // Hard-gated on trios#509 — see WAVE9_GATED_ON_509 above.  When OFF (default)
  // we never insert Wave-9 rows; manifest in manifests/wave9-asymlogit-ngram.GATED-ON-509.json
  // is documentation-only.
  if (!WAVE9_GATED_ON_509) {
    return;
  }
  const w9Head = db.select().from(runs).where(sql`queue_id = ${String(WAVE9_QUEUE_START)}`).all();
  if (w9Head.length === 0) {
    let q9 = WAVE9_QUEUE_START;
    for (const fmt of WAVE9_FORMATS) {
      for (const algo of WAVE9_ALGOS) {
        for (const seed of WAVE9_SEEDS) {
          const queueId = String(q9++);
          db.insert(runs).values({
            queueId,
            canonName: `wave9-${WAVE9_STACK_TAG}-${fmt}-${algo}-${seed}`,
            format: fmt,
            algo,
            seed,
            hidden: 828,
            lr: 0.003,
            status: "queued",
            imageSha: WAVE9_IMAGE_SHA,
            startedAt: null,
            finishedAt: null,
          }).run();
        }
      }
    }
    if (q9 - 1 !== WAVE9_QUEUE_END) {
      throw new Error(`Wave-9 queue range mismatch: ended at ${q9 - 1}, expected ${WAVE9_QUEUE_END}`);
    }
  }
}

export function bootstrap() {
  ensureSchema();
  seedRegistry();
  seedChampion();
}
