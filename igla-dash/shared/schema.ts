import { sqliteTable, text, integer, real } from "drizzle-orm/sqlite-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// IGLA RACE format×algo dashboard schema.
// Anchor: phi^2 + phi^-2 = 3 · DOI 10.5281/zenodo.19227877.

// Numerical format registry (binary32, bfloat16, gf16, fp8_e4m3, ...).
export const formats = sqliteTable("formats", {
  name: text("name").primaryKey(),
  family: text("family").notNull(), // ieee, brain, golden, posit, integer
  bits: integer("bits").notNull(),
  notes: text("notes"),
});

// Optimizer/algorithm registry (adamw, muon, sgd, lion, ...).
export const algos = sqliteTable("algos", {
  name: text("name").primaryKey(),
  family: text("family").notNull(), // first_order, second_order, schedulefree
  notes: text("notes"),
});

// Run = a single training job (image+seed+format+algo+hidden+lr).
export const runs = sqliteTable("runs", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  queueId: text("queue_id").notNull().unique(),
  canonName: text("canon_name").notNull(),
  format: text("format").notNull(),
  algo: text("algo").notNull(),
  seed: integer("seed").notNull(),
  hidden: integer("hidden").notNull(),
  lr: real("lr").notNull(),
  status: text("status").notNull(), // queued, running, done, failed
  imageSha: text("image_sha"),
  startedAt: integer("started_at"), // unix seconds
  finishedAt: integer("finished_at"),
});

// Per-step BPB telemetry sample.  Written by trainer-igla / scarab-cron via /api/ingest.
export const bpbSamples = sqliteTable("bpb_samples", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  runId: integer("run_id"),
  queueId: text("queue_id"),
  format: text("format").notNull(),
  algo: text("algo").notNull(),
  seed: integer("seed").notNull(),
  hidden: integer("hidden").notNull(),
  lr: real("lr").notNull(),
  step: integer("step").notNull(),
  bpb: real("bpb").notNull(),
  emaBpb: real("ema_bpb"),
  sha: text("sha"),
  gateStatus: text("gate_status"), // above_target | below_target | champion
  ts: integer("ts").notNull(), // unix seconds
});

export const insertFormatSchema = createInsertSchema(formats);
export const insertAlgoSchema = createInsertSchema(algos);
export const insertRunSchema = createInsertSchema(runs).omit({ id: true });
export const insertBpbSampleSchema = createInsertSchema(bpbSamples).omit({ id: true });

export type Format = typeof formats.$inferSelect;
export type Algo = typeof algos.$inferSelect;
export type Run = typeof runs.$inferSelect;
export type BpbSample = typeof bpbSamples.$inferSelect;

export type InsertFormat = z.infer<typeof insertFormatSchema>;
export type InsertAlgo = z.infer<typeof insertAlgoSchema>;
export type InsertRun = z.infer<typeof insertRunSchema>;
export type InsertBpbSample = z.infer<typeof insertBpbSampleSchema>;

// Aggregated payloads served by the dashboard API.
export type MatrixCell = {
  format: string;
  algo: string;
  bestBpb: number | null;
  emaBpb: number | null;
  sampleCount: number;
  lastTs: number | null;
  lastSeed: number | null;
};

export type LeaderboardRow = {
  rank: number;
  format: string;
  algo: string;
  seed: number;
  hidden: number;
  lr: number;
  bestBpb: number;
  step: number;
  sha: string | null;
  gateStatus: string | null;
  ts: number;
};

export type ChampionPayload = {
  bpb: number;
  format: string;
  algo: string;
  seed: number;
  hidden: number;
  lr: number;
  step: number;
  sha: string | null;
  gateTarget: number;
  gateMargin: number;
  ts: number;
};

export type RunSummary = Run & {
  bestBpb: number | null;
  lastBpb: number | null;
  lastStep: number | null;
};
