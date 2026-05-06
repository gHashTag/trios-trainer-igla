# IGLA RACE — format × algo dashboard

Backend: Express + Drizzle ORM + better-sqlite3.
UI: **[rainfrog](https://github.com/achristmascarl/rainfrog)** — Rust TUI (ratatui + sqlx).
Deploy target: Railway (single service, persistent volume mounted at `/data`).

> R1 Rust-only · ONE SHOT [trios-trainer-igla#99 RAINFROG-ADOPT](https://github.com/gHashTag/trios-trainer-igla/issues/99)
> dropped Drizzle Studio (JS) in favour of rainfrog (Rust, 5k★, last push 2026-04-24).

Anchor: `φ² + φ⁻² = 3` · DOI [10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877).
SOT: `gHashTag/trios-trainer-igla/assertions/seed_results.jsonl` · Race tracker: `gHashTag/trios#508`.

---

## Schema (`shared/schema.ts`)

| Table         | Purpose                                                      |
| ------------- | ------------------------------------------------------------ |
| `formats`     | Numeric format registry (binary32, bfloat16, gf16, fp8…)     |
| `algos`       | Optimizer registry (adamw, muon, sgd, lion, soap…)           |
| `runs`        | Training jobs (queue_id, format, algo, seed, hidden, status) |
| `bpb_samples` | Per-step BPB telemetry, written by scarab/cron via ingest    |

---

## Local dev

```bash
npm install
npm run db:push         # create tables in ./data.db
npm run db:seed         # registries + champion + Wave-8 manifest
INGEST_TOKEN=devtoken npm run dev   # API on http://localhost:5000

cargo install --locked rainfrog   # one-time
cp etc/rainfrog.config.toml.example ~/.config/rainfrog/rainfrog_config.toml
./scripts/dash          # opens ./data.db in rainfrog (igla-sqlite, default)
./scripts/dash phd      # opens phd-postgres-ssot (after L-RF4)
```

Keybinds, query history, schema browser and multi-DB picker are documented in
[`docs/runbooks/rainfrog.md`](docs/runbooks/rainfrog.md).

---

## REST API

| Method | Path                  | Auth          | Purpose                              |
| ------ | --------------------- | ------------- | ------------------------------------ |
| GET    | `/api/health`         | —             | liveness + row counts                |
| GET    | `/api/champion`       | —             | best BPB row + gate margin           |
| GET    | `/api/leaderboard`    | —             | best BPB per (format, algo, seed)    |
| GET    | `/api/matrix`         | —             | format × algo aggregate grid         |
| GET    | `/api/runs`           | —             | run queue (filter by `?status=`)     |
| POST   | `/api/ingest`         | INGEST_TOKEN  | one BPB sample                       |
| POST   | `/api/ingest/batch`   | INGEST_TOKEN  | array of BPB samples                 |

Auth: send `Authorization: Bearer <INGEST_TOKEN>` or `X-Ingest-Token: <token>`.

### Ingest example (scarab/cron)

```bash
curl -X POST $RAILWAY_URL/api/ingest \
  -H "Authorization: Bearer $INGEST_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "format":"gf16","algo":"muon","seed":1597,
    "hidden":828,"lr":0.003,"step":4000,
    "bpb":1.7821,"emaBpb":1.79,"sha":"abc1234",
    "gateStatus":"below_target","queueId":"19341"
  }'
```

---

## Railway deploy

1. `railway init` in this directory.
2. Add a **Volume** mounted at `/data`.
3. Set service variables:
   - `INGEST_TOKEN` = strong random secret.
   - `DB_PATH` = `/data/igla.db`.
   - `NODE_ENV` = `production`.
   - `PORT` is injected by Railway (the server reads `process.env.PORT`).
4. Build & start:
   - **Build:** `npm ci && npm run build`
   - **Start:** `npm run start`
5. After first boot, exec into the container once and run
   `DB_PATH=/data/igla.db npm run db:push && npm run db:seed`
   (or wire it into `releaseCommand` once Railway adds support).
6. To browse the live DB: copy `igla.db` locally (`railway run cat /data/igla.db > data.db`) or
   point Drizzle Studio at the Railway shell with `DB_PATH=/data/igla.db npm run db:studio`.

---

## Wave-8-HONEST manifest (seeded)

40 jobs queued 19327…19366, image
`ghcr.io/ghashtag/trios-train@sha256:ecce23e9e72e61c662cfa7a149292087ccd3c1d7d5be24615bae0175700d5832`.
Formats × algos × seeds: `{binary32, binary16, bfloat16, gf16} × {adamw, muon} × {1597, 2584, 4181, 6765, 10946}`.
Champion baseline: `seed=43, format=binary32, algo=adamw, bpb=2.1919, sha=cd91c45, step=81000, hidden=828`.
Gate-2 target: `BPB < 1.85`.
