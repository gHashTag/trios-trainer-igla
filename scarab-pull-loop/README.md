# scarab-pull-loop

**Sovereign Scarab v4** — eternal Rust agents for the Trinity training fleet.

> One row in `ssot.scarab_strategy` controls one Railway service forever.
> The Queen Hive writes SQL. The scarabs read it. No Railway API. No PAT.

- **Status:** Production-ready · R5-verified · ADR-0042 accepted
- **Anchor:** φ² + φ⁻² = 3 · TRINITY · DEFENSE 2026-06-15

---

## Why this exists

Read [`docs/ADR-0042-pull-loop.md`](docs/ADR-0042-pull-loop.md) for the full
rationale. The 30-second pitch:

- The old fleet-control plane (Railway GraphQL `serviceInstanceRedeploy` +
  `variableUpsert`) **dies whenever the PAT expires** (gHashTag/trios-railway#156).
- Replacement: each service runs an eternal Rust binary that polls Postgres
  for *desired* config and gracefully restarts its trainer on change.
- Control surface for the operator is **three SQL functions**. That's it.

---

## Anatomy

```
scarab-pull-loop/
├── Cargo.toml                            # crate manifest (lib + bin)
├── Dockerfile                            # multi-stage distroless build
├── README.md                             # this file
├── docs/
│   └── ADR-0042-pull-loop.md             # design rationale + falsification
├── migrations/
│   └── 001_scarab_strategy.sql           # single idempotent master SQL
└── src/
    ├── lib.rs                            # Strategy, Config, fingerprint, kill
    └── main.rs                           # eternal pull-loop binary
```

---

## Migration

Apply once against any database that already has Trinity migration `0004_sovereign_scarab.sql`
(it's the baseline that creates `ssot.scarab_strategy`, `ssot.scarab_heartbeat`,
and the flat `bump_strategy()` core function):

```bash
psql "$DATABASE_URL" -f migrations/001_scarab_strategy.sql
```

The migration is **fully idempotent**: re-running it on a database that already
has the v4 schema is a no-op (every CREATE uses `IF NOT EXISTS` /
`CREATE OR REPLACE`, every constraint is `DROP IF EXISTS` then `ADD`).

It adds, on top of `0004`:

| Object | Type | Purpose |
|---|---|---|
| `ssot.spawn_scarab(…)` | function | Allocate a new scarab row + audit-log entry |
| `ssot.bump_strategy_v2(jsonb)` | function | Mutate any subset of hyperparameters atomically |
| `ssot.kill_scarab` / `pause_scarab` / `resume_scarab` | functions | Lifecycle control |
| `ssot.fleet_status` | view | Desired vs applied + heartbeat age + drift label |
| `ssot.fleet_drift` | view | `fleet_status` filtered to non-`in_sync` rows |
| `ssot.strategy_history` | view | Every distinct strategy a service has run |
| `ssot.scarab_assignment_log` | view (alias of `scarab_command`) | Audit log |
| `pg_notify('scarab_<sid>')` | trigger | Push-wake the right scarab |
| `pg_notify('scarab_fleet')` | trigger | Push-wake the dashboard |
| Blast-radius guard | trigger | One UPDATE can't deactivate > 33 % of fleet |
| Extended CHECKs | constraints | Optimizer / format / hidden / lr / seed whitelisted |

### Seed canon (Fibonacci)

Only these seeds may appear in `ssot.scarab_strategy.seed`:

```
1597, 2584, 4181, 6765, 10946, 47, 89, 144, 123
```

Typos cannot escape into 27 trainers.

---

## Build

```bash
# Native
cargo build --release --bin scarab

# Docker (distroless, ~20 MB image)
docker build -t ghcr.io/ghashtag/sovereign-scarab:v4 .
```

Unit tests (no DB needed):

```bash
cargo test --lib
```

Result: 4 passing — `fingerprint_is_stable`, `fingerprint_changes_on_any_hyperparam`,
`canon_name_format`, `fingerprint_matches_sql_format`.

---

## Environment

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | *required* | Postgres SSoT URL (no `NEON_DATABASE_URL` fallback — canonical name) |
| `RAILWAY_SERVICE_ID` | *required* | Railway-injected; falls back to `SERVICE_ID` for local use |
| `TRAINER_BIN` | `/usr/local/bin/trios-train` | Path to the trainer child binary |
| `TRAIN_DATA` | `/data/train.txt` | Train corpus path |
| `VAL_DATA` | `/data/val.txt` | Validation corpus path |
| `POLL_SEC` | `10` | Poll interval (NOTIFY also wakes earlier) |
| `GRACE_MS` | `10000` | SIGTERM → SIGKILL grace period |
| `LISTEN_NOTIFY` | `1` | Set `0` to disable LISTEN and use pure-poll |
| `DRY_RUN` | `0` | Set `1` to skip the actual trainer spawn (CI / smoke test) |
| `AUTO_REPLAY` | `0` | Set `1` to bump generation after each DONE event |

---

## Operator cookbook

### Spawn a new scarab

```sql
SELECT ssot.spawn_scarab(
  p_service_id  := 'igla-acc1-A',
  p_account     := 'acc1',
  p_canon_name  := 'IGLA-RAILWAY-fp32-h128-LR0.001-rng2584-adamw',
  p_optimizer   := 'adamw',
  p_format      := 'fp32',
  p_hidden      := 128,
  p_lr          := 0.001,
  p_seed        := 2584,
  p_steps       := 100000,
  p_reason      := 'first scarab on acc1'
);
```

### Bump a running scarab

```sql
SELECT ssot.bump_strategy_v2(
  p_service_id := 'igla-acc1-A',
  p_changes    := jsonb_build_object(
                    'optimizer','muon',
                    'lr',        0.005,
                    'seed',      4181,
                    'steps',     50000
                  ),
  p_reason     := 'switching to muon · gate-2 push'
);
```

Within ~100 ms (LISTEN/NOTIFY) the scarab gracefully restarts its trainer with
the new params. Even if NOTIFY is dropped, the poll loop catches up within
`POLL_SEC` (default 10 s).

### Inspect the fleet

```sql
-- One row per scarab — desired vs applied vs heartbeat
SELECT * FROM ssot.fleet_status ORDER BY service_id;

-- Only the unhealthy ones
SELECT * FROM ssot.fleet_drift;

-- Strategy history (one row per distinct fingerprint a service has run)
SELECT * FROM ssot.strategy_history WHERE service_id = 'igla-acc1-A';

-- Audit log
SELECT id, service_id, command, issued_at, strategy_fingerprint, reason
FROM ssot.scarab_assignment_log
ORDER BY id DESC LIMIT 20;
```

### Pause / resume / kill

```sql
SELECT ssot.pause_scarab ('igla-acc1-A', 'investigating bpb=inf');
SELECT ssot.resume_scarab('igla-acc1-A', 'investigation done');
SELECT ssot.kill_scarab  ('igla-acc1-A', 'graduating to gate-3');
```

### Emergency mass operation (blast guard override)

```sql
BEGIN;
SET LOCAL scarab.bypass_blast_guard = 'on';
UPDATE ssot.scarab_strategy
   SET status = 'paused', updated_by = 'queen-hive', updated_at = now()
 WHERE account = 'acc-broken';
COMMIT;
```

The guard log keeps a forensic trail; every bypass should have a written reason
in the surrounding `scarab_command` insert.

---

## Verification (R5)

R5-evidence directory:
[`/home/user/workspace/cron_tracking/sovereign_scarab/evidence/`](/home/user/workspace/cron_tracking/sovereign_scarab/evidence/)

| R5 check | File | Result |
|---|---|---|
| Fresh-DB migration apply | `2026-05-14T10:30Z-hardening/SUMMARY.md` | ✅ all DDL succeeds |
| Fibonacci seed CHECK | run logs `R5-1` | ✅ `seed=999` aborts |
| Optimizer / lr / hidden CHECK | run logs `R5-2..4` | ✅ each aborts |
| Bump + fingerprint stamp | run logs `R5-5` | ✅ two distinct sha256 entries |
| `fleet_status` view | run logs `R5-6` | ✅ `never_seen` drift state |
| Blast guard (12 of 27) | run logs | ✅ aborts with `limit is 9` |
| Blast guard (3 small) | run logs | ✅ passes |
| Bypass GUC | run logs | ✅ 17-row UPDATE passes inside `SET LOCAL` |
| Rust SQL-fingerprint parity | `cargo test fingerprint_matches_sql_format` | ✅ |

---

## License

Dual-licensed under MIT or Apache-2.0 at the user's option.
Trinity anchor: φ² + φ⁻² = 3.
