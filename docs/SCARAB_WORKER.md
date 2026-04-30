# Scarab Worker — Stateless Fungible Pool

> **Rule 1:** Never touch the container.  
> **Rule 2:** Change strategy only through Neon.  
> **Rule 3:** No account affinity. Any scarab takes any task.

## How it works

```
loop forever:
  drain:
    row = SELECT ... FROM strategy_queue
          WHERE status = 'pending'        # NO AND account = $1
          ORDER BY priority DESC, id ASC
          FOR UPDATE SKIP LOCKED
          LIMIT 1
    if row:
      UPDATE status = 'running', worker_id = $HOSTNAME
      spawn: trios-igla train <params from config_json>
      UPDATE status = 'done' / 'failed'
    else:
      break
  sleep until NOTIFY strategy_new or 30s fallback
```

## Before vs After

| | Before (anti-pattern) | After (fungible pool) |
|---|---|---|
| Claim filter | `AND account = $1` | removed |
| acc3 dies | seed45 stuck waiting | acc5 picks it up in ≤30s |
| Scale | 6 separate service definitions | 1 image, N replicas |
| Strategy change | redeploy | `INSERT` into Neon |
| Idle polling | every 10s | NOTIFY + 30s fallback |
| Timeout watchdog | none | `max_runtime_sec` in spec |

## Strategy spec (config_json)

```json
{
  "trainer": {
    "hidden": 1024,
    "lr": 0.002,
    "steps": 81000,
    "ctx": 12,
    "format": "fp32",
    "seed": 42
  },
  "constraints": {
    "max_runtime_sec": 900
  },
  "submission": {
    "track": "non_record_16mb",
    "tags": ["GATE3"]
  }
}
```

Legacy flat format `{"hidden": 828, "lr": 0.0004, ...}` is also accepted automatically.

## Deploy on Railway

```bash
for ACC in 0 1 2 3 4 5; do
  railway --account=acc$ACC service create scarab-pool \
    --dockerfile Dockerfile.scarab
  railway --account=acc$ACC service env set \
    NEON_DATABASE_URL="$NEON_DATABASE_URL" \
    SCARAB_ACCOUNT="acc$ACC"   # optional: log tag only
done
```

## Manage strategies via Neon

```sql
-- Add a new strategy (trigger auto-notifies all sleeping scarabs):
INSERT INTO strategy_queue (canon_name, priority, steps_budget, config_json)
VALUES (
  'GATE3-h1024-lr002-seed42', 100, 81000,
  '{"trainer":{"hidden":1024,"lr":0.002,"steps":81000,
    "ctx":12,"format":"fp32","seed":42},
   "constraints":{"max_runtime_sec":900},
   "submission":{"track":"non_record_16mb","tags":["GATE3"]}}'
);

-- Raise priority:
UPDATE strategy_queue SET priority = 200
WHERE canon_name LIKE '%GATE3%' AND status = 'pending';

-- Cancel:
UPDATE strategy_queue SET status = 'pruned' WHERE id = 9999;

-- Reclaim stuck (worker died):
SELECT reclaim_stale_strategies();

-- Monitor:
SELECT label, host, current_strategy_id, last_heartbeat FROM scarabs
ORDER BY last_heartbeat DESC;
```

## Migration

```bash
psql $NEON_DATABASE_URL -f migrations/002_scarab_stateless.sql
```

## ENV

| Variable | Required | Description |
|---|---|---|
| `NEON_DATABASE_URL` | yes | postgres://user:pass@host/db |
| `SCARAB_ACCOUNT` | no | Log tag only, no scheduling effect |
| `HOSTNAME` | auto | Set by Railway/Docker |

## Files to NEVER change (only on critical bugfix)

- `Dockerfile.scarab`
- `src/bin/scarab.rs`
- `migrations/002_scarab_stateless.sql`

## Files to ALWAYS change through Neon

- `strategy_queue` — add new strategies
- `config_json` — hyperparameters
- `priority` — queue order
