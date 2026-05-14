# scarab-pull-loop

**SOVEREIGN SCARAB** control-plane agent. Every scarab on Railway runs this
binary instead of being directly orchestrated through Railway GraphQL.

## Behaviour

Every `POLL_SEC` (default 30s):

1. `SELECT * FROM ssot.scarab_strategy WHERE service_id = $SERVICE_ID`
2. Write heartbeat to `ssot.scarab_heartbeat` (last_seen, current_gen, step, bpb, pid)
3. If `status = 'stop'` → graceful shutdown of subprocess + exit.
4. If `generation > current_gen` → kill running trainer subprocess, spawn new
   `trios-train` with the new params from the strategy row.

## Environment

| Var | Required | Default | Note |
|---|---|---|---|
| `DATABASE_URL` | yes | — | Railway Postgres / Neon plugin. **NOT** `NEON_DATABASE_URL`. |
| `SERVICE_ID` | yes | — | e.g. `igla-1`, `matrix-runner-acc2-19`. |
| `TRAINER_BIN` | no | `/app/trios-train` | path to release binary |
| `TRAIN_DATA` | no | `/tmp/honest_runs/train.txt` | |
| `VAL_DATA` | no | `/tmp/honest_runs/val.txt` | |
| `POLL_SEC` | no | `30` | strategy poll interval |
| `DRY_RUN` | no | unset | `1` → heartbeat only, don't spawn trainer |

## Anti-token-hell guarantee

This binary is the entire control surface. It does **not** call:
- Railway GraphQL (`backboard.railway.com`)
- GitHub Actions
- any service that requires `RAILWAY_TOKEN_*` or PAT

Only credential needed: `DATABASE_URL`, set once at Railway service-create
time via the Neon plugin. It does not rotate.

## Queen-Hive command path

```sql
-- Switch local-A to muon-cwd / gf16, bump generation:
SELECT ssot.bump_strategy(
  'local-A',
  p_optimizer := 'muon-cwd',
  p_format    := 'gf16',
  p_seed      := 144,
  p_by        := 'queen-hive'
);

-- Stop a scarab gracefully:
UPDATE ssot.scarab_strategy
SET status = 'stop', generation = generation + 1,
    updated_at = now(), updated_by = 'queen-hive'
WHERE service_id = 'igla-1';

-- Dead-scarab detector:
SELECT s.service_id, s.optimizer, s.format,
       h.last_seen, EXTRACT(EPOCH FROM (now() - h.last_seen))/60 AS stale_min
FROM ssot.scarab_strategy s
LEFT JOIN ssot.scarab_heartbeat h USING (service_id)
WHERE h.last_seen IS NULL OR h.last_seen < now() - interval '2 minutes';
```

## R5-evidence

See [ADR-CHAT-011](https://github.com/gHashTag/trios/blob/main/docs/adr/ADR-CHAT-011.md)
and the local-prototype evidence pack at
`/home/user/workspace/cron_tracking/sovereign_scarab/evidence/2026-05-14T09:42Z/`.

Anchor: `phi^2 + phi^-2 = 3 · TRINITY · DEFENSE 2026-06-15`
