# Zero-Downtime Deploy: Scarab Pool

Do NOT kill old workers while they have running experiments.

## Step 1 — Schema migration

```bash
psql $NEON_DATABASE_URL -f migrations/002_scarab_stateless.sql

# Verify trigger:
psql $NEON_DATABASE_URL -c "
  SELECT tgname, tgenabled
  FROM pg_trigger
  WHERE tgrelid = 'strategy_queue'::regclass;"
```

## Step 2 — Deploy new scarabs ALONGSIDE old workers

```bash
for ACC in 0 1 2 3 4 5; do
  railway --account=acc$ACC service create scarab-pool \
    --dockerfile Dockerfile.scarab
  railway --account=acc$ACC service env set \
    NEON_DATABASE_URL="$NEON_DATABASE_URL" \
    RUST_LOG=info \
    SCARAB_ACCOUNT="acc$ACC"
done
```

## Step 3 — Wait for old workers to drain

```sql
-- Repeat every 5 min. Wait for 0 rows.
SELECT id, canon_name, started_at
FROM strategy_queue
WHERE status = 'running'
ORDER BY started_at;
```

## Step 4 — Kill old workers (only after Step 3 returns 0 rows)

```bash
for ACC in 0 1 2 3 4 5; do
  railway --account=acc$ACC service delete trios-train-v2-acc${ACC}-s1597
done
```

## Step 5 — Scale up (optional)

```bash
railway --account=acc0 service create scarab-pool-2 --dockerfile Dockerfile.scarab
railway --account=acc0 service env set NEON_DATABASE_URL="$NEON_DATABASE_URL"
```

## Diagnostics

```sql
SELECT status, COUNT(*) FROM strategy_queue GROUP BY status;

SELECT label, host, current_strategy_id,
       extract(epoch from (now() - last_heartbeat))::int AS idle_sec
FROM scarabs ORDER BY last_heartbeat DESC;

-- Reclaim stuck experiments:
SELECT reclaim_stale_strategies();
```
