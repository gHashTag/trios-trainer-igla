# trinity-runner — L2: Claim → Train → Write Loop

**Constitutional mandate (Law 1):** No claim without verified source row.

## Usage

```bash
# Local test with Neon
NEON_DATABASE_URL="postgres://..." cargo run --release --bin trinity-runner

# Run once and exit (no loop)
RUN_ONCE=true cargo run --release --bin trinity-runner

# Custom poll interval
POLL_INTERVAL=60 cargo run --release --bin trinity-runner
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEON_DATABASE_URL` | Yes | — | Postgres connection string |
| `WORKER_ID` | No | (generated) | UUID for this worker |
| `POLL_INTERVAL` | No | 30 | Poll interval in seconds |
| `RUN_ONCE` | No | false | Exit after one claim attempt |

## Loop Logic

```rust
loop {
    match claim_next(worker_id).await? {
        Claimed(claim) => {
            train(claim.config)?;
            write_result(claim.id, outcome)?;
        }
        NoPending => { /* sleep */ }
    }
}
```

## Invariants

- **Law 1 (R5-honest):** Idempotent claim with `FOR UPDATE SKIP LOCKED`
- **Law 2 (single SoT):** All writes go to strategy_experiments
- **Retry logic:** Exponential backoff on connection failure

## Safety

- Double-claim prevented by `FOR UPDATE SKIP LOCKED`
- Connection retry with exponential backoff (max 10 attempts)
- Fail-loud on BPB NaN/infinity
