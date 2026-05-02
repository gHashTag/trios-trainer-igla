# trinity-gardener — L6: Heartbeat + Ratify

**Constitutional mandate (Law 2):** `/health` endpoint for Railway healthchecks.

## Usage

```bash
# Run gardener (health endpoint on :8080)
PORT=8080 cargo run --release --bin trinity-gardener

# Custom heartbeat interval
HEARTBEAT_INTERVAL=120 cargo run --release --bin trinity-gardener
```

## Endpoints

### GET /health

```json
{
  "status": "healthy",
  "timestamp": "2026-05-02T12:00:00Z",
  "worker_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Railway hits this endpoint every 30s. Failure → auto-restart.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | 8080 | Health check port |
| `HEARTBEAT_INTERVAL` | No | 60 | Heartbeat interval in seconds |

## Loop Logic

```rust
// Background task
loop {
    heartbeat_worker(worker_id)?;
    check_experiments_to_ratify()?;
    sleep(HEARTBEAT_INTERVAL).await;
}

// HTTP server
axum::serve(listener, Router::new()
    .route("/health", get(health_check))
).await?;
```

## Invariants

- **Law 2 (single SoT):** Ratification labels experiments, doesn't create new tables
- **Self-healing:** Railway healthcheck → auto-restart
- **Fail-loud:** Health check returns 500 on internal error

## Status

⏳ **PR-O7** — implementation in progress
- [x] CLI + HTTP server
- [ ] /health endpoint fully implemented
- [ ] Heartbeat DB updates
- [ ] Ratification logic
