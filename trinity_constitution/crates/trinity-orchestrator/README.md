# trinity-orchestrator — L3+L4+L5: Declarative Deploy

**Constitutional mandate (Law 2):** Declarative fleet management.

## Usage

```bash
# Reconcile fleet (apply desired state)
cargo run --release --bin trinity-reconcile

# Dry run (show diff, don't apply)
DRY_RUN=true cargo run --release --bin trinity-reconcile

# Custom manifest path
SERVICES_MANIFEST=manifest/production.toml cargo run --release --bin trinity-reconcile
```

## Manifest Structure

```toml
# manifest/services.toml
[[services]]
name = "runner-0"
image = "ghcr.io/gHashTag/trinity:latest"
replicas = 3
env = ["NEON_DATABASE_URL=${NEON_DATABASE_URL}"]
command = ["/usr/local/bin/trinity-runner"]

[[services]]
name = "gardener"
image = "ghcr.io/gHashTag/trinity:latest"
replicas = 1
env = ["PORT=8080"]
command = ["/usr/local/bin/trinity-gardener"]
```

## Reconcile Logic

1. Load manifest (`services.toml`)
2. Query current Railway state
3. Diff (current vs desired)
4. Apply changes (create/update/delete services)
5. Report drifts (if any)

## Invariants

- **Law 2 (single SoT):** Manifest is source of truth
- **Self-healing:** Drift auto-corrected on next reconcile
- **Fail-loud:** Any drift reported and logged

## Status

⏳ **PR-O6** — implementation in progress
- [x] CLI skeleton
- [ ] Manifest parser
- [ ] Railway GraphQL client
- [ ] Diff and apply logic
