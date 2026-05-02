# trinity-experiments — L0: Single Source of Truth

**Constitutional mandate (Law 2):** `strategy_experiments` is the ONLY table.

## Modules

- `schema.rs` — Experiment domain model
- `repo.rs` — ExperimentRepo trait + Postgres implementation
- `migration.rs` — 0001_initial.sql (immutable, ONE TIME ONLY)

## Schema

```sql
CREATE TABLE strategy_experiments (
  id BIGSERIAL PRIMARY KEY,
  canon_name TEXT NOT NULL UNIQUE,
  phd_chapter TEXT NOT NULL,
  inv_id TEXT NOT NULL,
  config_json JSONB NOT NULL,
  required_image_tag TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending','running','done','failed')),
  worker_id UUID,
  claimed_at TIMESTAMPTZ,
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  final_bpb DOUBLE PRECISION CHECK (final_bpb IS NULL OR (final_bpb > 0 AND final_bpb < 100)),
  final_step INTEGER,
  bpb_curve JSONB,  -- Embedded, not a separate table
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (status NOT IN ('done','failed') OR final_bpb IS NOT NULL OR last_error IS NOT NULL)
);
```

## Usage

```rust
use trinity_experiments::{PostgresExperimentRepo, ExperimentRepo};
use uuid::Uuid;

let repo = PostgresExperimentRepo::connect(&neon_url).await?;
let worker_id = Uuid::new_v4();

match repo.claim_next(worker_id).await? {
    ClaimResult::Claimed(claim) => {
        // ... train ...
        repo.complete(claim.id, final_bpb, final_step, bpb_curve).await?;
    }
    ClaimResult::NoPending => { /* sleep */ }
}
```

## Invariants

- **Law 1 (R5-honest):** No claim without verified source row
- **Law 2 (single SoT):** No derived tables
- **Law 4 (immutable):** Append-only, no UPDATE after done/failed

## Migration

```bash
# Run migration (ONCE, never again)
psql $NEON_DATABASE_URL -f crates/trinity-experiments/migrations/0001_initial.sql
```

**CONSTITUTIONAL WARNING:** Never ALTER this table after creation.
