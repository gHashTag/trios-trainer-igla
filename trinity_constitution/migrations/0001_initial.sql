-- Trinity Experiments — Single Source of Truth
-- Migration: 0001_initial.sql
-- Constitutional mandate: NEVER ALTER this table after creation

CREATE TABLE IF NOT EXISTS strategy_experiments (
  -- Identity
  id              BIGSERIAL PRIMARY KEY,
  canon_name      TEXT NOT NULL UNIQUE,

  -- PhD config (READ-ONLY after insert)
  phd_chapter     TEXT NOT NULL,           -- e.g. "Ch.21"
  inv_id          TEXT NOT NULL,           -- e.g. "INV-6"
  config_json     JSONB NOT NULL,          -- {seed, hidden, lr, steps, corpus, ...}
  required_image_tag TEXT NOT NULL,        -- pinned, no drift

  -- Lifecycle (one-way state machine)
  status          TEXT NOT NULL CHECK (status IN ('pending','running','done','failed')),
  worker_id       UUID,
  claimed_at      TIMESTAMPTZ,
  started_at      TIMESTAMPTZ,
  finished_at     TIMESTAMPTZ,

  -- Result (R5-honest: filled exactly once on done/failed)
  final_bpb       DOUBLE PRECISION CHECK (final_bpb IS NULL OR (final_bpb > 0 AND final_bpb < 100)),
  final_step      INTEGER,
  bpb_curve       JSONB,                   -- [{step:1000,bpb:3.30},...] — embedded
  last_error      TEXT,

  -- Constitution invariant: status='done' XOR final_bpb XOR last_error
  CHECK (status NOT IN ('done','failed')
         OR final_bpb IS NOT NULL
         OR last_error IS NOT NULL),

  -- Audit
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_se_pending
ON strategy_experiments (status, id)
WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_se_phd
ON strategy_experiments (phd_chapter, inv_id);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_strategy_experiments_updated_at ON strategy_experiments;
CREATE TRIGGER update_strategy_experiments_updated_at
    BEFORE UPDATE ON strategy_experiments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE strategy_experiments IS 'Single Source of Truth for all training runs. O(1) Constitutional table — never ALTER.';
