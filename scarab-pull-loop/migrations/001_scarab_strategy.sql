-- ============================================================================
-- SOVEREIGN SCARAB v3.1 · CONSOLIDATED MASTER MIGRATION
-- ============================================================================
-- ADR-CHAT-012 · DB-as-control-plane: 27 eternal Rust agents poll Postgres SSoT,
-- graceful-restart themselves when ssot.scarab_strategy.generation advances.
-- Queen Hive operates ONLY via SQL (3 functions). Zero Railway API calls.
--
-- This single file is fully idempotent and is the canonical entry point.
-- Earlier split files 0005/0006/0007 are superseded; do not apply them.
--
-- Anchor: phi^2 + phi^-2 = 3 · TRINITY · DEFENSE 2026-06-15
-- ============================================================================

BEGIN;

-- ============================================================================
-- 0. EXTENSIONS
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- digest() for strategy_fingerprint

CREATE SCHEMA IF NOT EXISTS ssot;

-- ============================================================================
-- 0a. BASE bump_strategy() — absorbed from migration 0004 to make 001 self-contained.
-- ============================================================================
CREATE OR REPLACE FUNCTION ssot.bump_strategy(
  p_service_id text,
  p_optimizer  text DEFAULT NULL,
  p_format     text DEFAULT NULL,
  p_hidden     int  DEFAULT NULL,
  p_lr         numeric DEFAULT NULL,
  p_seed       int  DEFAULT NULL,
  p_steps      int  DEFAULT NULL,
  p_status     text DEFAULT NULL,
  p_by         text DEFAULT 'queen-hive'
) RETURNS bigint LANGUAGE plpgsql AS $$
DECLARE new_gen bigint;
BEGIN
  UPDATE ssot.scarab_strategy SET
    optimizer  = COALESCE(p_optimizer, optimizer),
    format     = COALESCE(p_format, format),
    hidden     = COALESCE(p_hidden, hidden),
    lr         = COALESCE(p_lr, lr),
    seed       = COALESCE(p_seed, seed),
    steps      = COALESCE(p_steps, steps),
    status     = COALESCE(p_status, status),
    generation = generation + 1,
    updated_at = now(),
    updated_by = p_by
  WHERE service_id = p_service_id
  RETURNING generation INTO new_gen;
  RETURN new_gen;
END $$;

-- ============================================================================
-- 1. CORE TABLES (additive to 0004 — uses existing scarab_strategy if present)
-- ============================================================================

-- Strategy row (one per eternal scarab). 0004 already creates this table on
-- production; the block below is defensive for fresh databases.
CREATE TABLE IF NOT EXISTS ssot.scarab_strategy (
  service_id   text PRIMARY KEY,
  account      text NOT NULL,
  optimizer    text NOT NULL,
  format       text NOT NULL,
  hidden       int  NOT NULL,
  lr           numeric NOT NULL,
  seed         int  NOT NULL,
  steps        int  NOT NULL,
  status       text NOT NULL DEFAULT 'active',
  generation   bigint NOT NULL DEFAULT 1,
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_by   text NOT NULL DEFAULT 'queen-hive'
);

-- Heartbeat (one row per service_id, upsert by scarab every ~30s)
CREATE TABLE IF NOT EXISTS ssot.scarab_heartbeat (
  service_id        text PRIMARY KEY,
  last_seen         TIMESTAMPTZ NOT NULL DEFAULT now(),
  current_gen       bigint NOT NULL DEFAULT 0,
  current_step      int,
  current_bpb       double precision,
  pid               int,
  started_at        TIMESTAMPTZ,
  applied_version   bigint NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS scarab_heartbeat_stale ON ssot.scarab_heartbeat (last_seen);

-- Training results (one row per DONE event from a trainer child)
CREATE TABLE IF NOT EXISTS ssot.scarab_result (
  id           BIGSERIAL PRIMARY KEY,
  service_id   text NOT NULL,
  canon_name   text NOT NULL,
  optimizer    text NOT NULL,
  format       text NOT NULL,
  hidden       int  NOT NULL,
  lr           numeric NOT NULL,
  seed         int  NOT NULL,
  steps        int  NOT NULL,
  final_bpb    double precision,
  wall_s       int,
  generation   bigint NOT NULL,
  written_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_scarab_result_service
  ON ssot.scarab_result (service_id, written_at DESC);

-- Assignment audit log (INSERT-only). Historically named scarab_command;
-- exposed to operators via the strategy_history view.
CREATE TABLE IF NOT EXISTS ssot.scarab_command (
  id            BIGSERIAL PRIMARY KEY,
  service_id    text NOT NULL,
  command       text NOT NULL,            -- spawn | bump | pause | resume | kill
  old_strategy  JSONB,
  new_strategy  JSONB,
  reason        text,
  issued_by     text NOT NULL DEFAULT 'queen-hive',
  issued_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  strategy_fingerprint text
);
CREATE INDEX IF NOT EXISTS idx_scarab_command_service
  ON ssot.scarab_command (service_id, issued_at DESC);

-- ============================================================================
-- 2. APPLIED-VERSION TRACKING (lift over 0004)
-- ============================================================================
ALTER TABLE ssot.scarab_heartbeat
  ADD COLUMN IF NOT EXISTS applied_version BIGINT NOT NULL DEFAULT 0;

-- ============================================================================
-- 3. STATUS / SEED / OPTIMIZER / FORMAT / HIDDEN / LR / STEPS guards
-- ============================================================================

ALTER TABLE ssot.scarab_strategy DROP CONSTRAINT IF EXISTS scarab_strategy_status_check;
ALTER TABLE ssot.scarab_strategy ADD CONSTRAINT scarab_strategy_status_check
  CHECK (status IN ('active','paused','draining','killed','stop'));

ALTER TABLE ssot.scarab_strategy DROP CONSTRAINT IF EXISTS scarab_strategy_optimizer_check;
ALTER TABLE ssot.scarab_strategy ADD CONSTRAINT scarab_strategy_optimizer_check CHECK (
  optimizer IN ('adamw','muon','muon-cwd','soap','shampoo','lamb','prodigy',
                'sgdm','lion','signum','ranger','sophia','adafactor')
);

ALTER TABLE ssot.scarab_strategy DROP CONSTRAINT IF EXISTS scarab_strategy_format_check;
ALTER TABLE ssot.scarab_strategy ADD CONSTRAINT scarab_strategy_format_check CHECK (
  format IN ('fp32','fp16','bf16','fp8_e4m3','fp8_e5m2',
             'gf16','gf256','int4','int8','nf4','posit16','binary16')
);

ALTER TABLE ssot.scarab_strategy DROP CONSTRAINT IF EXISTS scarab_strategy_hidden_check;
ALTER TABLE ssot.scarab_strategy ADD CONSTRAINT scarab_strategy_hidden_check CHECK (
  hidden IN (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
);

ALTER TABLE ssot.scarab_strategy DROP CONSTRAINT IF EXISTS scarab_strategy_lr_check;
ALTER TABLE ssot.scarab_strategy ADD CONSTRAINT scarab_strategy_lr_check CHECK (
  lr > 0::numeric AND lr < 1.0::numeric
);

ALTER TABLE ssot.scarab_strategy DROP CONSTRAINT IF EXISTS scarab_strategy_steps_check;
ALTER TABLE ssot.scarab_strategy ADD CONSTRAINT scarab_strategy_steps_check CHECK (
  steps >= 100 AND steps <= 1000000
);

-- Fibonacci seed canon: pre-registered seeds used across the fleet.
-- This is intentionally a small whitelist — a typo cannot leak into 27 trainers.
ALTER TABLE ssot.scarab_strategy DROP CONSTRAINT IF EXISTS scarab_strategy_seed_check;
ALTER TABLE ssot.scarab_strategy ADD CONSTRAINT scarab_strategy_seed_check CHECK (
  seed IN (1597, 2584, 4181, 6765, 10946, 47, 89, 144, 123)
);

-- ============================================================================
-- 4. STRATEGY FINGERPRINT (lineage)
-- ============================================================================
-- SHA-256 over the immutable hyperparameters. Used to:
--   • detect "this scarab has run this exact strategy before" (dedup)
--   • build strategy_history view ("served strategies in lineage order")

CREATE OR REPLACE FUNCTION ssot.scarab_fingerprint(
  p_optimizer text, p_format text, p_hidden int,
  p_lr numeric, p_seed int, p_steps int
) RETURNS text LANGUAGE sql IMMUTABLE AS $$
  SELECT encode(
    digest(
      p_optimizer || '|' || p_format || '|' || p_hidden::text || '|' ||
      p_lr::text || '|' || p_seed::text || '|' || p_steps::text,
      'sha256'
    ),
    'hex'
  )
$$;

CREATE OR REPLACE FUNCTION ssot.scarab_command_fingerprint_trigger() RETURNS TRIGGER
LANGUAGE plpgsql AS $$
DECLARE s record;
BEGIN
  IF NEW.strategy_fingerprint IS NOT NULL THEN RETURN NEW; END IF;
  SELECT optimizer, format, hidden, lr, seed, steps
  INTO s FROM ssot.scarab_strategy WHERE service_id = NEW.service_id;
  IF FOUND THEN
    NEW.strategy_fingerprint :=
      ssot.scarab_fingerprint(s.optimizer, s.format, s.hidden, s.lr, s.seed, s.steps);
  END IF;
  RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS scarab_command_fp_trg ON ssot.scarab_command;
CREATE TRIGGER scarab_command_fp_trg
  BEFORE INSERT ON ssot.scarab_command
  FOR EACH ROW EXECUTE FUNCTION ssot.scarab_command_fingerprint_trigger();

-- ============================================================================
-- 5. LISTEN / NOTIFY  (low-latency wake-up channel)
-- ============================================================================
CREATE OR REPLACE FUNCTION ssot.scarab_notify_trigger() RETURNS TRIGGER
LANGUAGE plpgsql AS $$
DECLARE channel text;
BEGIN
  channel := 'scarab_' || replace(NEW.service_id, '-', '_');
  PERFORM pg_notify(channel, NEW.generation::text);
  PERFORM pg_notify('scarab_fleet',
    json_build_object('service_id', NEW.service_id,
                      'generation', NEW.generation,
                      'status',     NEW.status)::text);
  RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS scarab_notify_trg ON ssot.scarab_strategy;
CREATE TRIGGER scarab_notify_trg
  AFTER INSERT OR UPDATE ON ssot.scarab_strategy
  FOR EACH ROW EXECUTE FUNCTION ssot.scarab_notify_trigger();

-- ============================================================================
-- 6. BLAST-RADIUS GUARD (bulletproof, statement-level)
-- ============================================================================
-- A single UPDATE statement cannot deactivate more than CEIL(active*0.33)
-- scarabs (status moving away from 'active').
-- Override: SET LOCAL scarab.bypass_blast_guard = 'on' inside the transaction.
-- Small operations (< 3 rows) always pass.

CREATE OR REPLACE FUNCTION ssot.scarab_blast_guard_stmt() RETURNS TRIGGER
LANGUAGE plpgsql AS $$
DECLARE
  v_bypass text;
  v_deactivated int;
  v_active_total int;
  v_limit int;
BEGIN
  BEGIN
    v_bypass := current_setting('scarab.bypass_blast_guard', true);
  EXCEPTION WHEN OTHERS THEN
    v_bypass := NULL;
  END;
  IF v_bypass = 'on' THEN RETURN NULL; END IF;

  SELECT COUNT(*) INTO v_deactivated
  FROM new_table n
  JOIN old_table o ON n.service_id = o.service_id
  WHERE o.status = 'active' AND n.status <> 'active';

  -- Small changes are always allowed (<= 3 rows): operators routinely pause one
  IF v_deactivated <= 3 THEN RETURN NULL; END IF;

  -- Total active BEFORE this statement = currently active + just-deactivated
  SELECT COUNT(*) INTO v_active_total FROM ssot.scarab_strategy WHERE status = 'active';
  v_active_total := v_active_total + v_deactivated;

  -- Need a meaningful fleet to gate (small fleet — always allow)
  IF v_active_total < 9 THEN RETURN NULL; END IF;

  v_limit := GREATEST(1, CEIL(v_active_total::numeric * 0.33)::int);

  IF v_deactivated > v_limit THEN
    RAISE EXCEPTION
      'scarab blast guard: % rows deactivated in one statement, limit is % of % active. SET LOCAL scarab.bypass_blast_guard=on to override.',
      v_deactivated, v_limit, v_active_total;
  END IF;
  RETURN NULL;
END $$;

DROP TRIGGER IF EXISTS scarab_blast_guard_stmt_trg ON ssot.scarab_strategy;
CREATE TRIGGER scarab_blast_guard_stmt_trg
  AFTER UPDATE ON ssot.scarab_strategy
  REFERENCING NEW TABLE AS new_table OLD TABLE AS old_table
  FOR EACH STATEMENT
  EXECUTE FUNCTION ssot.scarab_blast_guard_stmt();

-- ============================================================================
-- 7. CONTROL-PLANE FUNCTIONS
-- ============================================================================
-- The base bump_strategy() (flat parameters) is created by migration 0004
-- on production. We add jsonb-friendly wrappers and lifecycle helpers.

CREATE OR REPLACE FUNCTION ssot.bump_strategy_v2(
  p_service_id text, p_changes jsonb, p_reason text DEFAULT NULL
) RETURNS bigint AS $$
DECLARE
  v_old JSONB;
  v_new_gen bigint;
BEGIN
  SELECT jsonb_build_object(
    'optimizer',optimizer, 'format',format, 'hidden',hidden,
    'lr',lr, 'seed',seed, 'steps',steps, 'status',status
  ) INTO v_old FROM ssot.scarab_strategy WHERE service_id = p_service_id;

  IF v_old IS NULL THEN RAISE EXCEPTION 'unknown scarab: %', p_service_id; END IF;

  v_new_gen := ssot.bump_strategy(
    p_service_id := p_service_id,
    p_optimizer  := p_changes->>'optimizer',
    p_format     := p_changes->>'format',
    p_hidden     := (p_changes->>'hidden')::int,
    p_lr         := (p_changes->>'lr')::numeric,
    p_seed       := (p_changes->>'seed')::int,
    p_steps      := (p_changes->>'steps')::int,
    p_status     := p_changes->>'status',
    p_by         := 'queen-hive'
  );

  INSERT INTO ssot.scarab_command (service_id, command, old_strategy, new_strategy, reason)
  VALUES (p_service_id, 'bump', v_old, p_changes, p_reason);

  RETURN v_new_gen;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION ssot.spawn_scarab(
  p_service_id text, p_account text, p_canon_name text,
  p_optimizer text, p_format text, p_hidden int, p_lr numeric,
  p_seed int, p_steps int, p_reason text DEFAULT NULL
) RETURNS bigint AS $$
DECLARE v_gen bigint;
BEGIN
  INSERT INTO ssot.scarab_strategy
    (service_id, account, optimizer, format, hidden, lr, seed, steps, status, generation)
  VALUES
    (p_service_id, p_account, p_optimizer, p_format, p_hidden, p_lr, p_seed, p_steps, 'active', 1)
  ON CONFLICT (service_id) DO UPDATE SET
    optimizer = EXCLUDED.optimizer, format = EXCLUDED.format,
    hidden = EXCLUDED.hidden, lr = EXCLUDED.lr,
    seed = EXCLUDED.seed, steps = EXCLUDED.steps, status = 'active',
    generation = ssot.scarab_strategy.generation + 1,
    updated_at = now(), updated_by = 'queen-hive'
  RETURNING generation INTO v_gen;

  INSERT INTO ssot.scarab_command (service_id, command, new_strategy, reason)
  VALUES (p_service_id, 'spawn',
          jsonb_build_object('optimizer',p_optimizer,'format',p_format,
                             'hidden',p_hidden,'lr',p_lr,'seed',p_seed,'steps',p_steps,
                             'canon_name',p_canon_name),
          p_reason);
  RETURN v_gen;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION ssot.kill_scarab(p_service_id text, p_reason text DEFAULT NULL)
RETURNS void AS $$
BEGIN
  PERFORM ssot.bump_strategy(p_service_id := p_service_id, p_status := 'killed', p_by := 'queen-hive');
  INSERT INTO ssot.scarab_command (service_id, command, reason) VALUES (p_service_id, 'kill', p_reason);
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION ssot.pause_scarab(p_service_id text, p_reason text DEFAULT NULL)
RETURNS void AS $$
BEGIN
  PERFORM ssot.bump_strategy(p_service_id := p_service_id, p_status := 'paused');
  INSERT INTO ssot.scarab_command (service_id, command, reason) VALUES (p_service_id, 'pause', p_reason);
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION ssot.resume_scarab(p_service_id text, p_reason text DEFAULT NULL)
RETURNS void AS $$
BEGIN
  PERFORM ssot.bump_strategy(p_service_id := p_service_id, p_status := 'active');
  INSERT INTO ssot.scarab_command (service_id, command, reason) VALUES (p_service_id, 'resume', p_reason);
END $$ LANGUAGE plpgsql;

-- ============================================================================
-- 8. OPERATOR VIEWS  (canonical README names + back-compat)
-- ============================================================================
-- Drop legacy 0004 view shapes (column lists differ — CREATE OR REPLACE fails)
DROP VIEW IF EXISTS ssot.scarab_dead     CASCADE;
DROP VIEW IF EXISTS ssot.scarab_drift    CASCADE;
DROP VIEW IF EXISTS ssot.scarab_lineage  CASCADE;
DROP VIEW IF EXISTS ssot.fleet_drift     CASCADE;
DROP VIEW IF EXISTS ssot.fleet_status    CASCADE;
DROP VIEW IF EXISTS ssot.strategy_history CASCADE;
DROP VIEW IF EXISTS ssot.scarab_assignment_log CASCADE;

-- fleet_status: one row per scarab — desired vs applied + heartbeat freshness + drift label
CREATE OR REPLACE VIEW ssot.fleet_status AS
SELECT
  s.service_id,
  s.status,
  s.generation                                AS desired_version,
  COALESCE(h.applied_version, 0)              AS applied_version,
  s.generation - COALESCE(h.applied_version, 0) AS version_lag,
  h.last_seen,
  CASE
    WHEN h.last_seen IS NULL THEN NULL
    ELSE EXTRACT(EPOCH FROM (now() - h.last_seen))::int
  END                                         AS heartbeat_age_s,
  CASE
    WHEN h.last_seen IS NULL                                THEN 'never_seen'
    WHEN h.last_seen < now() - interval '90 seconds'         THEN 'dead'
    WHEN s.generation > COALESCE(h.applied_version, 0)
         AND h.last_seen < now() - interval '5 minutes'      THEN 'bump_dropped'
    WHEN s.generation > COALESCE(h.applied_version, 0)       THEN 'bump_pending'
    ELSE                                                          'in_sync'
  END                                         AS drift_state
FROM ssot.scarab_strategy s
LEFT JOIN ssot.scarab_heartbeat h USING (service_id);

-- fleet_drift: only scarabs that are NOT in_sync (alert-ready)
CREATE OR REPLACE VIEW ssot.fleet_drift AS
SELECT * FROM ssot.fleet_status WHERE drift_state <> 'in_sync';

-- strategy_history: every distinct strategy a service_id has ever run
CREATE OR REPLACE VIEW ssot.strategy_history AS
SELECT
  service_id,
  strategy_fingerprint,
  MIN(issued_at)  AS started_at,
  MAX(issued_at)  AS ended_at,
  COUNT(*) FILTER (WHERE command = 'bump') AS bump_count
FROM ssot.scarab_command
WHERE strategy_fingerprint IS NOT NULL
GROUP BY service_id, strategy_fingerprint;

-- back-compat aliases (older agents still use these)
CREATE OR REPLACE VIEW ssot.scarab_drift   AS SELECT * FROM ssot.fleet_status;
CREATE OR REPLACE VIEW ssot.scarab_lineage AS SELECT * FROM ssot.strategy_history;
CREATE OR REPLACE VIEW ssot.scarab_dead AS
  SELECT service_id, status, last_seen, heartbeat_age_s AS seconds_since
  FROM ssot.fleet_status
  WHERE drift_state IN ('dead','never_seen');

-- README-canonical alias for the audit log table
CREATE OR REPLACE VIEW ssot.scarab_assignment_log AS
SELECT * FROM ssot.scarab_command;

COMMIT;

-- ============================================================================
-- END OF MIGRATION 001_scarab_strategy.sql
-- ============================================================================
