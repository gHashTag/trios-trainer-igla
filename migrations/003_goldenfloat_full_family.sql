-- ============================================================================
-- GOLDEN FLOAT FAMILY · FULL ROSTER · MIGRATION 003
-- ============================================================================
-- Anchor: phi^2 + phi^-2 = 3 · TRINITY · DEFENSE 2026-06-15
-- DOI: 10.5281/zenodo.19227877
--
-- Idempotent patch — expands ssot.scarab_strategy.format whitelist to include
-- the complete GoldenFloat family: GFTernary · GF4 · GF8 · GF12 · GF16 ·
-- GF20 · GF24 · GF32 · GF64 · GF128 · GF256.
--
-- Maps PhD chapters:
--   Glava 06 — "Golden Mantissa: GoldenFloat Family GF4--GF64"
--   Glava 09 — "Golden Seal: GF vs MXFP4 Ablation"
--   Glava 23 — "GF(16) Algebra" (Coq-proven INV-3 / gf16_safe_domain)
--
-- Why: prior whitelist only carried gf16 + gf256, blocking Queen Hive
-- experiments on the intermediate / extended formats — closes the "11 GF
-- structs in Rust, 2 GF entries in DB" mismatch.
-- ============================================================================

BEGIN;

ALTER TABLE ssot.scarab_strategy DROP CONSTRAINT IF EXISTS scarab_strategy_format_check;

ALTER TABLE ssot.scarab_strategy ADD CONSTRAINT scarab_strategy_format_check CHECK (
  -- Golden Float Family — ВСЯ семья (PhD Glava 06 / 09 / 23)
  format IN (
    -- Classical IEEE binary{16,32,64,128,256}
    'fp16','fp32','fp64','fp80','bf16',
    'binary16','binary128','binary256',
    -- FP8 / FP6 / FP4 OCP variants + MXFP
    'fp8_e4m3','fp8_e5m2','mxfp8',
    'fp6_e2m3','fp6_e3m2','fp4_e2m1',
    -- Golden Float Family — every member
    'gfternary',
    'gf4','gf8','gf12','gf16','gf20','gf24',
    'gf32','gf64','gf128','gf256',
    -- Posit / NF / integer
    'posit8','posit16',
    'int4','int8','uint8',
    'nf4'
  )
);

-- Audit row in scarab_command so the strategy_history view records the schema bump
INSERT INTO ssot.scarab_command (service_id, command, reason, issued_by)
SELECT '__schema__', 'bump',
       'goldenfloat-fill-gaps: full family whitelist (gfternary + gf4/12/20/24/128 added; gf32/64/8 already produced upstream)',
       'goldenfloat-fill-gaps'
WHERE NOT EXISTS (
  SELECT 1 FROM ssot.scarab_command
  WHERE service_id = '__schema__' AND reason LIKE 'goldenfloat-fill-gaps%'
);

COMMIT;

-- ============================================================================
-- END 003_goldenfloat_full_family.sql
-- ============================================================================
