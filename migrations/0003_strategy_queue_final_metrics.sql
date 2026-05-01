-- 0003: Add final_bpb / final_step columns to strategy_queue
-- Scarab writes these after trainer completes (Bug E fix).
-- Idempotent: safe to run multiple times.

ALTER TABLE strategy_queue
  ADD COLUMN IF NOT EXISTS final_bpb  DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS final_step INTEGER;

COMMENT ON COLUMN strategy_queue.final_bpb IS 'Best BPB at end of training (written by scarab after trainer exits)';
COMMENT ON COLUMN strategy_queue.final_step IS 'Last eval step written to bpb_samples (written by scarab after trainer exits)';
