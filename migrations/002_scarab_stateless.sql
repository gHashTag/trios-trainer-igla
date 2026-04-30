-- 🪲 Migration 002: Stateless Scarab Pool
-- Запуск: один раз, необратимо.

BEGIN;

-- 1. Переименовывание workers -> scarabs
ALTER TABLE IF EXISTS workers RENAME TO scarabs;

ALTER TABLE scarabs
    ADD COLUMN IF NOT EXISTS scarab_id UUID DEFAULT gen_random_uuid(),
    ADD COLUMN IF NOT EXISTS host TEXT,
    ADD COLUMN IF NOT EXISTS label TEXT,
    ADD COLUMN IF NOT EXISTS current_strategy_id BIGINT;

-- Удаляем acc-привязку если была
ALTER TABLE scarabs DROP COLUMN IF EXISTS account;
ALTER TABLE scarabs DROP COLUMN IF EXISTS railway_acc;

-- 2. Переименовывание experiment_queue -> strategy_queue
ALTER TABLE IF EXISTS experiment_queue RENAME TO strategy_queue;

-- Добавляем worker_id для трейсировки
-- (hostname воркера, который взял задачу)
ALTER TABLE strategy_queue
    ADD COLUMN IF NOT EXISTS worker_id TEXT,
    ADD COLUMN IF NOT EXISTS error_msg TEXT;

-- Удаляем acc-привязку (больше не нужна)
ALTER TABLE strategy_queue DROP COLUMN IF EXISTS account;

-- 3. Fast claim index
CREATE INDEX IF NOT EXISTS idx_strategy_pending
    ON strategy_queue (priority DESC, id ASC)
    WHERE status = 'pending';

-- 4. NOTIFY trigger: будит всех спящих скарабеев
CREATE OR REPLACE FUNCTION notify_strategy_new() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('strategy_new', NEW.id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_strategy_new ON strategy_queue;
CREATE TRIGGER trg_strategy_new
    AFTER INSERT OR UPDATE OF status ON strategy_queue
    FOR EACH ROW WHEN (NEW.status = 'pending')
    EXECUTE FUNCTION notify_strategy_new();

-- 5. Обновляем застревшие running -> pending чтобы они были переприняты
-- (воркер умер — задача не завершилась — сброс через 1 ч)
CREATE OR REPLACE FUNCTION reclaim_stale_strategies() RETURNS void AS $$
BEGIN
    UPDATE strategy_queue
    SET status = 'pending', worker_id = NULL, started_at = NULL
    WHERE status = 'running'
      AND started_at < NOW() - INTERVAL '1 hour';
END;
$$ LANGUAGE plpgsql;

-- Вызываем сразу чтобы возобновить всё что было в running
SELECT reclaim_stale_strategies();

COMMIT;

-- Верификация:
-- SELECT COUNT(*) FROM strategy_queue WHERE status = 'pending';
-- SELECT * FROM scarabs ORDER BY registered_at DESC LIMIT 5;
