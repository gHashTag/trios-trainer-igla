# 🚀 Zero-Downtime Deploy: Scarab Pool

## Правильный порядок (parallel deploy)

Запрещено убивать старые воркеры пока есть running эксперименты.

### Шаг 1: Миграция в Neon

```bash
psql $NEON_DATABASE_URL -f migrations/002_scarab_stateless.sql
```

### Шаг 2: Деплой новых scarab РЯДОМ со старыми

```bash
# НОВЫЕ scarab-pool — Dockerfile.scarab, SCARAB_ACCOUNT опционален
for ACC in 0 1 2 3 4 5; do
    railway --account=acc$ACC service create scarab-pool \
        --dockerfile Dockerfile.scarab
    railway --account=acc$ACC service env set \
        NEON_DATABASE_URL="$NEON_DATABASE_URL" \
        SCARAB_ACCOUNT="acc$ACC"   # опционально, только для логов
done
```

Новые scarabs сразу начнут брать задачи из `strategy_queue`.

### Шаг 3: Подождать окончания старых

```sql
-- Ожидаем пока не останется ни одного running
SELECT id, canon_name, status, started_at
FROM strategy_queue   -- (experiment_queue до миграции)
WHERE status = 'running'
ORDER BY started_at;
-- Когда выдаст 0 строк → переходить к шагу 4
```

### Шаг 4: Удалить старые воркеры

```bash
# ТОЛЬКО после того как все running = 0
for ACC in 0 1 2 3 4 5; do
    railway --account=acc$ACC service delete trios-train-v2-acc${ACC}-s1597
done
```

### Шаг 5: Масштабирование

```bash
# +1 скарабей на любом acc:
railway --account=acc0 service create scarab-pool-2 --dockerfile Dockerfile.scarab
railway --account=acc0 service env set NEON_DATABASE_URL="$NEON_DATABASE_URL"
# Он сразу начнёт брать задачи из общего пула
```

## Диагностика

```sql
-- Кто живой:
SELECT label, host, current_strategy_id, last_heartbeat
FROM scarabs
ORDER BY last_heartbeat DESC;

-- Очередь:
SELECT status, COUNT(*) FROM strategy_queue GROUP BY status;

-- Застрявшие (воркер умер, задача не завершилась):
SELECT reclaim_stale_strategies();
```

## Почему НЕЛЬЗЯ убивать старые сразу

Старые `trios-train-v2-accN` сейчас **running** эксперименты (id 1990/1992/1994, GATE-2 wave).  
Убий → потерял 81k steps воркы = время в мусор + незаписанный результат.  
Параллельный деплой — никто ничего не теряет.
