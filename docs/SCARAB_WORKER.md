# 🪲 Scarab Worker — Stateless Fungible Pool

> **Правило №1:** Контейнер не трогаем НИКОГДА.  
> **Правило №2:** Стратегию меняем ТОЛЬКО через Neon.  
> **Правило №3:** Нет account affinity. Любой скарабей берёт любую задачу.

## Принцип работы

```
loop forever:
  drain:  # выбираем все pending задачи поочередно
    row = SELECT ... FROM strategy_queue
          WHERE status = 'pending'      # ← НЕТ AND account = $1 !
          ORDER BY priority DESC, id ASC
          FOR UPDATE SKIP LOCKED
          LIMIT 1
    if row:
      UPDATE status='running', worker_id=$HOSTNAME
      spawn: trios-igla train <params from config_json>
      UPDATE status='done' / 'failed'
    else:
      break
  # Ждём NOTIFY или 30 с fallback
  tokio::select! { notify_rx.recv(), sleep(30s) }
```

## Что поменялось по сравнению

| | Старое | Новое |
|---|---|---|
| acc filter | `AND account = $1` | ✘ удалён |
| acc3 умер | seed45 ждёт acc3 | acc5 берёт seed45 за 30с |
| Scale | 6 service definitions | 1 image, N replicas |
| Изменение стратегии | редеплой | INSERT в Neon |
| Hot-polling | каждые 10с | NOTIFY + 30с fallback |
| Watchdog | нет | `max_runtime_sec` в spec |

## Strategy Spec Format (JSONB)

```json
{
  "trainer": {
    "hidden": 1024,
    "lr": 0.002,
    "steps": 81000,
    "ctx": 12,
    "format": "fp32",
    "seed": 42,
    "val_split_seed": "0xDEADBEEF"
  },
  "constraints": {
    "max_runtime_sec": 900,
    "min_step_for_done": 9000
  },
  "submission": {
    "track": "non_record_16mb",
    "tags": ["GATE3", "phi-foundation"]
  }
}
```

Старый легаси-формат `{"hidden":828, "lr":0.0004, ...}` **тоже работает** автоматически.

## Деплой на Railway

```bash
# Один сервис — N реплик
# единственный обязательный ENV:
NEON_DATABASE_URL=postgres://...

# опционально для логов:
SCARAB_ACCOUNT=acc0   # не влияет на scheduling

# Масштабирование = добавить сервис в Railway с тем же Dockerfile
# И всё. Больше ничего не нужно.
```

## Управление через Neon

```sql
-- Добавить новую стратегию:
INSERT INTO strategy_queue (canon_name, priority, steps_budget, config_json)
VALUES (
  'GATE3-h1024-lr002-seed42', 100, 81000,
  '{"trainer":{"hidden":1024,"lr":0.002,"steps":81000,
    "ctx":12,"format":"fp32","seed":42},
   "constraints":{"max_runtime_sec":900},
   "submission":{"track":"non_record_16mb","tags":["GATE3"]}}'
);
-- Триггер автоматически будит всех спящих скарабеев через NOTIFY

-- Повысить приоритет:
UPDATE strategy_queue SET priority = 200
WHERE canon_name LIKE '%GATE3%' AND status = 'pending';

-- Отменить:
UPDATE strategy_queue SET status = 'pruned' WHERE id = 9999;

-- Ручной сброс застрявших running:
SELECT reclaim_stale_strategies();

-- Мониторинг:
SELECT label, host, current_strategy_id, last_heartbeat
FROM scarabs
ORDER BY last_heartbeat DESC;
```

## Миграция

```bash
psql $NEON_DATABASE_URL -f migrations/002_scarab_stateless.sql
```

## ENV переменные

| Переменная | Обязательна | Описание |
|---|---|---|
| `NEON_DATABASE_URL` | ✅ | postgres://user:pass@host/db |
| `SCARAB_ACCOUNT` | ✘ | Только для логов, не влияет на scheduling |
| `HOSTNAME` | авто | Railway/Docker подставляет сам |

## Что нельзя менять (только при критическом багфиксе)

- `Dockerfile.scarab`
- `src/bin/scarab.rs`
- `migrations/002_scarab_stateless.sql`

## Что изменять всегда через Neon

- `strategy_queue` — INSERT новых стратегий
- `priority` — управление очередью
- `config_json` — все гиперпараметры
