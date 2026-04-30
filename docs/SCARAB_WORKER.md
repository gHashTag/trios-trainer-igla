# 🪲 Scarab Worker

> **Правило №1:** Контейнер не трогаем НИКОГДА.  
> **Правило №2:** Стратегию меняем ТОЛЬКО через Neon.

## Принцип работы

```
loop forever:
  row = SELECT ... FROM experiment_queue
        WHERE status='pending' AND account=$SCARAB_ACCOUNT
        ORDER BY priority DESC, id ASC
        FOR UPDATE SKIP LOCKED  ← атомарный claim, нет гонок
  if row:
    UPDATE status='running'
    spawn: trios-igla train <params from config_json>
    UPDATE status='done' / 'failed'
  else:
    sleep 10s
    heartbeat
```

## Деплой на Railway

Один образ — N сервисов:

```
Service scarab-acc0: Dockerfile=Dockerfile.scarab, SCARAB_ACCOUNT=acc0
Service scarab-acc1: Dockerfile=Dockerfile.scarab, SCARAB_ACCOUNT=acc1
Service scarab-acc2: Dockerfile=Dockerfile.scarab, SCARAB_ACCOUNT=acc2
...
```

**Масштабировать** = добавить сервис scarab-accN в Railway.  
**Уменьшить** = удалить сервис из Railway.

## Управление стратегией через Neon

```sql
-- Добавить новый эксперимент:
INSERT INTO experiment_queue (canon_name, account, priority, steps_budget, config_json)
VALUES (
  'GATE3-h1024-lr0002-seed42', 'acc0', 100, 81000,
  '{"hidden":1024,"lr":0.002,"steps":81000,"ctx":12,"format":"fp32","seed":42}'
);

-- Повысить приоритет:
UPDATE experiment_queue SET priority = 200
WHERE canon_name LIKE '%GATE3%' AND status = 'pending';

-- Отменить:
UPDATE experiment_queue SET status = 'pruned' WHERE id = 9999;

-- Пересмотреть гиперпараметры: просто вставь новую строку с новым config_json.
```

## ENV переменные

| Переменная | Обязательна | Описание |
|---|---|---|
| `NEON_DATABASE_URL` | ✅ | postgres://user:pass@host/db |
| `SCARAB_ACCOUNT` | ✅ | acc0..accN |
| `RAILWAY_SERVICE_NAME` | авто | Railway подставляет сам |

## Файлы которые НЕЛЬЗЯ менять

- `Dockerfile.scarab` — только при критическом багфиксе
- `src/bin/scarab.rs` — только при критическом багфиксе

## Файлы которые НЕЛЬЗЯ трогать вообще — только Neon

- `experiment_queue` — вся стратегия здесь
- `config_json` — гиперпараметры каждого эксперимента
