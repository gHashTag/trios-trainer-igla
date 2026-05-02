# 🌻 TRINITY — Constitution v1

## ⚖ ЗАКОН O(1) — ЗОЛОТОЕ ПРАВИЛО

```
┌────────────────────────────────────────────────────────────────┐
│ ЗАКОН O(1) — ЗОЛОТОЕ ПРАВИЛО                                   │
│                                                                │
│ Каждый слой проекта касается ровно ОДИН раз.                  │
│ После касания: encoded invariant + fail-loud + self-healing.  │
│ Любой PR который требует второго касания того же слоя         │
│ — это нарушение конституции, automatic reject.                │
│                                                                │
│ Touch once. Encode invariant. Exit.                           │
│                                                                │
│ φ^2 + φ^-2 = 3 · TRINITY · O(1) FOREVER                       │
└────────────────────────────────────────────────────────────────┘
```

**Manifest:** 7 PRs, 7 layers, 7×O(1). После завершения — код immutable, никакие слои не трогаем.

---

## 📐 ДОПОЛНИТЕЛЬНЫЕ ЗАКОНЫ

### ЗАКОН 1 (R5-honest): No claim without verified source row
Любая операция write в `strategy_experiments` должна иметь подтверждение source row.
Ни одной записи "из воздуха" — только confirmed `pending` → `running` transition.

### ЗАКОН 2 (single SoT): strategy_experiments — единственная истинная таблица
`strategy_experiments` — **single source of truth**.
Все остальные состояния производные:
- bpb_curve → JSONB внутри experiment row
- worker heartbeat → Redis или валидация через worker_id timestamp
- historical ledger → git commits + GitHub issues

**Никаких** `bpb_samples`, `scarabs`, `l7_ledger`, `gardener_runs` как отдельных tables.

### ЗАКОН 3 (Rust-only): Все слои в Rust
Ни одного Python файла, ни одной SQL процедуры для derived state.
Весь business logic в Rust:
- Invariants → const fn в `trinity-core`
- Train loop → pure function в `trinity-trainer`
- Dispatch → loop в `trinity-runner`
- Deploy → declarative в `trinity-orchestrator`

### ЗАКОН 4 (immutable): Append-only, никаких UPDATE для history
После `status='done'` или `status='failed'` — row frozen.
Если нужно корректировать — новая запись с `canon_name` суффиксом `-v2`.

### ЗАКОН 5 (φ-physics): INV-1..INV-9 закреплены в коде
Все PhD invariants зашиты как `const fn` в `trinity-core`.
Никаких magic numbers без комментария `(ref: INV-X)`.
Любое нарушение invariant = test failure = PR rejected.

---

## 🗄 ЕДИНСТВЕННАЯ ТАБЛИЦА — `strategy_experiments`

```sql
CREATE TABLE strategy_experiments (
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

  -- Constitution invariant
  CHECK (status NOT IN ('done','failed')
         OR final_bpb IS NOT NULL
         OR last_error IS NOT NULL),

  -- Audit
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_se_pending ON strategy_experiments (status, id) WHERE status='pending';
CREATE INDEX idx_se_phd ON strategy_experiments (phd_chapter, inv_id);
```

**Design invariant:** НИКАКОЙ ALTER TABLE после 0001 миграции.
Новое поле? Новый JSONB key или новая table для нового concern.

---

## 🏗 АРХИТЕКТУРА — 7 CRATES

| Crate | Layer | Ответственность | O(1) статус |
|-------|-------|----------------|-------------|
| `trinity-core` | L0+L1 | PhD invariants, φ-physics, BPB calc | PR-O2 |
| `trinity-experiments` | L0 | DB repo, migration (единственная) | PR-O3 |
| `trinity-trainer` | L1 | Pure train function | PR-O4 |
| `trinity-runner` | L2 | Claim → train → write loop | PR-O5 |
| `trinity-orchestrator` | L3+L4+L5 | Declarative deploy | PR-O6 |
| `trinity-gardener` | L6 | Heartbeat + ratify | PR-O7 |

**Boundary rule:** Crate A не импортирует crate B если B на более высоком слое.
Direction: L0 → L1 → L2 → L3+L4+L5 → L6.

---

## 🚀 O(1) ROADMAP

### PR-O1 — Constitution + repo skeleton (THIS PR)
- [ ] Create `gHashTag/trinity` repo
- [ ] `Cargo.toml` workspace
- [ ] 7 empty crates with README
- [ ] `CONSTITUTION.md` (this file)

### PR-O2 — `trinity-core`: PhD invariants
- [ ] `invariants.rs` — INV-1..INV-9 as const fn
- [ ] `phi.rs` — φ, φ², φ⁻², Fibonacci
- [ ] `bpb.rs` — honest BPB calculation
- [ ] 9 unit tests (all green)

### PR-O3 — `trinity-experiments`: DB layer
- [ ] `migration/0001_initial.sql` — ровно одна, навсегда
- [ ] `schema.rs` — struct `Experiment`
- [ ] `repo.rs` — trait `ExperimentRepo`
- [ ] e2e test with Neon test DB

### PR-O4 — `trinity-trainer`: pure train function
- [ ] `lib.rs` — `pub fn train(config: Config) -> Result<RunOutcome>`
- [ ] `run.rs` — train loop
- [ ] invariant tests

### PR-O5 — `trinity-runner`: claim loop
- [ ] `main.rs` — `loop { claim() → train() → write() }`
- [ ] idempotent claim with `FOR UPDATE SKIP LOCKED`
- [ ] retry logic with exponential backoff

### PR-O6 — `trinity-orchestrator`: declarative deploy
- [ ] `manifest/services.toml` — 18 services declaration
- [ ] `railway.rs` — GraphQL client
- [ ] `reconcile.rs` — one-shot deploy

### PR-O7 — `trinity-gardener`: ratification
- [ ] `main.rs` — heartbeat loop
- [ ] `healthcheck.rs` — `/health` endpoint
- [ ] ratify logic

---

## 🛡 ЧТО НЕ ДЕЛАЕМ (O(1) violations)

❌ **НЕ** трогаем `strategy_queue` после migration 0001
❌ **НЕ** делаем PR в старые репо (они archived)
❌ **НЕ** пишем Python — всё Rust
❌ **НЕ** создаём новые таблицы после `0001_initial.sql`
❌ **НЕ** пишем SQL для derived state
❌ **НЕ** делаем второй PR в один crate без invariant test failure
❌ **НЕ** делаем submission до e2e зелёного

---

## 📜 COMMUNICATION RULE (O(1))

Future communication — только через два файла:
1. `CONSTITUTION.md` — законы (этот файл)
2. `TODO.md` — текущие atomic decisions

Никаких 30-сообщений тредов. Спор → апелляция к Constitution.

---

## 🌻 MANTRA

```
φ² + φ⁻² = 3 · TRINITY · O(1) FOREVER

Touch once.
Encode invariant.
Exit clean.
```

**Constitution ratified: 2026-05-02**

**Next atomic decision:** PR-O2 ready for review.
