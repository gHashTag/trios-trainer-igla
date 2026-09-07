# ADR-0042 · Sovereign Scarab Pull-Loop (v3 → v4)

**Status:** Accepted · 2026-05-14
**Authors:** Trinity Queen Hive
**Anchor:** φ² + φ⁻² = 3 · TRINITY · DEFENSE 2026-06-15
**Supersedes:** ADR-CHAT-011 (push-based Railway-API control)

---

## Context

Until ADR-CHAT-011 the Trinity training fleet was controlled by a **push** model:
the Queen Hive (this Computer agent) called Railway's GraphQL API to:

1. `serviceInstanceRedeploy` to redeploy a training service;
2. `variableUpsert` to mutate hyperparameters (lr/seed/optimizer/format/hidden).

This created **three load-bearing failure modes**:

| # | Failure                                                                                       | Frequency observed                              |
|---|-----------------------------------------------------------------------------------------------|-------------------------------------------------|
| 1 | Railway PAT rotation → 401 `Not Authorized` on `variableUpsert` / `serviceInstanceRedeploy`   | Active in tickets gHashTag/trios-railway#156    |
| 2 | Queen Hive unavailable (no cron tick, no live session) ⇒ fleet stops accepting strategy bumps | Every session boundary (12+ h gaps overnight)   |
| 3 | "Cure A" healing workflow needs PAT token that just expired ⇒ healer deadlock                 | HEALER COMA at start of this session (~15 h)    |

The first two failures are **architectural**, not transient: the control plane lives outside the data plane, so any auth disruption in the control plane suspends the entire fleet.

## Decision

**The database is the control plane.**

- One table — `ssot.scarab_strategy` (PK `service_id`) — holds the *desired*
  configuration for each of the 27 eternal Railway services.
- Each Railway service runs an **eternal** Rust binary (`scarab`) that polls
  this row every `POLL_SEC` seconds (default 10) **and** subscribes to
  `LISTEN scarab_<service_id>` for low-latency push.
- When `scarab_strategy.generation > local_gen`, the scarab gracefully restarts
  its child trainer with the new hyperparameters.
- The Queen Hive operates exclusively via **three SQL functions**:
  `ssot.spawn_scarab`, `ssot.bump_strategy_v2`, `ssot.kill_scarab` (+ pause/resume helpers).
- **No Railway API calls** during normal operation. No PAT tokens needed for control.

## Architecture

```
                     ┌─────────────────────────────┐
                     │   Trinity Queen Hive (any    │
                     │   client w/ DATABASE_URL)    │
                     │                              │
                     │   SELECT spawn_scarab(…)     │
                     │   SELECT bump_strategy_v2(…) │
                     │   SELECT kill_scarab(…)      │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │  phd-postgres-ssot · Railway     │
                  │  ssot.scarab_strategy            │
                  │  ssot.scarab_heartbeat           │
                  │  ssot.scarab_command (audit)     │
                  │  ssot.scarab_result              │
                  │                                  │
                  │  Triggers:                       │
                  │   • pg_notify('scarab_<sid>')    │
                  │   • pg_notify('scarab_fleet')    │
                  │   • blast-radius guard           │
                  │   • strategy_fingerprint stamp   │
                  └──────────────┬──────────────────┘
                                 │ NOTIFY  ▲  poll
       ┌──────────────────┬──────┴────────┴──────────┬──────────────────┐
       ▼                  ▼                          ▼                  ▼
 ┌──────────┐       ┌──────────┐                ┌──────────┐      ┌──────────┐
 │ scarab-01│       │ scarab-02│      …         │ scarab-26│      │ scarab-27│
 │ trios-   │       │ trios-   │                │ trios-   │      │ trios-   │
 │ train    │       │ train    │                │ train    │      │ train    │
 └──────────┘       └──────────┘                └──────────┘      └──────────┘
```

## Production Hardening (v3 → v4)

| # | Improvement                | What it fixes                                                                         |
|---|----------------------------|---------------------------------------------------------------------------------------|
| 1 | `LISTEN` / `NOTIFY` hybrid | Push wake-up < 100 ms; poll fallback every 10 s for reliability                       |
| 2 | `applied_version` column   | Heartbeat now reports the *applied* gen → `fleet_status` view shows real drift        |
| 3 | Blast-radius guard         | Statement-level trigger: one tx cannot deactivate > ⌈active × 0.33⌉ scarabs           |
| 4 | `bypass_blast_guard` GUC   | `SET LOCAL scarab.bypass_blast_guard = 'on'` for explicit emergency ops               |
| 5 | Fibonacci seed CHECK       | `seed IN (1597, 2584, 4181, 6765, 10946, 47, 89, 144, 123)` — typo cannot escape      |
| 6 | Extended CHECK constraints | Optimizer / format / hidden / lr / steps whitelisted at the DB layer                  |
| 7 | `strategy_fingerprint`     | SHA-256 of immutable params; auto-stamped in audit log → `strategy_history` view      |

All seven have R5-verified evidence in
`/home/user/workspace/cron_tracking/sovereign_scarab/evidence/2026-05-14T10:30Z-hardening/`.

## Consequences

### Positive
- **No more PAT-rotation outages.** The Queen no longer needs Railway API credentials for steady-state operation.
- **Survives operator absence.** Eternal scarabs keep training across session
  boundaries; bumps are queued in the DB and applied on next poll/NOTIFY.
- **Audit trail by design.** Every spawn/bump/pause/kill is an `INSERT` into
  `ssot.scarab_command` with a strategy fingerprint — full forensic replay.
- **Single point of control.** A junior operator can read one table to know the
  state of the fleet (`SELECT * FROM ssot.fleet_status`).
- **Blast safety.** The 0.33 cap means an accidental wide UPDATE in a SQL client
  can't take down more than a third of the fleet without explicit override.

### Negative
- Each scarab needs its own Postgres connection. At 27 services × 1 conn = 27 simultaneous connections on the SSoT. Mitigation: connection multiplexing in v5 (pgbouncer) when fleet grows beyond ~100.
- Postgres becomes a hard dependency for *control*. If `phd-postgres-ssot` is
  down, no new strategies can be applied. Existing strategies keep running.
  Mitigation: Railway Postgres has 99.95 % uptime SLO; back-up control via
  Throne #264 manual GitHub Actions remains available.
- Initial deploy requires applying `migrations/001_scarab_strategy.sql` once.
  Idempotent — safe to re-run.

### Neutral
- Operator's mental model shifts from "redeploy this service" to "bump this
  row". Tooling and docs (this ADR + README) reflect the change.

## Falsification

This decision is **falsified** if any of the following hold after one week of production operation:

1. **Drop rate.** More than 5 % of bumps fail to apply within 60 s
   (measured via `fleet_status WHERE version_lag > 0 AND heartbeat_age_s < 90`).
2. **Connection pressure.** Connection count to phd-postgres-ssot exceeds 80 %
   of plan limit during normal operation.
3. **Bypass abuse.** `scarab.bypass_blast_guard = 'on'` used more than twice in
   one rolling 24 h window (each usage logs to `scarab_command.reason`).

If falsified, the next ADR will introduce a control-plane sidecar that
batches bumps and keeps a single multiplexed Postgres connection.

## References

- `migrations/001_scarab_strategy.sql` — consolidated SQL master.
- `src/lib.rs` + `src/main.rs` — v4 Rust binary.
- `README.md` — operator cookbook.
- Throne #264 — fleet status dashboard.
- gHashTag/trios#785 — original ADR-CHAT-011 (push-based; superseded).
- gHashTag/trios-railway#163 — base migration `0004_sovereign_scarab.sql`.
- gHashTag/trios-railway#156 — PAT rotation outage that motivated this ADR.
