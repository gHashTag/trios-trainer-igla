# TODO.md — Atomic Decisions (O(1) communication)

**Rule:** Future ME, future YOU, future agents — read only:
1. `CONSTITUTION.md` — immutable laws
2. `TODO.md` — this file, current atomic decisions

No 30-message threads. Dispute → appeal to Constitution.

---

## CURRENT ATOMIC DECISIONS

### D1: Repo name (DECIDED)
**Decision:** `gHashTag/trinity`
**Rationale:** Short, brand-aligned, reflects 3 invariants
**Status:** ✅ Done

### D2: Existing repos (DECIDED)
**Decision:** Archive `trios`, `trios-trainer-igla`, `trios-railway`, `trios-rs` — read-only
**Rationale:** Don't lose historical data (PhD chapters, ledger rows), but don't touch
**Status:** ⏳ Pending manual action

### D3: Data migration (DECIDED)
**Decision:** Only migrate rows with PhD anchor (Ch.X / INV-Y mapping) — ~10-20 rows
**Rationale:** Keep PhD-anchored experiments, drop noise (1875+ historical rows)
**Status:** ⏳ Pending migration script

---

## PENDING ATOMIC DECISIONS

### D4: Railway project structure
**Options:**
- (a) One project, 18 services (single fleet)
- (b) Multiple projects by layer (one per crate)
- (c) Hybrid — runners in project, gardener separate

**Recommendation:** (a) — one project simplifies auth and monitoring

### D5: Neon DB strategy
**Options:**
- (a) Single DB for all environments (dev/staging/prod)
- (b) Separate DB per environment
- (c) Neon branch per environment

**Recommendation:** (b) — isolation, easier rollback

### D6: CI/CD approach
**Options:**
- (a) GitHub Actions → build → push GHCR → Railway deploy
- (b) Manual reconcile via `trinity-reconcile` CLI
- (c) Railway auto-deploy on GHCR push

**Recommendation:** (a) — full automation, declarative state

---

## COMPLETED ATOMIC DECISIONS

See git history. All decisions are immutable.

---

**Last updated:** 2026-05-02
