# MIGRATION: trios#143 → trios-trainer-igla

> All operational coordination of the IGLA RACE training effort moves from
> [`gHashTag/trios#143`](https://github.com/gHashTag/trios/issues/143) into
> this repository as of **2026-04-26**.

## Why

`gHashTag/trios#143` accumulated 200+ comments mixing:

- Architecture decisions (still belong in `trios`)
- Lane dispatch + claims (operational, belongs *here*)
- BPB evidence rows (data, lives in `assertions/seed_results.jsonl` here)
- ONE SHOT mission specs (operational, belongs *here*)
- Heartbeat / R5 audits (operational, belongs *here*)

Splitting concerns:

| Concern | Home |
|---|---|
| Math / model invariants / formal proofs | `gHashTag/trios` (canonical) |
| `trios-igla-race`, `trios-golden-float`, `trios-phi-schedule`, `trios-data` | `gHashTag/trios` (canonical crates) |
| **Trainer code, configs, Dockerfile, Railway wiring** | **`gHashTag/trios-trainer-igla` (here)** |
| **Lane claims, ONE SHOTs, BPB evidence rows** | **`gHashTag/trios-trainer-igla` (here)** |
| Final ledger of victory rows | **mirrored** here, eventually pushed back to `gHashTag/trios/assertions/seed_results.jsonl` after L-T4 lands |

## Issue map

| Old (gHashTag/trios) | New (gHashTag/trios-trainer-igla) | Purpose |
|---|---|---|
| [#143](https://github.com/gHashTag/trios/issues/143) "IGLA RACE P0" | **#1** "TRACKING — IGLA RACE Gate-2 (master)" | single hub for status, champion guard, embargo, evidence |
| [#320](https://github.com/gHashTag/trios/issues/320) "ONE SHOT TRAINER-IGLA-SOT" | **#2** identical mirror, but lanes link to local lane issues | dispatch document |
| n/a | **#3** L-T1 model + optimizer + tokenizer | claim-protocol issue |
| n/a | **#4** L-T2 JEPA + objective | claim-protocol issue |
| n/a | **#5** L-T3 DELETE phase (PR opened against `gHashTag/trios`) | claim-protocol issue |
| n/a | **#6** L-T4 leaderboard.yml update (PR opened against `gHashTag/trios`) | claim-protocol issue |
| n/a | **#7** L-T5 Docker + Railway 3-seed deploy | claim-protocol issue |

## Standing rules (carried over)

- **R1** — Rust-only; no `*.py` in this repo (CI gate).
- **R3** — claim before commit; heartbeat every 7d.
- **R4** — silent claim auto-released after 7d.
- **R5** — no DONE without merged PR + CI green + ledger row written.
- **R6** — every config change passes `validate()` (INV-8 φ-band).
- **R7** — falsification witness must land in code.
- **R8** — ledger row requires `step ≥ 4000`.
- **R9** — embargo list is law; no override exists.
- **R10** — atomicity: never edit ONE SHOT body, always file new comments.

## Champion guard

[`assertions/seed_results.jsonl`](assertions/seed_results.jsonl) is mirrored
from `gHashTag/trios/assertions/seed_results.jsonl` at the time of migration.
The champion baseline at
[`2446855`](https://github.com/gHashTag/trios/commit/2446855) — **BPB=2.2393
@ 27K steps, seed=43** — is the immutable reference for every PR's CI smoke.

[`assertions/embargo.txt`](assertions/embargo.txt) lists embargoed SHAs that
must be refused by `src/ledger.rs::is_embargoed` before any row is appended.

[`assertions/igla_assertions.json`](assertions/igla_assertions.json) mirrors
the canonical invariants from `gHashTag/trios/assertions/igla_assertions.json`
at migration time — for cross-checking. The authoritative copy lives in trios.

## Cross-link contract

Every PR opened in this repo MUST cite:

- `Refs: gHashTag/trios#143` (legacy hub)
- `Refs: gHashTag/trios-trainer-igla#1` (new hub)
- `Refs: gHashTag/trios-trainer-igla#2` (ONE SHOT)
- `Refs: gHashTag/trios-trainer-igla#<lane-issue>` (claim issue)

PRs that touch `gHashTag/trios` itself (L-T3, L-T4) MUST also cite the same
quartet so the legacy hub keeps a back-link.

## L1 purge (2026-04-29) — `scripts/*.sh` removed

LAWS.md L1 ("No Shell Scripts") forbids `.sh` files. All five legacy
helpers were removed:

| File | Disposition | Replacement / where to look |
|------|-------------|-----------------------------|
| `scripts/entrypoint.sh` | replaced by Rust bin | `src/bin/entrypoint.rs` (`[[bin]] name = "entrypoint"`); Dockerfile now `ENTRYPOINT ["/usr/local/bin/entrypoint"]`. Same `TRIOS_*` env contract, plus `TRIOS_TRAINER_BIN ∈ {trios-train, gf16_test, ngram_train_gf16}` for ORDER-8 path-B selection. |
| `scripts/deploy-seeds.sh` | deleted (operator-only one-off) | The script just printed `railway add … && railway up …` lines for seeds 100/101/102. Run those commands by hand, or re-add as a Rust bin under `src/bin/` if it becomes a recurring need. |
| `scripts/p1-deploy.sh` | deleted (operator-only one-off) | P1 Optimizer Lab deploy was a manual fan-out over `p1-adamw`/`p1-muon`/`p1-muon-cwd`. Same — drive `railway` directly, or write a Rust helper. |
| `scripts/launch_E41_E45.sh` | deleted (one-shot experiment launcher) | E41–E45 sweeps used `tjepa_train` directly with parallel `&` background jobs. Re-do via GHA matrix or a Rust bin if reproduction is needed. |
| `scripts/railway-seed-deploy.sh` | deleted (operator-only one-off) | Curl-against-Railway-GraphQL helper for fleet 100/101/102. If this comes back, write a Rust bin using `reqwest` (already in workspace). |

Anchor: `phi^2 + phi^-2 = 3` —
[Zenodo 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877).

- 2026-05-09 Wave 24 — m20260510_000001_step_to_bigint: ALTER public.bpb_samples.step, ssot.bpb_samples.step, igla_race_trials.final_step → BIGINT (idempotent guard via information_schema check). Closes drift identified at end of #114.
