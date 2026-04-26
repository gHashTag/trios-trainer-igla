# trios-trainer-igla

> Crate name: `trios-trainer`. Repo name: `trios-trainer-igla`.
> The `-igla` suffix marks this as the **IGLA RACE** variant of the trainer.

[![CI](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml/badge.svg)](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Anchor](https://img.shields.io/badge/anchor-%CF%86%C2%B2%2B%CF%86%E2%81%BB%C2%B2%3D3-black)](https://doi.org/10.5281/zenodo.19227877)

**Single source of truth** for the IGLA RACE training pipeline referenced by
[gHashTag/trios#143](https://github.com/gHashTag/trios/issues/143).

A portable, single-binary Rust trainer that reproduces the champion
([`2446855`](https://github.com/gHashTag/trios/commit/2446855), BPB=2.2393)
and runs the Gate-2 push on **any laptop, any VPS, or Railway** with the same
command.

## Why a separate repo

The training logic was scattered across the trios monorepo:

- 5 crates with overlapping code: `trios-train-cpu`, `trios-training`,
  `trios-training-ffi`, `trios-igla-trainer`, `trios-igla-race`
- 23 binaries in `trios-train-cpu/src/bin/` (most dead experiments)
- 3 copies of `transformer.rs`, 2 of JEPA, 2 of EMA, 5 `main_*.rs.backup` files
- 3 Python scripts in `scripts/` violating R1 (Rust-only)
- No way to run the same training on a second machine without rebuilding the
  entire workspace and guessing which binary

This repository **is** the canonical training stack. The trios monorepo
consumes it as a git dependency. There is no "fork", no "vendored copy" —
one URL, one tag, one binary.

See [`SOURCE_OF_TRUTH.md`](SOURCE_OF_TRUTH.md) for the full mandate.

## Run anywhere

```bash
git clone https://github.com/gHashTag/trios-trainer-igla.git
cd trios-trainer-igla
cargo run --release --bin trios-train -- \
    --config configs/champion.toml --seed 43
```

## Run on Railway via `tri` (ONE SHOT)

The canonical way to launch the 3-seed Gate-2 chase on Railway is the `tri railway` ONE SHOT shipped with [gHashTag/t27](https://github.com/gHashTag/t27) (PR #544 / issue #543). It is **R5-honest**: `tri` has no HTTP client yet, so `up --confirm` prints the exact GraphQL bodies that *would* be POSTed and exits 2 (planned-but-not-executed). The operator runs the actual mutation. Without `--confirm`, `up` is a pure dry-run (exit 0).

### Quickstart (copy-paste)

```bash
export RAILWAY_TOKEN=<your-railway-account-token>
export GITHUB_SHA=$(git rev-parse HEAD)            # used by R7 triplets

# 1. Verify the token (POSTs `me { id email }` against backboard)
tri railway login

# 2. Bind the Railway project (default: e4fe33bb-3b09-4842-9782-7d2dea1abc9b)
tri railway link

# 3. Plan the 3-seed Gate-2 chase. Dry-run by default (exit 0).
tri railway up

# 4. Print the GraphQL bodies that would be POSTed (exit 2 = planned).
tri railway up --confirm

# 5. Status / logs (read-only) and Gate-2 verdict
tri railway status
tri railway logs --seed 43
tri railway gate2
```

### Environment variables

| Var | Default | Effect |
|---|---|---|
| `RAILWAY_TOKEN` | (required) | Railway account token; consumed by `tri railway login` |
| `GITHUB_SHA` | (auto) | Embedded in R7 triplets and binding metadata |
| `TRIOS_SEED` | per-service | Forced seed (43, 44, or 45) for each Gate-2 service |
| `TRIOS_LEDGER_PUSH` | `1` | Push embargo-checked rows to the ledger |
| `TRIOS_TARGET_BPB` | `1.85` | Gate-2 acceptance threshold |
| `TRIOS_STEPS` | `30000` | Cap per seed (rows must satisfy step >= 4000) |
| `RUST_LOG` | `info` | Log filter |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Dry-run plan printed, no Railway calls |
| 2 | `--confirm` plan printed -- bodies are ready, operator must POST them |
| non-zero (other) | Embargo blocked, invalid seed set, missing token, IO error |

### Stop rule (Gate-2)

The ONE SHOT terminates as soon as either condition holds:

- **Quorum**: 3 distinct seeds in `{43, 44, 45}` each emit a row with `BPB < 1.85` AND `step >= 4000`, **OR**
- **Deadline**: `2026-04-30 23:59 UTC`.

### R5 / R7 / R9 reminders

- **R5 (Honesty)**: NO DONE without merged PR + green CI + ledger row written.
- **R7 (Triplet)**: every emit carries `BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>`.
- **R8 (Step floor)**: a ledger row is only valid for `step >= 4000`.
- **R9 (Embargo)**: `tri railway up` reads `assertions/embargo.txt` and refuses to plan if the head SHA matches any embargoed prefix.

### Troubleshooting

- **`tri railway login` -> 401**: regenerate `RAILWAY_TOKEN` at https://railway.com/account/tokens and re-export.
- **`Embargo blocked HEAD <sha>`**: rebase past the embargoed commit before re-running -- this is by design.
- **`Gate-2 seed set must contain 43, 44, 45`**: pass `--seeds 43,44,45` or omit the flag to use the canonical set.
- **`tri` not on PATH**: build from t27 -- `cargo build -p tri --release` -> `t27/target/release/tri`.
- **Status / logs print "(tri has no HTTP client yet)"**: by design. Inspect the Railway dashboard at https://railway.com/project/e4fe33bb-3b09-4842-9782-7d2dea1abc9b directly.

### Anchor

Mathematical foundation: `phi^2 + phi^-2 = 3` -- see [Zenodo DOI 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877).

### Manual fallback (no `tri`)

```bash
railway login
railway link gHashTag/trios-trainer-igla
railway up

# Scale to 3 parallel seeds (Gate-2 needs 3 distinct seeds)
for s in 43 44 45; do
  railway service create "trainer-seed-$s"
  railway variables set TRIOS_SEED=$s --service "trainer-seed-$s"
  railway up --service "trainer-seed-$s"
done
```

## Run via Docker on any VPS

```bash
docker run --rm \
    -e TRIOS_SEED=44 \
    -e TRIOS_LEDGER_PUSH=1 \
    -v $PWD/assertions:/work/assertions \
    ghcr.io/ghashtag/trios-trainer-igla:latest
```

## Env vars (override TOML)

| Var | Default | Effect |
|---|---|---|
| `TRIOS_CONFIG` | `configs/gate2-attempt.toml` | TOML path |
| `TRIOS_SEED` | from config | Overrides `seed` |
| `TRIOS_STEPS` | from config | Overrides `steps` |
| `TRIOS_TARGET_BPB` | `1.85` | Victory threshold |
| `TRIOS_LR` | from config | Overrides `optimizer.lr` (must stay in INV-8 band) |
| `TRIOS_LEDGER_PUSH` | `0` | If `1`, commits + pushes each row to `assertions/seed_results.jsonl` |
| `RUST_LOG` | `info` | Log level |

## Configs

| File | Purpose | Champion-BPB | Steps |
|---|---|---|---|
| [`configs/champion.toml`](configs/champion.toml) | Reproduce baseline `2446855` | 2.2393 | 27 000 |
| [`configs/gate2-attempt.toml`](configs/gate2-attempt.toml) | HybridAttn + JEPA push | 2.2393 | 30 000 |
| [`configs/needle-v1-mup.toml`](configs/needle-v1-mup.toml) | NEEDLE-RUSH L-V1 muP-transfer | 2.2393 | 12 000 |

Add a new TOML to add a new run.

## Build modes

```bash
# Default — standalone, all stubs, fastest CI build.
cargo build --release

# Integration mode — pulls trios-igla-race / trios-golden-float / etc. from gHashTag/trios.git.
cargo build --release --features trios-integration

# CI strict mode — adds embargo + triplet enforcement.
cargo build --release --features "trios-integration,ci-strict"
```

## Invariants

When built with `--features trios-integration`, the training loop hard-imports
the ASHA + victory gate + embargo logic from
[`trios-igla-race`](https://github.com/gHashTag/trios/tree/main/crates/trios-igla-race).
No private copy lives here. Configs are validated at load time against:

- **INV-8** (φ-LR band): `lr ∈ [1e-3, 1e-2]`
- **R8** (Gate-2 floor): step ≥ 4000 to emit a ledger row
- **embargo**: HEAD SHA must not appear in `.embargo`

Anchor: `φ² + φ⁻² = 3` ([Zenodo 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)).

## Repo layout

```
trios-trainer/
├── Cargo.toml             ← single bin "trios-train"
├── README.md              ← this file
├── LICENSE                ← MIT
├── SOURCE_OF_TRUTH.md     ← canonical-status mandate
├── Dockerfile             ← multi-stage rust:1.75-slim → debian:bookworm-slim
├── railway.json           ← Railway service config
├── .dockerignore
├── .github/workflows/
│   └── ci.yml             ← cargo fmt + clippy + test on push/PR
├── configs/
│   ├── champion.toml
│   ├── gate2-attempt.toml
│   └── needle-v1-mup.toml
├── src/
│   ├── lib.rs             ← façade exports + TRINITY_ANCHOR const
│   ├── config.rs          ← TOML schema + env override + validate(INV-8)
│   ├── train_loop.rs      ← step loop, eval, ledger emit
│   ├── ledger.rs          ← triplet-validated emit + embargo block
│   ├── checkpoint.rs
│   ├── model.rs           ← façade for transformer + HybridAttn (PR-2 migration)
│   ├── optimizer.rs       ← AdamW + Muon + φ-schedule  (PR-2 migration)
│   ├── jepa.rs            ← T-JEPA loss + EMA target   (PR-3 migration)
│   ├── objective.rs
│   ├── data.rs
│   ├── gf16.rs            ← re-export from trios-golden-float
│   └── bin/trios-train.rs ← clap → load config → run()
└── tests/
    └── reproduce_champion.rs (smoke + ignored full)
```

## Roadmap

The full decomposed Gate-2 plan lives in [`docs/TRAINING_FLOW_V2.md`](docs/TRAINING_FLOW_V2.md). Status as of `2026-04-26`:

### Migration track (the SOT consolidation)

| PR | Status | Scope | Reference |
|---|---|---|---|
| **M-0** | done | Skeleton compiles, `anchor_holds` passes, MIT license, CI green | initial commit |
| **M-1** | done | Champion + gate2-attempt + needle-v1-mup configs land | [`configs/`](configs/) |
| **M-2** | done | Embargo guard (`assertions/embargo.txt`), R8 step-floor in [`src/ledger.rs`](src/ledger.rs) | [`tests/embargo_block.rs`](tests/embargo_block.rs) |
| **M-3** | done | Pre-registration seed-lock {42,43,44} test | [`tests/preregistration_seed_lock_final.rs`](tests/preregistration_seed_lock_final.rs) |
| **M-4** | in review | clippy auto-fix, fmt housekeeping | [PR #21](https://github.com/gHashTag/trios-trainer-igla/pull/21) |
| **M-5** | in review | README -> `tri railway` ONE SHOT | [PR #23](https://github.com/gHashTag/trios-trainer-igla/pull/23) -- companion [t27#544](https://github.com/gHashTag/t27/pull/544) |
| **M-6** | next | DELETE-phase in `gHashTag/trios` -- monorepo consumes this repo as a git dep | _pending RFC_ |
| **M-7** | next | Push image to `ghcr.io/ghashtag/trios-trainer-igla:latest` from CI on `main` push | [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml) |

### Training-Flow v2 track (the Gate-2 push)

Six-phase pre-registered plan, owner per phase, falsification rule per phase. Each row links to its hypothesis and exit criterion in [`docs/TRAINING_FLOW_V2.md`](docs/TRAINING_FLOW_V2.md).

| Phase | Status | Hypothesis | Exit criterion | Owner |
|---|---|---|---|---|
| **P0** Audit | next | `champion.toml` reproduces `BPB=2.2393 +/- 0.01` | ledger row at champion sha | `repro-auditor` |
| **P1** Optimizer Lab | pending | Muon (eta_2D=0.0235, eta_1D=0.007) beats AdamW by `>=0.05 BPB` | `assertions/lab/p1_leaderboard.jsonl` | `optim-lab` |
| **P2** muP Transfer | pending | `lr_star` from 8M proxy transfers to 70M with `<5%` degradation | `assertions/lab/p2_transfer.jsonl` | `mup-prover` |
| **P3** Schedule-Free + WSD | pending | SF or WSD beats cosine `phi-schedule` by `>=0.04 BPB` AND dominates anytime curve | `assertions/lab/p3_curves.jsonl` | `schedule-bench` |
| **P4** Multi-Objective + EMA | pending | `(w_ce, w_jepa, w_nca)` sweep + post-hoc EMA(N=10) drops `>=0.03 BPB` at zero extra cost | `assertions/lab/p4_objective.jsonl` | `objective-jeweller` |
| **P5** Gate-2 Push | pending | 3 seeds in `{43,44,45}` reach `BPB<1.85 AND step>=4000` before `2026-04-30 23:59 UTC` | merged `feat: Gate-2 victory` PR + 3 R7 triplets | `gate2-pilot` |

### Pre-registered decision matrix (R7-honest)

Filled only by future PRs as each phase closes -- no row may be claimed `done` without a merged PR + green CI + ledger / lab JSONL artifact.

| Phase | Hypothesis margin | Outcome (BPB delta) | Decision | PR |
|---|---|---|---|---|
| P0 | reproduce 2.2393 +/- 0.01 | _pending_ | _pending_ | _pending_ |
| P1 | Muon - AdamW <= -0.05 | _pending_ | _pending_ | _pending_ |
| P2 | muP transfer < 5% deg | _pending_ | _pending_ | _pending_ |
| P3 | SF/WSD - cosine <= -0.04 | _pending_ | _pending_ | _pending_ |
| P4 | objective+EMA <= -0.03 | _pending_ | _pending_ | _pending_ |
| P5 | 3 seeds < 1.85 | _pending_ | _pending_ | _pending_ |

## License

MIT — see [`LICENSE`](LICENSE).
