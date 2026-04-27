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

# Build
cargo build --release

# Local training (single seed)
./target/release/trios-train --seed 43 --steps 27000 --hidden 384 --lr 0.004 --attn-layers 2

# Or use tri CLI
./target/release/tri train --seed 43 --steps 27000 --hidden 384
```

## `tri` CLI — Railway deploy + local train

`tri` is the IGLA Race CLI bundled in this repo. It deploys training containers
to [Railway](https://railway.com) with one command per seed.

```bash
cargo build --release --bin tri
```

### Commands

| Command | Description |
|---------|-------------|
| `tri deploy init` | Create Railway project `trios-trainer` |
| `tri deploy seed --seed 43` | Deploy a single seed container |
| `tri deploy all` | Deploy all Gate-2 seeds (42, 43, 44) |
| `tri deploy status` | List deployed Railway services |
| `tri deploy logs --seed 43` | Stream logs for a seed's container |
| `tri deploy remove --seed 43` | Remove a seed's container |
| `tri train --seed 43 --steps 27000` | Local training (no Railway) |
| `tri race start` | Start ASHA worker loop |
| `tri race status` | Show leaderboard (needs DATABASE_URL) |
| `tri race best` | Show best trial (needs DATABASE_URL) |

### Deploy to Railway (ONE SHOT)

```bash
# 1. Init Railway project (first time only)
./target/release/tri deploy init

# 2. Deploy all 3 seeds
./target/release/tri deploy all

# Or deploy individually with custom params
./target/release/tri deploy seed --seed 43 --steps 27000 --hidden 384 --lr 0.004 --attn-layers 2
./target/release/tri deploy seed --seed 44 --steps 27000 --hidden 384 --lr 0.004 --attn-layers 2
./target/release/tri deploy seed --seed 45 --steps 27000 --hidden 384 --lr 0.004 --attn-layers 2

# 3. Check status
./target/release/tri deploy status

# 4. Watch logs
./target/release/tri deploy logs --seed 43
```

Each seed runs in its own Railway container with these env vars:

| Var | Default | Effect |
|-----|---------|--------|
| `TRIOS_SEED` | `43` | Training seed |
| `TRIOS_STEPS` | `27000` | Number of training steps |
| `TRIOS_HIDDEN` | `384` | Hidden dimension |
| `TRIOS_LR` | `0.004` | Learning rate (INV-8: must be in [0.001, 0.01]) |
| `TRIOS_ATTN_LAYERS` | `2` | Number of attention layers |
| `TRIOS_OPTIMIZER` | `adamw` | Optimizer: `adamw`, `muon`, or `muon-cwd` |
| `TRIOS_EVAL_EVERY` | `1000` | Eval interval |

### Stop rule (Gate-2)

Training terminates when either:
- **Victory**: 3 distinct seeds each reach `BPB < 1.85` at `step >= 4000`
- **Deadline**: `2026-04-30 23:59 UTC`

## Search the needle (`trios-igla`)

A second binary, `trios-igla`, is the **read-only query tool** for the IGLA
RACE ledger. It never mutates `assertions/seed_results.jsonl` — it only
filters, lists, and emits the canonical triplet so operators and CI can
answer “did we find the needle yet?” without ad-hoc shell pipelines.

Triplet (R7):

```text
BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>
```

| Command | Effect | Exit code |
|---|---|---|
| `trios-igla search --seed 43 --bpb-max 1.85 --step-min 4000` | Filter ledger; one triplet per match. Flags: `--seed`, `--bpb-max`, `--step-min`, `--sha`, `--gate-status`. | 0 hit, 2 no-match |
| `trios-igla list --last 5` | Last N rows in triplet form (default 10). | 0 |
| `trios-igla gate --target 1.85` | Gate-2 quorum: PASS iff ≥3 distinct seeds satisfy `bpb < target` AND `step >= 4000`. | 0 PASS, 2 NOT YET |
| `trios-igla check 2446855` | R9 embargo refusal against `assertions/embargo.txt`. | 0 clean, 1 embargoed |
| `trios-igla triplet 0` | Canonical R7 triplet for a row index (0-based). | 0 |

Common flags:
- `--ledger <path>` (default `assertions/seed_results.jsonl`)
- `--embargo <path>` (default `assertions/embargo.txt`)

### Quickstart

```bash
cargo build --release --bin trios-igla
BIN=./target/release/trios-igla

# Did anyone hit the target yet?
$BIN search --bpb-max 1.85 --step-min 4000

# Print last 5 rows for a glance
$BIN list --last 5

# Gate-2 verdict in CI
$BIN gate --target 1.85 || echo "NOT YET—keep training"

# Refuse to act on an embargoed SHA
$BIN check 477e3377   # exit 1
$BIN check 2446855    # exit 0 (champion)
```

Gate-2 anchors used by `trios-igla gate`:
- target: `1.85` (`igla::DEFAULT_TARGET_BPB`, overridable via `--target`)
- step floor: `4000` (`igla::STEP_MIN_FOR_LEDGER`, R8)
- quorum: `3` distinct seeds (`igla::GATE2_SEED_QUORUM`)
- champion: `2.2393 @ 27K seed=43 sha=2446855` ([`gHashTag/trios@2446855`](https://github.com/gHashTag/trios/commit/2446855))

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

### Library API

Everything `trios-igla` does is also exposed from the library so the
training loop, CI gates, and the `trios-train` binary can reuse it:

```rust
use trios_trainer::igla::{
    self, SearchFilter, gate2_seed_count, is_embargoed, render_triplet,
};

let rows = igla::read_ledger("assertions/seed_results.jsonl".as_ref())?;
let count = gate2_seed_count(&rows, igla::DEFAULT_TARGET_BPB);
assert!(count >= igla::GATE2_SEED_QUORUM, "Gate-2 not yet reached");
```

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
trios-trainer-igla/
├── Cargo.toml             ← lib + bins (trios-train, tri, hybrid_train, etc.)
├── Dockerfile             ← multi-stage Rust build for Railway
├── nixpacks.toml          ← Railway nixpacks build config
├── README.md              ← this file
├── configs/
│   ├── champion.toml      ← reproduce baseline BPB=2.2393
│   ├── gate2-attempt.toml ← Gate-2 push config
│   ├── gate2-final.toml   ← P5 frozen config (384d, 4L, muon)
│   ├── needle-v1-mup.toml ← muP transfer experiment
│   └── lab/               ← P1-P4 experiment configs
│       ├── p1-adamw.toml
│       ├── p1-muon.toml
│       ├── p1-muon-cwd.toml
│       ├── p2-proxy-8m.toml
│       ├── p2-proxy-24m.toml
│       ├── p2-target-70m.toml
│       ├── p3-cosine.toml
│       ├── p3-sf.toml
│       ├── p3-wsd.toml
│       ├── p4-objective.toml
│       └── p4-ema.toml
├── assertions/
│   ├── seed_results.jsonl ← R7 ledger (P0/P5 only)
│   ├── igla_assertions.json ← Coq invariant bridge
│   ├── champion_lock.txt
│   ├── baseline_profile.json
│   └── lab/               ← P1-P4 lab results (not R7-validated)
├── docs/
│   ├── TRAINING_FLOW_V2.md ← P0-P5 decomposed plan
│   └── audit/              ← P0 audit documents
└── src/
    ├── lib.rs              ← module declarations + TRINITY_ANCHOR
    ├── config.rs           ← TOML schema + env override + INV-8 validate
    ├── train_loop.rs       ← AdamW + Muon + NCA training loop
    ├── optimizer.rs        ← AdamW + Muon (NS-5) + MuonCwd + Schedule-Free + WSD
    ├── objective.rs        ← CE + JEPA + NCA combined loss
    ├── checkpoint.rs       ← save/load + post-hoc EMA (ema_average/ema_sweep)
    ├── mup.rs              ← muP transfer: per-group LR scaling
    ├── invariants.rs       ← Coq-proven constants (INV-1..10)
    ├── ledger.rs           ← R7 triplet emit + embargo
    ├── model.rs            ← HybridAttn + transformer
    ├── race/               ← ASHA + Neon + lessons + victory
    ├── jepa/               ← T-JEPA predictor + EMA + masking + loss
    └── bin/
        ├── tri.rs          ← tri CLI (deploy/train/race)
        ├── trios-train.rs  ← standalone training binary
        ├── tjepa_train.rs  ← T-JEPA training (TASK-5D)
        └── railway_start.sh ← Railway container entrypoint
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


<!-- ci-trigger-19 -->
