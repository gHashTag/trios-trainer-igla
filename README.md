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

## Run on Railway

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

| Phase | Status | Scope |
|---|---|---|
| **PR-0** | ✅ done | Skeleton compiles, anchor test passes |
| **PR-1** | 🟡 next | Migrate model + optimizer + tokenizer (`git mv` from `trios-train-cpu`) |
| **PR-2** | ⬜ | Migrate JEPA + objective; merge `trios-igla-trainer::jepa_runner` |
| **PR-3** | ⬜ | Champion-config full run reproduces ≈ 2.2393 ± 0.01 |
| **PR-4** | ⬜ | DELETE phase in `gHashTag/trios` (consolidation PR) |
| **PR-5** | ⬜ | Push image to ghcr.io + wire 3-seed Railway deployment |

## License

MIT — see [`LICENSE`](LICENSE).

