# Source-of-Truth Mandate

> This document declares the canonical-status of `gHashTag/trios-trainer-igla`
> with respect to the IGLA RACE training pipeline.
> Effective: 2026-04-26. Anchor: φ² + φ⁻² = 3.

## Statement

`gHashTag/trios-trainer-igla` is the **single source of truth** for:

- The transformer + HybridAttn model architecture used by Gate-2 attempts
- The AdamW + Muon + φ-LR optimizer schedule
- The T-JEPA loss + EMA target predictor
- The combined objective function
- The BPE tokenizer + dataloader pipeline
- The triplet-validated ledger emit
  (`BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>`)
- The embargo enforcement against `assertions/seed_results.jsonl`
- The TOML run-config schema (`champion`, `gate2-attempt`, `needle-v1-mup`)
- The Dockerfile + Railway service config

## What this repo does NOT own

These remain canonical in [`gHashTag/trios`](https://github.com/gHashTag/trios):

- Runtime invariants (`trios-igla-race::invariants`, INV-1 … INV-10)
- ASHA scheduler + victory gate (`trios-igla-race::asha`, `trios-igla-race::victory`)
- GoldenFloat16 number type (`trios-golden-float`)
- φ-schedule primitives (`trios-phi-schedule`)
- Precision router (`trios-precision-router`)
- `assertions/seed_results.jsonl` itself (the ledger file)
- `assertions/embargo.jsonl` (the embargo list)

`trios-trainer` consumes those crates as **versioned git dependencies**
(via the `trios-integration` feature). It never re-implements them.

## Migration plan from `gHashTag/trios`

| File path inside `gHashTag/trios` | Outcome | When |
|---|---|---|
| `crates/trios-train-cpu/src/transformer.rs` | `git mv` → `trios-trainer/src/model.rs` | PR-1 |
| `crates/trios-train-cpu/src/hybrid_attn.rs` | `git mv` → `trios-trainer/src/model_hybrid_attn.rs` | PR-1 |
| `crates/trios-train-cpu/src/optimizer.rs` | `git mv` → `trios-trainer/src/optimizer.rs` | PR-1 |
| `crates/trios-train-cpu/src/forward.rs` | `git mv` → `trios-trainer/src/forward.rs` | PR-1 |
| `crates/trios-train-cpu/src/backward.rs` | `git mv` → `trios-trainer/src/backward.rs` | PR-1 |
| `crates/trios-train-cpu/src/objective.rs` | `git mv` → `trios-trainer/src/objective.rs` | PR-2 |
| `crates/trios-train-cpu/src/jepa/*` | `git mv` → `trios-trainer/src/jepa/` | PR-2 |
| `crates/trios-train-cpu/src/tokenizer.rs` | `git mv` → `trios-trainer/src/data/tokenizer.rs` | PR-1 |
| `crates/trios-train-cpu/src/bin/hybrid_train.rs` | rewrite as `trios-trainer/src/bin/trios-train.rs` | PR-1 |
| `crates/trios-igla-trainer/src/jepa_runner.rs` | merge into `trios-trainer/src/jepa/runner.rs` | PR-2 |
| `crates/trios-train-cpu/src/gf16.rs` | DELETE; re-export from `trios-golden-float` | PR-1 |
| `crates/trios-training/` (entire crate, ~80 KB) | DELETE | PR-3 |
| `crates/trios-training-ffi/` (Zig stub) | DELETE | PR-3 |
| `crates/trios-train-cpu/src/bin/{22 dead binaries}` | DELETE | PR-3 |
| `crates/trios-igla-race/src/main_*.rs.backup` | DELETE | PR-3 |
| `scripts/igla_race_worker.py` | DELETE (R1 violation) | PR-3 |
| `scripts/igla_train.py` | DELETE (R1 violation) | PR-3 |
| `scripts/train_gpt.py` | DELETE (R1 violation) | PR-3 |

After PR-3 in `gHashTag/trios`, the trios monorepo no longer contains any
training code. The `leaderboard.yml` GitHub Action is updated to:

```yaml
- name: Build trainer
  run: cargo install --git https://github.com/gHashTag/trios-trainer-igla trios-train --locked
- name: Run gate-2 attempt
  run: trios-train --config gate2-attempt.toml --seed ${{ matrix.seed }}
```

## Forking / reuse policy

External forks of `trios-trainer` for other model/dataset combinations are
welcome. Any fork that wishes to feed back into the IGLA RACE ledger
(`gHashTag/trios/assertions/seed_results.jsonl`) MUST:

- Build with `--features trios-integration`
- Pass the embargo + triplet validation from `src/ledger.rs` unmodified
- Cite [`gHashTag/trios#143`](https://github.com/gHashTag/trios/issues/143)
  in the row's `agent` field

## Ownership

- Anchor: `φ² + φ⁻² = 3` — [Zenodo 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)
- Maintainer: [@gHashTag](https://github.com/gHashTag)
- License: MIT
