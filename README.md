# trios-trainer-igla

[![CI](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml/badge.svg)](https://github.com/gHashTag/trios-trainer-igla/actions/workflows/ci.yml)
[![Anchor](https://img.shields.io/badge/anchor-%CF%86%C2%B2%2B%CF%86%E2%81%BB%C2%B2%3D3-black)](https://doi.org/10.5281/zenodo.19227877)

IGLA RACE trainer. Tracks [gHashTag/trios#143](https://github.com/gHashTag/trios/issues/143).
Anchor: `phi^2 + phi^-2 = 3`.

> **Canonical Zenodo SOT:** [zenodo.org/communities/trinity-s3ai](https://zenodo.org/communities/trinity-s3ai/). The anchor badge resolves to record [19227877 (VSA Operations v5.0, B007)](https://doi.org/10.5281/zenodo.19227877), which is canonical inside the SOT community.

## Calibration reference (checkpoint-backed)

> **This is a small-model calibration fixture, not a capability claim.** A
> ~196.6K parameter byte-level model on ~1.1 MB of Shakespeare is a
> reproducibility harness: it exists to prove that the same inputs yield the same
> weights, not to compete with anything. Read every number in this section as a
> measurement of the *pipeline*, not of the *model*.

**raw val_bpb = 2.6348 at 12 000 steps** -- seed 47, hidden=384, ~196.6K
parameters, AdamW, and **one effective attention layer**: the layer-2 block
(`wq2/wk2/wv2/wo2`) is allocated and carried through every forward pass but is
provably inert. `HybridAttn::with_config` zero-fills all eight blocks and
`HybridModel::new` randomizes only `wq/wk/wv/wo`, so with `wo2 = 0` the layer-2
gradients are identically zero at init and weight decay (`wd * lr * 0`) cannot
break the symmetry -- those weights are still exactly zero after training. That
claim is checked by a test rather than stated in prose: see
`run_single_emits_a_loadable_artifact_and_freezes_layer_two` in
`src/train_loop.rs`. The `params=196608` the trainer prints already excludes
them.

Backed by an on-disk artifact, not by a log line:

| Field | Value |
|-------|-------|
| Sidecar | `checkpoints/r4-docs-repro/12000.json` (schema `trios-checkpoint-record/3`) |
| Checkpoint sha256 | `8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c` |
| `final_val_bpb` | `2.6347548961639404` |
| Train corpus sha256 | `1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d` (1 015 394 bytes) |
| Val corpus sha256 | `2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502` (100 000 bytes) |
| `steps_total` | `12000` |
| `gf16_floor_every` | `1` (the default) |
| `data_synthetic` | `false` |
| Platform | `macos/aarch64`, `rustc 1.96.0 (ac68faa20 2026-05-25)`, libc `undetermined` |
| Trainer binary sha256 | `6c45b74b719d3b3e80e2785a490f36302ec8fb74a7b81a3942487603ae6a2348` |

The command that produced it, with the ledger unreachable so nothing outside
this checkout could have supplied a number:

```bash
cargo build --release --bin trios-train
env -u DATABASE_URL -u NEON_DATABASE_URL -u TRIOS_DATABASE_URL \
  TRIOS_CANON_NAME=r4-docs-repro ./target/release/trios-train \
  --seed 47 --steps 12000 --hidden 384 --attn-layers 2 --eval-every 1000 \
  --lr 0.003 --train-data data/tiny_shakespeare.txt \
  --val-data data/tiny_shakespeare_val.txt
```

### ASCII-only fixture: what "bits per byte" counts here

`val_bpb` is a per-token cross-entropy in bits, averaged over 40 windows of 129
tokens sampled at a fixed stride across the held-out corpus (`evaluate` in
`src/train_loop.rs`). It is a sample of the corpus, not a pass over all of it.

On this trainer a token is a byte folded into 128 classes: `load_data` maps
every input byte through `(b as usize) % 128`, and the model predicts over
`VOCAB = 128`.

On `tiny_shakespeare` that fold is the identity -- every byte in the pinned
corpus is below 128 -- so token and byte coincide and "bits per byte" is
literal. **On any corpus that is not pure ASCII the fold is not injective**, and
the number stops meaning what its name says. Verified against the binary hashed
above by folding the UTF-8 encoding of the Russian name for Russia:

```
Rossiya (Cyrillic)  UTF-8 : D0 A0 D0 BE D1 81 D1 81 D0 B8 D1 8F
                  % 128   : 50 20 50 3E 51 01 51 01 50 38 51 0F
                  as text : P SP P > Q SOH Q SOH P 8 Q SI
```

(`SP` is the space character `0x20`; `SOH` and `SI` are the C0 control codes
`0x01` and `0x0F`, which have no printable form.)

Every `0xD0` lead byte folds onto ASCII `P` (`0x50`), every `0xD1` onto `Q`
(`0x51`), and `0xA0` -- the continuation byte of the capital letter the word
starts with -- folds onto a space (`0x20`). The collision is total, not
occasional: each of the 128 classes has exactly two pre-images, `x` and
`x + 128`.

So the model is never asked to distinguish `0xD0` from `P`. Up to one bit per
byte of the real stream is discarded before training, and what the trainer
prints as bits-per-byte is bits-per-*folded*-byte. It is comparable to other
numbers measured on this fixture and **not** comparable to any published BPB on
non-ASCII or full 8-bit text. This is a statement about the measured behaviour
of the binary above, not a claim that anything has been fixed.

### The recipe changed, and the change is legible step by step

This headline is **not** the ~2.61 this section used to lead with, and the
difference is the most useful thing in this file.

`gf16_floor()` rewrites `embed` / `proj` / `lm_head` / `ctx` in place once a run
is past `floor(0.7 * steps)`. That mutation used to be gated on the eval
cadence, which made an observation knob change the weights. The fix moved it
onto its own knob and defaulted that knob to `1`
(`GF16_FLOOR_EVERY_DEFAULT`, `src/train_loop.rs`), so the floor now fires on
*every* step past the mark rather than once per eval. **The fix therefore
changed the default recipe**, and the pipeline no longer produces ~2.61 at
12 000 steps.

The prediction that follows is falsifiable: a 12 000-step run crosses the mark
at `floor(0.7 * 12000) = 8400`, so a post-fix run must agree with a pre-fix run
exactly up to 8400 and diverge after it. Measured against the two archived
pre-fix runs still on disk in `checkpoints/`:

| Step | `igla-honest-provenance` | `igla-honest-20260802` | this run |
|------|--------------------------|------------------------|----------|
| 0 (init) | -- | -- | 7.0002 |
| 1 000 | -- | -- | 3.3097 |
| 3 000 | -- | 3.077059507369995 | 3.0771 |
| 4 000 | 2.9554262161254883 | -- | 2.9554 |
| 6 000 | -- | 2.8116235733032227 | 2.8116 |
| 8 000 | 2.645456314086914 | -- | 2.6455 |
| **8 400** | *first floored step -- `floor(0.7 * 12000)`* | | |
| 9 000 | -- | 2.6429085731506348 | **2.6563** |
| 12 000 | 2.616914749145508 | 2.614117383956909 | **2.6347548961639404** |

Archived columns are the `bpb` fields of the on-disk sidecars; this run's
intermediate rows are the 4-decimal figures its own log printed, and its
12 000-step row is the sidecar `final_val_bpb`.

BPB agreeing to four decimals is weak evidence. The checkpoint bytes are not:

| Step | this run | archived | archived run | |
|------|----------|----------|--------------|---|
| 3 000 | `de2595a964da65df` | `de2595a964da65df` | `igla-honest-20260802` | identical |
| 4 000 | `9eb162017b53edc1` | `9eb162017b53edc1` | `igla-honest-provenance` | identical |
| 6 000 | `060ac8dc118d6adb` | `060ac8dc118d6adb` | `igla-honest-20260802` | identical |
| 8 000 | `c27ed8fc85692823` | `c27ed8fc85692823` | `igla-honest-provenance` | identical |
| 9 000 | `42bb8852a8c93925` | `63e8597a3232d5f2` | `igla-honest-20260802` | **differ** |
| 12 000 | `8a86fe691aef64fc` | `c1b1a12b914924ac` | `igla-honest-provenance` | **differ** |
| 12 000 | `8a86fe691aef64fc` | `b2fe52b0760dc369` | `igla-honest-20260802` | **differ** |

sha256, first 16 hex digits; the full values are in the sidecars next to each
`.bin`. The intermediate checkpoints come from a second run identical to the
headline run except for `TRIOS_CHECKPOINT_EVERY=1000`, and its 12 000-step
checkpoint is byte-identical to the headline run's -- which is also the
determinism check for this pair.

Four exact byte matches followed by divergence at exactly the predicted step is
the conformance argument this repository is for: the change to the recipe is
not asserted, it is located.

### Recovering a pre-fix artifact, and failing to recover the other one

If the floor cadence really is the only thing the fix changed, then setting
`TRIOS_GF16_FLOOR_EVERY` to a pre-fix run's eval cadence should reproduce that
run's bytes with the post-fix binary. That was executed, not assumed. All four
rows below are seed 47, 12 000 steps, same corpus, same binary
(`6c45b74b...`), differing only in the floor cadence:

| `TRIOS_GF16_FLOOR_EVERY` | `final_val_bpb` at 12 000 | checkpoint sha256 | matches an archived run? |
|---|---|---|---|
| `1` (default) | `2.6347548961639404` | `8a86fe691aef64fc` | -- this is the new headline |
| `1000` | `2.614117383956909` | `b2fe52b0760dc369` | **yes -- `igla-honest-20260802/12000.bin`, byte for byte** |
| `3000` | `2.614041805267334` | `7a95c99b962d565f` | no |
| `4000` | `2.5910677909851074` | `3ecc1c925e42ff57` | no |

One seed, one corpus, four cadences, four different artifacts, spanning
2.5911 to 2.6348 bpb. That spread is what "the cadence is part of the recipe"
costs, and the `1000` row shows the escape hatch works: the post-fix trainer
reproduced a pre-fix artifact exactly, down to the last digit of the `bpb` in
its sidecar. The byte match is also what *identifies* that run's cadence as
1000 -- its own record never said so.

The other archived run was **not** recovered. `igla-honest-provenance/12000.bin`
(`c1b1a12b914924ac`, sidecar `bpb` 2.616914749145508) -- the artifact this
section used to lead with -- does not come back at cadence `1`, `1000`, `3000`
or `4000`. Its `trios-checkpoint-record/1` sidecar does not state the cadence it
was run at, so recovering it means guessing, and four guesses were wrong. That
is the concrete price of an unrecorded recipe field, and it is why `ckpt_replay`
grades that record `INCOMPARABLE` rather than `MISMATCH`: there is no claim
there to be right or wrong about.

> **`--eval-every` is observation-only.** It selects when the held-out corpus is
> measured. It does not enter the recipe, and two runs that differ only in it
> produce byte-identical checkpoints. The knob that *is* part of the recipe is
> **`gf16_floor_every`** (`TRIOS_GF16_FLOOR_EVERY`, default `1`), because
> `gf16_floor()` mutates the weights. The trainer prints both in its banner, and
> `src/train_loop.rs` gates the mutation on the former under a comment that
> reads "NOT `args.eval_every`".
>
> **The parameters that must match before two runs may be compared are
> `gf16_floor_every` and `steps_total`** -- the second because the 70% mark is
> `floor(0.7 * steps_total)`, so changing the step budget moves the point where
> the floor starts. Both are recorded in `trios-checkpoint-record/3`; neither is
> recorded in schema 1.
>
> *History, fixed 2026-08-02:* the floor used to be gated on the eval cadence,
> and the pair BPB 2.6141 vs 2.6169 -- two seed-47 runs recorded a few minutes
> apart at different eval cadences -- was the symptom. 2.6141 has since been
> reproduced exactly by the post-fix binary at `TRIOS_GF16_FLOOR_EVERY=1000`;
> 2.6169 has not been reproduced at any cadence tried, because its schema/1
> record does not state one. Those two numbers are retained here as the dated
> evidence of a defect that no longer exists. They are **not** a live warning
> about `--eval-every`.

> **Retracted:** this section previously led with a "Champion BPB" headline in
> the low 2.2s, measured under a seed the binary now refuses (Canon #93 forbids
> {42, 43, 44, 45}). That figure is listed as RETRACTED in
> [`docs/audit/HONEST_FINDINGS.md`](docs/audit/HONEST_FINDINGS.md), and no
> checkpoint artifact was ever produced behind it.

## Quick start

```bash
git clone https://github.com/gHashTag/trios-trainer-igla.git
cd trios-trainer-igla

# Download data, then split it BYTE-DISJOINT: train is everything but the last
# 100 KB, val is that last 100 KB. Do not use `head -c 100000 train > val` --
# that makes val a prefix of train, which is the leak that tainted the
# 2026-04-30 ledger (#60). The trainer now refuses such a split at startup.
#
# The corpus is pinned by checksum: every BPB in this repo is measured against
# these exact bytes. train ++ val reconstructs the canonical corpus byte for
# byte (sha256 86c4e6aa..., 1115394 bytes).
mkdir -p data
curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    > data/full.txt
shasum -a 256 -c - <<'EOF'
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  data/full.txt
EOF
SIZE=$(wc -c < data/full.txt)
head -c $((SIZE - 100000)) data/full.txt > data/tiny_shakespeare.txt
tail -c 100000            data/full.txt > data/tiny_shakespeare_val.txt
shasum -a 256 -c - <<'EOF'
1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d  data/tiny_shakespeare.txt
2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502  data/tiny_shakespeare_val.txt
EOF
rm data/full.txt

# Build just the trainer this quickstart uses. (`cargo build --release` with no
# --bin also builds the experimental binaries, which are not part of this path.)
cargo build --release --bin trios-train

# Train a single seed. Canon #93: seeds {42, 43, 44, 45} are forbidden and the
# binary exits non-zero on them; the allowed set is {47, 89, 123, 144}.
# Drop --steps to something small (e.g. 200) for a first smoke run.
./target/release/trios-train --seed=47 --steps=81000 --hidden=384 --lr=0.003 --optimizer=adamw

# Train all 3 Canon #93 seeds. --eval-every may differ freely between them; what
# must be held fixed to keep them comparable is TRIOS_GF16_FLOOR_EVERY and
# --steps. See the calibration section above.
for s in 47 89 123; do
  ./target/release/trios-train --seed=$s --steps=81000 --hidden=384 --lr=0.003 --optimizer=adamw
done
```

## Railway deploy (3 seeds in parallel)

```bash
# Prerequisites
brew install railway # or: npm i -g @railway/cli
railway login

# Link project (first time)
railway link  # select "trios-trainer"

# Create services (once). Canon #93 seeds only -- a service pinned to 42/43/44
# will start and immediately exit non-zero inside the container.
for s in 47 89 123; do
  railway add --service "igla-seed-$s"  # choose "Empty Service"
  railway variables set --service "igla-seed-$s" TRIOS_SEED=$s
done

# Deploy all 3 seeds
for s in 47 89 123; do
  railway up --service "igla-seed-$s" --detach
done

# Watch logs
railway logs --service igla-seed-47
railway logs --service igla-seed-89
railway logs --service igla-seed-123
```

## Binaries

### `trios-train` -- main trainer

```bash
./target/release/trios-train [OPTIONS]

Options:
      --seed <SEED>            Seed. 0 = 3-seed sweep. Canon #93 forbids
                               {42,43,44,45} [env: TRIOS_SEED=] [default: 47]
      --steps <STEPS>          Training steps [env: TRIOS_STEPS=] [default: 54000]
      --hidden <HIDDEN>        Hidden dim [env: TRIOS_HIDDEN=] [default: 828]
      --lr <LR>                Learning rate [env: TRIOS_LR=] [default: 0.003]
      --attn-layers <N>        Attention layers [env: TRIOS_ATTN_LAYERS=] [default: 2]
      --eval-every <N>         Eval interval [env: TRIOS_EVAL_EVERY=] [default: 1000]
                               Observation only; it does not enter the recipe
      --optimizer <OPT>        adamw | muon | muon-cwd [env: TRIOS_OPTIMIZER=] [default: adamw]
      --train-data <PATH>      [env: TRIOS_TRAIN_PATH=] [default: data/tiny_shakespeare.txt]
      --val-data <PATH>        [env: TRIOS_VAL_PATH=] [default: data/tiny_shakespeare_val.txt]
      --ctx <CTX>              Accepted for seed-agent compat; ignored [env: TRIOS_CTX=]
      --format <FORMAT>        Fake-quant format pass-through [env: TRIOS_FORMAT_TYPE=]
      --neon <NEON>            Neon DSN for bpb_samples writes [env: TRIOS_NEON_DSN=]
      --config <TOML>          Config file (overrides flags) [env: TRIOS_CONFIG=]
      --sweep                  3-seed sweep {47, 89, 123} (Canon #93)
```

Run `./target/release/trios-train --help` for the authoritative list; the block
above is a copy and can drift.

The floor cadence has no CLI flag: it is set through `TRIOS_GF16_FLOOR_EVERY`
and defaults to `1`. The trainer prints the resolved value next to `eval_every`
in its startup banner, so a typo is visible at step 0.

### `ckpt_replay` -- spot-check verifier (auditor side)

Re-derives one checkpoint from its own sidecar and grades the result. It
re-executes the `trios-train` binary as a subprocess with a cleared environment,
so it tests what an outside auditor can actually test: a binary, a corpus and a
JSON file.

```bash
cargo build --release --bin ckpt_replay --bin trios-train
./target/release/ckpt_replay --record checkpoints/<run>/<step>.json
```

| Verdict | Exit | Meaning |
|---------|------|---------|
| `VERIFIED on <os>/<arch>` | 0 | the record's own parameters reproduce its bytes on this host |
| `MISMATCH` | 1 | they do not; both hashes are printed |
| `INCOMPARABLE` | 2 | the record does not describe its own inputs; nothing can be graded |
| `REFUSED` | 3 | the replay would exceed `--max-steps` (default 2 000) |

The archived `igla-honest-provenance/12000.json` returns **`INCOMPARABLE`**: a
`trios-checkpoint-record/1` sidecar states neither `steps_total` nor
`gf16_floor_every`, and both are first-order -- `steps_total` fixes where the
70% mark falls, `gf16_floor_every` fixes how often the weights are rewritten
past it. The headline sidecar at the top of this file is schema 3 and states
both. A `VERIFIED` verdict is always printed
with the platform it was obtained on, because the same seed and corpus produce
different checkpoint hashes on macOS and on Linux. The scale, the evidence and
the scope are in
[`docs/REPRODUCIBILITY-GRADING.md`](docs/REPRODUCIBILITY-GRADING.md).

### `trios-igla` -- ledger query tool

```bash
./target/release/trios-igla <COMMAND>

Commands:
  search   Filter ledger rows (--seed, --bpb-max, --step-min, --sha)
  list     Last N rows (--last N)
  gate     Gate-2 quorum check (--target BPB)
  check    Embargo refusal for SHA
  triplet  Print R7 triplet for row index
```

### Other binaries

| Binary | Purpose |
|--------|---------|
| `hybrid_train` | N-gram + HybridAttn + ReLU^2 + Muon trainer |
| `seed_emit` | Emit a ledger row (--seed, --bpb, --step, --sha) |
| `ledger_check` | Validate ledger format |
| `qk_gain_check` | Check QK-gain against INV-13 (--lr, --gain) |

## Results

The only results in this repo with a checkpoint artifact behind them are the
seed-47 calibration rows at the top of this file. The former
"Results (Railway, 2026-04-27)" table -- a 4x3 grid of roughly 2.2-2.7 BPB
values under the forbidden seeds 42/43/44 -- has been removed: those runs
pre-date `checkpoint::save` doing anything (it was a stub returning `Ok(())`),
so no weights exist for any cell in it, and its headline row belongs to the
retracted champion family. See
[`docs/audit/HONEST_FINDINGS.md`](docs/audit/HONEST_FINDINGS.md).

Determinism, which *is* verified: identical parameters reproduce
byte-identical checkpoints and identical BPB across independent runs.

## Environment variables

Defaults live in the Rust binaries, not in the image -- the Dockerfile
deliberately bakes in no `TRIOS_*` values so that operator-supplied aliases win
the precedence race (see the Wave-33B comment in `Dockerfile`).

| Var | Default | Used by |
|-----|---------|---------|
| `TRIOS_SEED` | 47 (trios-train CLI) / **43, see below** (entrypoint) | trios-train, entrypoint |
| `TRIOS_STEPS` | 54000 (trios-train CLI) / 81000 (entrypoint) | trios-train, entrypoint |
| `TRIOS_LR` | 0.003 | trios-train, entrypoint |
| `TRIOS_HIDDEN` | 828 (trios-train CLI) / 384 (entrypoint) | trios-train, entrypoint |
| `TRIOS_OPTIMIZER` | adamw | trios-train, entrypoint |
| `TRIOS_EVAL_EVERY` | 1000 | trios-train (observation only) |
| `TRIOS_GF16_FLOOR_EVERY` | 1 | trios-train (**recipe**; see the calibration section) |
| `TRIOS_CHECKPOINT_EVERY` | 0 (final step only) | trios-train (observation only) |
| `TRIOS_TRAIN_DATA` | /work/data/tiny_shakespeare.txt | entrypoint, railway-sweep |
| `TRIOS_VAL_DATA` | /work/data/tiny_shakespeare_val.txt | entrypoint, railway-sweep |
| `TRIOS_TRAINER_BIN` | trios-train | entrypoint |
| `RUST_LOG` | info | all binaries |

> **Known defect (not yet fixed):** `src/bin/entrypoint.rs:22` resolves
> `TRIOS_SEED` with a fallback literal of `"43"`, which Canon #93 forbids.
> Removing the baked `ENV TRIOS_SEED=43` from the image is therefore necessary
> but not sufficient: a container started with no seed set still forwards that
> forbidden seed to the trainer, which then exits non-zero on its own guard.
> **Always set `TRIOS_SEED` explicitly** to one of {47, 89, 123, 144} until the
> entrypoint fallback is changed.

## Docker

```bash
docker build -t trios-trainer .
# TRIOS_SEED is required in practice -- see the entrypoint defect note above.
docker run --rm -e TRIOS_SEED=47 -e TRIOS_STEPS=81000 trios-trainer
```

The image pins the corpus by sha256 and will fail the build rather than train on
bytes that differ from the ones every recorded BPB was measured against.

## Tests

```bash
cargo test --release --lib    # library test suite
```

The `--ignored` "champion reproduction" test is deliberately not advertised
here: it asserts BPB `2.2393`, which
[`docs/audit/HONEST_FINDINGS.md`](docs/audit/HONEST_FINDINGS.md) marks
RETRACTED, so a green run of it would certify a number the repo has withdrawn.

## Status

The honest position, stated plainly:

- **Verified:** training is bit-deterministic. Identical parameters produce
  byte-identical checkpoints and identical BPB across independent runs.
- **Verified:** the train/val split is byte-disjoint, checked over the full
  corpus (the previous check sampled with `step_by(256)` and would have missed
  an overlap 255 times out of 256).
- **Verified:** checkpoints are now actually written -- atomic
  `tmp + fsync + rename`, SHA-256 taken over bytes re-read from disk, magic
  `TRIOSCKP`, truncation rejected on load, JSON sidecar always emitted.
- **Not established:** any capability claim. The Gate-2 target (BPB < 1.85) is
  not met, and the numbers that once claimed to approach it are retracted.
- **Not citable:** BPB `1.5492`, reported as an "honest Gate-2 pass" in the
  PR's own `LEAK_INVESTIGATION.md`. At that commit `train_loop.rs` contained no
  `bpb_sample` call at all -- the wiring landed two days later -- so those rows
  came from an out-of-repo stdout parser with no artifact behind them.

See [`docs/TRAINING_FLOW_V2.md`](docs/TRAINING_FLOW_V2.md) for the training plan
and [`docs/audit/HONEST_FINDINGS.md`](docs/audit/HONEST_FINDINGS.md) for the
full retraction ledger.

## License

MIT
