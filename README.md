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

**raw val_bpb = 2.63 +/- 0.13 bpb (k = 2) at 12 000 steps**
*(equivalently `2.63 +/- 0.07 bpb` at `k = 1`; the coverage factor is part of
the figure and quoting the number without it is quoting a different quantity.
Eval grid: 40 windows of 129 tokens, stride 2 496, coverage 5.16% of the
100 000-byte held-out corpus)* -- seed 47, hidden=384, ~196.6K
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
| Sidecar | `checkpoints/r6-headline/12000.json`; its own `schema` field records version **6** of the checkpoint-record format. A sidecar generated today carries a higher version - the current tag is the value of `CHECKPOINT_RECORD_SCHEMA` in `src/checkpoint.rs`, the only place it is defined. Read the `schema` field of the record in front of you rather than assuming this row's version |
| **Quoted figure** | **`2.63 +/- 0.13` bpb at `k = 2`** (`+/- 0.07` at `k = 1`), from the combined standard uncertainty `u_c = 0.066 bpb` of [`docs/EVAL-UNCERTAINTY.md`](docs/EVAL-UNCERTAINTY.md) section 3b. Quantified components only: `0.0551` (this record's own `val_bpb_stderr`) and `0.0358` (the between-grid sigma, transferred from a 1 200-step checkpoint). Corpus choice, seed, platform, step count and selection are **not quantified**, so `U` is a lower bound |
| Eval grid | `eval_chunks` 40 windows x `eval_seq` 129 tokens = `eval_tokens` 5 160, stride 2 496, coverage **5.16%** |
| `val_bpb_stderr` (within-grid) | `0.05509733036160469` |
| `final_val_bpb` (raw `f32` expansion) | `2.6347548961639404` -- see "why three figures" below |
| Checkpoint sha256 | `8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c` |
| Train corpus sha256 | `1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d` (1 015 394 bytes) |
| Val corpus sha256 | `2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502` (100 000 bytes) |
| `seed` / `hidden` / `attn_layers` / `lr` | `47` / `384` / `2` / `0.003`, AdamW |
| `steps_total` | `12000` |
| `gf16_floor_every` | `1` (the default) |
| `eval_every` | `1000` |
| `data_synthetic` | `false` |
| Platform | `macos/aarch64`, `rustc 1.96.0 (ac68faa20 2026-05-25)`, libc `undetermined` |
| Trainer binary sha256 | `5a59f44eaa993e242da18fa55d682ba34b7d7934e9cbd417a62f8a8999d44225` |
| `source_sha256` (recorded, not re-derived) | `f0d8a29fa478f0d176014ad5b108728c9acfe4de3f97e9e7e410a3a528129737` |
| `git_sha` / `git_dirty` | `3c1f751cf4376c13d26e247c2cd86357ab51dd20` / **`true`** |
| Run log | `checkpoints/r6-headline/run.log` (full stdout, every `EVAL-PLAN:` line) |

### Why three significant figures, and not seventeen

The sidecar stores `final_val_bpb` as the decimal expansion of an `f32`. Fourteen
of those digits describe the float, not the model. What the measurement supports
is `2.63 +/- 0.13` bpb at `k = 2`, and that is the form this README quotes --
exactly the recommendation in
[`docs/EVAL-UNCERTAINTY.md`](docs/EVAL-UNCERTAINTY.md) section 3b.

**`[CORRECTED 2026-08-03 -- this passage used to read `2.63 +/- 0.04`.]`** That
was wrong in a way worth stating rather than silently overwriting: `0.04` is not
a band at all. It is the rounded value of **one component** -- the
**between-grid** sigma of this estimator, `0.0358 bpb`, measured over seven
grids on fixed weights *at 1 200 steps*, and therefore transferred to this
12 000-step artifact by assumption. Published on its own it was **smaller than
this record's own within-grid stderr**, `val_bpb_stderr = 0.0551`, which sits in
the table above: a counterparty who opened the sidecar read a bigger error bar
than the README quoted. The two quantified components combine as

```
u_c = sqrt(0.0551^2 + 0.0358^2) = 0.066 bpb   (k = 1)
U   = k * u_c                   = 0.13 bpb    (k = 2)
```

and `U` is a **lower bound**: corpus choice, seed, platform, step count and
selection are named in the budget and quantified in none of it. Either way,
nothing at the fourth decimal is a measurement.

**The eval grid is now part of the record, which is the point of re-running.**
`eval_chunks` is an *observation* parameter: it changes what is measured and
never the weights. `docs/EVAL-UNCERTAINTY.md` measures that this knob alone
moves the reported number by up to 0.048 bpb, while BPB values in this
repository are routinely compared at the fourth decimal, 0.0001 -- roughly 500
times finer than the undeclared knob's effect. A record that does not state its
own sampling plan cannot be audited at the precision it is quoted at, and no
argument fixes that; only a re-run does.

**Superseded:** the previous headline record,
`checkpoints/r5-adv-recheck/12000.json` (schema `trios-checkpoint-record/4`,
trainer `73fae5db...`, `final_val_bpb` `2.6347548961639404`), is retained on
disk and in the tables further down. It is superseded **not** because its number
was wrong -- the run above re-derived byte-identical weights,
`8a86fe69...`, from a *different* trainer binary -- but because its schema has
no `eval_chunks`, no `eval_seq` and no `val_bpb_stderr` field, so the record
cannot state the sampling plan its own number came from. Same bytes, same
number, an auditable record instead of an unauditable one.

**The working tree was dirty when this record was produced** (`git_dirty: true`,
against `git_sha 3c1f751`). No number on this page is citable until the commit
that produced it is named: a dirty tree means the recorded `git_sha` describes
the last commit, not the source that ran. `source_sha256` pins the source that
ran, but it is recorded by the trainer and never re-derived by the verifier, so
it is an attestation, not a proof.

This is not a property of one record. **Of the 51 sidecars under `checkpoints/`,
47 record `git_dirty: true`, 4 record `null`, and none records `false`** -- so
by this repository's own definition of L3 no measurement here had ever been
produced from a tree a counterparty could obtain.
[`docs/CLEAN-TREE-PROVENANCE.md`](docs/CLEAN-TREE-PROVENANCE.md) closes that
gap with a procedure that needs no commit, and states plainly what the resulting
artifact does and does not license.

This is the record the repository's own verifier grades. Re-derived from it
after the run (record timestamp `2026-08-02T23:58:45Z`), all 12 000 steps
re-executed, verdict `VERIFIED on macos/aarch64`, exit 0, with
`final_val_bpb 2.6348 confirmed by replay (4 dp)` and the eval grid confirmed
(`the replay was driven to it with TRIOS_EVAL_CHUNKS=40 and ran at
eval_chunks=40 eval_seq=129`). The full transcript, including the sha256 of the
`ckpt_replay` build that produced the verdict, is in
[`checkpoints/r6-headline/VERDICT.txt`](checkpoints/r6-headline/VERDICT.txt);
see also
[`docs/REPRODUCIBILITY-GRADING.md`](docs/REPRODUCIBILITY-GRADING.md) for the
earlier grading of the superseded record.

```bash
./target/release/ckpt_replay --record checkpoints/r6-headline/12000.json \
  --trainer ./target/release/trios-train --max-steps 12000
```

`--max-steps` is not optional here. It defaults to 2 000, so a 12 000-step
record without it exits 3 `REFUSED` before any work is done -- see the verdict
table under [`ckpt_replay`](#ckpt_replay----spot-check-verifier-auditor-side).

`checkpoints/` is gitignored, so the sidecar above is not in the tree of a fresh
clone. The command that produces it, with the ledger unreachable so nothing
outside this checkout could have supplied a number. **The `r6-headline` run
above was not made with the `env -u` guard** -- its sidecar reads
`"ledger": "written"`, meaning a DSN was set and the row was published, whereas
the superseded `r5-adv-recheck` record reads `"ledger": "skipped-no-dsn"`. That
direction of traffic is outbound only (the trainer writes rows, it never reads a
BPB back), and the number in the sidecar is the one its own stdout printed --
`checkpoints/r6-headline/run.log` line `DONE: seed=47 bpb=2.6348` -- but an
auditor who wants the stricter provenance should re-run with the `env -u` prefix
shown here:

```bash
cargo build --release --bin trios-train --bin ckpt_replay
env -u DATABASE_URL -u NEON_DATABASE_URL -u TRIOS_DATABASE_URL \
  TRIOS_CANON_NAME=r6-headline ./target/release/trios-train \
  --seed 47 --steps 12000 --hidden 384 --attn-layers 2 --eval-every 1000 \
  --lr 0.003 --train-data data/tiny_shakespeare.txt \
  --val-data data/tiny_shakespeare_val.txt
```

**Run that command twice and the second run exits 1. That is the evidence guard
working, not a bug, and it has to be stated here because re-running IS the
demonstration.** In a fresh clone the first run is safe: `checkpoints/` is
gitignored, so nothing under it exists yet. A SECOND run into an existing
`checkpoints/<canon>` refuses, with (verbatim, from a 2-step probe under
`TRIOS_CHECKPOINT_DIR=/tmp/ckpt_doc_probe TRIOS_CANON_NAME=doc-probe`; the
headline recipe fails the same way, just after 12 000 steps instead of 2):

```console
$ run() { env -u DATABASE_URL TRIOS_CHECKPOINT_DIR=/tmp/ckpt_doc_probe \
    TRIOS_CANON_NAME=doc-probe ./target/release/trios-train \
    --steps 2 --hidden 16 --attn-layers 1 --seed 47; }
$ run >/dev/null; echo "exit=$?"
exit=0
$ run >/dev/null; echo "exit=$?"
Error: final-step checkpoint failed

Caused by:
    refusing to overwrite sidecar "/private/tmp/ckpt_doc_probe/doc-probe/2.json":
    it already records ledger="skipped-no-dsn" and this write carries
    ledger="pending" (2 key(s) differ in total: ledger, ts). The document on disk
    is canon_name="doc-probe" step=2; this write is canon_name="doc-probe" step=2.
    A sidecar may only GAIN information, never restate it - the record is the
    evidence for the artifact beside it, and this function does not get to rewrite
    evidence in order to report success. Give the run its own TRIOS_CANON_NAME or
    TRIOS_CHECKPOINT_DIR, or set TRIOS_ALLOW_SIDECAR_OVERWRITE=1 to replace the
    record deliberately. Nothing was written and nothing was deleted.
exit=1
```

(The message is one long line in the terminal; it is wrapped here and nowhere
else altered.) The differing-key list is **at least** `ledger, ts`. Re-running
the same probe while a source file changed in between produced
`3 key(s) differ in total: ledger, source_sha256, ts` -- the same guard also
reporting that the tree it was asked to certify had moved.

Note *what* refuses. The `.bin` write is **accepted** -- `save_scoped` permits a
re-save of identical bytes, because the file that would result is the file
already there, which is the determinism result itself. The refusal is on the
**evidence document** beside it, and it is unconditional on a re-run for a
reason that has nothing to do with the weights: `ts` always moves, and the first
sidecar write of any run carries `ledger: "pending"` while the record already on
disk is terminal (`"skipped-no-dsn"` here, `"written"` for `r6-headline`). A
sidecar may only GAIN information, so a terminal record cannot be rewritten back
to `pending`.

Two escapes, and they are different claims. Use a fresh `TRIOS_CANON_NAME` (or a
fresh `TRIOS_CHECKPOINT_DIR`) to get a second, independent artifact and then
compare the two `sha256` fields -- that is how same-machine determinism is
demonstrated, and it keeps both records. Set `TRIOS_ALLOW_SIDECAR_OVERWRITE=1`
only when you deliberately want the old record gone; it prints a `WARN` naming
the differing keys, and the document that was there is not recoverable.

`TRIOS_CHECKPOINT_DIR` is the **root**, not the run directory: the canon name is
appended to it, so the artifacts above land in
`$TRIOS_CHECKPOINT_DIR/<canon>/<step>.bin` and `.../<step>.json` (default root
`checkpoints/`).

`TRIOS_EVAL_CHUNKS` is deliberately **not** set: the default grid (40 windows)
is what the record above declares, and changing the default would make every
existing BPB in this repository incomparable with every new one -- the same
mistake as gating `gf16_floor` on `--eval-every`. The defect was that the grid
was never *stated*, not that 40 is the wrong number. Set
`TRIOS_EVAL_CHUNKS=0` for full coverage (775 non-overlapping windows, 99.98% of
the corpus, no sampling error) when you want the exact corpus mean rather than
a number comparable with these.

A regenerated sidecar carries the `trainer.sha256` of *your* build. `ckpt_replay`
re-hashes the executable it is handed rather than trusting the path it was
given, and refuses to replay anything whose hash does not match the record.

**That refusal is why the following two facts have to be stated up front rather
than discovered by an auditor.**

*First: as of 2026-08-03 the L3 verdict is demonstrated INTRA-laboratory only.*
Until that date no second laboratory could have matched `trainer.sha256`, for a
reason that had nothing to do with the science:

```console
$ strings -a target/release/trios-train | grep -c '/Users/playra'
588
```

588 absolute paths of the original builder's home directory were compiled into
`.rodata` -- 504 of them under `.cargo/registry`, the rest this crate's own
source paths, mostly via panic location strings. There was no `.cargo/config.toml`,
no `--remap-path-prefix` and no `SOURCE_DATE_EPOCH` anywhere in the repository.
An auditor with identical sources, an identical `Cargo.lock --locked` and the
pinned `rustc` from `rust-toolchain.toml` still got a different binary, because
their `$HOME` is spelled differently. Different hash, no verdict.

The standard fix, `--remap-path-prefix`, is now applied by
[`scripts/repro_build.sh`](scripts/repro_build.sh) and takes that count to 0.
Dependency vendoring, the other half of the standard recipe, is **not** done.
**No third-party rebuild has yet matched this binary.** What has been measured
is a two-laboratory trial on a single macOS host: the two builds agreed on
every byte of code and data and differed in 48 bytes -- the Mach-O `LC_UUID`
and the code-signature hash derived from it. That is a much smaller gap than
588 embedded paths, and it is still not a match. See
[docs/REPRODUCIBILITY-GRADING.md](docs/REPRODUCIBILITY-GRADING.md#scope-of-the-verdict).

*Second: the published headline record names a `trainer.sha256` that is not the
binary a fresh `cargo build --release` produces today.* The record above names
`5a59f44e...`; `src/` has continued to change since it was snapshotted, so the
reproduction command printed above returns `TRAINER MISMATCH` against a
freshly built `target/release/trios-train`. This is the verifier working, not
failing: it re-hashes the executable it is handed rather than trusting the path
it was given. The `5a59f44e` binary is archived outside the repository (its
path is recorded verbatim in the sidecar's `trainer.path`) and the grading
transcript in `checkpoints/r6-headline/VERDICT.txt` names the exact bytes that
produced the verdict. The record is a historical attestation of one executable,
not a command that passes against an arbitrary later build. The previous
headline's trainer, `73fae5db...`, has the same status.

**Scope.** What the `VERIFIED` above demonstrates is *repeatability* in the
ISO 5725 / VIM sense -- same operator, same equipment, same conditions -- not
*reproducibility*, and a second laboratory has in fact already disagreed with
it byte for byte. Read
[Scope of the verdict](docs/REPRODUCIBILITY-GRADING.md#scope-of-the-verdict)
before quoting any of these numbers.

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
12 000-step row is the sidecar `final_val_bpb`. The seventeen-digit entries are
`f32` decimal expansions, not measurements to seventeen places.

**Pairing condition for this table:** every row was measured with the identical
sampling plan, so the sampling term is common-mode and the step-by-step
agreement is meaningful. Read across grids instead and the applicable band is
the unpaired one, `+/- 0.13` bpb at `k = 2`
([`docs/EVAL-UNCERTAINTY.md`](docs/EVAL-UNCERTAINTY.md) section 3b), which the
agreements below would survive and the *disagreements* would not.

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

| `TRIOS_GF16_FLOOR_EVERY` | checkpoint sha256 | matches an archived run? | `final_val_bpb` at 12 000 (`f32` expansion) |
|---|---|---|---|
| `1` (default) | `8a86fe691aef64fc` | -- this is the new headline | `2.6347548961639404` |
| `1000` | `b2fe52b0760dc369` | **yes -- `igla-honest-20260802/12000.bin`, byte for byte** | `2.614117383956909` |
| `3000` | `7a95c99b962d565f` | no | `2.614041805267334` |
| `4000` | `3ecc1c925e42ff57` | no | `2.5910677909851074` |

**The finding is the hash column, and it is a byte comparison.** One seed, one
corpus, four cadences, four *different artifacts*: changing the floor cadence
provably changes the weights, with no statistics involved and no error bar to
argue about. The `1000` row shows the escape hatch works -- the post-fix trainer
reproduced a pre-fix artifact exactly, byte for byte -- and that byte match is
also what *identifies* that run's cadence as 1000, since its own record never
said so.

**The BPB column is corroboration only, and it is weaker than it looks.** Its
span is 2.5911 to 2.6348, i.e. 0.0437 bpb, against a measured between-grid
estimator sigma of 0.0358
([`docs/EVAL-UNCERTAINTY.md`](docs/EVAL-UNCERTAINTY.md)). That is ~1.2 sigma
across the *whole* span and far less between adjacent rows, so these four
readings are not separable from one another on the metric alone. **Pairing
condition:** all four rows were measured with the identical sampling plan --
same val corpus, same `eval_chunks`, same `eval_seq`, same stride -- so the
sampling term is common-mode and cancels, which is the only reason the
fourth-decimal agreement of the `1000` row against its archived twin means
anything. Quote these numbers only with that condition attached. A reader who
re-measures on any other grid is in the **unpaired** case, where the applicable
band is `+/- 0.13` bpb at `k = 2` and every row above is indistinguishable from
every other. The seventeen-digit values are the decimal expansions of `f32`s, not
measurements to seventeen places.

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
# --max-steps must be raised past the record's steps_total, or the run is
# REFUSED before it starts. The default budget is 2 000.
./target/release/ckpt_replay --record checkpoints/r5-adv-recheck/12000.json \
  --trainer ./target/release/trios-train --max-steps 12000
```

| Verdict | Exit | Meaning |
|---------|------|---------|
| `VERIFIED on <os>/<arch>` | 0 | the record's own parameters reproduce its bytes on this host |
| `MISMATCH` | 1 | they do not; both hashes are printed |
| `INCOMPARABLE` | 2 | the record does not describe its own inputs; nothing can be graded |
| `REFUSED` | 3 | the replay would exceed `--max-steps`, which defaults to 2 000 -- so **every** record in this repo with `steps_total` above 2 000, the headline included, is `REFUSED` until the budget is raised explicitly |

The archived `igla-honest-provenance/12000.json` returns **`INCOMPARABLE`**: a
`trios-checkpoint-record/1` sidecar states neither `steps_total` nor
`gf16_floor_every`, and both are first-order -- `steps_total` fixes where the
70% mark falls, `gf16_floor_every` fixes how often the weights are rewritten
past it.

**And so does every `trios-checkpoint-record/3` record still on disk**,
including `r4-docs-repro/12000.json`, which is the sidecar this file led with
until 2026-08-03. `trainer.sha256` was added to the verifier's
`REQUIRED_FIELDS` after those records were written, so they are now ungradable
by the tool that once graded them:

```console
$ ./target/release/ckpt_replay --record checkpoints/r4-docs-repro/12000.json --max-steps 12000
INCOMPARABLE: trainer.sha256 not recorded; this artifact cannot be graded from its own provenance record
```

That is stated here rather than buried, because it is a real property of the
method and not an accident of this repository: **a provenance schema that adds
a required field ungrades its own predecessors.** Every record written before
the field existed drops a level, no matter how honest the run behind it was.
The bytes are unaffected -- `r4-docs-repro/12000.bin` and the new headline
`r5-adv-recheck/12000.bin` have the identical sha256 `8a86fe69...`, confirmed
with `shasum -a 256` on both files -- but the *grade* is a property of the
record, not of the artifact. The current headline is
`trios-checkpoint-record/4` and does carry the field.

A `VERIFIED` verdict is always printed with the platform it was obtained on,
because the same seed and corpus produce different checkpoint hashes on macOS
and on Linux -- measured, not assumed, and the CI evidence is in the grading
doc. The scale, the evidence and the scope are in
[`docs/REPRODUCIBILITY-GRADING.md`](docs/REPRODUCIBILITY-GRADING.md); read
[Scope of the verdict](docs/REPRODUCIBILITY-GRADING.md#scope-of-the-verdict)
before treating any `VERIFIED` here as a reproducibility claim.

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

Determinism, which *is* verified **on one host**: identical parameters
reproduce byte-identical checkpoints and identical BPB across independent runs
on the same machine. Across machines it is verified *false* -- the same
documented command on x86_64 Linux produces `bb14ab18...` where aarch64 macOS
produces `8a86fe69...`. In ISO 5725 / VIM terms this repository demonstrates
*repeatability*, not *reproducibility*; both results, and why the difference is
not pedantry, are in
[Scope of the verdict](docs/REPRODUCIBILITY-GRADING.md#scope-of-the-verdict).

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

- **Verified, scoped to one host:** training is bit-deterministic. Identical
  parameters produce byte-identical checkpoints and identical BPB across
  independent runs on the same machine. This is *repeatability* in the
  ISO 5725 / VIM sense, not *reproducibility*.
- **Verified false across machines:** the same documented command on x86_64
  Linux (glibc 2.39, same rustc, `--locked`, corpus checked against the
  manifest) produces a different checkpoint --
  [CI run 30767491098](https://github.com/gHashTag/trios-trainer-igla/actions/runs/30767491098),
  `MISMATCH`. **The bytes do not survive the crossing at all.** The metric
  appears to, to within `0.003` bpb at 12 000 steps, but that number carries a
  condition and is worthless without it: it is **paired** -- both arms ran the
  identical sampling plan, so the sampling term is common-mode and cancels.
  Against the unpaired estimator sigma of `0.0358`, `0.003` bpb is `0.08`
  sigma, so the honest reading is *the experiment could not resolve a
  difference*, *not* that the platforms agree to three decimals. Quote `0.003`
  only with the pairing condition attached; where the sampling plans are not
  identical the band is the unpaired `+/- 0.13` bpb at `k = 2`. See
  [`docs/EVAL-UNCERTAINTY.md`](docs/EVAL-UNCERTAINTY.md) section 4a, and
  `docs/CROSS-ARCH-DIVERGENCE.md` for the reminder that the same pair of
  checkpoints differs in 43.70% of its parameters.
- **Verified, and new:** a checkpoint produced from a **named source tree**.
  Of the 51 sidecars under `checkpoints/`, 47 record `git_dirty: true`, 4
  record `null` and none records `false`, so intra-laboratory L3 had never
  actually been demonstrated here. A throwaway clone parked at `3c1f751` with
  an empty `git status --porcelain` now yields a record whose commit and tree
  agree, reproduced byte-identically twice
  (`7e567530acd2d265a08832dd845ac2d89945fee810f06bc428dbe63ef6774ec8`). It is a
  2 000-step run from a commit that predates the current honesty fixes, so it
  demonstrates provenance discipline and is **not** a headline record --
  see [`docs/CLEAN-TREE-PROVENANCE.md`](docs/CLEAN-TREE-PROVENANCE.md).
- **Verified:** the headline artifact re-derives from its own record on the
  platform it declares -- `ckpt_replay` returns `VERIFIED on macos/aarch64`,
  exit 0, after re-executing all 12 000 steps. What that attests is an
  *executable*, not a source tree: `source_sha256` is recorded, never
  re-derived.
- **Verified:** the train/val split is byte-disjoint, checked over the full
  corpus (the previous check sampled with `step_by(256)` and would have missed
  an overlap 255 times out of 256).
- **Verified:** checkpoints are now actually written -- atomic
  `tmp + fsync + rename`, SHA-256 taken over bytes re-read from disk, magic
  `TRIOSCKP`, truncation rejected on load, JSON sidecar always emitted.
- **Not established:** any capability claim. **GATE-2 as specified is
  unreachable on this architecture given the measured calibration.** This is
  not a gap left to close, and framing it as one is what made a measurement
  defect look like progress. The arithmetic: the honest headline is
  `2.63 +/- 0.13` bpb at `k = 2`; `docs/EVAL-UNCERTAINTY.md` measures that a
  **100% verbatim train/val overlap** -- total contamination, the most a leak
  can possibly buy -- is worth only ~0.12 bpb on this architecture. That puts
  the floor of the entire instrument at ~2.51 -- a floor which inherits the
  headline's own `+/- 0.13`, so read it as ~2.5, not as three decimals. The
  conclusion survives the wider band because the distances below are several
  multiples of `U`, not fractions of it. `BPB_CHAMPION = 2.5193`
  (`src/invariants.rs`, itself documented there as retracted, so an upper bound
  on what was ever even *claimed*) sits within 0.005 of it -- a distance far
  inside `U = 0.13`, i.e. indistinguishable from the floor rather than below it.
  The Gate-2 threshold `DEFAULT_TARGET_BPB = 1.85` is ~0.66 bpb below that
  floor (~5 x `U`), and `IGLA_TARGET_BPB = 1.5` is ~1.0 below it (~8 x `U`). Neither is reachable by training a
  ~196.6K-parameter byte model harder; both are reachable only by a measurement
  defect -- which is exactly how the `1.5492` in the next bullet came about.
  The constants are retained because other call sites read them, and are
  documented as unreachable rather than as pending.
- **Not citable:** BPB `1.5492`, reported as an "honest Gate-2 pass" in the
  PR's own `LEAK_INVESTIGATION.md`. At that commit `train_loop.rs` contained no
  `bpb_sample` call at all -- the wiring landed two days later -- so those rows
  came from an out-of-repo stdout parser with no artifact behind them.

See [`docs/TRAINING_FLOW_V2.md`](docs/TRAINING_FLOW_V2.md) for the training plan
and [`docs/audit/HONEST_FINDINGS.md`](docs/audit/HONEST_FINDINGS.md) for the
full retraction ledger.

## License

MIT
