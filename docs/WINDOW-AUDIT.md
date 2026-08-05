# The window audit

**The defect.** Verification cost exactly what production cost.

`ckpt_replay` is described in this repository as a spot-check verifier. It is
not one: it re-executes from step 0, because the `TRIOSCKP` container
serialises weights and nothing else -- no AdamW first and second moments, no
per-instance step counters, no batch-sampler state -- and until now no binary
in the crate accepted a warm start. `grep -rn resume --include=*.rs src/`
returned one comment and zero lines of code.

So to check one claim about a 12000-step run, an auditor paid for a
12000-step run. For a 1B-parameter model that is the vendor's entire training
budget, per claim, and the predictable consequence is that nobody re-runs
anything and the scheme collapses into a self-signed affidavit. The
reproducibility argument this repository makes is only worth what its
verification costs; at 1.0x it is worth nothing.

**The fix is a format addition, not research.** The trainer's whole mutable
state is small: 12 `train_loop::AdamW` instances (embed, `ctx0..ctx5`, proj,
attn_down, attn_up, head, attn_w -- `m`, `v` and a bias-correction counter
each) plus the single `u64` LCG at `train_loop.rs` that drives batch sampling.
Written down beside the checkpoint, that turns audit cost from *T* steps into
*N* steps.

## What it buys, quantitatively

Split a *T*-step run into *W* windows of *N* steps, `W = T / N`. The vendor
publishes a checkpoint and a resume record at each boundary. The auditor
picks *k* windows -- **after** the artifacts are published, and by a rule the
vendor cannot predict -- and re-executes only those.

| quantity | value |
|---|---|
| audit compute, as a fraction of training | `kN / T = k / W` |
| detection probability, vendor cheated in exactly one window | `k / W` |
| detection probability, vendor cheated in *m* windows | `1 - C(W-m, k) / C(W, k)` |

Two readings of the same table:

* **12000 steps, 120 windows of 100, challenge 6.** The auditor spends 5% of
  the training budget and catches a single fabricated window one time in
  twenty. That is weak.
* **The vendor does not know which 6.** Cheating is only useful if it changes
  the published outcome, and changing the outcome of a long run generally
  means many windows, not one. At `m = 12` (10% of the run fabricated) and
  `k = 6`, detection is `1 - C(108,6)/C(120,6) = 0.48`. At `m = 24`, it is
  0.75. The scheme is cheap deterrence against wholesale fabrication, not a
  proof of honesty.

Say that plainly rather than dressing it up: **a window audit bounds how much
of a run can be fake before the vendor is likely caught. It does not certify
that any unchallenged window is real.** The one thing it does certify is the
challenged windows themselves, and it certifies those completely -- byte
equality, not agreement within a tolerance.

## What it does NOT prove

**1. It is a same-machine claim.** The demonstration in
`evidence/window-audit/` ran both halves on one aarch64 macOS machine with one
binary. It establishes that the resume record is *complete* -- that no state
which decides the weights was left out -- and nothing more.

The cross-architecture boundary documented in `docs/CROSS-ARCH-DIVERGENCE.md`
is **unchanged by this work**. The same seed-47 12000-step recipe produced
`bb14ab18...` on x86_64 Linux and `8a86fe69...` on aarch64 macOS with a
byte-identical compiler, `--locked` dependencies and all three corpus hashes
agreeing. Nothing in `TRIOSRSM` addresses that, and a resumed segment executed
on a different architecture will diverge for exactly the same reasons a whole
run does.

**2. An auditor on their own hardware cannot grade on bits.** This is the
practical case -- the auditor has their own machine, not the vendor's -- and
it is where the claim has to be stated carefully:

> A window re-executed on the auditor's own hardware may be graded ONLY on
> metric agreement within a stated uncertainty budget. Never on bit equality.

The budget to name is **Budget I, `U = 0.13` bpb (k = 2)** from
`docs/EVAL-UNCERTAINTY.md` -- the scope of which is precisely "reproducing a
stated reading on the SAME corpus hash". That is the right budget here because
a window audit re-reads a *declared* corpus: the resume record carries the
train and val SHA-256 and refuses a run that reads anything else, so the
corpus contributes nothing to the disagreement. Quoting the wider Budget II
would be over-generous to the vendor; quoting `+/- 0.07` (k = 1) would be
over-strict, and quoting bit equality across machines would fail every honest
vendor on earth, including this one.

Bit equality is the grade **only** when the auditor re-executes on the
declared platform -- same os, same arch, same rustc, same `source_sha256`,
all four of which every sidecar records.

**3. AdamW only.** `TRIOSRSM` version 1 serialises the `train_loop::AdamW`
instances. The Muon path holds a momentum buffer and its own step counter in
`optimizer::MuonOptimizer`, and version 1 carries neither, so
`--resume-from` on `--optimizer muon` / `muon-cwd` is **refused** with a
non-zero exit that names the missing state. It is not warm-started from a
zeroed buffer: that would run to completion, print a plausible BPB, and be a
segment of no run at all. A Muon run remains auditable only at 1.0x.

**4. It does not make an unaudited artifact auditable retroactively.** Every
`.bin` this repository published before this format existed has no `.resume`
beside it. `resolve_resume_pair` refuses those by name rather than starting
cold, because starting cold silently zeroes the moments.

## The format

Weights stay exactly where they were. **`to_checkpoint_bytes` was not
touched**, so every digest already published -- `bb14ab18...`, `8a86fe69...`,
`7e567530...`, the seals in `evidence/SEALS.txt` -- is still valid. The
resume state lands in a separate file, `{step}.resume` beside `{step}.bin`,
with its own magic `TRIOSRSM`, its own `u32` version, and its own SHA-256 over
the bytes as re-read from disk. The byte layout is documented in
`src/checkpoint.rs`, in the same form as the `TRIOSCKP` layout above it.

It carries: seed, step, the sampler state `rng_s`, planned total steps,
`--eval-every`, `TRIOS_GF16_FLOOR_EVERY`, the model shape, base lr, weight
decay, the GF16 and QAT flags, the train and val corpus digests, the SHA-256
of the weight checkpoint it pairs with, the running EMA and running minimum,
and for each of the 12 optimizer instances its name, its step counter and its
full `m` and `v` vectors.

Save discipline is `save_scoped`'s: tmp file in the same directory, `fsync`
before close, atomic rename, best-effort directory fsync, then hash the file
as it exists on disk.

### Every way it refuses

A refusal is a non-zero exit whose message begins `RESUME REFUSED` and names
one reason. **There is no permissive mode and no fallback**, because the
fallback -- resuming with zeroed moments -- looks exactly like success.

| reason | raised when |
|---|---|
| `bad-magic` | the file is not a `TRIOSRSM` container |
| `unsupported-version` | a version this build does not read |
| `truncated` / `trailing-bytes` | the file length disagrees with its own directory |
| `digest-mismatch` | the trailing SHA-256 does not cover the bytes present |
| `malformed-field` | a non-ASCII name, a non-hex digest, a nonzero reserved byte |
| `weight-digest-mismatch` | the record pairs with different weights than the `.bin` read |
| `corpus-mismatch` | a different train or val SHA-256 |
| `shape-mismatch` | hidden, d_model, layers, vocab, dim, num_ctx, or the optimizer instance names and lengths |
| `cadence-mismatch` | a different `--eval-every` or `TRIOS_GF16_FLOOR_EVERY` |
| `recipe-mismatch` | a different seed, total steps, lr, weight decay, GF16 flag or QAT format |
| `nothing-to-resume` | the record is already at the final step |
| `muon-path` | `--resume-from` with a Muon optimizer |
| `missing-file` | either half of the `(bin, resume)` pair is absent |

Two of those deserve a note. **`cadence-mismatch`** is there because
`gf16_floor` rewrites `embed`, `proj`, `lm_head` and every `ctx` slab in
place: its cadence is part of the recipe, and `--eval-every` is checked
alongside it because that is the knob whose coupling to the weights caused
`2.6141 vs 2.6169` (see `docs/OBSERVATION-INDEPENDENCE.md`). **`recipe-mismatch`**
covers `steps_total` because `cosine_lr(step, steps_total, base_lr, warmup)`
means a segment run to a different total applies different learning rates to
the same step numbers -- a divergence with no error message attached.

## Using it

```bash
# vendor: publish a checkpoint and a resume record every 100 steps
TRIOS_CHECKPOINT_EVERY=100 trios-train --seed 47 --steps 12000 ...

# auditor: re-execute the window 4900 -> 5000, and only that window
trios-train --seed 47 --steps 12000 --eval-every 1000 \
  --train-data <declared> --val-data <declared> \
  --resume-from checkpoints/<run>/4900.bin
```

`--steps` is the ORIGINAL total, not the window length. Either half of the
pair may be named: `4900.bin` and `4900.resume` resolve to the same pair.
`--resume-from` is refused with `--sweep`, `--seed 0` and `--config`, all of
which describe more than one run.

## The measurement

Full commands, digests and platform in
`evidence/window-audit/PROVENANCE.txt`. Recipe: seed 47, hidden 64, 200 steps,
`--eval-every 50`, checkpoints every 100, the byte-disjoint tinyshakespeare
train/val pair of the three-way split (`train_core` sha256 `21f0788a...`,
`val` sha256 `2088af36...`; see `data/README.md`). The BPB figures below are
not comparable to the two-way headline of 2.63 and are not offered as a
quality claim - they exist so that two runs can be compared to each other.

| artifact | sha256 |
|---|---|
| monolith `100.bin` | `d25ea6d6f805e8c54b0ecdff8f29b74b0442ad0cd402d2f874c4fe5e4069e901` |
| monolith `200.bin` | `180e8af9e78299d6a4166fb33ab8113d885e638263b5eef11ca99e1c26234a0e` |
| segment `200.bin` (resumed at 100) | `180e8af9e78299d6a4166fb33ab8113d885e638263b5eef11ca99e1c26234a0e` |

**Byte-identical.** So are the two `200.resume` records
(`7a5bf8fed881f22fd3df6c5b2d25edb88e7099401648845282c726a5f269903b`), which
is the stronger statement: the segment reproduced the moments, the counters
and the sampler state, not only the weights. The printed `val_bpb` agreed to
every digit at both steps the two runs share -- 3.7974 at 150, 3.7925 at 200.

`tests/window_audit_resume.rs` re-runs the whole comparison in a tempdir on an
8-step budget split at 4, asserts byte equality, and asserts one refusal per
reason in the table above. It is a test, not a report: if the segment ever
stops reproducing the monolith, that test fails rather than this document
being edited.

## What is worth doing next

The honest gap is the auditor's own machine. Everything above is one machine,
and the interesting audit is the one an outsider runs. Two concrete steps:

1. Re-execute a challenged window in the existing
   `.github/workflows/cross-arch-repro.yml` job and record what the metric
   disagreement actually is, against `U = 0.13`. The answer is presently
   unknown; it may be well inside the budget or it may not.
2. Localise where a cross-architecture segment first diverges -- first
   differing tensor and element index, at 0 steps and at small N -- which
   separates an init/RNG cause from a floating-point contraction or
   reduction-order cause. See `docs/DIVERGENCE-LOCALIZATION.md`.
