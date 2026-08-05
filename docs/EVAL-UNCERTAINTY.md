# The uncertainty budget of `val_bpb`

> ## READ THIS BEFORE QUOTING ANY NUMBER BELOW `[ADDED 2026-08-03]`
>
> **The seven-grid study and the headline are TWO DIFFERENT MODELS.**
>
> | | checkpoint | steps | `val_bpb` |
> |---|---|---|---|
> | the seven-grid study (sections 1, 2, 2a) | `e55d91d8...` | **1,200** | ~3.06 |
> | the headline artifact (section 3) | `8a86fe69...` | **12,000** | 2.6348 |
>
> Ten times the training, ~0.44 bpb apart. **`sigma = 0.0358` was never measured
> on the checkpoint it is attached to.** Carrying it from one to the other is an
> **ASSUMPTION**, and section 2's earlier claim that it is "a property of the
> estimator, not of one model" is **falsified by the records' own
> `val_bpb_stderr` field** -- see section 2b, "The transfer assumption is
> measurably wrong".
>
> **The headline record carries its own, larger sampling term.**
> `checkpoints/r6-headline/12000.json` states `val_bpb_stderr =
> 0.05509733036160469` at `eval_chunks = 40`. That is **larger than the
> `+/- 0.04` this document used to publish as a `k = 1` band.** The honest
> combined figures are **`u_c = 0.066` (k = 1)** and **`U = 0.13` (k = 2)**, not
> `0.04` and `0.07`. A counterparty who opens the sidecar reads a bigger error
> bar than the pitch quotes; the correction is in
> [section 3b](#3b-the-budget-proper) and the decision rule that follows from it
> is in [section 3c](#3c-the-decision-rule-draft-clause).
>
> ## TWO BUDGETS, AND WHICH ONE APPLIES `[REVISED 2026-08-06]`
>
> `u_c = 0.0657`, `U = 0.13` answers exactly one question: **how far may a
> second laboratory's reading of the SAME hashed corpus fall from ours before
> the difference is real?** It does **not** answer *how far is this model's BPB
> from the number we quote?*, because it excluded corpus choice as "not
> quantified". **Corpus choice is now quantified, and it is the largest
> component in the budget by a factor of five.** The same weights -- both
> `12000.bin` hashing to `902cfb69a84b...` -- read against two 100,000-byte
> slices of the same book give **2.638520** and **2.919297**
> ([HELD-OUT-PROTOCOL.md](HELD-OUT-PROTOCOL.md) section (iii), lines 315-321, in
> this same directory), a difference of **0.280777 bpb**: **4.3x** the `u_c`
> above and **2.1x** the expanded `U`.
>
> | budget | scope | `u_c` (k = 1) | `U` (k = 2) |
> |---|---|---:|---:|
> | **I -- declared corpus** | reproducing a stated reading on the SAME corpus hash | **0.0657** | **0.13** |
> | **II -- corpus choice included** | comparing BPBs measured on DIFFERENT corpora, or reading a BPB as a property of the MODEL | **0.2883** | **0.58** |
>
> The conformity clause of [section 3c](#3c-the-decision-rule-draft-clause)
> keeps `U = 0.13`, and it is entitled to only because clause L2-1 already
> requires the corpus `sha256`: **when the eval corpus is declared and hashed,
> corpus choice contributes nothing to reproducing the number.** Quote `0.13`
> with that condition attached, or quote `0.58`.

**What this document is.** A measured error bar for the number this trainer
publishes as its result, plus -- in [section 3b](#3b-the-budget-proper) -- the
budget the title promises: every component enumerated, Type A separated from
Type B, the quantified part combined, the unquantified part named rather than
estimated, and a coverage factor stated. Section 3c then does the one thing a
dispersion cannot do on its own: it issues a **decision**, with an acceptance
interval, a guard band and a declared consumer's risk, in the vocabulary of
JCGM 106:2012. Every figure below was produced by the runs recorded in
"How to reproduce" at the bottom, on 2026-08-03, with the working tree at
`git_sha 3c1f751cf4376c13d26e247c2cd86357ab51dd20` (dirty - the schema 6 change
this document accompanies was uncommitted at measurement time, which the
sidecars record as `git_dirty: true`).

**Why it exists.** `evaluate()` averaged a hardcoded 40 windows of `SEQ + 1` =
129 tokens. That is 5,160 bytes: **5.16% of the 100,000-byte validation
corpus**, on a fixed evenly-spaced grid. The count and the window length were
literals in `src/train_loop.rs` and appeared in no field of `CheckpointRecord`.
The result was published as `2.6347548961639404` - seventeen digits, no
uncertainty - and the crate compares `BPB_CHAMPION = 2.5193` against new runs at
the fourth decimal.

A quantity measured on an unstated 5% sample, with no dispersion reported, is
not a measurement result. It is a reading.

---

## 1. The experiment -- ON A 1,200-STEP CHECKPOINT, NOT THE HEADLINE

One fixed set of weights, seven different window grids. **Those weights are
`e55d91d8...` at 1,200 steps, `val_bpb ~ 3.06`. They are NOT the headline
artifact `8a86fe69...` at 12,000 steps, `val_bpb = 2.6348`.** Every sigma in
sections 2 and 2a is a property of the 1,200-step model as read by this
estimator, and applies to the headline only by assumption -- an assumption
section 2b measures and rejects.

* Seed 47, 1,200 steps, `hidden = 384`, two allocated attention blocks, one
  effective, `lr = 0.003`, `optimizer = adamw`, `gf16_enabled = true`,
  `gf16_floor_every = 1`, `vocab = 128`, `attn_scale = 0.1`, `attn_seq = 8`.
  196,608 effective parameters of 212,992 serialized -- the layer-2 block is
  allocated and provably frozen; see the test
  `run_single_emits_a_loadable_artifact_and_freezes_layer_two` in
  `src/train_loop.rs`.
* Train corpus `data/tiny_shakespeare.txt`, 1,015,394 bytes, sha256
  `1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d`.
* Val corpus `data/tiny_shakespeare_val.txt`, 100,000 bytes, sha256
  `2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502`, and its
  six leading prefixes.
* Platform: macos/aarch64, rustc 1.96.0 (ac68faa20 2026-05-25), release profile.

**The weights were provably identical across all seven runs.** Training reads
only the train corpus; the val stream is an observation. All seven checkpoints
hash to

```
e55d91d83515769925a41971f28a94f0a7f31dac05f4120f376201c166504f7d   (852,272 bytes)
```

so every difference in the table below is the estimator moving, not the model.

---

## 2. Seven grids on one 1,200-step checkpoint

**What varied across these seven rows was the validation CORPUS -- the pinned
100,000-byte file and its six leading prefixes -- not the window grid at a fixed
corpus.** `eval_chunks` is 40 throughout, so the stride is forced to change
because the corpus length changed. That makes `sigma = 0.0358` an **upper bound
on the sensitivity to grid choice**, not a measurement of it: it contains both
the grid movement and whatever the six truncated corpora contribute. The
in-scope measurement -- same corpus, grid varied -- is in section 2a and is much
smaller. See [budget row 1](#3b-the-budget-proper).

Each row is a full run of the trainer whose only difference is the val prefix.
`eval_chunks` is 40 in every row - the published default - so `n` is constant
and the spread across rows is the estimator's own between-grid variability.
`stride` is the distance between window starts, which is what actually changes.

| val bytes | stride | windows | eval bytes | coverage | **val_bpb** | within-grid stdev | within-grid stderr |
|----------:|-------:|--------:|-----------:|---------:|------------:|------------------:|-------------------:|
| 100,000 | 2,496 | 40 | 5,160 | 5.16% | **3.078032** | 0.256175 | 0.040505 |
|  90,000 | 2,246 | 40 | 5,160 | 5.73% | **3.084223** | 0.302480 | 0.047826 |
|  80,000 | 1,996 | 40 | 5,160 | 6.45% | **3.054324** | 0.270345 | 0.042745 |
|  70,000 | 1,746 | 40 | 5,160 | 7.37% | **3.074461** | 0.261377 | 0.041327 |
|  60,000 | 1,496 | 40 | 5,160 | 8.60% | **2.987364** | 0.230873 | 0.036504 |
|  50,000 | 1,246 | 40 | 5,160 | 10.32% | **3.028411** | 0.231778 | 0.036647 |
|  40,000 |   996 | 40 | 5,160 | 12.90% | **3.080693** | 0.250082 | 0.039541 |

```
n            = 7 grids
mean         = 3.055358 bpb
sample stdev = 0.035834 bpb      <- sigma
stderr       = 0.013544 bpb      (= stdev / sqrt(7))
min          = 2.987364
max          = 3.084223
range        = 0.096859 bpb
```

**Full coverage, same weights, same corpus:** 775 non-overlapping windows,
99,975 of 100,000 tokens (99.98%), `val_bpb = 3.084421`, within-grid
stderr 0.009055. This is the exact mean over the corpus - the number the 5%
sample is estimating. The seven 40-window readings sit between 0.01 and 2.71
sigma away from it.

Round 6 ran the same experiment on a different checkpoint and got stdev 0.0338
over a range of 0.0901. This measurement is 0.0358 over 0.0969. The two agree
**with each other**, and that agreement was previously written up here as
"sigma ~= 0.035 bpb is a property of the estimator, not of one model."
**`[RETRACTED 2026-08-03 -- see section 2b]`.** Both of those checkpoints are
early ones (~3.06 and ~3.28 bpb). Two early models agreeing says nothing about a
converged one, and the records now carry a field that settles it.

### 2a. The sample size sweep

Same weights, same full 100,000-byte corpus, varying `--eval-chunks`. This
isolates `n` from the corpus:

| eval_chunks | stride | eval bytes | coverage | val_bpb | within-grid stderr | deviation from full coverage |
|------------:|-------:|-----------:|---------:|--------:|-------------------:|-----------------------------:|
|  10 | 9,987 |  1,290 |  1.29% | 3.087329 | 0.056937 | +0.002908 |
|  20 | 4,993 |  2,580 |  2.58% | 3.036312 | 0.050365 | -0.048109 |
|  **40** | **2,496** | **5,160** | **5.16%** | **3.078032** | **0.040505** | **-0.006389** |
|  80 | 1,248 | 10,320 | 10.32% | 3.055597 | 0.027443 | -0.028824 |
| 160 |   624 | 20,640 | 20.64% | 3.079580 | 0.019752 | -0.004841 |
| 320 |   312 | 41,280 | 41.28% | 3.078354 | 0.015004 | -0.006067 |
| 775 (`=0`, full) | 129 | 99,975 | 99.98% | 3.084421 | 0.009055 | 0.000000 |

Note the non-monotonicity: `n = 10` lands closer to the true corpus mean than
`n = 80` does. That is what sampling noise looks like, and it is why a single
reading at any `n` cannot be trusted to its fourth decimal.

**This is the in-scope measurement of grid choice, and it is the one the budget
was missing.** Every row here holds the corpus fixed at the pinned 100,000-byte
file whose sha256 is `2088af36...` and moves only the grid -- which is exactly
what a reproducer who honours the corpus hash but picks its own `eval_chunks`
does. At the published default the `n = 40` reading deviates from full coverage
by **-0.006389 bpb**, which is **0.18 sigma**. The seven-row `0.0358` in section
2 is therefore an **upper bound** on grid choice at fixed corpus, inflated by
the corpus truncation the seven rows also varied; it is not a measurement of the
component a hash-honouring reproducer actually experiences.

**Attribution warning.** The `0.0405` in the `n = 40` row is **the 1,200-step
checkpoint's** within-grid stderr. `[CORRECTED 2026-08-03]` This document
previously carried it into the bridging argument for the headline. **The
headline record's own value is `0.055097`** (`checkpoints/r6-headline/12000.json`),
36% larger. The sentence that used to stand here -- that the recorded
`val_bpb_stderr` is "a usable proxy for the real uncertainty at the default
plan", the two figures being "the same order" -- is **`[RETRACTED 2026-08-03]`**:
it was true of the 1,200-step record and it is what let this document treat the
record's own sampling term as redundant with `sigma` and leave it out of the
budget. It is not redundant, it is not the same order, and at the headline it is
the **larger** of the two. See sections 2b and 3b.

---

### 2b. The transfer assumption is measurably wrong `[ADDED 2026-08-03]`

Schema 6 made every record state its own within-grid sampling term. Reading that
field off the sidecars, at the published default `eval_chunks = 40`, on the same
pinned val corpus, for seed-47 checkpoints of the same architecture:

| step | checkpoint | `final_val_bpb` | `val_bpb_stderr` | record |
|-----:|---|---:|---:|---|
| 0 | `4f854c82` | 7.000166 | 0.001236 | `checkpoints/loc-a/0.json` |
| 10 | `efef1cba` | 6.257606 | 0.018360 | `checkpoints/loc-a/10.json` |
| 300 | `fe0e6640` | 3.416600 | 0.031967 | `checkpoints/timing-probe/300.json` |
| 1,200 | `e55d91d8` | 3.078032 | 0.040505 | section 2a, `n = 40` row |
| 2,000 | `7e567530` | 2.974386 | 0.037877 | `checkpoints/verify-round3-2000/2000.json` |
| **12,000** | **`8a86fe69`** | **2.634755** | **0.055097** | **`checkpoints/r6-headline/12000.json`** |

The sampling term is **not flat in the step count**. It rises from 0.0012 at
init to 0.0551 at 12,000 steps, and the headline's own value is **1.36x** the
1,200-step value the seven-grid study measured and **1.45x** the 2,000-step
value. The mechanism is not mysterious: as the model gets better, its per-window
loss becomes more heterogeneous across the corpus, so the scatter over 40
windows grows even as the mean falls.

**Consequence.** `sigma = 0.0358`, measured at 1,200 steps, is not merely
*assumed* onto the headline -- it is assumed onto a checkpoint whose own
measured sampling term is **larger** than it. The transfer is not conservative.
Section 3b therefore stops treating the record's own stderr as a redundant
proxy and enters it as a component.

Reproduce the table with:

```bash
python3 - <<'PY'
import json, subprocess
paths = sorted(subprocess.check_output(
    ['find', 'checkpoints', '-name', '*.json']).decode().split())
for p in paths:
    d = json.load(open(p))
    if d.get('eval_chunks') == 40:
        print(d['step'], d.get('final_val_bpb'), d.get('val_bpb_stderr'),
              d['sha256'][:8], p)
PY
```

## 3. The consequence, in one sentence

**At sigma = 0.036 bpb, differences below ~0.036 are inside the estimator's own
noise and are not resolvable at the precision they are quoted; the two headline
claims survive at ~3.2 sigma, but every fourth-decimal comparison in this crate
does not.**

**`[SCOPE, ADDED 2026-08-03]` That sigma is the 1,200-step checkpoint's.**
Against the headline artifact's own combined `u_c = 0.0657` (section 3b,
Budget I) the same three "yes" rows fall to **1.77, 1.76 and 1.83 sigma** --
below the `2 sigma` bar this section itself sets two paragraphs down. **On the
headline artifact, NONE of the claimed differences in the table below is
resolvable.** The table is retained as written because it is the correct
arithmetic for the checkpoint it was measured on, and because seeing both
scalings side by side is the point.

**`[REVISED 2026-08-06]` Budget I is the right scale for this table, and saying
why is the point of publishing two budgets.** Every row below compares two
readings taken on the **same** hashed eval corpus, so the corpus-choice term of
`0.2807` (section 3b, row 2) is common-mode and drops out. Applied where it does
not belong -- against Budget II's `u_c = 0.2883` -- all three "yes" rows would
fall to about **0.40 sigma**, which would be the wrong arithmetic, not a
stricter one. The rule is the same one this whole document keeps arriving at:
name the corpus, then pick the budget.

The arithmetic, so the sentence can be checked:

| claimed difference | delta (bpb) | in sigma (0.0358) | resolvable? |
|---|---:|---:|---|
| `gf16_floor` on vs off, "worth ~0.116 bpb" | 0.1160 | 3.24 | yes, at ~3 sigma |
| `BPB_CHAMPION` 2.5193 vs headline 2.6348 | 0.1155 | 3.22 | yes, at ~3 sigma |
| 100% verbatim train/val overlap, ~0.12 bpb | 0.1200 | 3.35 | yes, at ~3 sigma |
| eval-cadence pair 2.6141 vs 2.6169 | 0.0028 | 0.08 | **no** |
| anything quoted at the 4th decimal | 0.0001 | 0.003 | **no** |

Read the "yes" rows carefully. Three sigma from a **single** reading each is not
three sigma from a replicated experiment: none of those numbers was measured
more than once, on more than one grid, and 3.2 sigma with `n = 1` on each arm is
a suggestive difference, not an established one. The honest statement is that
they are large enough to be worth replicating, and that nothing below ~0.07 bpb
(2 sigma) should be claimed at all without repeating the measurement on several
grids.

The specific casualty is the eval-cadence pair. Two seed-47 runs differing only
in `--eval-every` produced 2.6141 and 2.6169, and that 0.0028 gap was cited as
evidence that the eval cadence changed the weights. **It cannot carry that
weight: 0.08 sigma is indistinguishable from zero.** The claim that
`--eval-every` mutates the artifact is still true, but it is established by the
CHECKPOINT HASHES differing, not by those two BPB values.

And the headline itself: `2.6347548961639404` supports, at most,

```
2.63 +/- 0.07 bpb
    TWO components, k = 1:
      0.0551  the headline record's OWN sampling term (val_bpb_stderr,
              measured on 8a86fe69 at eval_chunks = 40)
      0.0358  the seven-grid dispersion, TRANSFERRED from a 1,200-step
              checkpoint by assumption -- and section 2b shows the
              assumption runs the WRONG WAY
    a LOWER BOUND: the seven grids are nested prefixes, hence correlated,
      and a correlated sample understates spread (section 4)
    excludes corpus choice [REVISED 2026-08-06: now QUANTIFIED at 0.2807 --
      add it and this k = 1 band becomes +/- 0.29, section 3b Budget II],
      and excludes seed, platform and step count, none of which is
      quantified anywhere in this document
    scope: this band applies ONLY on the declared corpus 2088af36...;
      it is not the uncertainty of the model's BPB
    sampling plan: 40 windows x 129 tokens = 5,160 of 100,000 bytes (5.16%)
```

**`[CORRECTED 2026-08-03 -- this block used to read `2.63 +/- 0.04`.]` The
previous `k = 1` band of `+/- 0.04` was SMALLER than the sampling stderr the
headline record itself carries (`0.0551`).** A counterparty who opened
`checkpoints/r6-headline/12000.json` read a bigger error bar than this document
published. That is the strongest sentence here precisely because it is a number
that moved against us: **`+/- 0.04` at `k = 1` was not conservative for the
headline artifact, and no amount of framing makes it so.**

Read the label, not just the number: **that `+/- 0.07` is two components, not a
budget.** The budget -- every component enumerated, Type A separated from
Type B, the quantified part combined, and an expanded uncertainty with its
coverage factor stated -- is section 3b immediately below, and the **decision
rule** that turns it into a verdict is section 3c. The figures are different
quantities and this repository uses several, so whichever is meant has to be
named at every point of use.

Three significant figures. The other fourteen digits describe an `f32`'s
decimal expansion, not the model.

---

## 3b. The budget proper

Sections 1 to 3 measure **one** component. Section 4 lists the others and
quantifies none of them. That is a dispersion, and a dispersion is not a budget:
a budget enumerates every component, records how each was evaluated, combines
the ones that were quantified, and says plainly which were not. This section is
that table, in the vocabulary of the GUM (JCGM 100:2008), so that the headline
in section 3 cannot be quoted without the scope that section 4 attaches to it.

**Type A** = evaluated by statistical analysis of repeated observations.
**Type B** = evaluated by any other means: judgement, a single observation, a
specification, prior knowledge. The distinction is load-bearing here because
`[REVISED 2026-08-06]` only **three** rows are Type A, and the four remaining
Type B rows are not merely small -- they are **not quantified at all**.

**The measurand.** `final_val_bpb` of the artifact identified by
`sha256 = 8a86fe69...`, evaluated on the corpus identified by
`sha256 = 2088af36...`, under the sampling plan `eval_chunks = 40`,
`eval_seq = 129`, `eval_tokens = 5160`. Change any of those four hashes or
numbers and this budget does not apply. `[REVISED 2026-08-06]` **Change the
corpus hash specifically, and the budget that applies is Budget II below**, not
Budget I: the corpus is part of the measurand, not part of the apparatus.

| # | Component | Type | Quantified? | u_i (bpb) |
|---|-----------|------|-------------|-----------|
| 0 | **Window sampling on the headline artifact** -- the record's own `val_bpb_stderr` at `n = 40` | A | **yes** | **0.0551** |
| 1 | **Corpus truncation + grid**, upper bound -- the seven-prefix study, TRANSFERRED from a 1,200-step checkpoint | A | **yes, but not on this artifact** | **0.0358** |
| 2 | **Corpus choice** -- which held-out text the metric is computed on `[REVISED 2026-08-06]` | A | **yes**, on identical weights | **0.2807** |
| 3 | **Seed / initialisation** | B | **no** | not quantified |
| 4 | **Platform** -- `(os, arch, libc, toolchain)` | B | **no** | not quantified |
| 5 | **Step count** -- where in training the reading is taken | B | **no** | not quantified |
| 6 | **Selection / multiplicity** -- the metric was optimised on the split it is reported on | B | **no** | not quantified, see 3d |

Row by row, with the reason each lands where it does:

0. **Window sampling on the headline artifact `[ADDED 2026-08-03]`.** The
   standard error of the mean over the 40 windows the headline eval actually
   read, computed by the trainer at the moment of measurement and stored in the
   record: `checkpoints/r6-headline/12000.json` -> `val_bpb_stderr =
   0.05509733036160469`. Independently reproduced in
   `checkpoints/verify-round3-ref/12000.json` (schema `/7`, same checkpoint
   `8a86fe69`, same value to all 17 digits). **This is the only quantified row
   measured ON the artifact the budget is about.** It was previously excluded as
   a redundant proxy for row 1; section 2b shows it is not redundant and, at the
   headline, is the larger of the two.
1. **Corpus truncation and grid, as an upper bound `[RELABELLED 2026-08-03]`.**
   Sample standard deviation over the seven runs of section 2, with the weights
   held provably identical (one checkpoint hash, seven runs). **The label was
   wrong.** Those seven rows varied the validation **corpus** -- the pinned
   100,000-byte file and its six leading prefixes -- not the window grid at a
   fixed corpus. `0.0358` therefore bounds grid choice from above and mixes in
   corpus truncation; it is **not a measurement of grid choice**. The in-scope
   figure, measured at fixed corpus in section 2a, is that the default `n = 40`
   reading sits **0.006389 bpb (0.18 sigma)** from full coverage -- so a
   reproducer who holds the corpus hash fixed and picks its own `eval_chunks`
   experiences far less than `0.0358`, while this budget quantifies a component
   that reproducer never meets and until now excluded the one it does. The row
   is also a **lower bound in the other direction**: the seven prefixes are
   nested, so the readings are correlated rather than independent draws, and a
   correlated sample understates spread. Both statements are true at once, which
   is why the row is retained and relabelled rather than deleted -- and why it is
   **kept in the combination**, as the conservative choice.
2. **Corpus choice `[REVISED 2026-08-06 -- was Type B, "not quantified"]`.** The
   sentence that stood here -- that the between-corpus spread "is a judgement
   and it is entered here as a judgement, no number, because none was measured"
   -- was **false on the day it was written**. It had been measured, in this
   same directory:
   [HELD-OUT-PROTOCOL.md](HELD-OUT-PROTOCOL.md) sections (ii) and (iii), lines
   315-321. Two runs of the frozen binary `a380ed47...`, same seed, same
   training bytes, same step count, same sampling plan, differing **only** in
   `--val-data`, produced **byte-identical weights**: both `12000.bin` hash to
   `902cfb69a84bdcf4d6ab524576ff375d612fd8f485dcbc7e02f35f6379599146`, 852,272
   bytes, `cmp` exit 0. Read against the two corpora, those identical weights
   give

   | corpus | sha256 | BPB | stderr |
   |---|---|---:|---:|
   | `data/tiny_shakespeare_test.txt` | `00365e8a...` | 2.638520 | 0.042386 |
   | `data/tiny_shakespeare_val.txt` | `2088af36...` | 2.919297 | 0.046898 |

   **difference 0.280777 bpb** against a combined stderr of 0.063214 -- a ratio
   of 4.44, so it is not sampling noise. **The identical `.bin` hash is what
   makes this a measurement of the corpus and of nothing else:** the model's
   contribution to the difference is not estimated to be small, it is exactly
   zero, because it was the same model. The row is **Type A** -- evaluated from
   observations, not from judgement. Two riders travel with it, and both make it
   a coarse figure rather than a wrong one:
   * **One specimen, two corpora.** Those weights were trained on
     `data/tiny_shakespeare_train_core.txt` (915,394 bytes), not on the
     headline's `data/tiny_shakespeare.txt`, so `0.2807` is **transferred** onto
     `8a86fe69...` exactly as row 1 is, and `HELD-OUT-PROTOCOL.md` section 3
     forbids comparing the two artifacts' BPB values directly. What is
     transferred is the *sensitivity to corpus*, not either reading.
   * **`n = 2` corpora, and the full difference is entered, not the two-point
     standard deviation.** The sample standard deviation of two readings
     differing by `d` is `d/sqrt(2) = 0.198539`; this budget enters the full
     `0.280777`. That is a **declared policy choice in the conservative
     direction**, of the same kind as the quadrature choice below -- not an
     arithmetic result. With one degree of freedom it is an observed magnitude,
     not a dispersion over the population of possible eval corpora, which is
     certainly wider still: these two slices were cut from the same book on the
     same day, and a genuinely different text would not be expected to sit
     closer.
3. **Seed.** One seed (47), one initialisation, no replicate. Seed-to-seed
   spread of the trained model is a separate quantity that this document did not
   measure.
4. **Platform.** macos/aarch64 only. One *paired* cross-platform observation
   exists -- four step-wise deltas whose sample standard deviation is
   `0.0031 bpb`, section 4a -- but a paired residual is not a component of an
   unpaired budget, and substituting it for one would understate this row by
   about an order of magnitude. Not quantified.
5. **Step count.** Row 1 was measured at 1,200 steps, where `val_bpb ~ 3.06`;
   the headline is at 12,000 steps, where it is 2.6348. This used to be argued
   away as "weak evidence that this component is small", on the strength of two
   early checkpoints (round 6 at ~3.28, this document at ~3.06) agreeing to
   within 6%. **`[RETRACTED 2026-08-03]`** Section 2b measures the step
   dependence directly off the records and finds the sampling term rising
   monotonically from 0.0012 at init to 0.0551 at 12,000 steps. The component is
   **not small and its sign is known**: transferring an early sigma to a
   converted checkpoint **understates**. Still not quantified as a `u_i` -- the
   seven-grid experiment has not been repeated at 12,000 steps -- but it may no
   longer be described as probably negligible.
6. **Selection / multiplicity `[REVISED 2026-08-06]`.** The sentence that stood
   here flatly denied that a test split existed. **It is withdrawn as false**:
   `data/tiny_shakespeare_test.txt` exists, is hashed in `data/MANIFEST.sha256`
   and in `data/README.md` as `00365e8a...`, and has been read once under the
   pre-registered one-look rule of
   [HELD-OUT-PROTOCOL.md](HELD-OUT-PROTOCOL.md). What remains true, and is all
   that may now be claimed, is that **the trainer has no `--test-data` flag**
   and that **every figure in THIS document, including the headline
   `8a86fe69...`, is still reported on the corpus it was selected on**. The
   component stays unquantified for the reason in section 3d: a multiplicity
   correction needs `N`, the number of runs compared, and `N` is unknown.

One quantity that deliberately does **not** appear as a row:
* **Full-coverage stderr** (0.009055). With respect to *this* corpus the
  sampling error at full coverage is exactly zero. The 0.009055 is a standard
  error with respect to the wider population the corpus is a sample OF, which is
  row 2 restated -- and it restates it **thirty times too small**
  `[REVISED 2026-08-06]`: row 2 is now measured at `0.2807`, directly, on two
  real corpora and identical weights. Entering 0.009055 for row 2 would have
  been the worst available option, which is why the empty cell was preferable to
  it until a measurement existed.

**Combination `[RECOMPUTED 2026-08-06]`.** Only quantified components may be
combined -- and **which** components are in scope depends on the question being
asked, so this section now publishes **two** budgets rather than one. The four
remaining Type B rows (3 to 6) contribute nothing to either arithmetic and
everything to the reading of both. Publishing two labelled budgets is the honest
resolution and it is stronger than either alternative: quoting only the small
one hides the dominant term, and quoting only the large one would inflate the
uncertainty of the one thing this repository can actually demonstrate.

**Budget I -- DECLARED CORPUS.** Rows 0 and 1. *Scope: reproducing a stated
reading of a stated artifact on the SAME hashed eval corpus under the SAME
sampling plan.* Row 2 is **correctly excluded here**, and the reason must be
written down rather than assumed: **when the eval corpus is declared and hashed,
corpus choice contributes nothing to reproducing the number**, because the
reproducer reads the same bytes we did. `0.2807` is a real uncertainty of the
BPB *as a statement about the model* and exactly zero uncertainty of the BPB *as
a statement about a named (model, corpus) pair*.

```
u_c = sqrt( 0.0551^2 + 0.0358^2 )       =  0.065709 -> 0.0657 bpb
                                           combined standard uncertainty, k = 1
k                                       =  2        coverage factor
U   = k * u_c = 0.131418                -> 0.13 bpb expanded uncertainty

    val_bpb = 2.63 +/- 0.13 bpb   (k = 2, DECLARED CORPUS, quantified only)
    val_bpb = 2.63 +/- 0.07 bpb   (k = 1, DECLARED CORPUS, quantified only)
```

**Budget II -- CORPUS CHOICE INCLUDED.** Rows 0, 1 and 2. *Scope: comparing a
BPB against one measured on a DIFFERENT eval corpus, or reading a BPB as a
statement about the MODEL rather than about a (model, corpus) pair.*

```
u_c = sqrt( 0.0551^2 + 0.0358^2 + 0.2807^2 ) =  0.288288 -> 0.2883 bpb
U   = k * u_c = 0.576577                     -> 0.58 bpb  (k = 2)

    val_bpb = 2.6 +/- 0.58 bpb    (k = 2, CORPUS CHOICE INCLUDED)
    val_bpb = 2.6 +/- 0.29 bpb    (k = 1, CORPUS CHOICE INCLUDED)
```

The value is quoted to two significant figures in Budget II deliberately:
against `U = 0.58` the second decimal of `2.63` is not supported, and writing
`2.63 +/- 0.58` would be quoting a digit the budget has just withdrawn.

**Budget II is 4.39x Budget I**, and that ratio is the most useful single number
in this document: **a BPB is a property of a (model, corpus) pair, and stripping
the corpus off it costs more than four times the entire rest of the budget
combined.** Budget I is in turn 1.83x the `U = 0.0716` this document published
before 2026-08-03, and its `k = 1` figure is 1.64x the `+/- 0.04` still quoted
elsewhere in this repository. Both numbers moved against us, twice, for the same
reason: a record that was already on disk was finally read instead of a smaller
figure being substituted for it.

**A rider on the combination itself, because rows 0 and 1 are not independent.**
Both estimate window-sampling scatter, by different routes: row 0 is the scatter
*inside* one grid, row 1 the scatter *between* grids that also changed corpus
length. Adding overlapping components in quadrature **over-counts**; taking only
the larger **under-counts** whatever of row 1 is genuinely separate. The two
bracketing treatments are:

```
Budget I,  envelope (rows 0,1 fully redundant, take larger) : u_c = 0.0551, U = 0.110
Budget I,  quadrature (rows 0,1 independent)                : u_c = 0.0657, U = 0.131
Budget II, envelope base + row 2                            : u_c = 0.2861, U = 0.572
Budget II, quadrature base + row 2                          : u_c = 0.2883, U = 0.577
```

**This document adopts the quadrature figure**, and that is a **declared policy
choice, not a measurement**: for a conformity decision the conservative
direction is the larger uncertainty, because an understated `u_c` shrinks the
guard band of section 3c and silently raises the consumer's risk. The
correlation coefficient between the two routes has not been measured and would
be needed to do better than bracket it.

`[ADDED 2026-08-06]` **Row 2 raises no such correlation question and dominates
either way.** Rows 0 and 1 overlap because both estimate window-sampling
scatter; row 2 was measured with the sampling plan and the weights both held
fixed, so it is a different quantity by construction rather than by assumption,
and combining it in quadrature is uncontroversial. Note also that the whole
envelope-vs-quadrature debate moves `U` by 0.021 bpb in Budget I and by 0.005 in
Budget II, while row 2 moves it by 0.445. **The component this document argued
about hardest for three days is an order of magnitude smaller than the one it
left in an empty cell.**

Four riders travel with that number and are not optional:

1. **Both `U` figures are lower bounds on the expanded uncertainty, twice over
   `[REVISED 2026-08-06]`.** Row 1 is itself a lower bound with respect to
   nesting; rows 3 to 6 are *excluded*, not estimated, from both budgets; and
   row 2, now present in Budget II, rests on two corpora cut from one book. `U`
   is not "the" uncertainty of `val_bpb`; it is the part of it that has been
   measured, under the scope its budget declares.
2. **Row 1 is transferred, not measured on this artifact.** The seven-grid study
   ran on `e55d91d8` at 1,200 steps. Only row 0 was measured on `8a86fe69`. If
   the seven-grid experiment were repeated at 12,000 steps, section 2b's
   monotone trend predicts a **larger** row 1, hence a larger `u_c` -- so the
   transfer is not conservative and `0.066` is expected to grow, not shrink,
   when the missing experiment is run.
3. **`k = 2` is optimistic and is chosen by convention, not derived.** It is the
   conventional factor for approximately 95% coverage under a normal
   distribution with large degrees of freedom, and **neither condition holds
   here**. Row 1 rests on at most 6 degrees of freedom (`n = 7`), for which the
   two-sided 95% Student-t factor is 2.45 (standard table value, not a
   measurement of ours), and the *effective* degrees of freedom are fewer than 6
   because the samples are nested and correlated. The effective degrees of
   freedom are **not quantified**: computing them (Welch-Satterthwaite) needs a
   correlation structure this experiment did not measure.
4. **Several bands are in circulation and they are different quantities
   `[REVISED 2026-08-06]`.** `+/- 0.58` (k = 2, Budget II, corpus choice
   included), `+/- 0.29` (k = 1, Budget II), `+/- 0.13` (k = 2, Budget I,
   declared corpus), `+/- 0.07` (k = 1, Budget I), the superseded `+/- 0.04`
   (k = 1, one transferred component), and the paired `0.003` of section 4a.
   **A band without its scope is not a band**, and the two most likely misuses
   are opposite errors: quoting `0.13` while comparing across corpora
   understates by 4.4x, and quoting `0.58` for a same-corpus reproduction
   inflates an uncertainty that is genuinely absent there.
   [REPRODUCIBILITY-GRADING.md](REPRODUCIBILITY-GRADING.md) adopts `+/- 0.04` as
   an unpaired cross-laboratory *acceptance tolerance*. **That adoption now needs
   re-examination and this document does not own that file:** `+/- 0.04` was
   defended there as "the tight direction is the conservative one" for an
   acceptance test, but section 3c shows that a tolerance of `0.04` against
   `U = 0.13` yields an **empty** guarded acceptance interval -- it is not tight,
   it is undecidable. Whichever band is meant must be named at the point of use.

**What would move a row out of "not quantified".** Each needs a new experiment,
and putting an invented figure in an empty cell would be strictly worse than
leaving it empty:

* Row 2 -- **DONE `[2026-08-06]`.** The prescription written here was "the same
  weights evaluated on an independent held-out corpus". That experiment had
  already been run: `docs/HELD-OUT-PROTOCOL.md` sections (ii) and (iii). Row 2
  is quantified at `0.2807` and Budget II exists because of it. The lesson is
  recorded rather than tidied away -- **the missing measurement was not missing,
  it was in the next file in the same directory**, and an empty cell survived
  three days of revision because nobody re-read the neighbours.
* Row 3 -- k seeds trained to the same step budget, sigma taken across them.
* Row 4 -- the *unpaired* cross-platform comparison, each arm free to choose its
  own grid, rather than the paired one already run.
* Row 5 -- the seven-grid experiment repeated at 12,000 steps rather than 1,200.
  This is now the highest-value one: section 2b makes a falsifiable prediction
  about its outcome.
* Row 6 -- `[REVISED 2026-08-06]` a split that never entered any selection
  decision now **exists** (`data/tiny_shakespeare_test.txt`, `00365e8a...`), but
  quantifying the row additionally needs `N`, the number of runs compared, and
  `N` is unknown. Section 3d.

---

## 3c. THE DECISION RULE (draft clause)

**Status: DRAFT CLAUSE. Provisional. Not a measurement.** Every figure in this
section is either arithmetic on the `u_c` of section 3b -- **Budget I, the
declared-corpus budget** `[REVISED 2026-08-06]` -- or a **declared policy
choice**, and each is labelled. Nothing here was measured; what was measured is
in sections 1 to 3b.

**Why it is here at all.** Sections 1 to 3b speak GUM -- `u_c`, `k`, `U` -- and
then stop. A dispersion is not a verdict. A conformity assessment must say what
happens when a reproducer's reading lands at a given distance from the declared
value, and that requires three things a dispersion does not supply: an
**acceptance interval**, a **guard band**, and a **declared consumer's risk**.
The framing below follows **JCGM 106:2012**, *Evaluation of measurement data --
The role of measurement uncertainty in conformity assessment*.

**Scope: the L2 tier only.** The grading scheme distinguishes

* **L3 -- byte identity.** The reproducer's checkpoint `sha256` equals the
  declared one. **Measurement uncertainty is irrelevant by construction:** a
  digest either matches or it does not, there is no interval and no guard band.
  This is the tier the same-machine determinism result reaches.
* **L2 -- metric reproduction.** The bytes differ (a different CPU
  architecture, a different libc, a different toolchain) but the reported metric
  is claimed to reproduce "within tolerance". **This is the only tier where
  uncertainty enters, and it is the tier with no rule written for it.** That is
  what follows.

### The clause

> **L2-1. Measurand.** `final_val_bpb` of the artifact identified by
> `sha256`, evaluated on the corpus identified by its `sha256`, under the
> sampling plan the record declares (`eval_chunks`, `eval_seq`, `eval_tokens`).
> A submission that does not state all five is **REFUSED**, not graded.
>
> **L2-2. Tolerance limits (POLICY CHOICE).** `T = 0.20 bpb` either side of the
> declared value. This is **declared, not derived**: it is the smallest round
> figure that leaves a non-empty acceptance interval after the guard band of
> L2-3, and its justification is that it is well inside the ~0.44 bpb that
> separates a 1,200-step model from a 12,000-step one, so a tolerance of 0.20
> still refuses an artifact that is a materially different model.
>
> **L2-3. Guard band (ARITHMETIC), AND THE SCOPE IT DEPENDS ON
> `[REVISED 2026-08-06]`.** `w = U = k * u_c = 2 x 0.0657 = 0.13 bpb`, from
> **Budget I** of section 3b. This follows from the budget; it is not chosen.
> **It holds only because L2-1 requires the eval corpus `sha256`.** Both
> laboratories therefore read the identical bytes, the corpus-choice component
> of `0.2807` (row 2) is common-mode, and it cancels exactly -- it is not
> neglected, it is absent. **Withdraw the corpus hash from L2-1 and this clause
> collapses**: Budget II applies instead, `w = U = 0.58`, and `A = T - w =
> 0.20 - 0.58 = -0.38`, an empty acceptance interval under which no artifact
> whatsoever could be graded. That is the arithmetic reason a submission without
> a corpus hash is REFUSED rather than graded loosely, and it is a stronger
> reason than the one L2-1 originally gave.
>
> This clause does **not** grade a submission whose eval corpus differs from the
> declared one. Such a comparison is out of scope of the L2 tier entirely, not
> merely wider-banded: at `T = 0.20` against `U = 0.58` the scheme has nothing
> to say, and the honest response is to refuse the comparison rather than widen
> `T` until it passes.
>
> **L2-4. Acceptance interval (ARITHMETIC).** `A = T - w = 0.20 - 0.13 =
> 0.07 bpb` either side of the declared value. This is **guarded acceptance** in
> the sense of JCGM 106: the acceptance interval is strictly inside the tolerance
> interval, by exactly the expanded uncertainty. (The clause number within
> JCGM 106 is deliberately not cited -- the document was not read for this work,
> only its framing is used, and an unchecked clause reference is the kind of
> citation this repository exists to refuse.)
>
> **L2-5. Declared consumer's risk (POLICY CHOICE + ARITHMETIC).** With a guard
> band of `k = 2` standard uncertainties and the measurement error taken as
> normal, the **specific** consumer's risk at a tolerance limit is bounded by
> `Phi(-2) = 2.28%`. **Declared: <= 2.5% per limit.** The **global** consumer's
> risk is deliberately **not** declared: it requires a prior distribution over
> true values across the population of submissions, which does not exist for a
> scheme with no submissions. Anyone quoting a global risk for this clause is
> quoting a number nobody measured.
>
> **L2-6. Verdicts.** Let `d = |reading - declared value|`.
>
> | condition | verdict |
> |---|---|
> | `d <= A` (0.07) | **CONFORM** -- L2 metric reproduction demonstrated at the declared consumer's risk |
> | `A < d <= T` (0.07 to 0.20) | **INDETERMINATE** -- inside tolerance but inside the guard band. Accepting here would exceed the declared risk; rejecting would exceed the producer's. No verdict is issued; L2-7 applies |
> | `d > T` (0.20) | **NON-CONFORM** |
>
> An **INDETERMINATE** verdict is a result, not a failure of the scheme. An
> instrument that cannot return "I cannot tell" is not a conformity instrument.
>
> **L2-7. The escape from INDETERMINATE is pairing, not more tolerance.** If
> both laboratories evaluate the **identical** sampling plan on the
> **identical** corpus hash, the window-sampling term is common-mode and
> cancels. The measured paired residual is `0.0031 bpb` (section 4a), giving
> `u_c = 0.0031`, `w = 0.0062`, and against a tolerance of `T = 0.04` an
> acceptance interval of `A = 0.034` -- **a factor of 21 tighter than the
> unpaired clause, from the same data.** This is the clause's incentive
> structure and its main practical content: **a submission that declares its
> sampling plan can be graded twenty times more sharply than one that does
> not.** It is also why L2-1 refuses a submission that omits the plan, rather
> than grading it loosely.

### What this clause costs us, stated plainly

The `+/- 0.04` tolerance this repository has been quoting **cannot be
conformity-assessed at the current uncertainty.** The arithmetic:
`A = T - w = 0.04 - 0.13 = -0.09`. A negative acceptance interval is an empty
one: under a `k = 2` guard band **no artifact whatsoever could be accepted
against a 0.04 tolerance**, including a perfect reproduction. So `+/- 0.04` was
not a conservative tolerance. It was an undecidable one, and the reason it never
looked undecidable is that no guard band was ever computed.

Three ways out, in order of honesty:

1. **Widen the tolerance** to `T = 0.20` as above and accept that the scheme
   resolves only differences larger than 0.07 bpb unpaired.
2. **Require pairing** (L2-7), which restores a `0.04` tolerance with a real
   `0.034` acceptance interval.
3. **Shrink `u_c`** by raising `eval_chunks`. Section 2a measures the lever
   directly: `n = 40` carries a within-grid stderr of 0.0405 on the 1,200-step
   checkpoint, `n = 775` (full coverage) carries 0.009055 -- a factor of 4.5.
   Full coverage is the single largest available reduction and it costs only
   compute. **This is a recommendation, not a measurement:** the equivalent
   sweep has not been run on the 12,000-step artifact.

---

## 3d. What the budget still does NOT contain: the trainer has no `--test-data` flag

**`[REVISED 2026-08-06 -- this section used to be headed with a flat denial that
a test split existed, and that denial is withdrawn as false.]`** A third corpus
now exists and is hashed:
`data/tiny_shakespeare_test.txt`, 100,000 bytes,
`00365e8aa883ffe50d954234fe810d9b2413450c944a67f0a621d433464e9603`, listed in
`data/MANIFEST.sha256` and in `data/README.md`, carved from
`tiny_shakespeare.txt[915394, 1015394)` and read exactly once under the
pre-registered one-look rule of
[HELD-OUT-PROTOCOL.md](HELD-OUT-PROTOCOL.md). Deleting the false half of the old
sentence leaves the true half, which is still a real scoping limit:

* **`--test-data` does not exist.** Confirmed again 2026-08-06:
  `grep -n 'test-data\|test_data' src/bin/trios-train.rs src/train_loop.rs`
  returns nothing. There is a train flag and a val flag and no third one; the
  held-out run reached the third corpus by passing it to `--val-data`, which
  works but records nothing that distinguishes a held-out stream from a
  selection stream. **The corpus is the part that was missing; the flag, and
  with it the machine-checkable declaration L2-8 demands, still is.**
* **Every number in THIS document is still computed on
  `data/tiny_shakespeare_val.txt`,** including the headline `8a86fe69...`. The
  held-out figure lives in `HELD-OUT-PROTOCOL.md` and belongs to a different
  specimen, trained on the 915,394-byte `train_core`; that document's section 3
  forbids comparing the two.
* `data/README.md` now marks **two** files `Fit for eval? YES` -- `..._val.txt`,
  and `..._test.txt` with the qualifier "and only once". Of the other five, two
  are training corpora (`fineweb_train.bin`, `tiny_shakespeare_train_core.txt`),
  one is `tiny_shakespeare.txt` (the two-way train split), one is a
  byte-identical duplicate of `fineweb_train.bin`, and one is a 160-byte
  degenerate pangram fixture.

So the split every figure in this document is **reported** on is still the split
every design decision was **selected** on: architecture, learning rate, `gf16`
on or off, the floor cadence, and every historical champion figure. A minimum or
a best-of taken over many runs scored on one split is a **maximum-of-N order
statistic**, biased optimistic by construction and biased further the more runs
were compared. This is the same defect that made `best_val_bpb` unfit to quote
(section 5), scaled from one run to the whole programme.

**The multiplicity component cannot be entered as a number, for a reason worth
recording: N is unknown.** The repository's own
[IGLA_V6_FINAL_RESULTS.md](../IGLA_V6_FINAL_RESULTS.md) states that the run
count is unreconciled -- "7,927 logs / 5 DONE" there, "1878-run fleet" in
trios-railway issue #109, "~1,851 experiments" in the skill-library retraction
-- "three different figures for overlapping populations. None is derived from an
experiment ledger. Do not quote any of them as the fleet size." A multiplicity
correction needs N. **We do not have N**, so row 6 stays unquantified and no
historical champion figure -- `BPB_CHAMPION = 2.5193` included -- may be
presented as an unbiased estimate of anything.

### The second draft clause

> **L2-8. A submission names and hashes a split that never entered selection.**
> The record must identify, by path, byte count and `sha256`, an evaluation
> corpus that was **not** used to choose any hyperparameter, architecture,
> checkpoint or run of the submitted model, and must declare the selection
> corpus separately. Where the two are the same file, the reported metric is
> graded as a **selected** figure and carries no conformity verdict at all --
> it is a training diagnostic, not a result.
>
> **Status: DRAFT, and this repository currently FAILS it -- but by less than it
> did `[REVISED 2026-08-06]`.** Stating a clause we do not yet pass is the point:
> it is a specification, not a certificate. The old status line said passing it
> "requires a third corpus and a `--test-data` flag, neither of which exists
> today". **The third corpus now exists and is hashed** (`00365e8a...`), and one
> run has been graded against it under a pre-registration. What is still missing
> is the **flag**: without `--test-data` the record cannot declare the selection
> corpus and the held-out corpus as separate fields, so the separation rests on
> a prose protocol rather than on a machine-checkable record -- which is exactly
> the class of claim this document exists to distrust.

---

## 4. What this sigma does and does not cover

Covered:

* Which windows of the val corpus were read (the grid), at fixed `n`.
* Which prefix of the val corpus was available.

**Not** covered:

* **Corpus choice `[REVISED 2026-08-06]`.** Still not covered *by this sigma*:
  all seven grids are drawn from one 100,000-byte tinyshakespeare tail. But it
  is **no longer unmeasured**, and the guess written here -- "certainly larger"
  -- was right by a factor this document should have known: the same weights
  (`902cfb69a84b...`) read against two 100,000-byte slices differ by
  **0.280777 bpb**, `HELD-OUT-PROTOCOL.md` (iii). It enters section 3b as row 2
  and produces Budget II. Anything below quoting `0.0358` or `0.0657` is
  therefore a **same-corpus** statement, by construction.
* **Nesting.** The seven prefixes are nested, so the rows are correlated
  samples, not independent draws. A correlated sample understates spread, so
  0.0358 is a **lower bound** on the true between-grid sigma.
* **Seed.** One seed (47), one initialization. Seed-to-seed spread of the
  trained model is a separate quantity.
* **Platform.** macos/aarch64 only. The cross-libc experiment already showed the
  same seed, corpus and source producing different checkpoint hashes on macOS
  and glibc; a different checkpoint has a different BPB, and that difference is
  not in this budget.
* **Step count `[REWRITTEN 2026-08-03]`.** Measured at 1,200 steps,
  `val_bpb ~ 3.06`; the headline is at 12,000 steps, `val_bpb = 2.6348`. This
  bullet used to argue from round 6 (~3.28) and this document (~3.06) agreeing
  to within 6% that "sigma is roughly flat in this range". **That inference is
  withdrawn:** two early checkpoints agreeing with each other says nothing about
  a converged one, and section 2b now measures the step dependence directly off
  the records -- the within-grid sampling term rises monotonically from 0.0012
  at init to 0.0551 at 12,000 steps. A converged model **does** have a larger
  window-to-window scatter. Still unquantified as a between-grid sigma at
  12,000 steps, but no longer plausibly negligible, and its **sign is known**.
* **Selection.** The metric is reported on the split it was selected on, and no
  test split exists. Section 3d.

At full coverage the sampling error with respect to **this corpus** is exactly
zero - the mean over 775 tiling windows is the corpus mean. The 0.009055
reported there is the standard error with respect to the wider population the
corpus is a sample OF, which is a different and larger question.

---

## 4a. When the paired tolerance applies, and when only the unpaired band does

This section adds scope. It restates no measurement and introduces none: every
number in it is either from section 2 above, from the records read in section
2b, or from the table in
[REPRODUCIBILITY-GRADING.md](REPRODUCIBILITY-GRADING.md).

The sigma measured here, `0.0358 bpb`, is a dispersion over seven runs that
varied the **val corpus length** (and therefore the grid): it is what the
reading moves by when the estimator reads a different sample of the val corpus.
That makes it the right band for one comparison and the wrong band for another,
and the difference is whether the two arms read the same windows.

**Paired -- the same grid on both arms.** Two runs that evaluate the identical
windows of the identical val corpus, with the identical `eval_chunks`,
`eval_seq` and `eval_every`, share the sampling term. It is common-mode and it
cancels, so the residual is much smaller than sigma. The cross-platform
experiment is of this kind: the four step-wise deltas between aarch64 and
x86_64 are `+0.0002`, `-0.0035`, `+0.0030`, `+0.0030`, whose sample standard
deviation is `0.0031` -- about a twelfth of `0.0358`. That ratio is not a
contradiction of this document; it is what a paired design is for, and it is
the reason `REPRODUCIBILITY-GRADING.md` can quote a `0.003` tolerance at all.

**Unpaired -- anything else.** The moment either laboratory changes the eval
grid, the sampling term stops cancelling and the applicable band is the
unpaired one:

```
2.63 +/- 0.07 bpb     (k = 1, Budget I, TWO components, a LOWER BOUND; SAME
                       hashed corpus, sampling plan stated -- see section 3b)
2.63 +/- 0.13 bpb     (k = 2, Budget I, the band the decision rule of 3c uses)
2.6  +/- 0.58 bpb     (k = 2, Budget II, DIFFERENT eval corpora -- section 3b)
```

**`[CORRECTED 2026-08-03 -- this block used to read `+/- 0.04`, one component.]`**
It omitted the headline record's own `val_bpb_stderr = 0.0551`, which is larger
than the band it quoted.

The first two bands cover a second laboratory choosing its own `eval_chunks`, a
different stride, a different prefix of the corpus, or simply not recording
which windows it read -- **all on the corpus whose hash the record declares.**
`[REVISED 2026-08-06]` They do **not** cover a different eval corpus: that is
the third band, `+/- 0.58`, and the reason it is four times wider is a
measurement, not a precaution -- `HELD-OUT-PROTOCOL.md` (iii), 0.280777 bpb
between two slices of one book at fixed weights. None of the three covers a
different seed or a different step budget; see section 4 for what the budgets
leave out, and section 3d for the flag that does not exist.

Two consequences worth stating plainly:

* A cross-laboratory agreement quoted at the third decimal is only meaningful
  if the sampling plan is quoted with it. Without the plan, `0.003` and `0.07`
  are indistinguishable claims, and only the looser one is defensible. This is
  the measurement behind clause L2-7 in section 3c: pairing is worth a factor of
  twenty in the acceptance interval, and it costs nothing but declaring the
  plan.
* Conversely, the measured cross-platform difference of `0.0030 bpb` is
  `0.084 sigma` and is therefore **not a difference**: the honest statement is
  that the experiment could not resolve one, not that the platforms agree to
  three decimals. The same run's checkpoints differ in 43.70% of their
  parameters (`docs/CROSS-ARCH-DIVERGENCE.md`) -- 93,071 of 212,992 differ in
  value, and a further 2,073 differ bitwise only in the sign of zero and are
  numerically equal, so two definitions of "differ" are in circulation and
  `43.70%` is the conservative one, which is why it is the figure quoted; the
  reconciliation and the census are in
  [`CANONICAL-SERIALIZATION.md`](CANONICAL-SERIALIZATION.md). That divergence
  is the reminder that
  agreement in this metric is agreement in this metric and nothing else.

The one-line rule `[REWRITTEN 2026-08-06]`: **name the corpus first, then pick
the band. Quote 0.003 only with the pairing condition attached; quote +/- 0.07
(k = 1) or +/- 0.13 (k = 2) when the eval corpus hash is the same and only the
sampling plan differs, as a lower bound on two components; quote +/- 0.58
(k = 2) the moment the corpora differ, or whenever a BPB is being read as a
property of the MODEL rather than of a (model, corpus) pair. Four unquantified
components sit on top of all of these (section 3b). The superseded `+/- 0.04`
must not be quoted at all: it is smaller than the headline record's own
`val_bpb_stderr`.**

---

## 5. What changed in the code

* `--eval-chunks` is now a declared parameter, exposed as `TRIOS_EVAL_CHUNKS`
  (see the deviation note in section 6). `0` means full coverage. **The default
  is still 40**, deliberately: the defect is that the coverage was never stated,
  not that 40 is wrong, and moving the default would silently make every
  recorded BPB incomparable with every new one.
* `evaluate()` returns the mean, the sample stdev, the count and the grid it
  walked, instead of a bare `f32`. A single-window eval reports `stdev: None`,
  not `0.0`.
* Every eval prints a new `EVAL-PLAN:` line carrying the plan and the
  uncertainty. The `DONE: seed=... bpb=<4dp> ...` token is untouched, format and
  all.
* `CheckpointRecord` schema 6 adds `eval_chunks`, `eval_tokens`, `eval_seq`,
  `val_bpb_stderr` and `optimizer_params`. All are sidecar-only, all are
  `#[serde(default)]`, none enters the hashed `TRIOSCKP` header and none is
  required for grading. An absent field is absent, not zero.
* `best_val_bpb` is renamed `min_observed_val_bpb` (with a serde alias, so
  existing sidecars still load). It is a minimum over noisy draws: with
  sigma = 0.036, the minimum of k readings is biased low by construction and
  biased further the more often the run evaluated. It is not a peer of
  `final_val_bpb` and must not be quoted as a run's result.

### 5a. `optimizer: "adamw"` was four numbers in five characters

Schema 6 records them. **They are not the phi-branded constants.**

`src/optimizer.rs` documents an `AdamWCpu` with `beta1 = 1/phi = 0.618` and
`weight_decay = 1/phi^3 = 0.23607`. The production trainer **never constructs
it**. `run_single` and `run_single_muon` in `src/train_loop.rs` use a private
`AdamW` with:

| parameter | value that ran | `optimizer::AdamWCpu` claims |
|---|---|---|
| beta1 | 0.9 | 0.6180339887 (`1/phi`) |
| beta2 | 0.999 | 0.999 |
| eps | 1e-8 | 1e-8 |
| weight_decay | 0.04 | 0.2360679775 (`1/phi^3`) |

The two also apply decay differently: `train_loop::AdamW` does
`p -= wd * lr * p` (decoupled and lr-scaled), `AdamWCpu` does `p -= wd * p`.

**No published `trios-train` BPB was produced with the phi constants.** The
sidecar records the values that ran and names their source in
`optimizer_params.source`, so the distinction survives in the record rather than
in a comment. This was found while implementing this item; the work item's own
premise (that the record's `adamw` string was hiding `beta1 = 0.618`) was
wrong in an interesting direction - it was hiding `beta1 = 0.9`.

### 5b. `val_bpb_stderr` was missing its finite-population correction `[ADDED 2026-08-03]`

`src/train_loop.rs` computed

```text
    stderr = s / sqrt(n)
```

unconditionally, while the doc comment on `eval_chunks_target` said
`TRIOS_EVAL_CHUNKS=0` is "the only setting with no sampling error at all". Both
could not be true, and the artifact was the one that was wrong: **measured on a
full-coverage run - 775 windows, 99,975 of 100,000 val tokens - the record
still carried `val_bpb_stderr = 0.0117720`.** That number is the spread BETWEEN
windows. It is not the uncertainty of the published mean, because at full
coverage there is no sampling left: every byte of the stream was read exactly
once and the "sample" mean IS the population mean.

`s / sqrt(n)` is the standard error of a mean drawn from an **infinite**
population. The val stream is finite and the grid samples it **without
replacement**. The corrected estimator is

```text
    stderr = (s / sqrt(n)) * sqrt(1 - n/N)
```

where `N` is the number of windows full coverage of the *same* stream would
walk -- `eval_plan(val_len, 0).chunks`, resolved from the same plan the eval
itself used, so the two cannot disagree. At `n == N` the factor is exactly
`0.0`, which makes the doc comment true. `stdev` is **unchanged**: the
between-window spread is a real quantity that this document uses elsewhere, it
is simply not the same quantity as the error of the mean.

Three consequences, stated plainly:

1. **Every record written before this change carries an uncorrected, slightly
   larger `val_bpb_stderr`.** There is no migration and none is wanted: the old
   value is what that run reported, and rewriting artifacts to match a later
   opinion is the failure mode this repository exists to avoid. A reader who
   wants the corrected figure multiplies by `sqrt(1 - eval_chunks/N)` using the
   `eval_chunks` the record already states.
2. **At the default coverage the correction is small and previously published
   bands are conservative, not wrong.** For `n = 40` against `N = 775` (the
   100,000-token val stream) the factor is **0.97385**. The headline record's
   `val_bpb_stderr = 0.05509733036160469` rescales to **0.05366** -- an
   arithmetic re-scaling of an existing number, not a new measurement. Nothing
   in section 3b's budget or section 3c's decision rule gets *wider*, so no
   published band needs to be withdrawn.
3. **Full coverage now reports `0.0`, and that is a statement about sampling
   only.** It does not mean the reading is exact. Every other term in the
   budget of section 3b -- grid aliasing, the single val split, the transfer
   assumption of section 2b, cross-architecture divergence -- is untouched by
   this correction and remains the dominant contribution. A `stderr` of `0.0`
   with a non-zero `stdev` beside it is exactly the intended reading: no
   sampling error, real dispersion between windows. Read the `0.0` as the
   finite-population statement it is (`n == N`) and not as a missing
   measurement -- and note that "full coverage" is 775 windows of 129 tokens =
   99,975 of 100,000 tokens, i.e. **99.975%, not 100%**: the 25 tokens at the
   tail cannot fill a window and are never evaluated, so `N` is the population
   the grid can reach rather than the whole file.

The unit test `evaluate_reports_the_spread_of_its_own_windows` in
`src/train_loop.rs` now pins both ends of the formula: the corrected value at a
strict subset, and exactly `0.0` at `n == N`.

---

## 6. Deviations from the work item

1. **`--eval-chunks` is `TRIOS_EVAL_CHUNKS`, not a clap flag.** The clap `Cli`
   struct lives in `src/bin/trios-train.rs`, which this item does not own, and
   `TrainArgs` is built by struct literal in three binaries it does not own
   either, so a new field would not compile. The knob is implemented as an env
   var, matching the existing `TRIOS_GF16_FLOOR_EVERY` / `TRIOS_ATTN_SEQ`
   pattern. Turning it into `--eval-chunks` is a two-line addition to
   `src/bin/trios-train.rs`: `#[arg(long, env = "TRIOS_EVAL_CHUNKS")] eval_chunks:
   Option<usize>` plus a `set_var` next to the existing `--format` re-export.
2. **The new tokens print before `DONE:`, not after.** `DONE:` is printed by
   `main()` in `src/bin/trios-train.rs` after `run_single` returns; this item
   cannot edit that file. The `EVAL-PLAN:` line is emitted from `train_loop` at
   each eval, so the final one lands immediately before `DONE:`. The `DONE:`
   token and its 4-decimal format are unchanged.
3. **`optimizer_params` is read from `train_loop`, not from `optimizer.rs`.**
   See section 5a: reading `optimizer.rs` would have written numbers into the
   provenance record that no run ever used.

---

## 7. How to reproduce

Every command below strips database DSNs and writes only to `/tmp`.

```bash
# 1. build
cargo build --release --bin trios-train

# 2. the seven grids (parallel; each run is independent)
mkdir -p /tmp/r7-eval-unc/val /tmp/r7-eval-unc/logs
for n in 100000 90000 80000 70000 60000 50000 40000; do
  head -c $n data/tiny_shakespeare_val.txt > /tmp/r7-eval-unc/val/val_$n.txt
done
for n in 100000 90000 80000 70000 60000 50000 40000; do
  env -u DATABASE_URL -u NEON_DATABASE_URL -u TRIOS_DATABASE_URL -u TRIOS_NEON_DSN \
    TRINITY_AUTOMIGRATE=0 TRIOS_CHECKPOINT_DIR=/tmp/r7-eval-unc TRIOS_CANON_NAME=prefix-$n \
    ./target/release/trios-train --seed 47 --steps 1200 --hidden 384 --attn-layers 2 \
    --lr 0.003 --eval-every 1200 --train-data data/tiny_shakespeare.txt \
    --val-data /tmp/r7-eval-unc/val/val_$n.txt > /tmp/r7-eval-unc/logs/prefix-$n.log 2>&1 &
done; wait

# 3. the weights must be identical - one hash, seven files
shasum -a 256 /tmp/r7-eval-unc/prefix-*/1200.bin | awk '{print $1}' | sort | uniq -c
#   7 e55d91d83515769925a41971f28a94f0a7f31dac05f4120f376201c166504f7d

# 4. the readings
grep -h "^EVAL-PLAN: seed=47 step=1200" /tmp/r7-eval-unc/logs/prefix-*.log

# 5. the sample-size sweep, including full coverage (TRIOS_EVAL_CHUNKS=0)
for c in 10 20 40 80 160 320 0; do
  env -u DATABASE_URL -u NEON_DATABASE_URL -u TRIOS_DATABASE_URL -u TRIOS_NEON_DSN \
    TRINITY_AUTOMIGRATE=0 TRIOS_CHECKPOINT_DIR=/tmp/r7-eval-unc \
    TRIOS_CANON_NAME=chunks-$c TRIOS_EVAL_CHUNKS=$c \
    ./target/release/trios-train --seed 47 --steps 1200 --hidden 384 --attn-layers 2 \
    --lr 0.003 --eval-every 1200 --train-data data/tiny_shakespeare.txt \
    --val-data data/tiny_shakespeare_val.txt > /tmp/r7-eval-unc/logs/chunks-$c.log 2>&1 &
done; wait
grep -h "^EVAL-PLAN: seed=47 step=1200" /tmp/r7-eval-unc/logs/chunks-*.log
```

The seven `prefix-*` sidecars under `/tmp/r7-eval-unc/` are the primary
evidence: identical `sha256`, identical `optimizer_params`, differing
`final_val_bpb`, and each one now stating the `eval_chunks` / `eval_tokens` /
`eval_seq` / `val_bpb_stderr` it was measured with.

**phi^2 + phi^-2 = 3 | TRINITY**
