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

* Seed 47, 1,200 steps, `hidden = 384`, 2 attention layers, `lr = 0.003`,
  `optimizer = adamw`, `gf16_enabled = true`, `gf16_floor_every = 1`,
  `vocab = 128`, `attn_scale = 0.1`, `attn_seq = 8`. 196,608 parameters.
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
Against the headline artifact's own combined `u_c = 0.0657` (section 3b) the
same three "yes" rows fall to **1.77, 1.76 and 1.83 sigma** -- below the
`2 sigma` bar this section itself sets two paragraphs down. **On the headline
artifact, NONE of the claimed differences in the table below is resolvable.**
The table is retained as written because it is the correct arithmetic for the
checkpoint it was measured on, and because seeing both scalings side by side is
the point.

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
    excludes corpus choice, seed, platform and step count -- none of which
      is quantified anywhere in this document
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
only two rows are Type A, and the Type B rows are not merely small -- they are
**not quantified at all**.

**The measurand.** `final_val_bpb` of the artifact identified by
`sha256 = 8a86fe69...`, evaluated on the corpus identified by
`sha256 = 2088af36...`, under the sampling plan `eval_chunks = 40`,
`eval_seq = 129`, `eval_tokens = 5160`. Change any of those four hashes or
numbers and this budget does not apply.

| # | Component | Type | Quantified? | u_i (bpb) |
|---|-----------|------|-------------|-----------|
| 0 | **Window sampling on the headline artifact** -- the record's own `val_bpb_stderr` at `n = 40` | A | **yes** | **0.0551** |
| 1 | **Corpus truncation + grid**, upper bound -- the seven-prefix study, TRANSFERRED from a 1,200-step checkpoint | A | **yes, but not on this artifact** | **0.0358** |
| 2 | **Corpus choice** -- which held-out text the metric is computed on | B | **no** | not quantified |
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
2. **Corpus choice.** All seven runs come from one 100,000-byte tinyshakespeare
   tail. Section 4 states the between-corpus spread is "certainly larger". That
   is a judgement and it is entered here as a judgement -- **no number**,
   because none was measured.
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
6. **Selection / multiplicity `[ADDED 2026-08-03]`.** There is no test split.
   See section 3d, which states this as a scoping limit and as the second draft
   clause a conformity scheme needs.

One quantity that deliberately does **not** appear as a row:
* **Full-coverage stderr** (0.009055). With respect to *this* corpus the
  sampling error at full coverage is exactly zero. The 0.009055 is a standard
  error with respect to the wider population the corpus is a sample OF, which is
  row 2 restated -- and row 2 is not quantified.

**Combination `[RECOMPUTED 2026-08-03]`.** Only quantified components may be
combined. The five Type B rows contribute nothing to the arithmetic and
everything to the reading of it, so the root-sum-square runs over rows 0 and 1:

```
u_c = sqrt( 0.0551^2 + 0.0358^2 )  =  0.065725 -> 0.066 bpb
                                      combined standard uncertainty, k = 1
k                                  =  2        coverage factor
U   = k * u_c = 0.131450           -> 0.13 bpb expanded uncertainty

    val_bpb = 2.63 +/- 0.13 bpb   (k = 2, quantified components only)
    val_bpb = 2.63 +/- 0.07 bpb   (k = 1, quantified components only)
```

**This is 1.83x the `U = 0.0716` this document previously published, and the
`k = 1` figure is 1.64x the `+/- 0.04` still quoted elsewhere in this
repository.** The number moved against us and it moved because the headline
record was finally read instead of a smaller one being substituted for it.

**A rider on the combination itself, because rows 0 and 1 are not independent.**
Both estimate window-sampling scatter, by different routes: row 0 is the scatter
*inside* one grid, row 1 the scatter *between* grids that also changed corpus
length. Adding overlapping components in quadrature **over-counts**; taking only
the larger **under-counts** whatever of row 1 is genuinely separate. The two
bracketing treatments are:

```
envelope (treat as fully redundant, take the larger) : u_c = 0.0551, U = 0.110
quadrature (treat as independent)                    : u_c = 0.0657, U = 0.131
```

**This document adopts the quadrature figure**, and that is a **declared policy
choice, not a measurement**: for a conformity decision the conservative
direction is the larger uncertainty, because an understated `u_c` shrinks the
guard band of section 3c and silently raises the consumer's risk. The
correlation coefficient between the two routes has not been measured and would
be needed to do better than bracket it.

Four riders travel with that number and are not optional:

1. **`U` is a lower bound on the expanded uncertainty, twice over.** Row 1 is
   itself a lower bound with respect to nesting, and rows 2 to 6 are *excluded*,
   not estimated. `U` is not "the" uncertainty of `val_bpb`; it is the part of it
   that has been measured.
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
4. **Several bands are in circulation and they are different quantities.**
   `+/- 0.13` (k = 2, two components), `+/- 0.07` (k = 1, two components), the
   superseded `+/- 0.04` (k = 1, one transferred component), and the paired
   `0.003` of section 4a.
   [REPRODUCIBILITY-GRADING.md](REPRODUCIBILITY-GRADING.md) adopts `+/- 0.04` as
   an unpaired cross-laboratory *acceptance tolerance*. **That adoption now needs
   re-examination and this document does not own that file:** `+/- 0.04` was
   defended there as "the tight direction is the conservative one" for an
   acceptance test, but section 3c shows that a tolerance of `0.04` against
   `U = 0.13` yields an **empty** guarded acceptance interval -- it is not tight,
   it is undecidable. Whichever band is meant must be named at the point of use.

**What would move a row out of "not quantified".** Nothing in this document.
Each needs a new experiment, and putting an invented figure in an empty cell
would be strictly worse than leaving it empty:

* Row 2 -- the same weights evaluated on an independent held-out corpus.
* Row 3 -- k seeds trained to the same step budget, sigma taken across them.
* Row 4 -- the *unpaired* cross-platform comparison, each arm free to choose its
  own grid, rather than the paired one already run.
* Row 5 -- the seven-grid experiment repeated at 12,000 steps rather than 1,200.
  This is now the highest-value one: section 2b makes a falsifiable prediction
  about its outcome.
* Row 6 -- a split that never entered any selection decision. Section 3d.

---

## 3c. THE DECISION RULE (draft clause)

**Status: DRAFT CLAUSE. Provisional. Not a measurement.** Every figure in this
section is either arithmetic on the `u_c` of section 3b or a **declared policy
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
> **L2-3. Guard band (ARITHMETIC).** `w = U = k * u_c = 2 x 0.0657 = 0.13 bpb`,
> from section 3b. This follows from the budget; it is not chosen.
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

## 3d. What the budget still does NOT contain: there is no test split

**Scoping limit, and it is the sharpest one remaining.** Every number in this
document is computed on `data/tiny_shakespeare_val.txt`. Measured 2026-08-03:

* `data/README.md` marks **exactly one** file `Fit for eval? YES` -- that same
  `tiny_shakespeare_val.txt`. Of the other five, one is the training corpus, one
  is a byte-identical duplicate of the training corpus, one is a 160-byte
  degenerate pangram fixture, and one is the train split.
* **`--test-data` does not exist.** `grep -n 'test-data\|test_data'
  src/bin/trios-train.rs src/train_loop.rs` returns nothing. There is a train
  flag and a val flag and no third one.

So the split the metric is **reported** on is the split every design decision
was **selected** on: architecture, learning rate, `gf16` on or off, the floor
cadence, and every historical champion figure. A minimum or a best-of taken over
many runs scored on one split is a **maximum-of-N order statistic**, biased
optimistic by construction and biased further the more runs were compared. This
is the same defect that made `best_val_bpb` unfit to quote (section 5), scaled
from one run to the whole programme.

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
> **Status: DRAFT, and this repository currently FAILS it.** Stating a clause we
> do not yet pass is the point: it is a specification, not a certificate. Passing
> it requires a third corpus and a `--test-data` flag, neither of which exists
> today.

---

## 4. What this sigma does and does not cover

Covered:

* Which windows of the val corpus were read (the grid), at fixed `n`.
* Which prefix of the val corpus was available.

**Not** covered:

* **Corpus choice.** All seven grids are drawn from one 100,000-byte
  tinyshakespeare tail. The between-corpus spread is not measured here and is
  certainly larger.
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
2.63 +/- 0.07 bpb     (k = 1, TWO components, a LOWER BOUND; sampling plan
                       stated with the number -- see section 3b)
2.63 +/- 0.13 bpb     (k = 2, the band the decision rule of section 3c uses)
```

**`[CORRECTED 2026-08-03 -- this block used to read `+/- 0.04`, one component.]`**
It omitted the headline record's own `val_bpb_stderr = 0.0551`, which is larger
than the band it quoted.

That covers a second laboratory choosing its own `eval_chunks`, a different
stride, a different prefix of the corpus, or simply not recording which windows
it read. It does **not** cover a different corpus, a different seed or a
different step budget - see section 4 for what this budget leaves out, and
section 3d for the split that does not exist.

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
  parameters (`docs/CROSS-ARCH-DIVERGENCE.md`), which is the reminder that
  agreement in this metric is agreement in this metric and nothing else.

The one-line rule `[REWRITTEN 2026-08-03]`: **quote 0.003 only with the pairing
condition attached; quote +/- 0.07 (k = 1) whenever the sampling plans are not
identical -- and quote it as a lower bound on two components, never as the
uncertainty of `val_bpb`, which is at least +/- 0.13 at `k = 2` and has five
unquantified components on top of that (section 3b). The superseded `+/- 0.04`
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
