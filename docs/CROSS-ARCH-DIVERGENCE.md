# How different is "a different artifact"?

**What this document is.** The measured distance between the two checkpoints
that a cross-architecture CI run proved were not byte-identical. Every number
below comes from `docs/cross-arch-divergence.json`, which is written by

```bash
python3 scripts/compare_checkpoints.py \
    evidence/xarch-run-30767491098/12000.bin \
    checkpoints/r5-adv-recheck/12000.bin
```

The script is standard-library Python with no numpy, so its output is a
function of the two input files and nothing else. Re-run it and the JSON is
byte-identical, or one of these two files has changed.

**Why it exists.** `docs/REPRODUCIBILITY-GRADING.md` reported the negative
result honestly and then understated it: "identical 852 272-byte artifact size
-- and a different artifact." A reader is entitled to hear that as *the same
model up to rounding*. It is not. **43.70% of the parameters differ** -- 93,071
of 212,992, differing *in value*, with a further 2,073 differing *bitwise* only
in the sign of zero and therefore numerically equal (see
[`CANONICAL-SERIALIZATION.md`](CANONICAL-SERIALIZATION.md)) -- the
relative L2 distance is **0.4756**, and every element of all four trained
attention matrices differs, some by a change of sign. "Same model up to
rounding" is not available as an answer, and the mechanism that produces the
divergence is more interesting than the divergence.

**The figure carries a definition, and both definitions are published, so take
the reconciliation before the objection.** Two counts exist and neither is
hidden: 93,071 differ in value, 95,144 (44.67%) differ bitwise, and the gap
between them is exactly the 2,073 signed-zero parameters, which are numerically
equal. `43.70%` is the conservative of the two and is therefore the figure
quoted here and everywhere else in this repository. The reconciliation and the
census are in
[`CANONICAL-SERIALIZATION.md`](CANONICAL-SERIALIZATION.md); section 2 below
repeats both counts side by side.

**What is proven here, and what is not.** That the two artifacts are not
identical is measured, and every number below is a measurement on those two
files. That the *cause* is CPU architecture and OS is a **hypothesis**, and this
document does not measure it. Each artifact is a single run: the aarch64 side
has been repeated on its own host many times, and the `ubuntu-latest` x86_64
side that produced `bb14ab18...` has been trained **exactly once**, by run
30767491098. One observation per arm cannot separate a between-architecture
effect from within-arm non-repeatability on the arm that was never repeated.

**The repeatability control has now run locally, and it passes -- on a
different x86_64 than the one above.** On 2026-08-03 an `x86_64-apple-darwin`
build was executed twice under **Rosetta 2** on the aarch64 macOS host, with
byte-identical flags, and the two checkpoints are byte-identical to each other
at step 0, step 10 and step 200 (`docs/DIVERGENCE-LOCALIZATION.md`). So "the
trainer is simply not repeatable on x86_64" is no longer consistent with the
evidence: an x86_64 build of this trainer repeats itself. Read every
"architecture" attribution below accordingly -- the run-to-run branch of the
alternative is now excluded, and what remains undetermined is the split between
**ISA, OS and libc**, which the 12000-step pair varies all three of at once.

**Be precise about which x86_64.** The local control is
**x86_64-under-Rosetta-2-on-macOS**, not the `ubuntu-latest` arm that produced
`bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3`. It is a
genuine second ISA target for the compiler -- LLVM emitted real x86_64 machine
code -- and holding the OS and the libm vendor fixed is exactly what makes it
informative. But Rosetta 2 is binary translation, not native x86_64 silicon: its
handling of SSE/AVX, denormals and rounding state may differ from a native Intel
or AMD part, and Apple's x86_64 libm slice is not glibc. The within-`ubuntu-latest`
control (`TRIOS_CANON_NAME=r4-docs-repro-b`, in the `repro` job of
`.github/workflows/cross-arch-repro.yml`) **still has not run**, and nothing
below rests on it. Nothing about the divergence itself is softened by any of
this: the artifacts differ, and by exactly as much as the tables say.

---

## 1. The two artifacts

| | A: x86_64 Linux | B: aarch64 macOS |
|---|---|---|
| file | `evidence/xarch-run-30767491098/12000.bin` | `checkpoints/r5-adv-recheck/12000.bin` |
| sha256 | `bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3` | `8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c` |
| bytes | 852,272 | 852,272 |
| origin | GitHub Actions run [30767491098](https://github.com/gHashTag/trios-trainer-igla/actions/runs/30767491098), `ubuntu-latest`, glibc 2.39 | this machine |
| toolchain | rustc 1.96.0 (ac68faa20 2026-05-25) | rustc 1.96.0 (ac68faa20 2026-05-25) |
| `git_sha` | `3c1f751cf4376c13d26e247c2cd86357ab51dd20`, `git_dirty: false` | `3c1f751cf4376c13d26e247c2cd86357ab51dd20`, `git_dirty: true` |
| `source_sha256` | `19aa22fb7cd187774b71cbde7cb89b664aff7260b57f55afd865dd447707108b` | `19aa22fb7cd187774b71cbde7cb89b664aff7260b57f55afd865dd447707108b` |
| `final_val_bpb` | 2.637763500213623 | 2.6347548961639404 |

The x86_64 half is not reproducible from this repository: it was produced by a
third party's machine and downloaded as a CI artifact. It is preserved verbatim
in `evidence/xarch-run-30767491098/`, with `PROVENANCE.txt` naming the run, the
workflow, the commit and every provenance field of its sidecar.

The comparison is **paired on everything the record can name**. Comparing the
two sidecars field by field, only these differ: `sha256`, `path`, `canon_name`,
`platform`, `trainer`, `ts`, `git_dirty`, `final_val_bpb`, `best_val_bpb`,
`ema_bpb`. Seed, step, steps_total, hidden, d_model, num_attn_layers, vocab,
optimizer, fake_quant_format, lr, attn_scale, attn_seq, gf16_floor_every,
eval_every, data_synthetic, format_version, git_sha, source_sha256 and all
three corpus hashes are identical. The aarch64 sidecar records `git_dirty:
true`; the identical `source_sha256` is what shows the dirt did not reach the
trainer source.

**Only the failing half is in a clone. Say this before an auditor finds it.**
The aarch64 arm lives in `checkpoints/r5-adv-recheck/`, which is gitignored:
`git ls-tree -r HEAD --name-only checkpoints` returns nothing. A copy has been
made verbatim at `evidence/xarch-aarch64-reference/`, with a `PROVENANCE.txt`
naming its schema, its missing fields and its `git_dirty` disclosure, and its
bytes are identical (`cmp`) to `checkpoints/r5-adv-recheck/12000.bin` -- so the
commands in section 6 and the contents of `docs/cross-arch-divergence.json` are
unchanged; only the path recorded in a regenerated JSON would differ. **But
that directory is staged in the author's index and is not at HEAD**, checked
the only way that answers the question:

```
$ git ls-tree -r HEAD --name-only evidence/xarch-aarch64-reference   # nothing
$ git ls-files          evidence/xarch-aarch64-reference             # 3 files
```

`git ls-files` counts the index and will report those three files to the author
and to nobody else; `git ls-tree -r HEAD` is what a stranger's clone actually
receives. Conflating the two is how the earlier "both halves are now tracked"
sentence came to stand here for two days. Until that directory is committed,
the honest statement of this comparison is: **the arm that failed
(`evidence/xarch-run-30767491098/`, x86_64, `bb14ab18...`) is at HEAD and can
be re-hashed by anyone; the arm that passed (aarch64, `8a86fe69...`) cannot.**

### 1.1 The `source_sha256` equality is sealed to one commit

The sentence above -- "the identical `source_sha256` is what shows the dirt did
not reach the trainer source" -- is the load-bearing claim that makes this a
*paired* comparison with one variable changed. It does not survive being
re-run with today's binary, and the reason is worth stating before anyone else
finds it.

`19aa22fb...` was computed under the digest domain **`trios-source-tree/1`**,
whose entire input set at `3c1f751` was `src/**/*.rs` (115 files) plus
`Cargo.toml`. The tag has since moved twice: `trios-source-tree/2` widened the
inputs to `Cargo.lock`, `rust-toolchain.toml`, `migration/src/**/*.rs` and the
compiled feature set, and **`trios-source-tree/3`**, the value at HEAD, added
`.cargo/config.toml`. `src/checkpoint.rs` says in terms that a `/1` and a `/2`,
or a `/2` and a `/3`, digest over one unchanged tree differ and that "the
domain tag is what stops the two being compared as if they were", and the test
`source_digest_domain_is_version_three` pins the current tag so the rule cannot
drift silently. So today's trainer, run on a checkout of `3c1f751`, would print
a different `source_sha256` -- correctly, because it is answering a wider
question.

What that seals, and what it does not:

* **Sealed.** No binary built from HEAD reproduces `19aa22fb...`. Checking out
  `3c1f751` and running that trainer is one way back.
* **Not sealed after all: the digest is recomputable from tracked bytes.** The
  `/1` algorithm is 12 lines and its inputs are all in git. Recomputing it over
  `git archive 3c1f751` reproduces `19aa22fb7cd187774b71cbde7cb89b664aff7260b5`
  `7f55afd865dd447707108b` exactly -- the recipe is in
  `evidence/xarch-aarch64-reference/PROVENANCE.txt`. That is a stronger result
  than a re-run would be: it shows a **dirty** aarch64 tree recorded the digest
  of the **clean** tracked tree, so at walk time every Rust source and the
  manifest were byte-identical to `3c1f751`.
* **Narrower than it looks.** A `/1` equality says nothing about `Cargo.lock`,
  the pinned toolchain file, `migration/`, the feature set or the build flags.
  The CI arm built with `--locked` and `rust-toolchain.toml` pins the compiler,
  which is why the toolchain strings agree; but that is evidence from the run
  log and the record, not from this digest.

### 1.2 Cross-*laboratory* `source_sha256` equality is unattainable by construction

This follows from `/3` and must be said before anyone writes it into a rule.
`BUILD_FLAGS_PATH` is `.cargo/config.toml`, and at HEAD it is a hashed digest
input. That file is **gitignored**, is **generated per host** by
`scripts/repro_build.sh`, and its content **encodes the builder's home
directory** -- it is a list of `--remap-path-prefix` flags whose left-hand
sides are this machine's expanded `$HOME`, `$CARGO_HOME` and `$RUSTUP_HOME`.
Two honest laboratories running the identical procedure will therefore compute
**different** `source_sha256` values, and a third laboratory that clones the
repository will not have the file at all -- which `digest_source_inputs`
deliberately hashes as ABSENT rather than as empty, so that too is a third
distinct value.

The consequence, stated as a rule:

> A `source_sha256` mismatch between two laboratories is **expected**.
> `source_sha256` is a **divergence detector within one host** -- it tells you
> whether the tree moved under you between two runs -- and it is **not** a
> recipe-identity check across hosts. It must not be used as a grading
> criterion in any conformity scheme.

The identity a second laboratory can actually be asked to match is the
declared, hashed *recipe*: `git_sha` plus corpus hashes plus the pinned
toolchain plus the declared flags. Not this digest.

---

## 2. The divergence

The two files agree on their entire 152-byte header **and** on their 152-byte
tensor directory. They are the same shape, the same seed, the same step, the
same declared recipe. The first differing byte is at offset **2,882**, which is
inside the payload, in parameter **644**, in `embed` - the very first tensor.

| quantity | value |
|---|---:|
| parameters | 212,992 |
| parameters differing in value | **93,071** |
| fraction differing | **43.70%** |
| parameters differing bitwise | 95,144 (44.67%) |
| of which differ only in the sign of zero | 2,073 |
| parameters that are zero in both files | 33,503 |
| relative L2, `\|\|A-B\|\|2 / \|\|B\|\|2` | **0.4756** |
| the same with A as denominator | 0.4715 |
| `\|\|A-B\|\|2` | 29.0348 |
| `\|\|A\|\|2` / `\|\|B\|\|2` | 61.5781 / 61.0463 |
| median relative difference | **0.5588** |
| max absolute difference | **0.6854** |

The same figures as they appear in `docs/cross-arch-divergence.json`, so the
table above can be checked against the record without reformatting anything:

```json
"params": 212992,
"params_differing": 93071,
"frac_differing": 0.4369694636418269,
"params_differing_bitwise": 95144,
"params_signed_zero_only": 2073,
"params_zero_in_both": 33503,
"rel_l2": 0.475618745134565,
"median_rel_diff": 0.5588088370023011,
"max_abs_diff": 0.6854054629802704,
"first_differing_byte": 2882,
"header_bytes_equal": true,
"directory_bytes_equal": true
```

Two definitions, because both are choices:

* **differing in value** means `a != b` as IEEE-754 numbers, which counts
  `+0.0` and `-0.0` as equal even though they are different bytes. The bitwise
  count is reported too; the 2,073 gap between them is exactly the signed-zero
  parameters. `43.70%` is the conservative figure and is the one quoted.
* **median relative difference** is the median over the 93,071 *differing*
  parameters of `|a-b| / max(|a|,|b|)`. Over all 212,992 it would be `0.0`,
  because more than half of them agree - a statistic that says nothing about
  the ones that do not. The max-magnitude denominator is bounded: a sign flip
  scores 2.0 rather than infinity, and a parameter that is zero in one file
  scores exactly 1.0.

The largest single disagreement is parameter 185,481, in `wk`:

```
x86_64 Linux   -0.3120293915271759
aarch64 macOS  +0.3733760714530945
```

That is not a rounding difference. It is a different number with a different
sign.

### Per tensor

| tensor | parameters | differing | % | max abs diff | on the 1/16 grid |
|---|---:|---:|---:|---:|:--|
| `embed` | 8,192 | 1,824 | 22.27% | 0.187500 | yes |
| `ctx0` | 8,192 | 2,050 | 25.02% | 0.250000 | yes |
| `ctx1` | 8,192 | 2,338 | 28.54% | 0.312500 | yes |
| `ctx2` | 8,192 | 2,433 | 29.70% | 0.250000 | yes |
| `ctx3` | 8,192 | 2,416 | 29.49% | 0.312500 | yes |
| `ctx4` | 8,192 | 2,429 | 29.65% | 0.312500 | yes |
| `ctx5` | 8,192 | 2,544 | 31.05% | 0.375000 | yes |
| `proj` | 24,576 | 14,108 | 57.41% | 0.375000 | yes |
| `attn_down` | 24,576 | **1** | 0.004% | 1.86e-09 | no |
| `attn_up` | 24,576 | 24,576 | **100.00%** | 0.294869 | no |
| `lm_head` | 49,152 | 21,968 | 44.69% | 0.562500 | yes |
| `wq` | 4,096 | 4,096 | **100.00%** | 0.632240 | no |
| `wk` | 4,096 | 4,096 | **100.00%** | 0.685405 | no |
| `wv` | 4,096 | 4,096 | **100.00%** | 0.580690 | no |
| `wo` | 4,096 | 4,096 | **100.00%** | 0.518969 | no |
| `wq2` | 4,096 | 0 | 0.00% | 0 | (all zero) |
| `wk2` | 4,096 | 0 | 0.00% | 0 | (all zero) |
| `wv2` | 4,096 | 0 | 0.00% | 0 | (all zero) |
| `wo2` | 4,096 | 0 | 0.00% | 0 | (all zero) |

Three things in that table are worth saying out loud before anyone else finds
them.

1. **Every element of `wq`, `wk`, `wv` and `wo` differs.** All four first-layer
   attention matrices, 16,384 parameters, 100%.
2. **`wq2`, `wk2`, `wv2` and `wo2` are identically zero in both files.** The
   record says `num_attn_layers: 2`, but the second layer's four matrices hold
   a single distinct value - zero - in both artifacts. 16,384 of the 212,992
   parameters (7.7%) agree trivially because nothing ever wrote to them. They
   are counted in the totals above, and they flatter every "fraction differing"
   figure by being there. This is a finding about the model, not about the
   platforms, and it is recorded here because it was found while measuring the
   platforms.
3. **`attn_down` differs in exactly one of 24,576 parameters**, by
   `1.862645149230957e-09` - one f32 unit in the last place at that magnitude
   (`-0.024582361802458763` against `-0.024582359939813614`). One tensor of the
   pair is separated by a single last-bit rounding decision. The tensor next to
   it, `attn_up`, is separated in 100% of its elements. That contrast is the
   whole subject of the next section.

---

## 3. The mechanism: a 1/16 grid, and what a last bit does to it

`gf16_floor` snaps parameters to multiples of `1/16 = 0.0625`. The artifacts
say so themselves, without any appeal to the training source. `embed` holds
**18 distinct values across 8,192 parameters**; `proj` holds **17 across
24,576**; `lm_head` holds **23 across 49,152**:

| tensor | parameters | distinct values (A) | distinct values (B) | modal spacing | share of adjacent gaps at that spacing |
|---|---:|---:|---:|---:|---:|
| `embed` | 8,192 | 18 | 18 | 0.0625 | 17/17 (A), 17/17 (B) |
| `proj` | 24,576 | 17 | 17 | 0.0625 | 16/16 (A), 16/16 (B) |
| `lm_head` | 49,152 | 23 | 23 | 0.0625 | 22/22 (A), 21/22 (B) |

The one gap in `lm_head` on aarch64 that is not `0.0625` is `0.125` - a single
empty bin, i.e. two occupied bins with an unoccupied one between them, not a
different spacing.

Counting all the tensors whose every value is an exact multiple of 1/16
(`embed`, `ctx0`..`ctx5`, `proj`, `lm_head`; 131,072 parameters, 61.5% of the
model, plus the four all-zero layer-2 matrices which are trivially on any
grid):

```
differing parameters in on-grid tensors                     52,110
of which the difference is an exact multiple of 1/16        52,110   (100.0%)

difference, in grid steps:   1 step  40,435
                             2 steps  9,122
                             3 steps  1,983
                             4 steps    437
                             5 steps     97
                             6 steps     28
                             7 steps      6
                             9 steps      2
```

Not one of those 52,110 disagreements is a small numerical difference. Every
single one is a whole number of bins, and 77.6% of them are exactly one bin.

That is the mechanism, and it is not subtle: **a last-bit floating-point
difference on either side of a bin boundary becomes a bin flip, and a bin flip
is a 50% to 100% relative change in that parameter.** It is why the median
relative difference among differing parameters is 0.5588 rather than something
near zero, and it is why `attn_down` - which is not quantised - can sit one ULP
away from its counterpart while `embed` - which is - shows 1,824 parameters
that moved a full sixteenth.

The quantisation is not the whole story, and saying otherwise would be
overclaiming. The four trained attention matrices and `attn_up` are *not* on
the grid, and they differ in 100% of their elements with a maximum absolute
difference of 0.685 including changes of sign. Whatever separates them is most
likely ordinary floating-point non-associativity compounded over 12,000 steps
rather than rounding to a grid. That is now better supported than it was.
`docs/DIVERGENCE-LOCALIZATION.md` runs the same seed for 10 and 200 steps across
an ISA boundary and finds the identical *pattern* at its origin: at 10 steps the
divergence is confined to exactly these un-quantised tensors -- `attn_up` at
45.4%, `wq`/`wk`/`wv`/`wo` at 38-46% -- while **every** on-grid tensor is
byte-identical, and the largest disagreement anywhere is `9.2e-07`, one unit in
the last place. By 200 steps precisely one grid parameter has flipped a bin, by
exactly `0.0625`. The table below is the same process 12,000 steps later. It
remains true that neither document isolates non-associativity from libm, which
is the other candidate that survives; FMA contraction is excluded, because
neither binary contains a fused-multiply-add instruction. What quantisation
contributes is the *amplification*, and that part is measured: it converts
differences too small to see into differences that dominate the relative-error
statistics, and it does so at 61.5% of the parameters.

---

## 4. What follows, in this repository's own voice

**Bit-identity of this artifact is a knife-edge property of the gf16
quantisation on this fixture, not a robust property of the pipeline.** For the
two artifacts to hash the same, 212,992 rounding decisions must fall the same
way, 12,000 times, in two separate executions. They did not, and there is no
reason grounded in the arithmetic why they should. The single ULP in
`attn_down` is the size of the perturbation the pipeline actually has to
tolerate; the 1/16 grid is a mechanism for turning that perturbation into a
43.70% artifact difference -- 93,071 of 212,992 parameters differing in value,
with a further 2,073 differing bitwise only in the sign of zero and therefore
numerically equal. `43.70%` is the conservative of the two definitions and is
the one used here; see section 2 above and
[`CANONICAL-SERIALIZATION.md`](CANONICAL-SERIALIZATION.md).

Whether the two executions in question differed because they ran on two
different instruction sets, or would have differed on one, is exactly the
question the within-x86_64 repeatability control is there to answer. On the
local Rosetta arm that control has now been run and **passes**: two x86_64
executions of the same command are byte-identical to each other at 0, 10 and 200
steps. So the worse of the two readings is off the table -- this is a property
that survives a second run on the same CPU, and fails only across a change of
CPU. The knife-edge conclusion is unchanged; it is now knife-edge for a reason
that has been localised rather than assumed.

A conformity regime built on "the hashes match" is therefore building on the
wrong criterion. It would not be measuring whether a development cycle is
reproducible; on the architectural reading it would be measuring whether every
laboratory in the scheme happened to buy the same CPU, and on the alternative
reading it would be measuring nothing repeatable at all. And it fails in the
dangerous direction as well as the safe one: a hash match proves the bytes are
the same, but on this pipeline a hash *mismatch* proves almost nothing about
whether the two laboratories ran the same procedure.

**The transferable claim is the metric-level one.** The same run that produced
this 43.70% byte divergence -- 93,071 of 212,992 parameters differing in value,
plus 2,073 more differing bitwise only in the sign of zero and hence
numerically equal; `43.70%` is the conservative count, reconciled in
[`CANONICAL-SERIALIZATION.md`](CANONICAL-SERIALIZATION.md) -- produced a
`final_val_bpb` difference of 0.0030,
which is 0.084 of the estimator's own measured sigma (0.0358 bpb,
`docs/EVAL-UNCERTAINTY.md`) - statistically indistinguishable from zero. Two
executions that cannot agree on a single parameter of `wk` agree on the
published result to well inside the noise of the instrument that measures it.
That statement survives whichever way the repeatability control lands, which is
why it is the one worth transferring.
That is the property a conformity scheme can actually require of a second
laboratory, and `docs/REPRODUCIBILITY-GRADING.md` states the scope under which
0.0030 is the right number and the scope under which it is not.

The honest summary is a demotion of the strongest-sounding claim and a
promotion of the useful one:

> Do not certify on artifact hashes. Certify on a declared metric tolerance,
> measured with a declared sampling plan, and require the artifact hash only
> within a fixed `(os, arch, libc, toolchain)`.

---

## 5. What this evidence is not

One measurement, honestly reported. Specifically:

* **The cross-architecture workflow has executed exactly once.**
  `.github/workflows/cross-arch-repro.yml` has exactly one run in this
  repository's history: run 30767491098, `push` to `fix/509-qat-v2`, head
  `3c1f751cf4376c13d26e247c2cd86357ab51dd20`, started 2026-08-02T21:15:56Z,
  duration 7m04s, conclusion `failure`. That single run is the entire
  cross-platform evidence base, and everything above is one paired
  observation with `n = 1` per arm.
* **There is still no within-`ubuntu-latest` repeatability control in that
  evidence base.** The historical `repro` job invoked `trios-train` exactly
  once. The workflow in the working tree now runs the documented seed twice on
  x86_64 Linux and compares the two checkpoints to each other; **that job has
  not been run.** `two-lab-repro` does not close this: it compares two
  *binaries* built on one runner and never trains. What *has* closed is the
  weaker, non-CI version of the same control: an `x86_64-apple-darwin` build run
  twice under Rosetta 2 on the aarch64 host is self-identical
  (`docs/DIVERGENCE-LOCALIZATION.md`). That removes "the trainer is not
  repeatable on x86_64" as an explanation of the mismatch above, without
  supplying the native-Linux measurement.
* **The local control is x86_64 under Rosetta 2, not native x86_64.** It is a
  real second ISA target for the compiler and it isolates the ISA from the OS
  and the libc, which is more than run 30767491098 can do. It is not native
  silicon, and Apple's x86_64 libm slice is not glibc's, so it constrains the
  `ubuntu-latest` arm rather than standing in for it.
* **`localize-divergence` cannot run on a fresh checkout.**
  `TRIOS_CHECKPOINT_INIT` is not committed to `src/train_loop.rs`
  (`git show HEAD:src/train_loop.rs | grep -c TRIOS_CHECKPOINT_INIT` -> 0,
  against 5 in the worktree), so that job fails at
  `test -f checkpoints/loc-divergence/0.bin` for a non-architectural reason.
  The local experiment answered that job's question on one ISA pair by running
  the same knob from the working tree; the job itself remains unrun.
* **The version that ran is not the version now in the tree.** The run's job
  executed these steps: record the environment, checkout, rust-cache, fetch and
  split the corpus, verify the corpus against `data/MANIFEST.sha256`, build the
  trainer, run the documented seed, compare the checkpoint, publish the
  sidecar. The workflow in the working tree has since gained a step proving the
  training environment carries no DSN and no automigrate, a step building the
  repository's auditor alongside the trainer, and a step grading the sidecar
  with that auditor. **None of those three ran.** The evidence in
  `evidence/xarch-run-30767491098/` was produced without them.
* **The `two-lab-repro` job has never executed at all.** It exists only in the
  working tree; the one historical run contains a single job, `repro`.
* **The workflow does not run on the default branch.** Its triggers are
  `workflow_dispatch` and pushes to `fix/509-qat-v2`. Nothing here is enforced
  on merge, and a claim that "CI checks reproducibility" is true only for that
  one branch.
* **Two platforms is not a population.** macos/aarch64 and linux/x86_64/glibc,
  one run each *in this file*. Run-to-run variation has since been measured
  elsewhere -- on native aarch64 macOS and on x86_64-under-Rosetta on the same
  host, and it is nil at 0, 10 and 200 steps -- but nothing anywhere measures a
  third architecture, a second libc, or a second compiler version, and nothing
  measures run-to-run variation on native x86_64 Linux.

This is a measurement, not a CI regime. The distinction is the point of the
document.

---

## 6. Reproducing every number above

```bash
# the divergence record; writes docs/cross-arch-divergence.json
python3 scripts/compare_checkpoints.py \
    evidence/xarch-run-30767491098/12000.bin \
    checkpoints/r5-adv-recheck/12000.bin

# the same record on stdout, to diff against the committed one
python3 scripts/compare_checkpoints.py \
    evidence/xarch-run-30767491098/12000.bin \
    checkpoints/r5-adv-recheck/12000.bin --json \
  | diff - docs/cross-arch-divergence.json

# the evidence is unmodified
shasum -a 256 evidence/xarch-run-30767491098/12000.bin
#   bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3

# the container decodes under the independent Python reader too
python3 interop/triosckp_reader.py \
    evidence/xarch-run-30767491098/12000.bin \
    --sidecar evidence/xarch-run-30767491098/12000.json
```

`scripts/compare_checkpoints.py` exits non-zero if either file fails a
container check - magic, format version, header length, tensor count, reserved
bytes, or the `file_len = 304 + 4 * sum(directory)` identity - and refuses to
compare two files whose tensor directories disagree.

---

**phi^2 + phi^-2 = 3 | TRINITY**
