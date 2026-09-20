# A bit-exact `exp` and `cos`, substituted and priced

**N2 IS A MATCH.** With the two softmax `exp` calls and the one `cosine_lr`
cosine on the training path routed through `det_math::exp_det` and
`det_math::cos_det` instead of the platform libm, the step-10 checkpoint is
**byte-identical across the ISA change**:

```
INIT   aarch64 4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457
INIT   x86_64  4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457   MATCH
STEP10 aarch64 ca3bc1c999796cc7ac702357f12d2ca48625685a4e7b83499d2bb3fc3aa4fd1f
STEP10 x86_64  ca3bc1c999796cc7ac702357f12d2ca48625685a4e7b83499d2bb3fc3aa4fd1f   MATCH
```

Without the feature, on the same host and the same day, step 10 was
`efef1cba...` on aarch64 and `5913542e...` on x86_64.

**The STEP10 det-math hash moved, and that is expected rather than a
regression.** It was `7459c716a04ac89b0f590a4e645c47176d013b167bd5a138270df33dec8d6258`
when the feature pinned `exp` alone. At ten steps `warmup = steps / 10 = 1`, so
`cosine_lr` takes its cosine branch on nine of the ten steps and pinning the
cosine changes nine learning rates. Any hash taken with the older feature
describes an older function. The DEFAULT build is unmoved - N1 still reports
the published `4f854c82...` and `efef1cba...` - which is the check that the
gating is inert where it is supposed to be.

**And the 12000-step schedule now carries across too.** N4 used to be the
reason a 10-step MATCH could not speak for the headline: the learning-rate
schedule diverged on its own, through `cosf`, 58 of 12000 rates differing with
the first at step 4041. With `cos_det` that diff is **0 of 12000**. The
sentence this licenses, and did not before, is narrow and worth stating
exactly: **the cross-ISA MATCH now extends past step 4041, because the site
that broke there is gone.** It is not a 12000-step checkpoint claim - see "What
this does and does not license" below.

Evidence, verbatim, under `evidence/det-math-isa/` and
`evidence/cos-det-isa/`. Probes: `scripts/det_math_isa_probe.py` and
`scripts/lr_schedule_isa_probe.py`.

---

## The exact scope sentence

One macOS host (Darwin 25.5.0, macOS 26.5.2, Apple T6000), one rustc
(1.96.0 `ac68faa20`, pinned by `rust-toolchain.toml`), one working tree, one
corpus verified against `data/MANIFEST.sha256`. The `--target` triple is the
only variable that moved. The x86_64 arm ran under **Rosetta 2 binary
translation, not native x86_64 silicon**, and Apple's x86_64 libm is not
glibc's. n=1 per arm for every hash, with an n=2 within-arm repeatability
control that passed on both arms; n=3 per arm for the timing. Ten training
steps, not twelve thousand.

That paragraph is the claim. Anything shorter is a different claim.

---

## What was measured

### N1 - the control that makes the rest readable

Default features, aarch64, 10 steps, run **before** anything else:

```
INIT   4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457
STEP10 efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac
```

Both are the published values, and they are STILL the published values after
`cosine_lr`'s cosine was gated on top of the two `softmax` edits. N1 is what
shows every one of those edits is inert on the default build. If it had failed,
nothing below would be worth reading, and the probe exits 2 rather than
continuing.

### N2 - the result

`--features det-math`, both targets, 10 steps. MATCH, hashes above. The
within-arm control ran n=2 per arm and reported SELF-MATCH at both step 0 and
step 10 on both arms, which is what licenses reading the cross-ISA line as a
cross-ISA effect rather than as one arm failing to repeat itself.

No stage trace was needed. It would have been run to name the first divergent
tensor had N2 been a MISMATCH.

### N3 - the price, and it is not the number a microbenchmark gives

1000-step trainer runs on aarch64, n=3 per arm:

| arm | runs (s) | min | median |
|---|---|---|---|
| default  | 32.97, 32.99, 32.76 | 32.76 | 32.97 |
| det-math | 32.17, 33.47, 31.77 | 31.77 | 32.17 |

ratio det-math/default: **0.97 min, 0.98 median**.

**The honest reading is "no measurable cost", not "3% faster".** The det-math
arm's own spread is 1.70s, which is larger than the 0.80s gap between the two
medians. A ratio of 0.98 at n=3 with that spread is not distinguishable from
1.00. The measurement bounds the cost; it does not resolve its sign.

This contradicts the expectation the work item was written with - that
constraining the arithmetic costs speed. The measurement wins. The reason is
that `exp` is a rounding error's worth of the trainer's work: the time goes to
matmuls, and the two softmax loops are not where it lives.

**Do not quote a per-call microbenchmark as this number.** A scout measured
`exp_det` against `expf` in isolation at roughly 2.6-3.4x per call. That ratio
is real and it is not the cost of this feature, because the trainer does not
spend a measurable fraction of its time in `exp`.

### N4 - the scope limit that used to kill the obvious generalization, and does not any more

`train_loop::cosine_lr` calls `f32::cos`. When `det-math` replaced `exp` and
nothing else, that call was the whole obstacle. The 10-step probe runs with
`warmup = steps / 10 = 1`, so it exercises the cosine branch on nine of its ten
steps and reports the lr bits identical at every one - exactly the shape of
evidence that tempts a reader into extending a 10-step MATCH to 12000 steps.

`src/bin/lr_schedule_dump.rs` dumps the headline schedule (`max_steps 12000`,
`warmup 1200`, `base_lr 0.003`, steps 1..=12000) by **calling `cosine_lr`
itself**, not a replica of it. The binary is built for both targets from one
tree and the two dumps diffed. `scripts/lr_schedule_isa_probe.py` runs BOTH
feature states and checks the default one against the published baseline before
it will report the other:

| arm | lines compared | lines differing | first differing step |
|---|---|---|---|
| default (libm `cosf`) - **the control** | 12000 | **58** | 4041 |
| `--features det-math` (`cos_det`) - **the result** | 12000 | **0** | none |

```
default   aarch64  LR 4041 3b25032e
default   x86_64   LR 4041 3b25032d
default   aarch64  LR 11966 3728ff81      the last and largest difference
default   x86_64   LR 11966 3728ffe3

det-math  aarch64  dump sha256 d1625dddb18767f5d05d1acfe4e58da9e7d31dddd1a2ed753af5a28720220391
det-math  x86_64   dump sha256 d1625dddb18767f5d05d1acfe4e58da9e7d31dddd1a2ed753af5a28720220391
```

Most default-arm differences are 1 ULP; the last one, at step 11966, is larger,
because near the end of the cosine the schedule is subtracting two nearly equal
numbers and a 1-ULP disagreement in `cosf` is amplified by the cancellation.

**Both arms were run in the same tree on the same host minutes apart, after
every edit.** That ordering is the point. A `cos_det` that closed the gap in
both builds would mean something other than `cos_det` moved, and the
measurement would not be about `cos_det` at all. The default arm still reports
58 at step 4041, so it is.

(58 and step 4041 were predicted by an earlier out-of-repo replay. They are
re-derived here from the trainer's own function, not restated. The independent
agreement is worth something; it is not the reason the numbers are in this
document.)

Full evidence with every sha256 and every build command:
`evidence/cos-det-isa/lr-schedule.txt`.

### What this does and does not license

**Licensed, and it was not before:** the cross-ISA MATCH extends past step
4041. The one site that was MEASURED to break the headline schedule is gone,
and the schedule is now the same 12000 numbers on both instruction sets.

**Not licensed:** that the 12000-step CHECKPOINT is byte-portable. The schedule
is 12000 numbers; the headline is 12000 optimizer steps of a whole model. The
longest cross-ISA checkpoint MATCH anyone has run remains **ten steps**. What
closing the schedule did was remove the only known reason to expect a longer
run to diverge - it did not run one. A 12000-step two-arm run is the next
measurement, and it has not been taken.

**Not licensed:** anything about native x86_64 Linux. See the scope sentence.

---

## What `exp_det` is, and what it costs in accuracy

`src/det_math.rs`. Argument reduction `k = round(x * log2 e)`, Cody-Waite
two-part `ln2` subtraction, degree-6 Horner polynomial, `2^k` built from the
exponent bits. Only `+ - * /` and integer shifts - IEEE 754 requires all five
to be correctly rounded, so the result is a function of the input bits alone.
No `mul_add`, no libm call of any kind, not even `round` or `floor`. The module
header states why each of those choices is forced rather than merely tidy.

**`exp_det` is up to 3 ULP from Apple's libm.** Measured, not bounded by
argument: over 400001 samples of `[-8.0, 8.0]`, 83007 disagree with `f32::exp`
and the maximum is 3 ULP, at x = -7.96984. `tests/det_math_bit_exact.rs`
asserts the bound and prints the measured maximum on every run.

It also has a deliberate underflow deviation. `exp_det` returns `+0.0` once the
reduction yields `k < -126`; the boundary is sharp and was read off the
implementation:

```
x = -87.683113098 (0xc2af5dc1)   exp_det 8.312046e-39   libm 8.312044e-39
x = -87.683120728 (0xc2af5dc2)   exp_det 0.0            libm 8.31198e-39
x = -103.972084045 (0xc2cff1b5)  libm itself first returns 0.0
```

On `[-103.972084045, -87.683120728]` this function returns zero where libm
returns a subnormal. Reaching further down would require a second rounding into
the subnormal range, and a double rounding is the one place this construction
could quietly become platform-dependent again.

## What `cos_det` is, and what it costs in accuracy

Same file, same constraints, none of them relaxed. Argument reduction
`k = round(x * 2/pi)`, Cody-Waite **three-part** `pi/2` subtraction, quadrant
select on `k & 3`, and two Horner nests written out longhand - cosine to
`r^10`, sine to `r^9`. Only `+ - * /` and integer shifts. No `mul_add`, no libm
call of any kind, and that includes `abs`: the magnitude test masks the sign
bit with an integer AND rather than calling `f32::abs`.

`PIO2_HI` carries twelve significant bits, which is what makes `kf * PIO2_HI`
exact for every `|k| <= 2^11` and `x - kf * PIO2_HI` exact with it. That
argument stops holding above a point, so **`cos_det` has a hard domain limit**:
`|x| > COS_MAX_ARG = 2048.0` returns `NaN`. It is a refusal, not a result. An
unbounded reduction needs Payne-Hanek, which is different work; approximating
past the point where the construction is provable is the defect class this
module exists to remove. `cosine_lr` passes `PI * p` with `p` in `[0, 1]`, so
the caller is three orders of magnitude inside the limit.

Nothing upstream of the call needed replacing. `std::f32::consts::PI * p` is a
single correctly-rounded f32 multiplication - portable already, by the same
IEEE 754 guarantee the whole module rests on. The libm dependence began at
`.cos()` and ended there.

**`cos_det` is up to 2 ULP from Apple's libm.** Measured on the arguments that
are actually cited rather than on an abstract sweep: over the 10801
cosine-branch arguments of the headline schedule, 2194 disagree with `f32::cos`
and the maximum is 2 ULP, at x = 1.3197598. `tests/det_math_bit_exact.rs`
asserts the bound and prints the measured maximum on every run.

Two values are exact and the schedule's endpoints depend on it:
`cos_det(0.0) == 1.0` and `cos_det(PI) == -1.0`, the same two libm returns. So
`cosine_lr` hands back exactly `base_lr` at `step == warmup` and exactly `1e-5`
at `step == max_steps` in both builds, and a test asserts those bit patterns
under both feature states.

### The plain consequence

**Enabling `det-math` by default would move every published hash.** A 3-ULP
change in `exp` changes the softmax, which changes the gradients, which changes
every checkpoint from step 1 onward, and a 2-ULP change in `cos` moves the
learning rate on top of it - 2194 of the 10801 cosine-branch steps get a
different multiplier. The headline record
(`checkpoints/r6-headline/12000.json`), every seal in `evidence/SEALS.txt` and
every cross-architecture hash in `docs/` would have to be re-minted, and every
published BPB re-measured. That is why the feature is **default OFF** and must
stay off until somebody decides to pay that price deliberately. It exists so the
question can be ASKED, not so it can be shipped.

---

## Breaking the guard

The plan was to change the last Horner coefficient `C6` by one ULP and watch the
frozen-vector assertion fail. **It did not fail. It passed, and it was right
to.**

`C6 * r^6` contributes at most 2.4e-6 to a polynomial whose value is about 1.
Perturbing `C6` by its own last bit is a relative change of 6e-8 on that term,
so it moves the answer by about 1.4e-13 relative - against an f32 ULP of 6e-8,
roughly six orders of magnitude too small to change a single output bit. There
was nothing to detect. A test that had "caught" it would have been reporting
noise.

So the sensitivity of the guard was measured instead of assumed. Each constant
was moved up by one ULP in a replica first verified to reproduce all 28 frozen
pairs exactly, and the effect counted over the 28 frozen points and over the
400001-point sweep:

| constant | frozen pairs moved (of 28) | sweep outputs moved (of 400001) |
|---|---|---|
| `LOG2E`  | 0  | **0** |
| `LN2_HI` | 17 | 371874 |
| `LN2_LO` | 0  | **0** |
| `C2`     | 1  | 12101 |
| `C3`     | 0  | 775 |
| `C4`     | 0  | 55 |
| `C5`     | 0  | 5 |
| `C6`     | 0  | **0** |

Two things fall out, and both changed the code:

1. **Three of the eight constants are unobservable at f32 precision.** A
   one-ULP move of `LOG2E`, `LN2_LO` or `C6` changes not one bit of output
   anywhere in the sweep. No test can guard them, and the test file says so
   rather than implying otherwise.
2. **Twenty-eight points were not enough.** A one-ULP move of `C3`, `C4` or
   `C5` moves 775, 55 and 5 outputs while moving none of the 28 frozen pairs
   and staying inside the 3-ULP bound - a real change to `exp_det` that the
   guard as first written would have waved through. That is why
   `sweep_digest_is_frozen` was added: one FNV-1a checksum over the output bits
   of all 400001 points.

Both were then run against the real test, not the replica:

**`C3` +1 ULP** - `frozen_vector_is_bit_identical` PASSES,
`sweep_digest_is_frozen` FAILS:

```
digest 0xca03da8d1a340686, frozen 0x0e011a2586914fb2
```

**`LN2_HI` +1 ULP** - all four tests fail, and the frozen-vector assertion names
the first differing input, as it is supposed to:

```
first differing input: exp_det(1e0) [input bits 0x3f800000] gave 0x402df853,
                       frozen value is 0x402df854
17 of 28 pairs moved
...
exp_det vs f32::exp over [-8.0, 8.0]: n=400001, differing=375451,
                       max |ULP|=11 at x=-7.99988
exp_det drifted to 11 ULP from libm at x=-7.99988; the documented bound is 3
```

Both perturbations were reverted and all four tests are green.

### And the same, done to `cos_det`

`COS_C4` - the `r^4` coefficient, `1/24` - was moved DOWN by one ULP,
`0x3d2aaaab -> 0x3d2aaaaa`, and the suite re-run in both feature states.

**Under `--features det-math`, two tests fail and each names its own digest:**

```
cos_det_headline_digest_is_frozen FAILED
  digest 0xe2354941a65a3d19, frozen 0x0c4a426e5af24b0e

headline_schedule_digest_is_frozen FAILED
  digest 0x7ac85341a5041ed3, frozen 0x73e100bcf8e669fd
```

**Under default features the tripwire still fires**, because `cos_det` is
compiled unconditionally and only the call site is gated:

```
cos_det_headline_digest_is_frozen FAILED
  digest 0xe2354941a65a3d19, frozen 0x0c4a426e5af24b0e
```

The coefficient was restored and all nine tests are green in both states.

**The interesting part is what did NOT fail.**
`cos_det_matches_libm_within_measured_bound_on_the_headline` PASSED with the
broken coefficient, in both feature states. A one-ULP move of `COS_C4` changes
`cos_det`'s output on the headline arguments while staying inside the 2-ULP
bound - which is the same finding the `C3` experiment produced for `exp_det`,
arrived at independently. **An accuracy bound cannot guard a bit-exact
function.** Only a digest can, and that is why there are two.

---

## The libm surface, enumerated by the linker

Everything above is about two functions somebody named. The question underneath
it is whether the list of names is complete, and until now the honest answer was
the one written into `src/train_loop.rs:842-847` beside the gated cosine:

> That is not a proof that no other call exists; see docs/DET-MATH.md, "What was
> assumed, and what was not".

The linker answers it in one command. `nm -u` lists every symbol the binary
needs and does not define, so a libm entry point called from *anywhere* in the
linked program - trainer code, a dependency, a monomorphised generic - is in
that list, and one called from nowhere is not. That is an enumeration of the
whole binary, not a grep of the files somebody thought to open.

```console
$ cargo build --release
$ nm -u target/release/trios-train | sort > evidence/libm-surface/nm-default.txt
$ cargo build --release --features det-math
$ nm -u target/release/trios-train | sort > evidence/libm-surface/nm-det-math.txt
$ cargo build --release          # leave the tree on the default build
$ comm -23 evidence/libm-surface/nm-default.txt evidence/libm-surface/nm-det-math.txt
_cosf
_expf
$ comm -13 evidence/libm-surface/nm-default.txt evidence/libm-surface/nm-det-math.txt
$
```

161 undefined symbols on the default build, 159 with `det-math`. **The feature
removes exactly `_cosf` and `_expf` and adds nothing.** That is what the three
`#[cfg]` sites claim, now measured on the linked artifact rather than read off
the source.

`nm` names the symbol and not the caller, so the attribution below is from the
disassembly - the enclosing label of each call:

```console
$ otool -tV target/release/trios-train | awk '
    /^_?_?ZN|^[_A-Za-z].*:$/ { label = $0 }
    /symbol stub for: _(logf|log|log2|pow|powf|expf|cosf|sqrtf)$/ {
      n = split($0, a, ": "); print a[n] "\t<- " label }' | sort | uniq -c
```

### The ranking, for the `det-math` build

| rank | symbol | call sites | reaches |
|---|---|---|---|
| 1 | `_log` (f64) | `objective::shannon_entropy`, 1 | **the weights** |
| 2 | `_logf` (f32) | `train_loop::evaluate`, 1 | reported metrics + the sidecar |
| 3 | `_log2` (f64) | `num_bigint::...::to_radix_le`, 1 | nothing in the trainer |
| 3 | `_pow` (f64) | `tokio::...::Context::run`, 2 | nothing in the trainer |
| - | `_powf` | none | **absent from the binary** |
| - | `_sqrtf` | none | absent; lowered to the hardware instruction |

**Rank 1 is the finding. `det-math` does not make the training path libm-free.**
`src/train_loop.rs:3023` calls `nca_entropy_loss` inside the unconditional per-step
body of `for step in start_step + 1..=args.steps`; that calls
`objective::shannon_entropy`, whose `p.ln()` is this `_log`; and at
`src/train_loop.rs:3035-3038` the gradient buffer `gp` is multiplied element-wise
by `1.0 + nca_loss_val.min(5.0)`. One f64 libm call, surviving the feature that
exists to remove libm, scaling every gradient. The comment at
`src/train_loop.rs:2991-2998` already records that this amplifier is
*discontinuous* - `nca_state` is a `.round()` into `k` buckets - so it is one of
the few places in the loop where a last-bit difference becomes a macroscopic one
in a single step.

This is not a prediction that it diverges. `xarch_probe`'s `shannon_entropy_f64`
primitive came out `SAME` across the two arms on its grid. It is a statement
about scope: the thing measured to agree is not the thing `det-math` replaced,
and nothing pins it.

Rank 2 cannot move a weight. `evaluate` produces `val_bpb`, which feeds
`ema_bpb`, `best_val_bpb`, `final_val_bpb`, the printed line and the sidecar. It
can still end a run - `guard_bpb` turns an out-of-range reading into an error -
so it decides whether an artifact exists, not what is in it.

Rank 3 is dependency code. Neither symbol touches trainer arithmetic.

### One cell of the ULP census prices a call site that does not exist

`_powf` is absent from both builds. The only callers of
`model_hybrid_attn::rope_angle` at `e9eec90` are inside `#[cfg(test)]`, which
`src/bin/xarch_probe.rs:574` already recorded; the linker confirms it for the
trainer. `src/bin/ulp_census.rs` nevertheless publishes a `powf` cell - 25 of
4096 differing, 0.610%, 1 ULP. **That cell prices a call site the shipped
trainer cannot reach.** It is kept on purpose, so a revived RoPE path arrives
with its number already taken, and its `BAND` line now names the dormant grid it
is about rather than implying a live one.

### The method is sensitive, not vacuous

An absence is worth nothing until the same lens is shown saying "present". The
control is `ulp_census` itself, which calls all five functions:

```console
$ nm -u target/release/ulp_census | grep -E '^_(cosf|expf|logf|powf|sqrtf)$'
_cosf
_expf
_logf
_powf
$ otool -tV target/release/ulp_census | grep -c fsqrt
8
```

`_sqrtf` is missing from a binary that takes 69632 square roots, because the
compiler emitted the instruction instead. Four libm functions appear, the
hardware-lowered one does not, in a binary where all five are demonstrably
called. The absence of `_powf` from the trainer is a fact about the trainer, not
an artefact of the lens.

The full lists, the attribution and the reproduction commands are in
`evidence/libm-surface/PROVENANCE.txt`. What this does **not** settle: anything
statically linked. compiler-rt routines such as `__powisf2` are inlined or
linked as defined local symbols and `nm -u` cannot see them by construction -
`evidence/xarch-probe/PROVENANCE.txt` records that lens error and the
full-symbol-table check that replaced it. Nor does it say anything about a
Linux/glibc build; no such capture has been taken.

---

## The `ln` census was measured on the wrong domain

`src/bin/ulp_census.rs` published `ln` at 11 of 65536 (0.01678%) over
`fill_range(_, 123, 0.25, 8.0)`, and the comment on `measure()` called that cell
"`ln` through the loss". It is not. The loss calls `.ln()` on a softmax
probability - `src/train_loop.rs:1436`, guarded three lines above by
`if !p.is_finite() || p <= 0.0` - so its argument lies in `(0, 1]`.

Two measurements, both taken after this section was written.

**The buffer does not span what it declares.** `Lcg::next_f32` documents itself
as returning `[-1.0, 1.0)`; `(s >> 33)` is 31 bits, so it never exceeds 0 and the
true range is `[-1.0, 0.0]`. Every `fill_range(n, seed, lo, hi)` therefore spans
`[lo, (lo + hi) / 2]`. From the published aarch64 dump:

| buffer | declared | **measured** |
|---|---|---|
| `pos` | `[0.25, 8.0)` | `[0.2509801, 4.1239854]` |
| `ranged` | `[-8.0, 8.0)` | `[-7.9978266, -0.0000534]` |

That cuts both ways, and the direction is the point:

* **the `exp` cell is fair, by accident.** `softmax` computes
  `(x - max_val).exp()`, whose argument is non-positive by construction, so the
  call site's domain is `(-inf, 0]` - and the truncated `ranged` buffer lands
  entirely inside it. Had `next_f32` spanned the `[-1, 1)` it claims, half the
  `exp` census would have been on arguments `softmax` cannot produce. The
  published `exp 30 4096 0.732%` needs no asterisk.
* **the `ln` cell is not.** `[0.2509801, 4.1239854]` overlaps `(0, 1]` over less
  than a fifth of its own width and reaches none of `(0, 0.25)`.

**The reachable-domain rate is 2.455x the censused one.** A third census band,
`REACH`, samples `ln` uniformly *in value* over `(0, 1]` at the same 65536
elements as the dense pass, so both sit at the same 0.00703% detectable-rate
floor and only the domain moves. Both arms, `aarch64-apple-darwin` natively and
`x86_64-apple-darwin` under Rosetta 2, same host, same pinned 1.96.0:

| band | domain | n | differing | rate | max ULP | first index |
|---|---|---|---|---|---|---|
| sparse `ln` | `[0.25, 4.125]` | 4096 | 0 | 0.00000% | 0 | - |
| dense `ln` | `[0.25, 4.125]` | 65536 | 11 | 0.01678% | 1 | 7091 |
| **reach `ln`** | **`(0, 1]`** | **65536** | **27** | **0.04120%** | **1** | **256** |

`0.04120 / 0.01678 = 2.455`. The published figure understates the rate at the
call site by a factor of two and a half, in the flattering direction. The
one-unit-in-the-last-place shape survives the move: max ULP is still 1.

Sampling is uniform in value and not a bit-pattern stride walk. A stride over
`u32` patterns is uniform in `log2|x|` and spends half its draws below `1e-19`,
which is not where a softmax probability lives and not a number that could be
placed beside the cells above. The measured deciles of the reach buffer are
0.100, 0.200, 0.300, ... 0.900.

Each cell now publishes a `BAND` line carrying the domain it samples, the call
site it claims to describe and that call site's reachable domain, and
`tests/xarch_probe_census.rs::census_bands_sample_the_domain_they_claim` fails
if the three stop agreeing. `cos` and the two broad `ln` cells now declare
`claim=none` with a reason, because neither describes a call site: the `cos`
buffer is a *superset* of `cosine_lr`'s `[0, pi]` rather than a subset of it, and
the real ten arguments are measured by `src/bin/lr_schedule_dump.rs`.

Reproduce:

```console
$ cargo build --release --target aarch64-apple-darwin --bin ulp_census
$ cargo build --release --target x86_64-apple-darwin  --bin ulp_census
$ ./target/aarch64-apple-darwin/release/ulp_census > /tmp/ulp-arm64.txt
$ ./target/x86_64-apple-darwin/release/ulp_census   > /tmp/ulp-x86_64.txt
$ diff <(grep '^BAND ' /tmp/ulp-arm64.txt) <(grep '^BAND ' /tmp/ulp-x86_64.txt)   # empty
```

**Not yet in `evidence/`.** The two dumps under `evidence/xarch-probe/` still
carry only the sparse and dense passes; both are reproduced byte for byte as a
prefix of the new output, so nothing published has moved, but the `REACH` and
`BAND` lines are not in them and the derived `ULP-CENSUS.txt` therefore has no
reach row. Regenerating those two files is the remaining step.

---

## What was assumed, and what was not

**Measured** - everything in the four sections above; the 3-ULP bound for
`exp_det` and the 2-ULP bound for `cos_det`; the underflow boundary; the
58-then-0 schedule diff on both arms in the same tree; that both `softmax` call
sites are the only `.exp()` calls on the training path (grep over
`src/train_loop.rs` and `src/model_hybrid_attn.rs`; the other `.exp()` hits in
`src/bin/` are other binaries, and `src/multi_seed.rs:331` is an `f64` call
inside `betai`, a Student's-t p-value helper that nothing in `train_loop` or
`trios-train` references).

**Assumed, and stated as an assumption** - that the aarch64/x86_64 pair on this
one macOS host generalizes to the aarch64-macOS / x86_64-Linux pair that CI run
30767491098 measured. It does not follow. Rosetta is not silicon and Apple's
libm is not glibc's. A native-Linux det-math arm has NOT been run, and until it
is, "the checkpoint is byte-portable with `det-math`" is a claim about one host.

**WAS assumed, now measured and FALSE** - that `cosine_lr`'s cosine was the LAST
libm elementary call on the training path. This paragraph used to end "that is
not the same as a proof that no other call exists, and no census has been run
over the whole trainer". The census has now been run, by the linker, and it
found one more: `objective::shannon_entropy`'s f64 `_log` is still dynamically
bound in the `det-math` build and it scales gradients every step. See "The libm
surface, enumerated by the linker" above. `sqrtf` and `powi` are unaffected -
`sqrtf` is not linked at all on any of the three binaries checked, and `powi`
lowers to a statically linked compiler-rt routine that `nm -u` cannot see, which
`evidence/xarch-probe/PROVENANCE.txt` records separately.

**Assumed, and stated as an assumption** - that the dynamic surface is the whole
surface. `nm -u` enumerates what the binary needs and does not define; it says
nothing about anything inlined or statically linked, and it was taken on
aarch64-apple-darwin only. A Linux/glibc capture has not been made.

**Not measured** - whether N2 still MATCHes at 100, 1000 or 12000 steps. What
changed with `cos_det` is that N4 no longer gives a positive reason to expect it
does not. There is now no KNOWN divergent site left below 12000 steps, and "no
known site" is a statement about what has been looked for, not about what is
there. The measurement that would settle it is a two-arm 12000-step run, and it
has not been taken.

---

## A hole in the evidence, found while assembling it - and since closed

The checkpoint sidecars **could not tell a det-math run from a default run**.
Both reported

```
platform.features = ci-strict=0,gf16=0,gpu=0,race=0,smoke=0,trios-integration=0
source_sha256     = fda0dd211fa530f969d6bc372584470ce2f293dbb6495892883bc157b894ad2f
```

while their weight digests were `efef1cba...` and `7459c716...`. Two artifacts
with different bytes, identical declared provenance.

The cause was that `declared_feature_states()` in `src/checkpoint.rs` was a
hand-maintained list of six features and `det-math` was not in it. Its own doc
comment claimed "adding a feature changes the digest even when nobody turns it
on"; adding one to `Cargo.toml` demonstrably did not, because the list did not
derive from the manifest. That is the repository's own defect class - a field
that looks authoritative and is silently incomplete.

**It has since been repaired by separate work on `src/checkpoint.rs`.** The
strings below are quoted from what the binary emitted in the run of
`scripts/det_math_isa_probe.py --only isa` that produced the STEP10 hash at the
top of this document (`checkpoints/det-math-isa/20260806T174425Z/`), not
transcribed from the source:

```
det-math-aarch64/10.json
  platform.features = ci-strict=0,det-math=1,gf16=0,gpu=0,race=0,smoke=0,trios-integration=0
  source_sha256     = 8397b441fbdc8ca498621c86199829195975d65a8a3312610b1840d8846ef98f

det-math-control/10.json   (default features)
  platform.features = ci-strict=0,det-math=0,gf16=0,gpu=0,race=0,smoke=0,trios-integration=0
  source_sha256     = dadf9f4e0aa261179368dfa1080c34d7702082ed1112fba7b94b3f697961ba89
```

The feature state is now declared and the two `source_sha256` values differ, so
a sidecar is once again able to say which build wrote it.

Two caveats on those quotes. The `source_sha256` values are of the WORKING TREE
at measurement time, which carried this work item's edits and other in-flight
work; they identify the run, they are not a published constant. And a sidecar
that declares `det-math=1` still says nothing about which VERSION of the
feature ran - `evidence/det-math-isa/` and `evidence/cos-det-isa/` remain the
only things tying a particular hash to a particular `exp_det`/`cos_det`, which
is why `tests/det_math_bit_exact.rs` freezes both digests against the code.
