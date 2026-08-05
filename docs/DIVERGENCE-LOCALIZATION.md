# Where does the cross-architecture divergence begin?

Status: **measured, on one ISA pair, under Rosetta 2.** Initialisation is
byte-portable between aarch64 and x86_64; the divergence is introduced by the
training arithmetic. The within-arm repeatability control that licenses reading
this as an architecture effect has been **run and passes** on both arms.

**The one-sentence answer.** The initial weights are byte-identical across the
two instruction sets -- `0.bin` hashes to
`4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457` on aarch64
and on x86_64 alike -- while after ten optimizer steps they are not
(`efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac` on aarch64
against `5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab` on
x86_64), so the cause is **not** initialisation or RNG but the floating-point
arithmetic of the training loop.

Read that together with "Scope, and what this does not replace" below. The
x86_64 arm measured here is an `x86_64-apple-darwin` build executed under
**Rosetta 2**, not a native x86_64 part. That is what makes the experiment
clean -- OS, libm vendor, kernel and filesystem are held fixed and only the ISA
target moves -- and it is also the reason this result **constrains** the native
x86_64 Linux arm rather than replacing it.

The whole experiment is one command, `scripts/local_isa_probe.py`, which
rebuilds both arms, re-runs all four measurements, checks its own
preconditions and prints the verdict. It is the executable form of everything
on this page; a reader who does not trust the tables is meant to run it rather
than to read harder.

## The question

GitHub Actions run 30767491098 measured that the documented seed-47 12000-step
checkpoint produced on x86_64 Linux is **not** byte-identical to the one
recorded on aarch64 macOS:

| host | sha256 of `12000.bin` | bytes |
|---|---|---|
| x86_64 linux | `bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3` | 852272 |
| aarch64 macos | `8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c` | 852272 |

The compiler was pinned byte-identical by `rust-toolchain.toml`, dependencies
were built `--locked`, and all three corpus hashes (train, val, and the union
hash that proves the split is a partition) passed before training started.

### The attribution that run cannot make

**That run changed two things at once and can attribute its result to
neither.** The hosts differ in instruction set **and** in operating system, and
with the OS in libm and in libc: Apple's arm64 libm on Darwin against glibc
2.39 on Linux. Naming both variables is not the same as controlling either. So
run 30767491098 is evidence *that* the artifacts disagree and is not evidence
about *why*, and no summary of this repository -- pitch, README, or workflow
header -- may call it a CPU-architecture result. The header of
`.github/workflows/cross-arch-repro.yml` has been rewritten to say so at the
point of use, because that is where the number gets quoted from.

There is a further reason to distrust the architecture story specifically, and
it is worth stating before any expensive work is scheduled on the strength of
it: rustc does not enable fast-math or floating-point contraction, so LLVM is
not licensed to reassociate sums or to fuse `a*b + c` into an FMA in the
trainer's own code. Disassembly confirms it held -- see "What the mechanism is
not" below, where neither binary is found to contain a single FMA instruction.
That makes libm the stronger prior of the two candidates, and it makes the
single-variable experiment worth running *before* anyone constrains the
arithmetic at a cost in speed.

That run also says nothing about **where** the divergence begins, because the
earliest artifact it produces is 12000 optimizer steps deep. Two different
defects fit the same observation, and they have different fixes:

* **The initial weights already differ.** Then the likely cause is
  initialisation and RNG -- laying seeded values into arrays -- and it is
  outright fixable. No floating-point contraction is involved in that step.
* **Initialisation is byte-identical and the divergence appears only after
  gradient steps.** Then the likely cause is the arithmetic: reduction order,
  FMA contraction, or a libm difference reached from the forward/backward pass.
  That is fixable only by constraining the arithmetic, at a cost in speed.

Until `TRIOS_CHECKPOINT_INIT=1` existed the trainer could not be asked. Both
training loops are `for step in 1..=args.steps` and the periodic-checkpoint
predicate required `step % n == 0` with `n > 0`, so the earliest artifact any
run could write was **after** one optimizer step. `TRIOS_CHECKPOINT_INIT=1`
writes `0.bin` before the first step.

## The local ISA experiment

Measured 2026-08-03 on one host: Apple M1 Pro, macOS 26.5.2 (build 25F84),
`Darwin arm64` kernel 25.5.0, rustc `1.96.0 (ac68faa20 2026-05-25)`, release
profile. Rosetta 2 present and running (`/usr/libexec/rosetta`, `oahd` live).

Two binaries were built from **one** working tree, back to back:

| target | sha256 of `trios-train` | `file` says |
|---|---|---|
| `aarch64-apple-darwin` | `8a357f46af263f26f3def2710090853e6fdc32cadfcd53b9c3628c89379d0bd3` | Mach-O 64-bit executable arm64 |
| `x86_64-apple-darwin` | `96d0365c9837de21c0d5dd2a678fc4efb450b2dc3d3b03e8acb6c622663f225e` | Mach-O 64-bit executable x86_64 |

**The two binary hashes differ, and that is expected and carries no
information.** Different machine code for a different instruction set cannot
hash the same. Only the *checkpoint* was ever claimed to be portable.

```bash
cargo build --release --bin trios-train
cargo build --release --target x86_64-apple-darwin --bin trios-train

# aarch64 anchor
env -i PATH="$PATH" HOME="$HOME" TRINITY_AUTOMIGRATE=0 \
  TRIOS_CANON_NAME=loc-arm-anchor TRIOS_CHECKPOINT_INIT=1 \
  ./target/release/trios-train \
  --seed 47 --steps 10 --hidden 384 --attn-layers 2 --eval-every 1000 \
  --lr 0.003 --optimizer adamw \
  --train-data data/tiny_shakespeare.txt \
  --val-data data/tiny_shakespeare_val.txt

# the two x86_64 runs; a vs b IS the repeatability control
for name in loc-x86-a loc-x86-b; do
  env -i PATH="$PATH" HOME="$HOME" TRINITY_AUTOMIGRATE=0 \
    TRIOS_CANON_NAME=$name TRIOS_CHECKPOINT_INIT=1 \
    /usr/bin/arch -x86_64 ./target/x86_64-apple-darwin/release/trios-train \
    --seed 47 --steps 10 --hidden 384 --attn-layers 2 --eval-every 1000 \
    --lr 0.003 --optimizer adamw \
    --train-data data/tiny_shakespeare.txt \
    --val-data data/tiny_shakespeare_val.txt
done
```

`env -i` is the same environment scrub the CI jobs use. It matters here: this
host has `DATABASE_URL` set, and an inherited ledger DSN would have made the
runs write rows. `GF16_ENABLED`, `TRIOS_GF16_DISABLE` and
`TRIOS_GF16_FLOOR_EVERY` were confirmed unset before the runs and are scrubbed
by `env -i` regardless.

Artifacts land under `checkpoints/`, which is gitignored. The canon names to
regenerate are `loc-arm-anchor`, `loc-x86-a`, `loc-x86-b` at 10 steps and
`loc-arm-200`, `loc-x86-200-a`, `loc-x86-200-b` at 200 steps.

### The four facts, as measured

| # | question | answer |
|---|---|---|
| 1 | x86 `0.bin` == x86 `0.bin` across two runs? x86 `10.bin` == x86 `10.bin`? | **SELF-MATCH**, both |
| 2 | x86 `0.bin` == aarch64 `0.bin`? | **MATCH** |
| 3 | x86 `10.bin` == aarch64 `10.bin`? | **MISMATCH** |
| 4 | where does the first difference fall? | payload byte 426292, parameter 106497, tensor `attn_up` |

The hashes those answers are read off:

| artifact | sha256 | bytes |
|---|---|---|
| `loc-arm-anchor/0.bin` | `4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457` | 852272 |
| `loc-x86-a/0.bin` | `4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457` | 852272 |
| `loc-x86-b/0.bin` | `4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457` | 852272 |
| `loc-arm-anchor/10.bin` | `efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac` | 852272 |
| `loc-x86-a/10.bin` | `5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab` | 852272 |
| `loc-x86-b/10.bin` | `5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab` | 852272 |

The aarch64 pair is the anchor already recorded in `checkpoints/verify-init/`
and reproduced here from the current tree before any x86_64 run was taken, so
the comparison is same-tree rather than against a stale reference.

At 200 steps, same flags, same scrub:

| artifact | sha256 |
|---|---|
| `loc-arm-200/200.bin` | `a645d688b06b4bed5c90a7bd29fcce2dd38425c9282bad7d10dd34d7d7284071` |
| `loc-x86-200-a/200.bin` | `94e31069977a53fbebd7536b2c8e2da10be656be228929ab98357f3a578edb40` |
| `loc-x86-200-b/200.bin` | `94e31069977a53fbebd7536b2c8e2da10be656be228929ab98357f3a578edb40` |

**Fact 1 is the one that licenses the others.** Two independent x86_64 runs
agree with each other byte for byte at step 0, at step 10 and at step 200. Both
sidecars record `arch = x86_64`, `os = macos`. Determinism is therefore a
property of this trainer on *both* instruction sets, not only on the one it was
developed on, and the step-10 mismatch cannot be dismissed as the trainer simply
being unrepeatable on x86_64. That reading -- by far the worse of the two, since
it would make determinism itself platform-contingent -- is now excluded on this
host.

**Fact 2 answers the question this document was opened to ask.** The initial
weights are byte-identical across the ISA boundary. Weight initialisation and
the seeded RNG are portable as written; there is nothing to fix there.

**Fact 3 places the cause in the arithmetic.** Ten optimizer steps are enough to
separate the two artifacts.

### Fact 4: where, and how big

`scripts/compare_checkpoints.py` (read-only, standard-library Python, no numpy)
on the step-10 pair:

```bash
python3 scripts/compare_checkpoints.py \
    checkpoints/loc-x86-a/10.bin checkpoints/loc-arm-anchor/10.bin --json
```

| quantity | 10 steps | 200 steps |
|---|---:|---:|
| header bytes equal | yes | yes |
| tensor directory bytes equal | yes | yes |
| first differing byte | 426,292 | 242,323 |
| first differing parameter | 106,497 | 60,504 |
| first differing tensor | `attn_up` | `proj` (a signed zero) |
| parameters | 212,992 | 212,992 |
| parameters differing in value | 18,231 | 40,864 |
| fraction differing | **8.559%** | **19.186%** |
| parameters differing bitwise | 18,231 | 40,868 |
| of which signed zero only | 0 | 4 |
| relative L2 | **3.718e-08** | **1.436e-03** |
| max absolute difference | 9.174e-07 (`attn_up`) | 6.250e-02 (`lm_head`) |
| median relative difference | 1.067e-07 | 1.259e-05 |
| quantised parameters differing | **0** | **1** |

At 10 steps the first differing byte falls in the payload and is a genuine
value difference. At 200 steps the *first* one is not: parameter 60,504 of
`proj` is `-0.0` on x86_64 and `+0.0` on aarch64, four bytes apart at the sign
bit and numerically equal. `proj` has zero value-differences at 200 steps; the
first difference that is a difference of number is elsewhere.

The largest single disagreement at 10 steps is parameter 109,291 in `attn_up`:

```
x86_64 (Rosetta)  -0.012438224628567696
aarch64 native    -0.012439141981303692
```

That is a difference of `9.17e-07` on a value of `1.2e-02`: a last-place
rounding decision, not a different number. Contrast the 12000-step CI pair,
where the largest disagreement was a change of sign.

Per tensor at 10 steps:

| tensor | parameters | differing | % | max abs diff | on the 1/16 grid |
|---|---:|---:|---:|---:|:--|
| `embed` | 8,192 | 0 | 0.000% | 0 | yes |
| `ctx0`..`ctx5` | 8,192 each | 0 | 0.000% | 0 | yes |
| `proj` | 24,576 | 0 | 0.000% | 0 | yes |
| `attn_down` | 24,576 | 0 | 0.000% | 0 | no |
| `attn_up` | 24,576 | 11,150 | **45.369%** | 9.174e-07 | no |
| `lm_head` | 49,152 | 0 | 0.000% | 0 | yes |
| `wq` | 4,096 | 1,868 | **45.605%** | 4.470e-08 | no |
| `wk` | 4,096 | 1,874 | **45.752%** | 4.470e-08 | no |
| `wv` | 4,096 | 1,580 | **38.574%** | 4.470e-08 | no |
| `wo` | 4,096 | 1,759 | **42.944%** | 4.470e-08 | no |
| `wq2`, `wk2`, `wv2`, `wo2` | 4,096 each | 0 | 0.000% | 0 | (all zero) |

**Every quantised tensor is byte-identical at 10 steps.** The 1/16 grid absorbs
the perturbation entirely: 131,072 parameters, 61.5% of the model, agree exactly
while the un-quantised tensors next to them disagree in 38% to 46% of their
elements at the last bit. By 200 steps exactly **one** grid parameter has
flipped a bin -- one element of `lm_head`, by exactly `0.0625` -- and by 12000
steps 52,110 of them have (`docs/CROSS-ARCH-DIVERGENCE.md`, section 3). The
amplification mechanism described there is now observed in progress rather than
inferred from its end state:

| steps | parameters differing | quantised parameters flipped | relative L2 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 10 | 18,231 (8.56%) | 0 | 3.718e-08 |
| 200 | 40,864 (19.19%) | 1 | 1.436e-03 |
| 12000 | 93,071 (43.70%) | 52,110 | 4.756e-01 |

"Parameters differing" in every row of this table means **differing in value**,
out of 212,992. On the 12000-step row a further 2,073 parameters differ
*bitwise* only in the sign of zero and are numerically equal, so the bitwise
count there is 95,144 (44.67%); `43.70%` is the conservative of the two
definitions and is the one quoted here and elsewhere. The reconciliation and
the census are in
[`CANONICAL-SERIALIZATION.md`](CANONICAL-SERIALIZATION.md).

The 12000-step row is the CI pair (x86_64 Linux against aarch64 macOS), not this
Rosetta pair, and is included to show the trajectory, not as a fourth point on
one curve.

### Independently replicated, 2026-08-05, by one command

The experiment above was assembled by hand. It has since been re-run start to
finish by `scripts/local_isa_probe.py`, which rebuilds both targets from the
working tree, runs each arm **twice**, and re-derives all four hashes:

```bash
python3 scripts/local_isa_probe.py --json evidence/xarch-local-isa/probe.json
```

Verdict `ISA_SUFFICIENT_TO_DIVERGE`, control passing on both arms, from a *newly
built* pair of binaries -- `709fcbf2...` (aarch64) and `1083862c...` (x86_64),
neither of which is the pair the tables above were measured with. The four
hashes came back unchanged: `4f854c82...` at init on both instruction sets,
`efef1cba...` against `5913542e...` at step 10. Two further back-to-back
invocations of the probe reproduced the same four hashes again.

The verdict string said ONLY where it now says SUFFICIENT when those runs were
taken; the archived record reads `ISA_SUFFICIENT_TO_DIVERGE` because
`probe.json` was **re-minted by re-running the probe**, not edited. See "The
verdict string was too strong" below for why the name changed. The re-run used a
third pair of freshly built binaries -- `faec0870...` (aarch64) and
`a213d226...` (x86_64) -- and returned all four hashes identical again.

This matters for one specific doubt. The measurements above were taken from a
dirty tree while other work was landing in it, which is a real objection to
them. The anchor has now been reproduced under **six** distinct trainer binaries
-- the re-run above added the third pair -- built from six recorded source
states, so the concurrent edits demonstrably do not reach the numerics.

One caveat on the archive, stated because the directory is the evidence: the
`.bin` files under `evidence/xarch-local-isa/` are the ones the *earlier* pair of
binaries wrote. They are byte-identical to what the re-run produced -- that is
what "all four hashes identical" means -- so they were not replaced, and their
sidecars therefore still carry the record schema of the day they were minted.
`probe.json` and `probe-stdout.txt` are from the re-run.

The probe also proves rather than assumes the two properties the experiment
depends on. It reads `platform.arch` back out of each run's own sidecar --
`aarch64` and `x86_64`, with `os=macos` on both -- so "the ISA was varied and
the OS was not" is an observation from inside the two processes, not an
inference from the command line. And it re-hashes both binaries after the last
run to confirm neither changed underneath the measurement, which on this tree
is a live hazard rather than a theoretical one.

Artifacts, sidecars and the probe's full stdout are archived under
`evidence/xarch-local-isa/`. **They are staged, not committed**, and the
distinction matters to a reader rather than to the author, so it is stated
here: `git ls-tree -r HEAD --name-only evidence/xarch-local-isa` returns
nothing, while `git ls-files` on the same path lists them, because it counts
the author's index. Until that directory is committed, this section's evidence
is reproducible by re-running the probe but is **not** obtainable from a clone.
Each invocation writes into its own timestamped
directory under `checkpoints/isa-probe/`, so re-running never overwrites or
deletes a previous measurement.

### And the metric, at every step count

| steps | aarch64 `final_val_bpb` | x86_64 `final_val_bpb` | difference |
|---:|---|---|---:|
| 0 | 7.000166416168213 | 7.000166416168213 | 0 |
| 10 | 6.257606029510498 | 6.257606029510498 | 0 |
| 200 | 3.5353050231933594 | 3.5353055000305176 | 4.8e-07 |

At 10 steps the published metric is **bit-identical** on two instruction sets
whose weight tensors already disagree in 8.56% of their parameters. At 200 steps
it differs by `4.8e-07`, which is `1.3e-05` of the estimator's own measured sigma
(0.0358 bpb, `docs/EVAL-UNCERTAINTY.md`). This is the same lesson the 12000-step
pair taught at 0.0030 bpb, visible from step 10: **the artifact hash is a far
more fragile object than the number the artifact is published for.**

### What the mechanism is not

Two candidate causes were named above. One of them can be excluded by reading
the binaries, which is cheaper and more direct than reasoning about it:

```bash
otool -tv target/release/trios-train \
  | grep -cE '\b(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)\b'
#   0
otool -tv target/x86_64-apple-darwin/release/trios-train \
  | grep -cE 'vfm(add|sub|nmadd|nmsub)'
#   0
```

**Neither binary contains a single fused-multiply-add instruction.** FMA
contraction in the trainer's own compiled code is therefore not the mechanism;
Rust does not enable floating-point contraction by default, and this confirms it
held here. What the platform libm does inside its own implementations is not
observable this way and is not excluded.

Both counts re-measured 2026-08-05 on the second pair of binaries
(`709fcbf2a95342a8bbf70dab6ea3acdc157b8e771b908386ae5e9bb72db36845`,
`1083862c3166dbafdc9e9db66b76c3ef78d9d7d89584902def8b77365a0833e4`): **0** and
**0**. The count is a property of how this trainer is compiled, not of the
particular build the claim was first read off.

#### Positive control: the check is capable of firing

A grep that returns 0 is worthless until you have seen it return 1. Both
patterns above were run against a deliberately FMA-carrying object, built on
this machine on 2026-08-05:

```bash
printf 'float f(float a,float b,float c){return a*b+c;}\n' > /tmp/fma_pos.c
clang -target x86_64-apple-darwin -O2 -mfma -c /tmp/fma_pos.c -o /tmp/fma_pos.o
otool -tv /tmp/fma_pos.o | grep -cE 'vfm(add|sub|nmadd|nmsub)'
#   1        the instruction is `vfmadd213ss %xmm2, %xmm1, %xmm0`
```

and, for the other arm, with no `-m` flag at all, because FMA is in the AArch64
baseline and needs no opting in:

```bash
clang -target arm64-apple-darwin -O2 -c /tmp/fma_pos.c -o /tmp/fma_pos_arm.o
otool -tv /tmp/fma_pos_arm.o \
  | grep -cE '\b(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)\b'
#   1        the instruction is `fmadd s0, s0, s1, s2`
```

That second control is what gives the aarch64 zero its force: `fmadd` was
available to the compiler on that target and was not emitted.

**A correction, recorded rather than quietly applied.** Until 2026-08-05 the
x86_64 arm above matched the literal `vfm` followed by a *lowercase-letters-only*
character class, fenced by word boundaries on both sides. That pattern **cannot
match any x86 FMA mnemonic**, because every one of them carries digits: in
`vfmadd213ss` there is no word boundary between the `d` and the `2`, so a
lowercase-only run can never reach a boundary at the end of the token. Run
against `/tmp/fma_pos.o` the old pattern returns **0** -- on an object that
provably contains `vfmadd213ss`. The old pattern is deliberately not reproduced
here in copyable form, so that nobody re-inherits it.

The check was vacuous, and it sat beside an aarch64 check that was sound, which
is the worst possible arrangement: two commands that look symmetric, only one of
which was capable of failing. The **conclusion is unchanged** -- the corrected
pattern still returns 0 on the real binary -- but on the x86_64 arm that
conclusion was, until this correction, unevidenced rather than wrong. A check
that cannot fire proves nothing, and it is more dangerous than no check at all,
because it looks like one.

**The stronger and cheaper argument needs no disassembly at all.** The x86_64
build targets baseline x86-64, and that target simply has no FMA feature
enabled:

```bash
rustc --print cfg --target x86_64-apple-darwin | grep target_feature
#   cmpxchg16b, fxsr, sse, sse2, sse3, sse4.1, ssse3   -- and no `fma`
```

Absence of FMA in that binary is therefore a property of the target
specification, guaranteed before a single byte is disassembled; the
disassembly only confirms that nothing overrode it. The aarch64 arm gets no
such guarantee, since `fmadd` *is* baseline there, which is precisely why the
disassembly is doing real work on that side and the positive control above
matters.

What remains, both evidenced, neither isolated:

* **Reduction order.** Both backends auto-vectorised the float arithmetic, and
  differently: 326 `fadd.4s` and 695 `fmul.4s` on aarch64 against 282 `addps`
  and 392 `mulps` on x86_64. Floating-point addition is not associative, so two
  different vectorisation schedules over the same sum are entitled to two
  different results. (On the second pair of binaries the same counts read
  324 / 692 / 282 / 392. They drift with the tree, as instruction counts do;
  what does not drift is that the two schedules differ.)
* **libm.** Both binaries import the same four symbols -- `_expf`, `_log`,
  `_logf`, `_pow` -- and resolve them against **different architecture slices**
  of the platform libm. None of those four is bit-specified by IEEE-754, so the
  two slices are permitted to differ in the last place and no standard says they
  must not.

Distinguishing these two is the next measurement, and this document does not
make it.

## Scope, and what this does not replace

**Rosetta 2 is binary translation, not native x86_64 silicon.** This has to be
read in both directions and neither one may be dropped.

What makes the experiment *good*: the compiler genuinely targeted a second
instruction set. LLVM performed x86_64 instruction selection, register
allocation and auto-vectorisation, and emitted real x86_64 machine code against
the `x86_64-apple-darwin` baseline feature set (`sse`, `sse2`, `sse3`, `ssse3`,
`sse4.1`, `cmpxchg16b`, `fxsr`). Meanwhile the OS, the kernel, the filesystem,
the libm *vendor* and the compiler version are all held fixed, and both arms ran
on one physical machine within seconds of each other. The CI comparison that
produced `bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3`
cannot attribute its mismatch because it varies architecture, OS and libc
simultaneously; this one varies the ISA target and holds the rest still. That is
why it can answer the init-versus-arithmetic question at all.

What makes it *not a substitute*: Rosetta translates x86_64 machine code to run
on ARM hardware. IEEE-754 pins the results of `+`, `-`, `*`, `/` and `sqrt`
exactly, so a correct translation of those is bit-exact by construction -- but
"the translation is correct" is an assumption this experiment does not test, and
Rosetta's handling of SSE and AVX semantics, of denormal and rounding-mode
state, and of anything a native part implements in hardware that Rosetta must
emulate, may differ from a native Intel or AMD CPU. Equally, the x86_64 slice of
Apple's libm is not glibc's libm: the `_expf` and `_logf` reached here are not
the ones an `ubuntu-latest` runner reaches.

So, explicitly, **which result each claim rests on**:

| claim | rests on |
|---|---|
| Initialisation is byte-portable across ISA | this local Rosetta experiment only |
| The divergence is introduced by training arithmetic, not init | this local Rosetta experiment only |
| The trainer is self-repeatable on an x86_64 build | this local Rosetta experiment only |
| Quantisation amplifies a last-bit difference into a bin flip | `docs/CROSS-ARCH-DIVERGENCE.md` at 12000 steps, corroborated here at 200 |
| A checkpoint produced on x86_64 **Linux** differs from the aarch64 macOS one | CI run 30767491098 only |
| The metric survives where the hash does not | both, independently |

**Do not read this experiment as having retired the native x86_64 Linux arm.**
It constrains it. Before this, a native-Linux mismatch was consistent with an
init/RNG bug, with an arithmetic difference, with a libc difference and with
plain non-repeatability. Two of those four were made unlikely here:
initialisation carried across an ISA change, and the trainer repeated itself on
an x86_64 build. What this experiment could not see -- because it holds the OS
fixed by construction -- is whether the *native* x86_64 arm behaves as the
translated one does, and whether glibc's libm contributes on top of the ISA.
Only a native x86_64 host answers that. **As of 2026-08-05 one has**; see
directly below.

### The CI job that asks this has now run, and it answered

This section used to say the job could not run, because
`TRIOS_CHECKPOINT_INIT=1` existed "only in the working tree". That blocker is
gone. Both commands the section printed as proof now return the opposite of
what it claimed, re-measured 2026-08-05:

```
$ git show HEAD:src/train_loop.rs | grep -c TRIOS_CHECKPOINT_INIT
5
$ git log --all -S TRIOS_CHECKPOINT_INIT --oneline
ba272b9 feat(repro): localize the cross-architecture divergence, and retract what it invalidates
```

The variable landed in `ba272b9`, that commit is on `origin/fix/509-qat-v2`, and
`actions/checkout@v4` therefore now fetches a trainer that honours it. The old
paragraph is preserved above in summary rather than deleted, because a document
that silently drops its own retracted claims cannot be audited.

**`localize-divergence` ran on 2026-08-05 in run `31004703001`, on native
x86_64 Linux, and reported:**

```
 init   x86_64 linux : 4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457
 init   aarch64 macos: 4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457
 step10 x86_64 linux : a32e9b2ab0043b91aaaae96f3e6945a419ef305e375fa2aaf2ba38a6deba19d1
 step10 aarch64 macos: efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac
 bytes  : 852272 / 852272
INIT MATCH
STEP10 MISMATCH
```

Environment recorded by the job itself: `rustc 1.96.0 (ac68faa20 2026-05-25)`,
`ldd (Ubuntu GLIBC 2.39-0ubuntu8.7) 2.39`.

**The job's GitHub conclusion is `failure`. That is the designed outcome, not a
malfunction.** The step exits 1 on `STEP10 MISMATCH` precisely so the finding
cannot be mistaken for a pass. Red here means *measured*.

Two things follow, and they are worth separating:

* **The native arm lands in the same cell as the Rosetta pair: MATCH /
  MISMATCH.** Initialisation is byte-identical across architecture, OS *and*
  libc taken together -- `4f854c82...` is the same init hash this document
  anchors locally on aarch64 macOS and under Rosetta. The divergence is
  introduced by the training arithmetic, within ten gradient steps. What the
  local probe showed with the OS held fixed, CI now shows with the OS varying
  too.
* **It does not isolate ISA from OS/libm.** This job varies both at once. The
  local Rosetta probe is what holds the OS fixed; the two together say the ISA
  alone is *sufficient*, and that adding a second OS and libc does not change
  the verdict. Neither says libm contributes nothing.

The step-10 x86_64 Linux hash `a32e9b2a...` also differs from the x86_64 Rosetta
step-10 hash `5913542e...` recorded above. Those two are not a controlled pair --
different OS, different libc, different host -- so the difference is recorded and
not attributed.

### Limits on the local reference itself

`git_sha = 3c1f751cf4376c13d26e247c2cd86357ab51dd20`, `git_dirty = true`. Both
arms were produced by an **uncommitted** working tree, and other work was
landing in that tree while these runs were taken. That was checked rather than
assumed, in three ways:

1. **The binaries did not change.** Both `trios-train` hashes above were
   re-verified after every run and were unchanged throughout. Every run in this
   document was executed by one of exactly two binaries, built back to back from
   a single source state.
2. **The anchor is stable across source states.** The aarch64 pair
   `4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457` /
   `efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac` has now
   been reproduced under four different recorded `source_sha256` values and two
   different trainer binaries, including the `checkpoints/verify-init/` pair
   taken earlier the same day. The concurrent edits do not reach the numerics.
3. **The x86_64 pair is stable too.** Re-running the x86_64 binary against a
   later, static tree reproduced `4f854c82...` and `5913542e...` exactly, at
   which point both binaries also recorded the *same* `source_sha256`
   (`1569500d614776d999ae7eff7ec675e0f3421aac4a2d40624b4bdd8caf92e643`),
   confirming that an earlier disagreement in that field was a concurrent edit
   landing between two runs and not an architectural effect. `source_sha256` is
   a run-time walk of the working directory; it describes the tree at the moment
   of the run, not the tree the binary was compiled from.

Both arms verified the corpus identically before training:

```
train  1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d
val    2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502
```

`--attn-layers 2`, `--eval-every 1000` and the two corpus paths are CLI defaults
and were passed explicitly, matching what the CI job runs. Note that
`--eval-every` is not a pure observation parameter on this trainer: `gf16_floor`
mutates weights in place on an eval-gated schedule late in training. At 10 and
200 steps with `--eval-every 1000` that gate never fires, so it is not a
confound here, but runs with a different eval cadence are not comparable to
these.

## Reading the four outcomes

Kept because the reasoning is what licenses row 3 of the table above, and
because a native x86_64 run will land in one of these cells.

| `INIT` | `STEP10` | What it means | Fixable how |
|---|---|---|---|
| MATCH | MATCH | The artifact carries across at both points, at 10 steps. Says nothing about 12000 -- run 30767491098 measured a mismatch there -- so this would mean the divergence accumulates later and the next step is to bisect the step count. | n/a; bisect |
| MISMATCH | MISMATCH | The divergence is present **before any gradient step**. Cause: initialisation / RNG, not floating-point arithmetic. | Outright: make weight initialisation platform-independent (integer RNG, explicit bit-level conversion). No speed cost. |
| **MATCH** | **MISMATCH** | **Observed twice: here on the Rosetta pair, and on native x86_64 Linux in CI run `31004703001` (2026-08-05).** Initialisation is byte-identical; the divergence is introduced by the training arithmetic within 10 steps. Cause: reduction order or libm (FMA contraction excluded by disassembly *and* by the x86_64 target spec, which carries no `fma` feature). | Only by constraining the arithmetic: fix summation order, avoid libm where a correctly-rounded substitute exists. Costs speed. |
| MISMATCH | MATCH | Contradictory: different starting weights cannot yield identical weights 10 steps later. Treat the run as broken and re-run before drawing any conclusion. | n/a; the experiment failed |

## Scope of every claim on this page

* Same-machine determinism, aarch64 native (Apple M1 Pro, macOS 26.5.2, rustc
  1.96.0, release): **measured**, at init, at 10 steps and at 200 steps.
* Same-machine determinism, x86_64 under Rosetta 2 on the same host:
  **measured**, at init, at 10 steps and at 200 steps, two independent runs at
  each. This is the repeatability control, and it passes.
* Same-machine determinism on **native** x86_64 Linux: **measured 2026-08-05**,
  run `31004703001`. The `repro` job of `.github/workflows/cross-arch-repro.yml`
  trained the documented seed twice on that arm; both runs produced
  `bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3` at 852272
  bytes -- "SELF-MATCH - two independent x86_64 runs produced identical bytes."
  That is also the hash run `30767491098` recorded three days earlier on a
  different runner instance. This bullet said "still not measured" until the
  job was run.
* Cross-platform behaviour at **0 and 10 steps**, native x86_64 Linux against
  aarch64 macOS: **measured 2026-08-05**, same run, `localize-divergence` job:
  `INIT MATCH`, `STEP10 MISMATCH`. This bullet previously said the job could not
  run at all; see the section above for the full output and for what it does and
  does not isolate.
* Cross-ISA behaviour at 0, 10 and 200 steps, macOS host, Rosetta:
  **measured**, and reported above. The 0- and 10-step pair is **replicated**
  by `scripts/local_isa_probe.py` across two pairs of binaries, five
  invocations, the same four hashes every time. The verdict string those runs
  emit is `ISA_SUFFICIENT_TO_DIVERGE`; in the first three it said ONLY in place
  of SUFFICIENT, which claimed more than the data supports -- see below.
* The **separation of ISA from OS/libm** as the cause of run 30767491098's
  mismatch: **partly measured.** The ISA alone is shown to be *sufficient* to
  move the artifact. Whether OS/libm *also* contributes there, and how much, is
  **not measured** -- this page holds the OS fixed by construction and so cannot
  see that contribution at all.

* Cross-platform behaviour at 12000 steps, x86_64 Linux against aarch64 macOS:
  the two checkpoints differ (run 30767491098, reproduced in run 31004703001).
  The x86_64 arm has now been repeated -- three times in total across two runs,
  all `bb14ab18...` -- so the within-arm control that this bullet used to lack
  is present on that side. The aarch64 arm is still n=1 *in CI*; it is repeated
  locally, but not by a job a reader can re-run. That the difference is
  attributable to architecture *rather than to OS or libc* is still not
  established by these runs; this page narrows it by showing that an ISA change
  alone is sufficient to produce a divergence, which is not the same as showing
  it is the only cause operating there.
* A native x86_64 CPU and a second libc: **measured 2026-08-05** (GitHub-hosted
  Ubuntu runner, `ldd (Ubuntu GLIBC 2.39-0ubuntu8.7) 2.39`), at 0, 10 and 12000
  steps. A second compiler version and a third architecture: **not measured** on
  this page.

### The verdict string was too strong

`scripts/local_isa_probe.py` emitted a machine-readable label that said the ISA
was the ONLY thing that diverged. The prose beside it always said the right
thing -- "that does not make it the only cause operating here" -- but a JSON key
is what a machine reads, and that key said ONLY. This page's own data denies it.
Three step-10 hashes exist for the same declared inputs:

```
efef1cba...  aarch64  macOS               native
5913542e...  x86_64   macOS, Rosetta 2    translated
a32e9b2a...  x86_64   Linux, glibc 2.39   CI run 31004703001
```

The last two carry the **same declared architecture** and differ anyway. At
fixed `arch=x86_64`, a change of OS and libc moves the bytes too, so varying the
ISA is *sufficient* to diverge and is demonstrably not the only thing that is.
The constant is now `ISA_SUFFICIENT_TO_DIVERGE`, and the probe prints the three
hashes above in its own verdict block rather than leaving the qualification in a
document the reader may not have open.

The retracted spelling is described rather than quoted, here and in the script,
for one checkable reason: it was never persisted. `git log --all -S` finds it in
no commit -- the probe and its evidence are themselves still uncommitted -- so
no artifact anywhere carries it and no reader can hold one to search for. That
is the same test applied to the checkpoint record schema in `src/checkpoint.rs`,
and it is the only thing that makes dropping a string safe.

The same run also pushed the Rosetta disclosure down onto each arm's
`declared_platform` entry. The record carried it once at the top level, so a
consumer slicing out `declared_platform["isa-probe-x86_64"]` read
`arch: x86_64, os: macos` with nothing to say it was not native silicon.

The word "reproducible" is not used unqualified anywhere in this document on
purpose. A checkpoint hash is a measurement, and a measurement without the
conditions it was taken under is not a result. The conditions here are the seed,
the step count, the corpus hashes, the compiler version, the profile, the host,
and -- newly, and non-negotiably -- whether the code was executing natively or
under translation.
