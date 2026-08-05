# What mechanism produces the cross-architecture divergence?

Status: **named, on one ISA pair, under Rosetta 2.** The divergence is carried
by the platform maths library, not by reduction order. Twelve arithmetic
primitives of the training path were probed in isolation; five disagree across
the instruction sets, and of those five exactly three are elementary --
`expf`, `powf`, `cosf` -- while the other two (`softmax`, one full forward pass)
are composites that contain them. Every primitive built only from IEEE-754
operations, including two deliberately different summation orders over the same
4096-element dot product, is **byte-identical**.

**The one-sentence answer.** `docs/DIVERGENCE-LOCALIZATION.md` established that
the divergence is introduced by the training arithmetic and named two surviving
candidates without separating them; this page separates them, and the answer is
the maths library: suppressing auto-vectorisation entirely (290 vector
instructions down to 42 on aarch64, 287 down to 8 on x86_64) changes **not one
output byte on either arm**, while `exp`, `powf` and `cos` differ across the
arms by exactly **one unit in the last place** on 0.73%, 0.61% and 1.47% of
4096 inputs respectively.

## Scope, and what this does not replace

**The x86_64 arm on this page is `x86_64-apple-darwin` executing under Rosetta 2
on Apple Silicon. It is not the native x86_64 Linux that produced the CI
mismatch** `bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3`
against `8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c`
(GitHub Actions run 30767491098). This experiment isolates **the ISA variable
only**: one host, one OS, one kernel, one compiler build, one libm *vendor*, two
instruction sets. That is what makes it able to attribute anything at all -- the
CI job moves architecture, OS and libc at once and so can report *that* the
artifacts disagree without saying why -- and it is also why it **constrains**
the native Linux arm rather than replacing it. glibc's `expf` is not Apple's
`expf`; a native x86_64 part with AVX2 and FMA available is not the SSE-only
baseline LLVM targeted here. Read the claim table at the end of this document
before quoting any line of it.

A second scope limit specific to this page: under Rosetta the x86_64 binary
resolves `_expf` out of the **x86_64 slice** of the platform libm, which is
itself x86_64 code being translated. This experiment therefore cannot separate
"Apple's x86_64 `expf` implements a different algorithm" from "Rosetta's
translation of Apple's x86_64 `expf` rounds differently at one step". Both sit
**outside the trainer's own compiled arithmetic**, which is the distinction the
experiment was run to make, but neither is individually established here.

## The question

Round 6 (`docs/DIVERGENCE-LOCALIZATION.md`) proved WHERE the divergence begins:
initial weights are byte-identical across the ISA boundary (`0.bin` hashes to
`4f854c82...` on both), and the artifacts separate only after gradient steps
(`efef1cba...` against `5913542e...` at step 10). It closed the init/RNG
hypothesis and left two open, explicitly unseparated:

* **reduction order** -- both backends auto-vectorised, and differently (326
  `fadd.4s` / 695 `fmul.4s` on aarch64 against 282 `addps` / 392 `mulps` on
  x86_64); floating-point addition is not associative, so two schedules over one
  sum are entitled to two answers;
* **libm** -- both binaries import `_expf`, `_log`, `_logf`, `_pow` and resolve
  them against different architecture slices; none of the four is bit-specified
  by IEEE 754.

FMA contraction was already excluded twice over: `grep -c mul_add
src/train_loop.rs` is `0`, Rust does not contract by default, and neither
release binary contains a single `fmadd`/`vfmadd` instruction.

Which of the two remaining candidates it is decides the price of the fix. If it
is reduction order, the fix reaches into every accumulation in the trainer and
costs throughput. If it is libm, the fix is three functions -- `expf` is reached
once per logit from `softmax`, `powf` once per RoPE frequency, `cosf` once per
step from `cosine_lr` -- and costs almost nothing.

A 852 272-byte checkpoint cannot answer this, because by the time it exists
every primitive has fed every other one. Running each primitive **alone** can.

## The instrument

`src/bin/xarch_probe.rs`, new with this document. It fills fixed f32 buffers
from an inline LCG (the two constants of `HybridModel::new`, restated in the
binary rather than imported, so that editing the trainer cannot silently
redefine the instrument used to audit it), runs each primitive in isolation, and
prints the SHA-256 of the raw little-endian f32 output bytes:

```text
PRIMITIVE <name> <sha256>
DETAIL    <name> n=<count> first_bits=0x<u32 bit pattern of output[0]>
```

It performs **no file I/O, no environment reads and uses no randomness**, so
nothing about the host can reach the numbers except the instruction set and the
maths library. It depends on no module of this crate.

The primitive set mirrors the real training path rather than a textbook one:

| # | primitive | what it isolates |
|---|---|---|
| 0 | `input_a`, `input_b`, `input_positive`, `input_ranged` | the buffers themselves, before any arithmetic -- if these disagree nothing below is interpretable |
| 1 | `dot_sequential_4096` | plain scalar accumulation; only `+` and `*`, both IEEE-exact |
| 2 | `dot_split8_4096` | the same dot with 8 partial sums -- the shape a vectoriser produces on its own |
| 3 | `exp_4096` | `f32::exp`, one call per element, no reduction |
| 4 | `ln_4096` | `f32::ln`, ditto |
| 5 | `sqrt_4096` | `f32::sqrt`, ditto |
| 6 | `sqrelu_4096` | **the activation the training path actually uses**: `if x > 0.0 { x * x } else { 0.0 }`. The trainer has no `tanh` anywhere, so this is what item (6) of the brief covers. Pure multiply and compare: the control |
| 7 | `softmax_4096` | `src/train_loop.rs::softmax` structurally: max fold, in-place `exp`, sequential sum, divide -- libm composed with a reduction |
| 8 | `forward_tiny_h64` | one full forward pass on fixed weights at `DIM = 64`, `VOCAB = 128`, `NUM_CTX = 6`: embed + weighted context mix + `layer_norm` + `proj` matvec + squared ReLU + `lm_head` matvec + `softmax` + the `.ln()` the loss is read through |
| + | `powf_4096`, `cos_4096` | the two other libm functions on the training path: `10_000.0_f32.powf(exp)` in RoPE (`src/model_hybrid_attn.rs`) and `.cos()` in `cosine_lr` |
| + | `layer_norm_64` | `src/train_loop.rs::layer_norm` at its real width: two reductions plus one `sqrt` |
| + | `matvec_384x64` | the `proj` matvec at its real shape, 384 rows of 64 |

Five unit tests ship with it, of which one is load-bearing:
`split_accumulator_changes_the_sum` **fails** if the sequential and 8-way-split
dot ever agree exactly, because at that moment the reduction-order probe would
be measuring nothing and this document's reading of it would be void. On this
host they differ (`0x447d1b06` against `0x447d1b0f`), so reduction order
demonstrably moves this number on this hardware. That is the precondition for it
being *able* to move the number between architectures -- and it did not.

## The commands

Measured 2026-08-03 on one host: Apple M1 Pro, macOS 26.5.2 (build 25F84),
`Darwin 25.5.0 arm64`, rustc `1.96.0 (ac68faa20 2026-05-25)` pinned by
`rust-toolchain.toml`, release profile, Rosetta 2 present
(`/usr/libexec/rosetta`: `oahd`, `runtime`, `translate_tool`).
`git_sha = 3c1f751cf4376c13d26e247c2cd86357ab51dd20`, tree dirty (see "Limits on
the local reference" below).

```bash
cargo build --release --bin xarch_probe
cargo build --release --target x86_64-apple-darwin --bin xarch_probe

./target/release/xarch_probe                     | tee /tmp/probe_arm64.txt
./target/x86_64-apple-darwin/release/xarch_probe  | tee /tmp/probe_x86.txt
diff /tmp/probe_arm64.txt /tmp/probe_x86.txt
```

The x86_64 binary is a Mach-O x86_64 executable run on Apple Silicon, so macOS
routes it through Rosetta 2; no `arch -x86_64` prefix is needed, and `file`
confirms which slice was built.

| target | sha256 of `xarch_probe` | `file` says |
|---|---|---|
| `aarch64-apple-darwin` | `c1bc686e5ae26f61b8ee1e76df6e34881a9604aa861a1e04bbfcacbc81329983` | Mach-O 64-bit executable arm64 |
| `x86_64-apple-darwin` | `9ca60975759a8c71326292f73a5f8e5750fdef52d293eff7309b32da90e189d5` | Mach-O 64-bit executable x86_64 |

The two binary hashes differ, which is expected and carries no information:
different machine code for a different instruction set cannot hash the same.

The vectorisation arm was built with the flag applied to **this binary only**,
using `cargo rustc` rather than `RUSTFLAGS`, so that the dependency graph was
not rebuilt with different codegen underneath a shared `target/` directory (all
the arithmetic under test lives in this binary, so the narrower scope loses
nothing):

```bash
cargo rustc --release --target x86_64-apple-darwin --bin xarch_probe \
    -- -C llvm-args=-force-vector-width=1
cargo rustc --release --bin xarch_probe \
    -- -C llvm-args=-force-vector-width=1
```

`-C llvm-args=-force-vector-width=1` was **accepted** by the pinned rustc; the
`-C opt-level=1` fallback was not needed. That it took effect was verified in
the disassembly rather than assumed:

| binary | vector float instructions |
|---|---|
| `xarch_probe` aarch64, default | 290 (`fadd.4s` + `fmul.4s`) |
| `xarch_probe` aarch64, `-force-vector-width=1` | **42** |
| `xarch_probe` x86_64, default | 287 (`addps` + `mulps` + `subps`) |
| `xarch_probe` x86_64, `-force-vector-width=1` | **8** |

Neither probe binary contains an FMA instruction (`fmadd`/`fmla` count 0 on
aarch64, `vfm*` count 0 on x86_64), matching the trainer.

## The result

Full per-primitive table, both arms, default codegen. Rows are marked `SAME`
where the two hashes are byte-identical and `DIFF` where they are not.

| primitive | aarch64-apple-darwin | x86_64-apple-darwin (Rosetta 2) | |
|---|---|---|---|
| `input_a` | `ef0dc7ce292aab8f99099908ef1dfb306ac7ed2f038b1ff332b15aef7188dfbd` | `ef0dc7ce292aab8f99099908ef1dfb306ac7ed2f038b1ff332b15aef7188dfbd` | SAME |
| `input_b` | `9af976a4697b5cb7aba3996fff1d5ea2a889873a10fbc5d59d157c436a427a8e` | `9af976a4697b5cb7aba3996fff1d5ea2a889873a10fbc5d59d157c436a427a8e` | SAME |
| `input_positive` | `fd506d52f2f96f2025240f4bf7808c8f5a4ce29783f779421dd3189434e8d57e` | `fd506d52f2f96f2025240f4bf7808c8f5a4ce29783f779421dd3189434e8d57e` | SAME |
| `input_ranged` | `dff27d4d8b7be4e80065bf6328d5c67cd0d57a01f0231e50ebc0a1bdc830bed0` | `dff27d4d8b7be4e80065bf6328d5c67cd0d57a01f0231e50ebc0a1bdc830bed0` | SAME |
| `dot_sequential_4096` | `b8194eadf31dac2763e4945d0a2faeb3436b6f46a048c7a7a301022f7420efbc` | `b8194eadf31dac2763e4945d0a2faeb3436b6f46a048c7a7a301022f7420efbc` | SAME |
| `dot_split8_4096` | `559837cedd3c8749e66a8d7f7d0bf927c76be6dec2a3c195fdb33577ce695ec3` | `559837cedd3c8749e66a8d7f7d0bf927c76be6dec2a3c195fdb33577ce695ec3` | SAME |
| `exp_4096` | `da21b7d2ae9578bbe6ebe089b65d9e78977aebc0b5e8655c939dfeb501374fc2` | `d62949525417a6a0542f84b1109aca6a151ea29bfab80020e85dd1a7bdcbcb50` | **DIFF** |
| `ln_4096` | `92d8570e4e8546c7244ab648bbc22ea1ea6acc69710841056c709ab8dcaa8de6` | `92d8570e4e8546c7244ab648bbc22ea1ea6acc69710841056c709ab8dcaa8de6` | SAME |
| `sqrt_4096` | `8673c1472b4452b4b8eca9917b8934d203faa67ea099c4d56c90bb4ea9daee09` | `8673c1472b4452b4b8eca9917b8934d203faa67ea099c4d56c90bb4ea9daee09` | SAME |
| `powf_4096` | `9c9938ff73747ca30fce88ecd2df01d907d1d5772bce1e752d324def573fcf94` | `7d2814e5136759896bcb2201d888f0ef123c13b027ceaffa5f2a772d10d9993e` | **DIFF** |
| `cos_4096` | `10b3c47cd1f01c9e8de03b73ae25bb1a5de766c6bee8c2e2c620fa28eef41132` | `88a24781d5e99089822d761430efa3f81d215de64932b3dae60edd2afcedcdc9` | **DIFF** |
| `sqrelu_4096` | `4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe` | `4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe` | SAME |
| `softmax_4096` | `27bfe3723da351408ed3e31760ea32f0dac04fe6d900fc02b9033a5061ffcc0a` | `f2eb0c3c764a457c14b7143138bbc2da7410d44f6bd6d25ff5a655e68efc7949` | **DIFF** |
| `layer_norm_64` | `d0a67af559306e42683c724f73782112f0a4f85a0c678f3038f9aca77004b9cc` | `d0a67af559306e42683c724f73782112f0a4f85a0c678f3038f9aca77004b9cc` | SAME |
| `matvec_384x64` | `f2c23c3efc235242150c74db2c0bc90ab5968139951fc03b76ffdea041f4597c` | `f2c23c3efc235242150c74db2c0bc90ab5968139951fc03b76ffdea041f4597c` | SAME |
| `forward_tiny_h64` | `9eeb6b05fb70205a92da5b43e183eb9c09de45fd7553ccae8edadd26f465aaa3` | `daa30f5753e02ed64825e0fba0a0019aa1048e30913c61111fa3d43f85b27d99` | **DIFF** |

Five of sixteen differ. The `-force-vector-width=1` arm produced the **same
sixteen hashes on both architectures** -- every one of them, byte for byte,
identical to the default-codegen run on its own arm -- so the table above is
also the table for the de-vectorised build, and the DIFF/SAME column is
unchanged.

### What agreed

* **Both dot products.** 4096 sequential multiply-accumulates and the same sum
  reassociated into 8 partial accumulators. Both agree across the ISA boundary,
  *and they disagree with each other* on each arm (`0x447d1b06` against
  `0x447d1b0f`) -- which proves the probe is sensitive to reduction order and
  that reduction order simply did not vary between the two backends here.
* **`matvec_384x64` and `layer_norm_64`**, the two reductions at the trainer's
  real shapes, including `layer_norm`'s `powi(2)` accumulation and its `sqrt`.
* **`sqrt_4096`.** `nm -u` shows `sqrtf` is *not* imported by either binary:
  LLVM lowers it to the hardware square-root instruction, which IEEE 754 pins
  exactly. This row is a correctness check on the whole method -- an IEEE-pinned
  operation that came out equal, as it must.
* **`sqrelu_4096`**, the trainer's activation. Pure multiply and compare.
* **`ln_4096`.** `_logf` *is* dynamically imported by both binaries and still
  agreed on all 4096 inputs. That is an **observation, not a guarantee**:
  nothing requires two `logf` implementations to agree, and a different input
  distribution could separate them.
* **All four input buffers**, confirming that the two arms measured the same
  numbers and that the integer-to-float conversion in the generator is exact.

### What differed, and by how much

Only libm. To quantify it, the same inputs were dumped element-wise by a
throwaway program built with the same pinned `rustc` for both targets (it lives
in `/tmp`, is not part of this repository, and is reproduced in full at the
bottom of this file so the numbers below can be re-derived):

| function | elements differing | of | share | max difference |
|---|---:|---:|---:|---|
| `exp` | 30 | 4096 | 0.732% | **1 ULP** |
| `powf` | 25 | 4096 | 0.610% | **1 ULP** |
| `cos` | 60 | 4096 | 1.465% | **1 ULP** |
| `ln` | 0 | 4096 | 0.000% | -- |
| `sqrt` | 0 | 4096 | 0.000% | -- |

Every single disagreement is exactly one unit in the last place. The first for
each: `exp` at index 279, `0x39cb81f8` against `0x39cb81f9`; `powf` at index
128, `0x4058d9b2` against `0x4058d9b1`; `cos` at index 35, `0x3e57a105` against
`0x3e57a104`. These are last-place rounding decisions, not different numbers --
the same character of difference `docs/DIVERGENCE-LOCALIZATION.md` measured in
the step-10 weights (max absolute difference `9.17e-07` on a value of
`1.2e-02`).

Both binaries import exactly four maths symbols, and the three that differ are
three of those four:

```console
$ nm -u target/release/xarch_probe | grep -E '_(expf|logf|cosf|powf|sqrtf)$'
_cosf
_expf
_logf
_powf
$ nm -u target/x86_64-apple-darwin/release/xarch_probe | grep -E '_(expf|logf|cosf|powf|sqrtf)$'
_cosf
_expf
_logf
_powf
```

### The composites confirm the reading rather than adding to it

`softmax_4096` differs and contains `exp`. `forward_tiny_h64` differs and
contains `softmax`. `layer_norm_64` and `matvec_384x64` agree and contain no
libm call that is not IEEE-pinned. **Every composite primitive that disagrees
contains a disagreeing libm call, and every composite that agrees contains
none.** No composite disagrees for a reason its parts do not already supply.

## The conclusion the data supports

1. **Reduction order is not the mechanism on this ISA pair.** This is asserted
   on two independent grounds: two deliberately different summation orders over
   the same data are byte-identical across the arms, and forcing the vector
   width to 1 -- which removed 85% of the aarch64 and 97% of the x86_64 vector
   float instructions -- changed no output byte on either arm. LLVM does not
   reassociate floating-point sums without fast-math, and this measures that it
   did not.
2. **The mechanism is the platform maths library.** `expf`, `powf` and `cosf`
   resolve to different architecture slices that round differently, by exactly
   1 ULP, on roughly 1% of inputs. `softmax` calls `f32::exp` on every logit of
   every forward pass, so this reaches the numerical core of the training loop
   on every single step.
3. **FMA contraction remains excluded**, now on a third piece of evidence: the
   probe binaries contain no FMA instruction either.
4. **The magnitude is consistent with the observed checkpoint divergence.** A
   1-ULP difference at step 1, amplified by 10 optimizer steps, is exactly the
   size of perturbation `docs/DIVERGENCE-LOCALIZATION.md` measured (relative L2
   `3.72e-08` at step 10). Consistency is not causation, and this document does
   not claim to have traced a specific ULP into a specific weight.

### What the data does not support

* **It does not explain the CI mismatch.** That run is native x86_64 Linux with
  glibc 2.39 against aarch64 macOS. The mechanism named here is *sufficient* to
  produce a divergence and is *present* on that pair too -- glibc's `expf` is
  not Apple's `expf` -- but the native pair also varies OS, libc and available
  CPU features simultaneously, and a native x86_64 build with FMA and AVX2
  enabled could reintroduce candidates this pair excluded. Nothing here retires
  the `localize-divergence` job of `.github/workflows/cross-arch-repro.yml`.
* **It does not prove that removing libm would make the trainer
  cross-architecture bit-identical.** It proves that libm is the only mechanism
  operating on *this* pair, over *these* inputs, in *these* primitives. The
  backward pass, the attention module's own code, and the optimizer were probed
  only through their primitives, not as compiled units.
* **It does not separate "Apple's x86_64 libm differs" from "Rosetta's
  translation of it differs."** See the scope section.
* **`ln` agreeing is not a guarantee about `logf`**, only a measurement on 4096
  inputs in `[0.25, 8.0)`.

### What it costs to close

This is the practical payoff of naming the mechanism, and it is much smaller
than the alternative would have been. Constraining reduction order means
touching every accumulation in the trainer and paying for it in throughput.
Constraining libm means replacing **three** functions with implementations
compiled into the binary rather than resolved from the host:

| function | call site | frequency |
|---|---|---|
| `f32::exp` | `softmax` in `src/train_loop.rs`, and the attention softmax in `src/model_hybrid_attn.rs` | once per logit per forward pass -- the hot one |
| `f32::powf` | RoPE frequency table, `src/model_hybrid_attn.rs` | once per (position, dimension) pair, cacheable |
| `f32::cos` | `cosine_lr`, `src/train_loop.rs` | once per step -- negligible |

`f32::ln` and `f32::sqrt` need no change on the evidence here (`sqrt` because it
is IEEE-pinned and compiles to hardware; `ln` because it was measured equal,
which is weaker and should be re-measured if it is ever relied on).

Doing this is **not done** and is not proposed as done. What this document
establishes is that the work is bounded and that `xarch_probe` is the regression
test that would prove it landed: after the change, `exp_4096`, `powf_4096`,
`cos_4096`, `softmax_4096` and `forward_tiny_h64` must join the other eleven
rows in the SAME column.

**This matters for how the boundary is described.** Cross-architecture bit
identity at fp32 is an engineering gap with published fixes -- see the prior-art
paragraph in `docs/REPRODUCIBILITY-GRADING.md` on Srivastava, Arora and Boneh
(arXiv:2403.09603) and on RepDL (arXiv:2510.09180) -- not a law of nature. The
honest position is that this trainer **declares** the boundary rather than
eliminating it, and now knows what eliminating it would cost.

## Limits on the local reference itself

`git_sha = 3c1f751cf4376c13d26e247c2cd86357ab51dd20`, `git_dirty = true`; other
work was landing in the tree while these runs were taken. Three things bound
that risk here, and the exposure is far smaller than in
`docs/DIVERGENCE-LOCALIZATION.md`:

1. **The probe depends on no crate module.** It imports `sha2` for hashing and
   nothing else; its LCG, its `layer_norm`, its `softmax` and its forward pass
   are written out inside the file. Concurrent edits to `src/train_loop.rs`
   cannot reach these numbers.
2. **Both binaries were re-built from the same tree state back to back**, and
   the aarch64 output was re-produced after the de-vectorised arm had
   overwritten and then restored the binary; the sixteen hashes were identical.
3. **`src/bin/tjepa_train.rs` did not compile at the time of these runs**, due
   to another item's in-flight change to `compute_grads` in `src/train_loop.rs`.
   That blocks `cargo test --test ledger_exit_code_binaries` (which builds all
   bin targets) but not `cargo build --bin xarch_probe`, `cargo test --bin
   xarch_probe`, or anything on this page. Recorded so the reader is not
   surprised by it and does not attribute it here.

## Scope of every claim on this page

| claim | rests on |
|---|---|
| Reduction order does not vary between the aarch64 and x86_64 backends for these primitives | this local Rosetta experiment only, default and `-force-vector-width=1` |
| `expf`, `powf`, `cosf` differ by 1 ULP between the two Apple libm slices | this local Rosetta experiment only, 4096 inputs each |
| `logf` and `sqrtf` agree between the two slices | this local Rosetta experiment only, 4096 inputs each; `sqrtf` additionally by IEEE 754 and by not being imported |
| FMA contraction is not present | disassembly of four binaries plus `grep -c mul_add src/train_loop.rs` = 0 |
| Initialisation is byte-portable; divergence starts after gradient steps | `docs/DIVERGENCE-LOCALIZATION.md` |
| A checkpoint produced on x86_64 **Linux** differs from the aarch64 macOS one | CI run 30767491098 only |
| The libm mechanism is what produced *that* mismatch | **NOT ESTABLISHED.** Sufficient and present, not shown to be operating alone |
| Cross-architecture bit identity is achievable at fp32 by constraining arithmetic | published prior art, read by abstract only -- see `docs/REPRODUCIBILITY-GRADING.md` |
| A native x86_64 CPU, a second libc, a second compiler version, a third architecture | **not measured** |

The word "reproducible" is not used unqualified anywhere on this page, for the
same reason it is not used unqualified in `docs/DIVERGENCE-LOCALIZATION.md`: a
hash is a measurement, and a measurement without its conditions is not a result.

## Appendix: the element-wise dump program

Reproduced in full because the ULP table above is the only number on this page
that `xarch_probe` does not itself print. Build it with the repository's pinned
toolchain -- run `rustc` from inside the checkout so `rust-toolchain.toml`
applies, otherwise a different default toolchain may not have the x86_64 target
installed:

```bash
rustc -O --target aarch64-apple-darwin -o /tmp/expdump_arm /tmp/expdump.rs
rustc -O --target x86_64-apple-darwin  -o /tmp/expdump_x86 /tmp/expdump.rs
/tmp/expdump_arm > /tmp/dump_arm.txt
/tmp/expdump_x86 > /tmp/dump_x86.txt
```

```rust
struct Lcg { state: u64 }
impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005)
                               .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        let s = self.next_u64();
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.next_f32()).collect()
}
fn fill_range(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mid = (lo + hi) * 0.5; let half = (hi - lo) * 0.5;
    fill(n, seed).into_iter().map(|v| mid + v * half).collect()
}
fn main() {
    let pos = fill_range(4096, 123, 0.25, 8.0);
    let ranged = fill_range(4096, 144, -8.0, 8.0);
    for (i, v) in ranged.iter().enumerate() { println!("exp {} {:08x}", i, v.exp().to_bits()); }
    for (i, v) in pos.iter().enumerate() { println!("ln {} {:08x}", i, v.ln().to_bits()); }
    for (i, v) in pos.iter().enumerate() { println!("sqrt {} {:08x}", i, v.sqrt().to_bits()); }
    for (i, v) in pos.iter().enumerate() { println!("powf {} {:08x}", i, 10_000.0f32.powf(*v / 16.0).to_bits()); }
    for (i, v) in ranged.iter().enumerate() { println!("cos {} {:08x}", i, v.cos().to_bits()); }
}
```

The two dumps were compared line by line, counting differing bit patterns per
function and taking the maximum absolute difference of the u32 bit patterns as
the ULP figure (valid here because every value involved is finite, positive-
exponent and of the same sign on both arms).
