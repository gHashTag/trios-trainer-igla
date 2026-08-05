# Decidability and traceability: the two arguments with no counter

**What this document is.** Two attacks on this project's reproducibility claim
that currently have no written answer, written down. The first is the one to
make in the room, because it is not a weakness: it is the reason the whole
approach is the right one. The second is a genuine gap, and it is stated first
in its own section and not defended.

Every number below is re-derived from files tracked in this repository. Nothing
is quoted from memory, from a chat log, or from an out-of-repo parser.

---

## (a) DECIDABILITY: identical metric, different artifact

### The measurement

`evidence/xarch-local-isa/` holds a local two-ISA probe run on one host on
2026-08-05: the same trainer source, built once for `aarch64-apple-darwin` and
once for `x86_64-apple-darwin`, each run for 10 steps at seed 47. The x86_64
arm executed under Rosetta 2 (`probe.json`, field `translation`), so the OS,
the libm vendor and the kernel are held fixed and the ISA is the variable.

| | A: `isa-probe-arm64-10` | B: `isa-probe-x86_64-10` |
|---|---|---|
| sidecar | `evidence/xarch-local-isa/isa-probe-arm64-10.json` | `evidence/xarch-local-isa/isa-probe-x86_64-10.json` |
| `platform.arch` | `aarch64` | `x86_64` |
| `platform.os` | `macos` | `macos` |
| `platform.libc_version` | `25.5.0` | `25.5.0` |
| checkpoint `sha256` | `efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac` | `5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab` |
| `bytes` | 852272 | 852272 |
| **`final_val_bpb`** | **6.257606029510498** | **6.257606029510498** |
| `min_observed_val_bpb` | 6.257606029510498 | 6.257606029510498 |
| `ema_bpb` | 6.716533660888672 | 6.716533660888672 |
| `val_bpb_stderr` | 0.01787959225475788 | 0.017879599705338478 |
| `git_sha` | `ba272b988971545801c9e337b0b42e35dc18abc4` | `ba272b988971545801c9e337b0b42e35dc18abc4` |
| `source_sha256` | `38f16194e88e4086f1b950016fe78cbdc9d9c646f00a96508757c5d237a92354` | `38f16194e88e4086f1b950016fe78cbdc9d9c646f00a96508757c5d237a92354` |
| `platform.rustflags_sha256` | `b18ac6a3a1eaa5f2126007da415a13b27913ec13f0412df94a9baaa7a8e8ac58` | `b18ac6a3a1eaa5f2126007da415a13b27913ec13f0412df94a9baaa7a8e8ac58` |
| `platform.features` | `ci-strict=0,gf16=0,gpu=0,race=0,smoke=0,trios-integration=0` | same |
| corpus train / val sha256 | `1a5aead1...` / `2088af36...` | identical |
| `trainer.sha256` | `709fcbf2a95342a8bbf70dab6ea3acdc157b8e771b908386ae5e9bb72db36845` | `1083862c3166dbafdc9e9db66b76c3ef78d9d7d89584902def8b77365a0833e4` |

Re-derive the premise:

```bash
python3 -c "import json; \
a=json.load(open('evidence/xarch-local-isa/isa-probe-arm64-10.json')); \
b=json.load(open('evidence/xarch-local-isa/isa-probe-x86_64-10.json')); \
assert a['final_val_bpb']==b['final_val_bpb']; \
assert a['sha256']!=b['sha256']; \
print(a['final_val_bpb'], a['sha256'][:8], b['sha256'][:8])"
```

Two further facts from the same directory, because they bound the reading:

* **At step 0 the two artifacts are byte-identical**, both
  `4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457`
  (`isa-probe-arm64-0.bin`, `isa-probe-x86_64-0.bin`; `probe.json` field
  `init.match: true`). Initialisation and RNG are not the cause. The
  divergence is created by the gradient steps.
* **`val_bpb_stderr` is NOT identical**: `0.01787959225475788` against
  `0.017879599705338478`. The arithmetic really did diverge inside the
  evaluation too. What is identical is the *point estimate at the precision it
  is published to*.

### The conclusion, in both directions

Both readings are true at once and neither may be dropped.

1. **The divergence is behaviourally inert at 10 steps.** Two artifacts that
   share no common hash produce the same reported quality metric to all 16
   published digits. Anyone who fears that a byte difference means a different
   model is answered by this table: on this fixture it does not.
2. **Therefore a conformity check by reported quality metric cannot detect an
   artifact difference.** This is the same fact read as an auditor rather than
   as an engineer. A regulator who requires "the second laboratory reproduces
   the declared metric" would pass these two runs as identical. They are not
   identical. Two different sets of 212,992 numbers were shipped under one
   number, and the check designed to notice could not.

### Why bit-identity is the only decidable invariant

A conformity criterion has to be **falsifiable by a stated procedure that two
parties can run and agree on**. Exactly one candidate here has that property.

* **Bit-identity is decidable.** `shasum -a 256` on two files. It terminates,
  it needs no parameters, and two auditors cannot disagree about its output.
  It admits no tolerance argument, no sampling plan and no expert judgement.
* **Behavioural equivalence is not, without a declared tolerance.** "The same
  model" means `|bpb_A - bpb_B| < t`. Every part of that needs declaring
  before the comparison: the value of `t`, the eval corpus and its hash, the
  sampling plan (`eval_chunks`, `eval_tokens`, `eval_seq` -- fields this
  project added to its own record precisely because they were missing), and
  the estimator's own sigma, without which `t` cannot be set at all. Choose
  `t` and you have chosen the verdict.

The gap between those two sentences is the entire technical content of the
problem. 243-FZ (26.07.2026) requires confirmation of conformity and supplies
no criteria; the implementing regulations are being written now. **The
undefined tolerance is the gap.** A scheme that writes "the metric must
match" has written nothing, because it has not said to what precision, on what
data, under what sampling plan. A scheme that writes "the hashes must match"
has written something decidable and, as `docs/CROSS-ARCH-DIVERGENCE.md`
measures, something that fails for two laboratories running the identical
procedure on different CPUs.

The defensible position is neither, and it is what this repository actually
does:

> Require bit-identity **within a declared, hashed environment**
> `(os, arch, libc, toolchain, flags)`. Require a **declared metric tolerance
> with a declared sampling plan** across environments. Publish the
> environment in the artifact so a reader can tell which of the two claims is
> being made.

The 10-step probe above is the cheapest possible demonstration that these are
two different claims and that the second cannot substitute for the first.

---

## (b) TRACEABILITY GAP: the instrument attests its own identity

Stated first, not defended.

Every sidecar this project writes records:

```json
"trainer": {
  "path": "target/release/trios-train",
  "sha256": "709fcbf2a95342a8bbf70dab6ea3acdc157b8e771b908386ae5e9bb72db36845",
  "provenance": "self-hashed"
}
```

`"self-hashed"` is accurate and it is the whole problem. The binary that
produced the artifact computed the hash of itself and wrote it into the record
it also produced. Nothing external corroborates any of it. Concretely, absent
from this project today:

* **no signature** over any artifact or sidecar,
* **no timestamping authority**, so no artifact can be proven to have existed
  before any particular moment,
* **no transparency log** or append-only public ledger of the records,
* **no third-party attestation** of the build,
* **no accredited laboratory** and no accreditation of the procedure.

SHA-256 gives **integrity**: it detects modification of bytes between two
points where both parties already trust the chain. It does not give
**traceability**: it does not tie a measurement to a declared reference
through an unbroken chain of documented comparisons. Traceability is a claim
about the *chain of custody of trust*, not about the strength of a hash
function. A self-hashed binary is a chain with one link, and both ends of it
are the same party.

The consequence is exact and should be said in those words: **all
reproducibility evidence in this repository is self-reported.** It is
internally consistent, it is unusually complete, and it is checkable by
anyone who trusts the artifacts to be what they claim. It has not been checked
by anyone who does not.

**The remedy, named and NOT yet done.** A third-party attestation over a
CI-built binary: the trainer built by a hosted runner from a named commit,
with the build provenance signed by the runner's identity (for example a
signed SLSA-style provenance statement or an equivalent attestation bound to
the workflow, not to a developer key), the resulting `trainer.sha256`
published in that attestation, and the sidecar recording that external
statement instead of, or alongside, its own self-hash. Until that exists,
`"self-hashed"` is the honest value and it is why this section is a gap and
not a feature.

---

## (c) The binary is not third-party bit-reproducible, even on the same architecture

A boundary, not a "not yet".

Section (a) treats the *checkpoint*. This section treats the *trainer binary*,
and the negative result there is independent of the ISA question entirely: two
laboratories on the **same** architecture, the same OS and the same pinned
compiler still do not produce the same executable bytes.

The concession is already in this repository, in the generated header of
`.cargo/config.toml`, which is the file that carries the
`--remap-path-prefix` flags that removed the *first* blocker (the builder's
home directory in `.rodata`):

```
# The flags rewrite these prefixes to /build, /cargo and /rustup before rustc
# embeds them, so the builder's home directory does not end up in .rodata.
# That removes the blocker; it is NOT by itself a third-party reproduction.
# Measured on macOS, two labs still differ in the 16-byte Mach-O LC_UUID and
# the code signature over it. See docs/REPRODUCIBILITY-GRADING.md.
```

Read that carefully. The **LC_UUID** is a 16-byte load command the linker
writes into every Mach-O image; the code signature is computed over the image
including it. So even with paths remapped, with the same source, the same
`Cargo.lock`, the same pinned `rustc 1.96.0 (ac68faa20 2026-05-25)` and the
same flags, two builds differ in at least those bytes, and `trainer.sha256`
differs with them.

What this costs, stated plainly:

* **`trainer.sha256` is not a portable identity.** It identifies a build on
  one host. Two honest laboratories will report different values for the same
  procedure, and a check that compares them will report a mismatch that means
  nothing. This is the same structural error as using `source_sha256` across
  laboratories (`docs/CROSS-ARCH-DIVERGENCE.md` 1.2), and it must not be a
  grading criterion for the same reason.
* **The chain from source to artifact has an unproven link on every platform,
  not only across platforms.** Cross-architecture checkpoint divergence is the
  headline result; this is quieter and it is present even in the easy case.
* **It is a property of the platform's object format and signing model**, not
  of this trainer. Naming it as a boundary is the correct treatment: a
  measurement is meaningless without stating the conditions under which it
  holds, and "the binary is bit-identical" is not a condition this project can
  offer to a second laboratory on macOS.

The claim that survives is the narrow one: **given a binary, the training is
bit-deterministic on that machine, and the binary and the platform are
declared and hashed in every artifact.** Getting from there to "given a
source tree, the binary is bit-determined" is outstanding work on macOS and
is not claimed here.

---

phi^2 + phi^-2 = 3 | TRINITY
