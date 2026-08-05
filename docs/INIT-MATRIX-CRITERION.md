# The initial coefficient matrix: the one artifact the statute names

Status: **measured, and connected to the statutory text for the first time in
this repository.** The initial coefficient matrix reproduces byte-for-byte
across a change of instruction set. The trained weights do not. Those are two
different claims about two different artifacts, and only one of them is named
in the law.

---

## 1. What the statute actually says

`[SECONDARY SOURCE 2026-08-06 -- the criterion below is quoted from press
reporting of the adopted text of 243-FZ (26.07.2026), corroborated across two
independent secondary sources. It has NOT been read against pravo.gov.ru or
ConsultantPlus. Do not quote it verbatim to a counterparty until the primary
text is read. Transliteration is Latin-only because this repository is
ASCII-only (L3 PURITY); the original is Russian.]`

The criterion, whole and not in fragment:

> polnaya tekhnicheskaya i tekhnologicheskaya vosproizvodimost tsikla
> razrabotki, vklyuchaya obuchenie i matritsu iskhodnykh koeffitsientov

In English, and this rendering is ours rather than an official translation:

> full technical and technological reproducibility of the development cycle,
> including training **and the matrix of initial coefficients**.

**The last five words are the finding.** No document in this repository
contained the phrase "matrix of initial coefficients" before this one, and no
actor in the surrounding landscape grades initialisation as a separate object
from training: not ISP RAS, not EQTY Lab, not Gensyn, not Gundersen and
Kjensmo, not the ACM badging scheme, not sigstore. Every one of them grades a
*process* or a *final artifact*. The statute names an intermediate artifact
that exists before the first optimizer step -- and that artifact is the one this
repository can already produce, hash and compare in seconds.

## 2. A correction to our own premise, made here rather than in a room

Two errors have been circulating in the framing of this work. Both are
corrected before the measurement, because a measurement attached to a wrong
claim is worse than no measurement.

**Error 1: "we supply the criteria for podtverzhdenie sootvetstviya."** We do
not, and we must never say that. `podtverzhdenie sootvetstviya` (confirmation
of conformity) is a **different criterion** from reproducibility. Its object is
conformity to Russian law and to traditional values, assessed in an order the
government is to establish. That is a legal and policy determination. Nothing
in this repository decides it, and a technical method that claimed to would be
correctly dismissed.

The sentence that is true, and which should replace it verbatim:

> **We supply the decision procedure for the reproducibility criterion, which
> has no procedure at all.**

That is narrower, it is defensible, and it is still the only criterion in the
adopted text with measurable technical content.

**Error 2: assuming an independent re-execution is mandated.** The reported
reading `[SECONDARY SOURCE 2026-08-06]` is that reproducibility is owed **by
the developer**. That cuts both ways and both cuts must be stated:

- **It helps on the cross-architecture boundary.** A developer demonstrating
  reproducibility of its own development cycle is demonstrating it on its own
  iron: same developer, same machines, same declared platform. That is exactly
  the regime in which this repository's L3 bit-identity holds, and it is the
  regime in which the cross-ISA `MISMATCH` recorded in
  [DIVERGENCE-LOCALIZATION.md](DIVERGENCE-LOCALIZATION.md) and
  [CROSS-ARCH-DIVERGENCE.md](CROSS-ARCH-DIVERGENCE.md) does not arise.
- **It hurts on market size.** If the duty falls on the developer, then on the
  face of the text no independent laboratory re-execution is required, and the
  paid third-party verification business that a conformity regime would
  otherwise create does not exist yet. The implementing regulations may add it.
  They have not been written. Sizing a market on the assumption that they will
  is a forecast, not a reading.

## 3. What was measured

Every hash below was re-derived with `shasum -a 256` against the files on disk
at the time of writing, not transcribed from another document.

```console
$ shasum -a 256 evidence/xarch-local-isa/*.bin
4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457  isa-probe-arm64-0.bin
efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac  isa-probe-arm64-10.bin
4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457  isa-probe-x86_64-0.bin
5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab  isa-probe-x86_64-10.bin
```

### 3.1 The initial coefficient matrix is byte-portable across the ISA boundary

| artifact | arch | sha256 | bytes |
|---|---|---|---|
| `isa-probe-arm64-0.bin` | aarch64 | `4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457` | 852272 |
| `isa-probe-x86_64-0.bin` | x86_64 | `4f854c82177fadc41de92f754884dd290f0bbc426916dc840f499bafa66d5457` | 852272 |

Equal. The seeded RNG and the weight-initialisation path carry across a change
of instruction set with no bytes to spare. There is nothing to fix in the
statute's named artifact, and this is decidable by one `shasum` invocation
rather than by a second training run.

The two step-0 sidecars also agree on the evaluated metric to all printed
digits (`final_val_bpb = 7.000166416168213` on both arms), which is what an
identical artifact evaluated by identical arithmetic on identical bytes ought
to give, and is reported here as a consistency check rather than as a second
result.

### 3.2 Ten optimizer steps are enough to separate the arms

| artifact | arch | sha256 |
|---|---|---|
| `isa-probe-arm64-10.bin` | aarch64 | `efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac` |
| `isa-probe-x86_64-10.bin` | x86_64 | `5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab` |

Different. Since step 0 matched and step 10 did not, the divergence is
introduced by the training arithmetic and not by initialisation or RNG. The
mechanism is analysed in [DIVERGENCE-MECHANISM.md](DIVERGENCE-MECHANISM.md) and
localised in [DIVERGENCE-LOCALIZATION.md](DIVERGENCE-LOCALIZATION.md); it is
not re-argued here.

The OS, the libm vendor, the kernel, the filesystem, the corpus hashes, the
compiler (`rustc 1.96.0 (ac68faa20 2026-05-25)`) and the source tree were all
held fixed across the two arms. Only the compilation target moved. The x86_64
arm is an `x86_64-apple-darwin` build executed under **Rosetta 2 binary
translation, not native x86_64 silicon**, which is recorded in the run's own
sidecar and is what makes the experiment single-variable.

### 3.3 The control that licenses reading this as an architecture effect

Two independent x86_64 runs, same host, same flags, agree with each other byte
for byte at step 0, at step 10 and at step 200. Re-derived:

```console
$ shasum -a 256 checkpoints/loc-x86-200-a/200.bin checkpoints/loc-x86-200-b/200.bin
94e31069977a53fbebd7536b2c8e2da10be656be228929ab98357f3a578edb40  loc-x86-200-a/200.bin
94e31069977a53fbebd7536b2c8e2da10be656be228929ab98357f3a578edb40  loc-x86-200-b/200.bin

$ shasum -a 256 checkpoints/loc-x86-a/10.bin checkpoints/loc-x86-b/10.bin
5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab  loc-x86-a/10.bin
5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab  loc-x86-b/10.bin
```

This excludes the worse of the two readings. Without it, the step-10 mismatch
could be explained by the trainer simply not being repeatable on x86_64 -- which
would make determinism itself platform-contingent and would sink the method
rather than bound it. It does not. Determinism is a property of this trainer on
**both** instruction sets.

### 3.4 Scope, which may not be dropped when this result is quoted

- **One ISA pair, on one host, with the OS held fixed.** aarch64 macOS against
  x86_64 macOS under Rosetta 2. It is not a statement about native x86_64
  silicon and not a statement about Linux.
- **The CI pair is a different experiment.** GitHub Actions run `30767491098`
  varies architecture **and** operating system (and with it libc and libm)
  simultaneously, so it cannot attribute its `MISMATCH` to either.
  [CROSS-ARCH-DIVERGENCE.md](CROSS-ARCH-DIVERGENCE.md) states that; this page
  does not overwrite it.
- **`evidence/xarch-local-isa/` is STAGED in the git index and is not at
  `HEAD`.** A stranger who clones this repository today cannot see any of the
  files quoted in section 3.1 or 3.2. Verified at the time of writing:

  ```console
  $ git cat-file -e HEAD:evidence/xarch-local-isa/isa-probe-arm64-0.bin
  fatal: path 'evidence/xarch-local-isa/isa-probe-arm64-0.bin' exists on disk,
  but not in 'HEAD'
  ```

  The `checkpoints/` artifacts quoted in section 3.3 are worse off still:
  `checkpoints/` is gitignored and ships nothing at all. Until these are
  committed, every hash on this page is a claim about the author's disk. That
  is stated here rather than discovered by an auditor.

## 4. The wedge, and the ceiling

**The wedge, in one sentence:** initial-matrix reproducibility is L3-strength
evidence -- SHA-256 equality on the artifact, no tolerance, no judgement call --
obtainable at L1 cost, decidable in seconds rather than in a second training
run, and it is the one artifact the statute names explicitly.

**The ceiling, in the next one, and volunteering it is the point:** L3 for a
distributed national foundation model is **not on offer**. Verification at L3
costs what training cost, because the verifier re-executes every step; the
honest ceiling at scale is **L2 with a declared tolerance and a declared
sampling plan**, which is exactly what this repository's own
[REPRODUCIBILITY-GRADING.md](REPRODUCIBILITY-GRADING.md) already says in
sections (e) and (iii) and qualifies further in L2.1, where a pair of artifacts
differing in 43.70% of their parameters passes L2.

The two sentences are the offer. A regime that adopts "matrix of initial
coefficients must reproduce bit-for-bit on a declared platform" gets a
criterion that is cheap, decidable, and already met by this trainer across an
instruction-set change. A regime that extends the same demand to the trained
weights of a foundation model is asking for something no one has shipped at
that scale, and it should hear that from us first.

---

## Related documents

- [DIVERGENCE-LOCALIZATION.md](DIVERGENCE-LOCALIZATION.md) -- where the
  divergence begins, and the full four-fact table this page draws on.
- [DIVERGENCE-MECHANISM.md](DIVERGENCE-MECHANISM.md) -- which floating-point
  primitives carry it.
- [CROSS-ARCH-DIVERGENCE.md](CROSS-ARCH-DIVERGENCE.md) -- the 12000-step CI
  pair that varies two things at once.
- [REPRODUCIBILITY-GRADING.md](REPRODUCIBILITY-GRADING.md) -- the L0-L3 scale,
  its prior art, and the scope of every verdict.
