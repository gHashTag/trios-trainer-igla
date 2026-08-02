# Reproducibility grading: L0 to L3

"Reproducible" is not a property a project can assert about itself. It is a
grade, and a grade needs a scale, a test for each level, and a stated scope.
This document defines the scale this repository grades itself on, names the
concrete command that decides each level, and states plainly where the trainer
stops.

The scale exists because of a specific failure: 1 851 experiments were run and
`checkpoint::save` was a stub returning `Ok(())` that nothing called. Every one
of those runs would have been described as "reproducible" by its authors. Not
one of them produced an artifact. That is L0, and nothing in the logs said so.

---

## The four levels

| Level | Claim | What must exist | Deciding command |
|-------|-------|-----------------|------------------|
| **L0** | none | nothing on disk | (no test can pass) |
| **L1** | *an artifact exists and is named by its digest* | `{step}.bin` plus a sidecar whose `sha256` is taken over the bytes **re-read from disk** | `shasum -a 256 <ckpt>.bin` equals the sidecar `sha256` |
| **L2** | *the metric reproduces within a stated tolerance* | L1, plus a recorded metric and a declared tolerance | re-run, compare `final_val_bpb` against the record within the tolerance |
| **L3** | *the bytes reproduce on a declared platform triple* | L2, plus every recipe input in the record, plus a declared platform | `ckpt_replay --record <sidecar>.json` returns `VERIFIED` |

A level is only meaningful with its scope attached. `L3 on
macos/aarch64/undetermined-libc` is a real claim. Bare "L3" is not.

---

## L1 -- an artifact, hashed from disk

The digest must be taken over the bytes **read back from the final path**, not
over the buffer that was about to be written. Those differ whenever the write
was short, the disk was full, or the process died between `write` and `rename`.
`checkpoint::save` writes `{step}.bin.tmp.{pid}`, `fsync`s it, `rename`s it into
place, and only then re-reads and hashes the result.

```bash
# The artifact and the record agree:
shasum -a 256 checkpoints/igla-honest-provenance/12000.bin
python3 -c "import json;print(json.load(open('checkpoints/igla-honest-provenance/12000.json'))['sha256'])"
```

`ckpt_replay` performs exactly this comparison before it does any other work,
and reports `ARTIFACT ALTERED` (exit 1) when the file no longer matches. An
artifact that fails L1 cannot be graded at any higher level, because the thing
under test is not the thing that was recorded.

## L2 -- the metric reproduces within a stated tolerance

L2 is weaker than L3 and is the level most published results actually mean. It
requires a tolerance to be stated in advance; without one, "close enough" is
decided after seeing the number.

This repository has one measured example of an L2-but-not-L3 pair. The same
seed, corpus and source tree at 300 steps produced **identical** BPB
(3.416599988937378) on macOS and on Linux while producing **different**
checkpoint bytes; at 3 000 steps the same pair produced 2.9097988605499268 and
2.9212632179260254 -- a gap of 0.0115 bpb. So the honest L2 tolerance for this
trainer across platforms is not tighter than ~0.012 bpb at 3 000 steps, and the
gap grows with the number of steps rather than shrinking.

```bash
# L2 by hand: re-run and compare the metric, not the bytes.
./target/release/trios-train --seed 47 --steps 3000 --hidden 384 \
  --train-data data/tiny_shakespeare.txt --val-data data/tiny_shakespeare_val.txt
# then compare `final_val_bpb` in the new sidecar against the recorded one.
```

## L3 -- the bytes reproduce, on a declared platform

```bash
cargo build --release --bin ckpt_replay --bin trios-train
./target/release/ckpt_replay --record checkpoints/<run>/<step>.json
```

`ckpt_replay` re-executes the `trios-train` **binary** as a subprocess with a
cleared environment, feeding it only the parameters the sidecar records, and
compares SHA-256 over the produced bytes. It returns one of:

| Verdict | Exit | Meaning |
|---------|------|---------|
| `VERIFIED` | 0 | the record's own parameters reproduce its bytes **on this host** |
| `MISMATCH` | 1 | they do not; both hashes are printed |
| `INCOMPARABLE` | 2 | the record does not describe its own inputs; there is nothing to grade |
| `REFUSED` | 3 | the replay would exceed `--max-steps` |
| `ERROR` | 4 | the check itself could not run |

`INCOMPARABLE` is the level-0 of grading itself, and it is what the artifact
this repository led with until 2026-08-03 returns:

```console
$ ./target/release/ckpt_replay --record checkpoints/igla-honest-provenance/12000.json
INCOMPARABLE: eval_every not recorded; this artifact cannot be graded from its own provenance record
  record:  "checkpoints/igla-honest-provenance/12000.json" (schema trios-checkpoint-record/1)
  missing: eval_every, steps_total, gf16_floor_every, trainer.sha256
  note:    this is a verdict, not a failure. The run may have been perfectly reproducible; its record simply does not say so.
```

That artifact carries the 2.6169 bpb figure the README used to lead with. It is
a real measurement backed by a real file, the verdict is `INCOMPARABLE`, and the
verdict is correct -- but the reason this document used to give for it was not.

The decisive omissions are `steps_total` and `gf16_floor_every`. Both are
first-order inputs to the recipe:

- **`gf16_floor_every`** is how often `gf16_floor()` fires past the 70% mark,
  and `gf16_floor()` rewrites `embed`/`proj`/`lm_head`/`ctx` **in place**. A
  record that does not state it does not state what was computed. It is its own
  knob (`TRIOS_GF16_FLOOR_EVERY`, default `1`); `src/train_loop.rs` gates the
  mutation on it under a comment reading "NOT `args.eval_every`".
- **`steps_total`** is where that mark falls: `floor(0.7 * steps_total)`. Two
  records agreeing on `gf16_floor_every` and disagreeing on the step budget
  still do not describe the same recipe.

A `trios-checkpoint-record/1` record states neither. `eval_every` is missing
too, and the verifier lists it first, but of the three it is the one missing
harmlessly: it selects when the held-out corpus is read, and two runs differing
only in it produce byte-identical checkpoints. Schema 3 records all three.

So the headline artifact is L1, not L3, and no amount of re-running will raise
it: the record is the thing that is incomplete, not the run.

> **Correction, 2026-08-03.** An earlier revision of this section justified this
> same verdict by calling the eval cadence a recipe parameter and citing BPB
> 2.6141 vs 2.6169. That was true of the pre-fix trainer and is false of this
> one. Those two numbers survive only as the dated symptom of a defect that has
> been fixed. The `INCOMPARABLE` verdict never depended on them.

### The current headline, and what its own record still does not say

Re-measured for this document on 2026-08-03, ledger unreachable, from a
privately hashed copy of the binary
(`6c45b74b719d3b3e80e2785a490f36302ec8fb74a7b81a3942487603ae6a2348`):

| Field | Value |
|-------|-------|
| Sidecar | `checkpoints/r4-docs-repro/12000.json` (`trios-checkpoint-record/3`) |
| Checkpoint sha256 | `8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c` |
| `final_val_bpb` | `2.6347548961639404` |
| `steps_total` / `gf16_floor_every` / `eval_every` | `12000` / `1` / `1000` |
| Platform | `macos/aarch64`, `rustc 1.96.0 (ac68faa20 2026-05-25)` |

The number moved because the fix that gave the floor its own knob also changed
the default recipe: `GF16_FLOOR_EVERY_DEFAULT = 1`, so the floor now fires on
every step past the mark rather than once per eval. The pipeline no longer
produces ~2.61 at 12 000 steps, and **anyone needing bit-compatibility with a
pre-fix run must set `TRIOS_GF16_FLOOR_EVERY` to that run's `eval_every` -- a
value no schema/1 record states.** That is the whole content of `INCOMPARABLE`
made concrete, and it was tested rather than asserted. Four seed-47 runs
identical except for the floor cadence:

| `TRIOS_GF16_FLOOR_EVERY` | `final_val_bpb` | checkpoint sha256 | recovers an archived pre-fix run? |
|---|---|---|---|
| `1` (default) | `2.6347548961639404` | `8a86fe691aef64fc` | -- |
| `1000` | `2.614117383956909` | `b2fe52b0760dc369` | **yes, `igla-honest-20260802/12000.bin` byte for byte** |
| `3000` | `2.614041805267334` | `7a95c99b962d565f` | no |
| `4000` | `2.5910677909851074` | `3ecc1c925e42ff57` | no |

One of the two archived artifacts came back exactly once the trainer was told
the cadence -- and the byte match is what identifies that cadence as 1000, since
the record never stated it. The other -- `igla-honest-provenance/12000.bin`,
`c1b1a12b914924ac`, sidecar `bpb` 2.616914749145508, the artifact this
document's `INCOMPARABLE` example is about -- did not come back at any of the
four, because its record does not say which one to use and guessing is not a
method. That is the difference between `INCOMPARABLE` and `MISMATCH` shown
rather than defined.

The step-by-step divergence -- byte-identical checkpoints against two archived
pre-fix runs at steps 3 000, 4 000, 6 000 and 8 000, then divergence from
step 9 000, with the 70% mark at `floor(0.7 * 12000) = 8400` -- is tabulated in
the README's calibration section. It is the cleanest L3-shaped evidence this
repository currently holds: a recipe change that is located to a step rather
than asserted.

This record grades better than the schema/1 one and still does not grade
`VERIFIED`. Measured on the same day:

```console
$ ./target/release/ckpt_replay --record checkpoints/r4-docs-repro/12000.json
INCOMPARABLE: trainer.sha256 not recorded; this artifact cannot be graded from its own provenance record
  record:  "checkpoints/r4-docs-repro/12000.json" (schema trios-checkpoint-record/3)
  missing: trainer.sha256
```

Every recipe field is present; what is missing is the identity of the binary
that produced the bytes. The cause is stated rather than guessed at: the
`trios-train` used for the run above and the `ckpt_replay` used here were built
from the same working tree at different moments while it was being edited, and
`trainer.sha256` was added to the trainer's record between the two builds. So
this verdict is skew, not a defect of schema 3 -- a run made with a trainer
built at the same moment as the verifier would carry the field.

It is left standing as measured. The point of a scale is that this costs one
line of output instead of a judgement call: the verdict names the single field
that is missing, and the reader does not have to take anyone's word for what
the artifact does and does not prove.

---

## This project's own finding: L3 holds only inside a fixed platform triple

**The trainer is bit-deterministic on a fixed platform and is not
bit-reproducible across platforms.** This is a finding of this project, measured
here, not a caveat borrowed from the literature.

The evidence directory `/Users/playra/xarch-evidence/` holds five checkpoints
produced from the same seed (47), the same corpus and the same source tree on
three toolchain/OS combinations. The hashes below were computed for this
document with `shasum -a 256` against that directory:

| File in `/Users/playra/xarch-evidence/` | Steps | sha256 |
|------|-------|--------|
| `macos/mac/xarch-mac/300.bin` | 300 | `fe0e66401d38e9d7ee2e0cd9667e1e118a549671f0e808a5714ecf09d19ae124` |
| `linux-rustc191/lin191/300.bin` | 300 | `9866cb2a0b13b5b3e8a630e74f9a8c792dc02c4cb43e9880f317e33be831d115` |
| `linux-rustc196/lin196/300.bin` | 300 | `9866cb2a0b13b5b3e8a630e74f9a8c792dc02c4cb43e9880f317e33be831d115` |
| `macos/mac3k/mac3k/3000.bin` | 3 000 | `f95c07ab15e1134a3109c7cbec5fb17dcde602849d0a091cd2a44ce8632c5b6a` |
| `linux-rustc191/lin3k/3000.bin` | 3 000 | `22294bce82d997f81ce928dddf4b0a9681659789ea8715479e6dd5ef03ecbfaa` |

Two readings:

1. **rustc 1.91 and rustc 1.96 on Linux agree byte for byte** (identical hash at
   300 steps). The compiler version is not the decisive variable.
2. **macOS and Linux disagree at every step count measured.** The OS/libc pair
   is the decisive variable.

The same conclusion is reproducible on demand rather than only on file. Running
the verifier on the Linux-produced record from a macOS host:

```console
$ ./target/release/ckpt_replay --record /Users/playra/xarch-evidence/linux-rustc191/lin191/300.json
recorded sha:  9866cb2a0b13b5b3e8a630e74f9a8c792dc02c4cb43e9880f317e33be831d115
replay sha:    fe0e66401d38e9d7ee2e0cd9667e1e118a549671f0e808a5714ecf09d19ae124
host:          macos/aarch64
record says:   (no platform fields)
MISMATCH on macos/aarch64
platform triple was NOT declared by the record, so this MISMATCH does not distinguish a bad record from a different host
```

Note the last line. A schema-2 record declares no platform, so this `MISMATCH`
is uninterpretable on its own: it is consistent with a dishonest record and with
an honest record replayed on the wrong machine, and nothing in the record tells
them apart. Schema 3 adds `platform.{os,arch,pointer_width,libc,toolchain}`
precisely so this verdict becomes decidable.

Consequently, **the only defensible L3 claim this trainer supports is scoped**:

> Bit-reproducible for a fixed `(os, arch, libc, toolchain)` and a fixed set of
> recorded recipe parameters. NOT bit-reproducible across `macos` and `linux`.

A regulator asking for "reproducibility of the development cycle" is asking a
question with four possible answers, and only one of them is a number. Naming
the level and the platform is the whole of the method.

---

## What the verifier still cannot decide

Stated so no reader has to discover them by being wrong:

- **It grades one checkpoint, not a run.** A `VERIFIED` verdict on step 300 says
  nothing about step 12 000 beyond what determinism implies.
- **It does not re-derive the source tree.** Schema 3 records
  `source_sha256`; `ckpt_replay` prints it and does not recompute it. The
  binary it re-executes is hashed and printed, which is the closer question.
- **It cannot see a compiler.** The trainer is invoked as an already-built
  binary. A record that names a toolchain is trusted about it.
- **Schemas 1 and 2 omit `lr`, `attn_scale` and `attn_seq`.** They are replayed
  at the trainer's defaults. That cannot corrupt a `VERIFIED` verdict -- bit
  identity could not arise if the values had differed -- but it makes a
  `MISMATCH` on an old record ambiguous, and the verifier says so in its output.
- **The replay costs what the original run cost.** A checkpoint at step S of a
  T-step run needs the full T steps, so `--max-steps` (default 2 000) refuses
  rather than silently spending an hour.
