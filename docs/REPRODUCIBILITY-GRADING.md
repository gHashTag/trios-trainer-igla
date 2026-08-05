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
| **L3** | *the bytes reproduce on a declared platform triple* | L2, plus every recipe input in the record, plus a declared platform | `ckpt_replay --record <sidecar>.json --max-steps <steps_total>` returns `VERIFIED` |

`--max-steps` defaults to 2 000 and is a refusal, not a limit: any record whose
`steps_total` exceeds it exits 3 `REFUSED` before doing any work, rather than
silently spending the cost of a full training run. Every long-run record in
this repository must be graded with it raised explicitly.

A level is only meaningful with its scope attached. `L3 on
macos/aarch64/undetermined-libc` is a real claim. Bare "L3" is not. Read
[Scope of the verdict](#scope-of-the-verdict) before quoting any level from
this document: in ISO 5725 / VIM terms what is demonstrated below is
*repeatability*, and the one cross-laboratory test that has been run came back
`MISMATCH`.

The scale also has exactly one axis. Every level above grades *re-derivability*
and none of them grades *validity* -- whether the procedure being re-derived
measures anything. A deterministic defect reproduces perfectly and is therefore
awarded the top grade. Read
[(v) The scale has one axis](#v-the-scale-has-one-axis-validity-is-the-other-and-l3-is-blind-to-it)
before treating `VERIFIED` as an endorsement of a method.

The two upper rungs also answer two different questions, and only one of them
is about the identity of the artifact. L3 is an integrity claim; L2 is a
portability claim, and a pair of artifacts differing in 43.70% of their
parameters passes it. Read
[L2.1 -- What metric-equivalence does not attest](#l21----what-metric-equivalence-does-not-attest)
before quoting an L2 pass as evidence about what a model contains.

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

## L2.1 -- What metric-equivalence does not attest

L2 is the rung a workable conformity scheme is most likely to land on, because
L3 costs a second full training run
([(e)](#e-verification-costs-what-training-cost)). That makes it the rung whose
scope has to be stated most carefully, and this repository's own measured L2
pass is the sharpest illustration available of what the rung buys and what it
does not.

The pass in question is the cross-architecture pair in
[The headline artifact, re-run on a second machine](#the-headline-artifact-re-run-on-a-second-machine-mismatch):
identical corpus, identical compiler, identical locked dependency graph, and
**93 071 of the 212 992 serialized parameters different** -- 43.70% of them,
every element of all four trained attention matrices among them, the largest
single disagreement a change of sign, read out of the record
`docs/cross-arch-divergence.json` rather than transcribed
(`"params": 212992`, `"params_differing": 93071`) -- while `final_val_bpb` moved by
`0.0030086` bpb (`2.637763500213623` against `2.6347548961639404`), which is
5.5% of that record's own `val_bpb_stderr` of `0.0551`. Under the unpaired
acceptance rule stated below, that pair **CONFORMS**. Four things follow. The
rung survives all four; what changes is what may be said on the strength of it.

**1. Bit-identity inside a declared and hashed platform triple is an INTEGRITY
claim.** SHA-256 equality on the artifact detects *any* change to the bytes --
a rounding difference, a truncated write, a swapped optimizer, and equally an
implanted one. It does not distinguish between them, and it does not need to:
it fails on all of them. That is the whole value of L3, and it is why L3 is
worth keeping even though it does not survive a change of CPU.

**2. Metric-equivalence within a stated tolerance is a PORTABILITY claim, and
explicitly not an integrity claim.** The pair above is the proof, measured in
this repository rather than argued: two artifacts differing in 43.70% of their
parameters agreed on aggregate bpb to 5.5% of one arm's own standard error. The
same agreement is exactly what a backdoored artifact produces. An adversary who
modifies weights while holding aggregate perplexity fixed is not defeating the
L2 test -- L2 does not test for that, and cannot be made to by tightening the
tolerance, because the honest cross-architecture pair sits well inside any
tolerance a backdoor would also sit inside. **No L2 pass, at any tolerance, is
evidence that two artifacts are the same artifact.**

**2a. The second budget makes point 2 worse, not better `[ADDED 2026-08-06]`.**
The smallest defensible L2 tolerance is bounded below by the expanded
uncertainty of the comparison, and
[EVAL-UNCERTAINTY.md](EVAL-UNCERTAINTY.md#3b-the-budget-proper) section 3b now
publishes **two** of those, not one: **Budget I**, `u_c = 0.0657`,
`U = 0.13 bpb` (`k = 2`), scope *same hashed eval corpus*; and **Budget II**,
`u_c = 0.2883`, `U = 0.58 bpb` (`k = 2`), scope *different eval corpora, or a
BPB read as a property of the model*. The honest cross-architecture difference
measured above, `0.0030086 bpb`, is **2.3% of `U` under Budget I and 0.52%
under Budget II**. Read the two together and point 2 hardens: the room an
adversary has to work in is not a hair's breadth of instrument noise but a band
between `0.13` and `0.58 bpb` wide, depending only on whether the scheme pinned
the corpus hash -- and `0.58` is **4.39x** `0.13`. A conformity scheme that
forgets to demand the eval corpus hash does not merely lose precision; it
quadruples the interval inside which a modified artifact reports as conforming.
Budget II is therefore an argument for clause L2-1, not against L2.

**3. Aggregate bpb over a 100 KB corpus is blind to behaviour on any chosen
input.** `final_val_bpb` is one scalar averaged over 40 windows of 129 tokens
-- 5.16% of a 100 000-byte held-out corpus. A behaviour that fires on inputs
outside those windows, or on inputs inside them but at a magnitude the average
absorbs, moves the scalar by less than the instrument can resolve. So **an L2
pass licenses a statement about a number, and no statement about what the model
does.** For a statute whose motivation is "this artifact is what its developer
says it is and contains nothing implanted", this is the load-bearing sentence
on this page: the L2 rung is worth least exactly where that motivation is
strongest, and the answer is not a better tolerance but a different test --
behavioural evaluation on adversarially chosen inputs, which nothing in this
repository performs and nothing in this scale grades.

**4. n = 1 per arm. The agreement is not established, and the experiment that
would establish it has not been run.** Both arms of the pair above were trained
exactly once at 12 000 steps. The `0.0030086` is one observation of a
difference whose between-arm variance is **unestimated**, and one observation
per arm cannot separate a between-architecture effect from within-arm
non-repeatability on the arm that was never repeated. The control that would
estimate it is a within-`ubuntu-latest` repeat of the documented seed, compared
against itself; `.github/workflows/cross-arch-repro.yml` in the working tree now
contains that job and **it has not been run**. The local Rosetta 2 control in
`docs/DIVERGENCE-LOCALIZATION.md` does not substitute for it: it establishes
byte-identity of two x86_64 executions on the *macOS* host at 0, 10 and 200
steps, which removes "this trainer is not repeatable on x86_64" as an
explanation, and estimates no variance of `final_val_bpb` at 12 000 steps on the
arm that produced `bb14ab18...`.

This is named here as an **outstanding experiment**, not as a caveat on a
result: until it runs, "the platforms agree within uncertainty" is a sentence
this repository is not entitled to. The sentence it is entitled to is the
weaker one already stated below -- the platforms *have not been shown to
disagree*, at an unpaired resolution of about 0.19 bpb, from one paired run per
arm.

None of this demotes L2. It states the rung's scope: **L2 answers "does this
recipe carry to another laboratory", L3 answers "is this the same artifact",
and neither answers "is this artifact safe".** A conformity scheme that reads
an L2 pass as an integrity finding has made a category error, and it is cheaper
to write that down here than to have it discovered across a table.

## L3 -- the bytes reproduce, on a declared platform

```bash
cargo build --release --bin ckpt_replay --bin trios-train
# --max-steps must be at least the record's steps_total; the default is 2 000,
# so the 12 000-step headline is REFUSED without it.
./target/release/ckpt_replay --record checkpoints/r5-adv-recheck/12000.json \
  --trainer ./target/release/trios-train --max-steps 12000
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

### The current headline, and the scope its verdict carries

Re-measured for this document on 2026-08-03, ledger unreachable, from a
privately hashed copy of the binary
(`73fae5db63b3279457a85757492b2cac88da9119e0ce71894d7c6cfcacf263ae`), read out
of the sidecar rather than transcribed from an earlier draft:

| Field | Value |
|-------|-------|
| Sidecar | `checkpoints/r5-adv-recheck/12000.json` (`trios-checkpoint-record/4`) |
| Checkpoint sha256 | `8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c` |
| `final_val_bpb` | `2.6347548961639404` |
| `steps_total` / `gf16_floor_every` / `eval_every` | `12000` / `1` / `1000` |
| `data_synthetic` | `false` |
| Train / val corpus sha256 | `1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d` / `2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502` |
| Trainer binary sha256 | `73fae5db63b3279457a85757492b2cac88da9119e0ce71894d7c6cfcacf263ae` |
| Platform | `macos/aarch64`, libc `undetermined`, `rustc 1.96.0 (ac68faa20 2026-05-25)` |

`checkpoints/` is gitignored, so this sidecar does not exist in a fresh clone.
It is regenerated by the command printed next to the same table in
[`README.md`](../README.md#calibration-reference-checkpoint-backed); a
regenerated record will carry the `trainer.sha256` of the rebuilt binary, not
`73fae5db...`, unless the rebuild reproduces this executable byte for byte.

Note what did **not** move -- and it is a three-point result, not a two-point
one. Three separately compiled trainers produced the same weights:

| Record | Schema | `trainer.sha256` | Checkpoint sha256 | bytes |
|---|---|---|---|---|
| `r4-docs-repro/12000.json` | 3 | not recorded (attributed to `6c45b74b...`) | `8a86fe69...` | 852 272 |
| `r5-adv-recheck/12000.json` | 4 | `73fae5db63b3279457a85757492b2cac88da9119e0ce71894d7c6cfcacf263ae` | `8a86fe69...` | 852 272 |
| `r6-headline/12000.json` | 6 | `5a59f44eaa993e242da18fa55d682ba34b7d7934e9cbd417a62f8a8999d44225` | `8a86fe69...` | 852 272 |

Re-verified for this document on 2026-08-03: `shasum -a 256` over all three
`.bin` files returns
`8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c`. One rider,
stated because it is the difference between a recorded fact and an attributed
one: schema 3 has no `trainer.sha256` field, so `6c45b74b...` in the first row
comes from the build that produced it and cannot be read out of that record --
which is exactly why that record grades `INCOMPARABLE` further down. The other
two hashes are read out of their own sidecars.

The stronger sentence this licenses, and it is already measured rather than
future work: **same-platform determinism here survives RECOMPILATION, not
merely re-execution.** Three distinct executables, differing in what they
record, agree on 852 272 bytes of weights. The builds differ in what the
trainer *records*, not in what it *computes*; the trainer hash changed and the
arithmetic did not. So what a rebuild breaks is `ckpt_replay`'s trainer-PIN
CHECK -- its refusal to execute a binary whose SHA-256 is not the record's --
and **not the reproduction**. Those are different failures, they are reported
differently, and the pin check fails before a single training step runs. The
`TRAINER MISMATCH` an auditor sees today against the published headline record
(see [(d.1)](#d1-measured-2026-08-03-l3-is-demonstrated-intra-laboratory-only))
is that pin check, not evidence that the weights would not come back.

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

### The worked example: a record this repository's own verifier grades

This is the transcript of grading the headline record on 2026-08-03, pasted as
emitted. It re-executed all 12 000 steps in 9m30s -- see
[Scope of the verdict](#scope-of-the-verdict) on why that runtime is a property
of the method rather than an implementation detail.

Three elisions, each marked in place and none of them load-bearing: the
per-1000-step progress lines for steps 2 000 to 11 000 with their two
accompanying `[ledger]` lines; the repeated `[ledger] WARN: canon_name=...
publishes with NO declared optimizer` / `[ledger] no connection (DSN unset)`
pair that follows every eval; and one `[migrator] TRINITY_AUTOMIGRATE=0`
line, dropped only because it contains a non-ASCII dash that this file may not
carry. The binary was run from a privately hashed copy at `/tmp/r5-audit/` so
that a concurrent `cargo build` could not swap the executable under the audit,
which is why the paths below are not `./target/release/`. One redaction in the
transcript: the trainer's `cwd=Some(...)` field printed the original builder's
absolute checkout path and reads `<CHECKOUT>` here, so that no command or output
on this page depends on one machine's directory layout. Nothing else was
altered.

```console
$ ./target/release/ckpt_replay --record checkpoints/r5-adv-recheck/12000.json \
    --trainer /tmp/r5-audit/trios-train --max-steps 12000 --workdir /tmp/r5-audit/wd
replaying 12000 steps with "/tmp/r5-audit/trios-train" ...
[trios-train] startup args=["/tmp/r5-audit/trios-train", "--seed", "47", "--steps", "12000", "--hidden", "384", "--attn-layers", "2", "--eval-every", "1000", "--optimizer", "adamw", "--train-data", "data/tiny_shakespeare.txt", "--val-data", "data/tiny_shakespeare_val.txt", "--lr", "0.003000000026077032"] cwd=Some("<CHECKOUT>")
[trios-train] Canon #93 OK: seed=47 (no SEED env, validated cli.seed)
[trios-train] parsed seed=47 steps=12000 hidden=384 lr=0.003 ctx=None optimizer=adamw neon=None
=== trios-train seed=47 steps=12000 hidden=384 lr=0.0030 attn_layers=2 eval_every=1000 gf16_floor_every=1 ===
train=1015394 val=100000
params=196608 (196.6K) attn_d=64 attn_layers=2
Initial val_bpb=7.0002
seed=47 step=1000 val_bpb=3.3097 ema_bpb=5.5906 best_val_bpb=3.3097 nca_h=1.008 t=35.4s
... steps 2000 to 11000 elided ...
seed=47 step=12000 val_bpb=2.6348 ema_bpb=2.6707 best_val_bpb=2.6348 nca_h=0.654 t=569.9s
[ledger] no connection (DSN unset) - skipping checkpoint_record
[ckpt] /tmp/r5-audit/wd/checkpoints/r5-adv-recheck/12000.bin sha256=8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c bytes=852272 ledger=skipped-no-dsn
DONE: seed=47 bpb=2.6348 steps=12000 opt=adamw
--- ckpt_replay verdict ---
record:        "checkpoints/r5-adv-recheck/12000.json" (schema trios-checkpoint-record/4)
artifact:      "checkpoints/r5-adv-recheck/12000.bin"
recipe:        seed=47 steps_total=12000 step=12000 hidden=384 attn_layers=2
               optimizer=adamw fake_quant_format=f32 gf16_floor_every=1 eval_every=1000 data_synthetic=false
lr:            0.003000000026077032 (from the record)
source_sha256: 19aa22fb7cd187774b71cbde7cb89b664aff7260b57f55afd865dd447707108b (recorded only; this binary does NOT re-derive it - the caveat applies to this field alone, trainer.sha256 below WAS re-hashed)
trainer:       "/tmp/r5-audit/trios-train"
               sha256=73fae5db63b3279457a85757492b2cac88da9119e0ce71894d7c6cfcacf263ae version=unreported (binary exposes no --version)
               re-hashed here and it MATCHES the record's trainer.sha256
vocab:         128 (alphabet the corpus was folded onto)
workdir:       "/tmp/r5-audit/wd"
recorded sha:  8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c (852272 bytes)
replay sha:    8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c (852272 bytes)
host:          macos/aarch64
record says:   os=macos arch=aarch64 libc=undetermined toolchain=rustc 1.96.0 (ac68faa20 2026-05-25)
VERIFIED on macos/aarch64
platform triple matches the record
```

Exit status `0`, captured by the harness that ran it as `exit=0`.

Two lines in that output are worth more than the verdict. `re-hashed here and
it MATCHES` is the verifier declining to trust the `--trainer` path it was
handed. `recorded only; this binary does NOT re-derive it` is the verifier
declining to claim something it did not check.

One line in it counts fewer parameters than the artifact carries, and both
numbers belong in the record. **`params=196608` is the EFFECTIVE count -- the
weights training can move. The serialized checkpoint holds 212 992
parameters.** The difference is the 16 384 layer-2 projection weights
(`wq2/wk2/wv2/wo2`): allocated, carried through every forward pass, and still
exactly zero after training, because `HybridAttn::with_config` zero-fills them
and `wo2 = 0` makes their gradients identically zero at init. The trainer
therefore excludes them from what it prints, while
[CROSS-ARCH-DIVERGENCE.md](CROSS-ARCH-DIVERGENCE.md) counts the 43.70%
divergence against all 212 992, since those are the bytes on disk. The freeze
is enforced by the test `run_single_emits_a_loadable_artifact_and_freezes_layer_two`
in `src/train_loop.rs` rather than asserted in prose, and `README.md` states the
same thing next to the headline figure. The transcript above is pasted as
emitted and is not edited to say so.

### The predecessor, and a schema that ungrades its own history

The record this document led with until 2026-08-03,
`checkpoints/r4-docs-repro/12000.json`, no longer grades at all:

```console
$ ./target/release/ckpt_replay --record checkpoints/r4-docs-repro/12000.json --max-steps 12000
INCOMPARABLE: trainer.sha256 not recorded; this artifact cannot be graded from its own provenance record
  record:  "checkpoints/r4-docs-repro/12000.json" (schema trios-checkpoint-record/3)
  missing: trainer.sha256
  note:    this is a verdict, not a failure. The run may have been perfectly reproducible; its record simply does not say so.
```

Every recipe field is present; what is missing is the identity of the binary
that produced the bytes. `trainer.sha256` was appended to the verifier's
`REQUIRED_FIELDS` (`src/bin/ckpt_replay.rs`) after those records were written.

**This is not confined to one file.** Grading the whole directory settles it by
census rather than by assertion:

```bash
for f in checkpoints/*/*.json; do
  echo "$f :: $(./target/release/ckpt_replay --record "$f" --max-steps 1 | head -1)"
done
```

Of the 29 records on disk on 2026-08-03, **27 return `INCOMPARABLE`**: the 18
`trios-checkpoint-record/3` records (`r3-repro-4000`, the twelve
`r4-docs-ckpt` steps, `r4-docs-repro`, the three `r4-emul-*` cadence runs and
`trios-train-rng47`) name `trainer.sha256` as the missing field, and the 9
`trios-checkpoint-record/1` records (`det-a`, `det-b`, `igla-honest-20260802`,
`igla-honest-provenance`) name `eval_every` first. Only the two schema-4
records -- `r4-bind/200.json` and the headline `r5-adv-recheck/12000.json` --
get past the field check at all, and reach `REFUSED` on the deliberately tiny
`--max-steps 1` budget used to make the census cheap.

Stated without softening, because it generalises past this repository: **a
provenance schema that adds a required field ungrades every record written
before it.** The artifacts are untouched -- `r4-docs-repro/12000.bin` still
hashes to `8a86fe69...`, byte-identical to the new headline -- and the runs
behind them may have been perfectly reproducible. The grade is a property of
the record, not of the run, and tightening the standard retroactively demotes
the past. Any conformity regime built on this method inherits that: a record
format is a commitment, and revising it invalidates the corpus already
attested under it unless the old records are re-issued.

It is left standing as measured. The point of a scale is that this costs one
line of output instead of a judgement call: the verdict names the single field
that is missing, and the reader does not have to take anyone's word for what
the artifact does and does not prove.

---

## L3 holds only inside a fixed platform triple

**The trainer is bit-deterministic on a fixed platform and is not
bit-reproducible across platforms.** That cross-platform floating-point results
diverge is old and well known; nothing below is claimed as a discovery. What is
this project's own is that the divergence was *measured on this trainer* rather
than assumed away, and that the measurement is wired into CI so the boundary
cannot quietly move.

"Well known" must not be allowed to slide into "unfixable". It is neither
unfixable in the literature (arXiv:2403.09603 reports exact FP32 training
replication across three GPU types by constraining the arithmetic -- see
[Prior art this method must be measured against](#prior-art-this-method-must-be-measured-against))
nor unattributed here: `docs/DIVERGENCE-MECHANISM.md` probes each primitive of
the training path in isolation and finds three named libm functions carrying the
divergence while every reduction is byte-identical. This section states a
**declared** boundary, not an inevitable one.

The evidence directory `evidence/xarch-rustc/` holds five checkpoints produced
from the same seed (47), the same corpus and the same source tree on three
toolchain/OS combinations, each with its sidecar JSON -- ten files. It lived
outside the repository until 2026-08-03, which meant no commit could carry it
and an auditor could not resolve the paths cited here; it is now in-tree. The
hashes below were computed for this document with `shasum -a 256` when the
directory was still external, and **every one of the five was re-verified
against the in-repository copy** after the move:

| File in `evidence/xarch-rustc/` | Steps | sha256 |
|------|-------|--------|
| `macos/mac/xarch-mac/300.bin` | 300 | `fe0e66401d38e9d7ee2e0cd9667e1e118a549671f0e808a5714ecf09d19ae124` |
| `linux-rustc191/lin191/300.bin` | 300 | `9866cb2a0b13b5b3e8a630e74f9a8c792dc02c4cb43e9880f317e33be831d115` |
| `linux-rustc196/lin196/300.bin` | 300 | `9866cb2a0b13b5b3e8a630e74f9a8c792dc02c4cb43e9880f317e33be831d115` |
| `macos/mac3k/mac3k/3000.bin` | 3 000 | `f95c07ab15e1134a3109c7cbec5fb17dcde602849d0a091cd2a44ce8632c5b6a` |
| `linux-rustc191/lin3k/3000.bin` | 3 000 | `22294bce82d997f81ce928dddf4b0a9681659789ea8715479e6dd5ef03ecbfaa` |

Two readings:

1. **rustc 1.91 and rustc 1.96 on Linux agree byte for byte** (identical hash at
   300 steps). The compiler version is not the decisive variable. This arm is
   clean: both runs are x86_64 Linux, so only the compiler moved.
2. **macOS and Linux disagree at every step count measured.** The two arms
   differ in OS, in libc **and in CPU architecture** simultaneously, so this
   comparison localises the cause to that group of three and **cannot single out
   OS/libc**. An earlier revision of this section claimed the OS/libc pair was
   "the decisive variable"; that overstated the evidence and has been corrected.
   `docs/DIVERGENCE-LOCALIZATION.md` shows that an ISA change **alone**, with OS
   and libc held fixed, is already sufficient to produce a divergence.

The same conclusion is reproducible on demand rather than only on file. Running
the verifier on the Linux-produced record from a macOS host:

```console
$ ./target/release/ckpt_replay --record evidence/xarch-rustc/linux-rustc191/lin191/300.json
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

### The headline artifact, re-run on a second machine: MISMATCH

The evidence above is local. The same experiment has since been run by a third
party -- GitHub's own x86_64 runners -- on the exact headline artifact, and it
failed, which is the useful outcome.

`.github/workflows/cross-arch-repro.yml` re-executes the documented 12 000-step
command on `ubuntu-latest` and compares the checkpoint against the aarch64
macOS hash. It holds the compiler fixed by `rust-toolchain.toml` and the
dependency graph fixed by `--locked`, and verifies the corpus against
`data/MANIFEST.sha256` first, so that a mismatch cannot be blamed on the data.
Run [30767491098](https://github.com/gHashTag/trios-trainer-igla/actions/runs/30767491098),
2026-08-02, `Linux x86_64`, `rustc 1.96.0 (ac68faa20 2026-05-25)`,
`ldd (Ubuntu GLIBC 2.39-0ubuntu8.7) 2.39`:

```console
train  expected=1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d got=1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d
val    expected=2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502 got=2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502
union  expected=86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed got=86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
...
==============================================================
 checkpoint : checkpoints/r4-docs-repro/12000.bin
 bytes      : 852272
 x86_64 linux : bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3
 aarch64 macos: 8a86fe691aef64fcb637b90d4cf62650c217c3b000b6252846a8ab70c186012c
==============================================================
MISMATCH - the checkpoint is NOT portable across architectures.
```

The workflow still writes under the old canon name `r4-docs-repro`, which is a
directory name and nothing more: the aarch64 reference hash it compares against,
`8a86fe69...`, is the headline checkpoint's, because `r4-docs-repro/12000.bin`
and `r5-adv-recheck/12000.bin` are the same bytes.

Identical corpus, identical compiler, identical locked dependency graph,
identical 852 272-byte artifact size, identical 152-byte header, identical
tensor directory -- and **43.70% of the 212 992 parameters differ**, at a
relative L2 distance of **0.4756**, with every element of all four trained
attention matrices changed and the largest single disagreement a change of
sign. "A different artifact" is not an overstatement of the result; it was an
understatement, by an order of magnitude. The measurement, its definitions, and
the quantisation mechanism that explains it are in
[CROSS-ARCH-DIVERGENCE.md](CROSS-ARCH-DIVERGENCE.md); the x86_64 artifact
itself is preserved in `evidence/xarch-run-30767491098/`, and every figure is
re-derivable with `scripts/compare_checkpoints.py`.

The metric, however, survives the crossing far better than the bytes do:

| | aarch64 macOS | x86_64 Linux / glibc 2.39 |
|---|---|---|
| step 1 000 | `3.3097` | `3.3099` |
| step 3 000 | `3.0771` | `3.0736` |
| step 8 000 | `2.6455` | `2.6485` |
| step 12 000 | `2.6348` | `2.6378` |
| checkpoint sha256 | `8a86fe69...` | `bb14ab18...` |

So this pair is **L2 at a PAIRED tolerance of about 0.003 bpb at 12 000 steps,
and L0 at L3**. That figure is load-bearing, it is twelve times tighter than
the estimator's own sigma, and it therefore needs its scope stated before
anyone else states it.

**The 0.003 bpb figure is a paired, same-grid, same-sampling-plan number.**
Both arms evaluate the *same* windows of the *same* validation tail: 40 windows
of 129 tokens on a fixed evenly-spaced grid, `eval_every = 1000`, identical
`val` corpus hash `2088af36...`. The window count was a literal in
`src/train_loop.rs` at that commit rather than a recorded field -- these are
schema 4 sidecars, and `eval_chunks` arrives in schema 6 -- so the evidence
that both arms used the same plan is that both sidecars record the same
`source_sha256` (`19aa22fb...`), i.e. the same eval code down to the byte. The
dominant term in the
estimator's uncertainty is *which* 5.16% of the corpus gets read, and in a
paired design that term is common-mode and cancels. The four deltas in the
table above are `+0.0002`, `-0.0035`, `+0.0030`, `+0.0030` -- sample standard
deviation `0.0031`, an order of magnitude tighter than the between-grid sigma,
which is exactly what pairing predicts and is the evidence that the pairing is
real rather than asserted.

**That pairing ends the moment either laboratory changes the eval grid.** A
second laboratory that reads a different sample of the val corpus -- a
different `eval_chunks`, a different stride, a different prefix -- is no longer
paired with this one, and the applicable band is the *unpaired* one. It is
derived here, rather than asserted, from the combined standard uncertainty of
[EVAL-UNCERTAINTY.md](EVAL-UNCERTAINTY.md#3b-the-budget-proper) section 3b:

```
u_c   = sqrt(0.0551^2 + 0.0358^2) = 0.0657 bpb   per laboratory, k = 1
                0.0551 = the record's own val_bpb_stderr at eval_chunks = 40
                0.0358 = the between-grid sigma, transferred from 1 200 steps
u_d   = sqrt(u1^2 + u2^2) = sqrt(2) * 0.0657 = 0.0930 bpb   of the DIFFERENCE
k     = 2
T_unpaired = k * u_d = 0.19 bpb
```

**Which budget this rule is entitled to, and why `[ADDED 2026-08-06]`.**
Section 3b publishes two budgets, and the `0.0657` above is **Budget I**, not
the whole budget:

| budget | scope | `u_c` (k = 1) | `U` (k = 2) | rows combined |
|---|---|---:|---:|---|
| **I -- declared corpus** | reproducing a stated reading on the SAME corpus hash, under a possibly different sampling plan | **0.0657** | **0.13** | 0 (`0.0551`) + 1 (`0.0358`) |
| **II -- corpus choice included** | comparing readings taken on DIFFERENT eval corpora, or reading a BPB as a property of the MODEL | **0.2883** | **0.58** | 0 + 1 + 2 (`0.2807`) |

`u_c(II) = sqrt(0.0551^2 + 0.0358^2 + 0.2807^2) = 0.288288`, so
`U(II) = 0.576577`; **Budget II is 4.39x Budget I**, and row 2 alone is 4.27x
the entire `u_c` of Budget I.

**This rule uses Budget I, and is entitled to only because clause L2-1 already
forces the eval corpus `sha256` to be declared and re-checked.** Both
laboratories therefore read the identical bytes, and the corpus-choice
component is not neglected as small -- it is **common-mode and cancels
exactly**, which is a different and much stronger reason. Withdraw the corpus
hash from L2-1 and the arithmetic collapses: the applicable per-laboratory
figure becomes `0.2883`, `u_d = sqrt(2) * 0.2883 = 0.4077`, an unpaired
tolerance of `k * u_d = 0.82 bpb`, and against the policy tolerance
`T = 0.20 bpb` of section 3c the guarded acceptance interval is
`A = T - w = 0.20 - 0.58 = -0.38` -- empty, exactly as the `+/- 0.04` rule
retracted below was empty. **The corpus hash is what makes the L2 tier
decidable at all.**

A submission whose eval corpus differs from the declared one is therefore **out
of scope of the L2 tier**, not merely graded more loosely inside it. If such a
comparison must be quoted anyway, it is quoted against Budget II and at two
significant figures -- `2.6 +/- 0.58 bpb (k = 2, corpus choice included)` --
because against `U = 0.58` the second decimal is not supported and writing
`2.63 +/- 0.58` would quote a digit the budget has just withdrawn.

> **UNPAIRED ACCEPTANCE RULE (adopted here).** Two laboratories reporting
> `final_val_bpb` on the same corpus hash but on independently chosen sampling
> plans **agree** when `|b1 - b2| <= 0.19 bpb`. `k = 2` is chosen because it is
> the coverage factor EVAL-UNCERTAINTY.md section 3b already adopts for `U`, so
> the tolerance and the budget it is compared against use one convention rather
> than two; section 3b's own rider that `k = 2` is conventional and optimistic
> (the effective degrees of freedom are fewer than 6 and were not computed)
> travels with this rule unchanged.

**`[CORRECTED 2026-08-03 -- this paragraph used to adopt `+/- 0.04 bpb` as the
unpaired acceptance tolerance.]`** That adoption was defended as "knowingly too
tight, which is the conservative direction for an acceptance test". **The
defence was arithmetically wrong.** An acceptance tolerance smaller than the
expanded uncertainty of the quantity being compared does not reject
borderline agreements -- it rejects *every* submission, including a perfect
reproduction: with a guard band `w = U = 0.13 bpb` the guarded acceptance
interval is `A = T - w = 0.04 - 0.13 = -0.09`, which is **empty**. See
EVAL-UNCERTAINTY.md section 3c, "What this clause costs us". A rule nothing can
pass is not conservative; it is undecidable, and it looked conservative only
because no guard band had ever been computed.

**The honest note that came with the old figure still applies, and applies to
the new one.** `0.19` rests on the same two quantified components and is a
**LOWER BOUND** for the same reasons: the `0.0358` term was computed from seven
*nested*, hence correlated, prefixes, and it was measured on a 1 200-step
checkpoint and transferred to a 12 000-step one; and the budget behind it
excludes **corpus choice, seed, platform and step count** entirely.
**`[CORRECTED 2026-08-06 -- this sentence used to end "none of which is
quantified anywhere".]`** Three of those four are still unquantified: **seed**
(one seed, 47, no replicate), **platform** (`(os, arch, libc, toolchain)`,
where only a *paired* residual of `0.0031 bpb` exists and a paired residual is
not a component of an unpaired budget) and **step count** (the seven-grid
experiment has never been repeated at 12 000 steps) are rows 3, 4 and 5 of
EVAL-UNCERTAINTY.md section 3b and each still reads `not quantified`. **Corpus
choice is not.** It is row 2, it is **Type A**, and it is `0.2807 bpb`:
identical weights -- both `12000.bin` hashing to `902cfb69a84b...`, 852 272
bytes -- read against two held-out corpora gave `2.638520` and `2.919297`, a
difference of `0.280777` against a combined stderr of `0.063214`
(HELD-OUT-PROTOCOL.md section (iii)). It is excluded from `0.19` **on purpose
and correctly**, because L2-1 pins the corpus hash and the component cancels;
it is not excluded because nobody measured it. The tolerance is the part of the
dispersion that has been measured **and is in scope**, not the dispersion. It
is also, independently, within rounding of the
`T = 0.20 bpb` that section 3c declares as a policy choice -- two different
routes to the same order of magnitude, which is corroboration and not a second
measurement.

**Pairing remains the way to be graded sharply, and is not available here as an
acceptance rule.** EVAL-UNCERTAINTY.md L2-7 sets out the incentive: a submission
that declares its sampling plan can be compared with the window-sampling term
common-mode, at roughly the paired residual of `0.0031 bpb` rather than at
`u_d`. This document does **not** turn that into a pass mark for the pair below,
because it would be circular: the `0.0031` is the sample standard deviation of
the very four deltas that would then be graded against it, with `n = 4` and no
independent replicate. Quoting `0.003` as a cross-laboratory tolerance without
the pairing condition would be claiming a precision the instrument does not
have; quoting it as one *with* the pairing condition still needs a second pair
that this repository has not run.

**And the observed cross-platform difference is not a difference.** At step
12 000 the two arms differ by `0.0030 bpb`, which is `0.0030 / 0.0657 = 0.046`
of one laboratory's combined standard uncertainty -- an **upper** bound on that
ratio, since `u_c` is itself a lower bound -- and `0.0030 / 0.19 = 1.6%` of the
unpaired acceptance tolerance above. So: **CONFORM under the unpaired rule, and
statistically indistinguishable from zero.** The correct reading of the table is
not "the platforms agree to within 0.003" but "the platforms have not been shown
to disagree at all, at an unpaired resolution of about 0.19 bpb (`k = 2`), from
one paired run per arm". Nothing here establishes that they *would* agree at
finer resolution; it establishes that this experiment cannot see a difference,
and says how big a difference it could have seen.

With that scope stated, the conclusion stands and gets sharper: a conformity
scheme that demands bit-identity across laboratories would fail this trainer --
and, per [CROSS-ARCH-DIVERGENCE.md](CROSS-ARCH-DIVERGENCE.md), would be failing
it on 212 992 rounding decisions rather than on anything about the method. One
that demands a declared metric tolerance, with a declared sampling plan and a
declared pairing condition, would pass it. The two demands are not the same
requirement, and only the second one is measuring the thing 243-FZ asks about.

---

## Scope of the verdict

Everything in this document is a claim about a measurement, and a measurement
without a stated scope is a slogan. This section states the scope of the
`VERIFIED` above, in the vocabulary a metrologist would use, before anyone else
has to.

### (a) This is repeatability, not reproducibility

ISO 5725 and the VIM distinguish two precision conditions:

- **Repeatability** -- same operator, same equipment, same procedure, same
  location, short interval.
- **Reproducibility** -- different laboratory, different operator, different
  equipment.

Every `VERIFIED` verdict in this repository was obtained under the first
condition: one aarch64 macOS host, one operator, one toolchain, with
`platform.libc` recorded literally as `"undetermined"`. Under the standard
vocabulary that is **repeatability**, and calling it "reproducibility" would be
a vocabulary attack on the reader.

The tool concedes this itself rather than leaving it to the docs;
`src/bin/ckpt_replay.rs` says in its own module header:

> A `VERIFIED` verdict is scoped to ONE platform triple.

Read the headline as: *the same operator, on the same machine, re-executing the
same recorded procedure, obtained the same bytes.* That is a real and
non-trivial property -- it is exactly the property 1 851 earlier experiments
could not demonstrate, because they wrote no artifact at all -- and it is not
the property the word "reproducible" names.

### (b) The cross-laboratory result exists, and it is negative

There is no ambiguity to hide behind here. The experiment was pre-registered as
CI, it has run, and it disagreed: run 30767491098 above, x86_64 Linux against
the aarch64 macOS reference, `MISMATCH`. Bit-level reproducibility across
laboratories is **measured false** for this trainer, not merely unproven.

### (c) A named mechanism that predicts exactly that

The divergence is not mysterious, and the reason is checkable in one command
rather than taken on faith:

```console
$ nm -u target/release/trios-train | grep -E '_(expf|logf|cosf)$'
_cosf
_expf
_logf
```

```console
$ otool -L target/release/trios-train
target/release/trios-train:
	/System/Library/Frameworks/SystemConfiguration.framework/Versions/A/SystemConfiguration (compatibility version 1.0.0, current version 1405.120.5)
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 5026.5.4)
	/usr/lib/libiconv.2.dylib (compatibility version 7.0.0, current version 7.0.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
```

`expf` and `logf` are **undefined** in the trainer and are resolved at run time
out of `/usr/lib/libSystem.B.dylib`. The softmax in `src/train_loop.rs` calls
`f32::exp` on every logit of every forward pass, so the numerical core of the
training loop is executed by an operating-system component that is pinned by
**none** of the three things this repository pins: not `rust-toolchain.toml`,
not `Cargo.lock`, and not the provenance record. Neither `expf` nor `libm` is
required by IEEE 754 to be correctly rounded, so two implementations may differ
in the last bit and remain conformant -- and one differing bit at step 1
propagates to a different checkpoint at step 12 000.

The record is honest about not knowing this. `platform.libc` reads
`"undetermined"` because `cfg!(target_env)` is empty on Apple targets, and
`src/checkpoint.rs` records that rather than guessing:

> `"undetermined"` on Apple targets, where `target_env` is empty: the compiler
> is told no libc identity there, and probing the running host would describe
> the machine reading the record rather than the one that linked the binary.

So the field the cross-platform experiment showed to be decisive is precisely
the field the record cannot fill in. A macOS `VERIFIED` is therefore scoped to
an unnamed libc: it says "the same host", and it cannot say which host.

### (d) What is attested is an executable, not a source tree

`trainer.sha256` is re-hashed by the verifier and closes executor
substitution -- two records naming different trainers were not produced by the
same program. `source_sha256` does not: it is **recorded, not re-derived**.
`ckpt_replay` prints it with that caveat inline, and `src/checkpoint.rs` states
the limit in the field's own documentation:

> this hashes the RUNNING EXECUTABLE, so it ties an artifact to a BINARY, not
> to a source tree. This crate has no `build.rs`, so nothing compiles a digest
> of the sources INTO the binary; the binary -> source link is still unclosed.

Said outright: **a `VERIFIED` verdict attests a binary. It does not attest that
the binary was built from the source in this repository.** Closing that link
needs a reproducible build -- compiling the pinned sources with the pinned
toolchain and obtaining the recorded `trainer.sha256` -- and this repository
does not do it. Anyone citing L3 here is citing a claim about an executable.

#### (d.1) Measured 2026-08-03: L3 is demonstrated INTRA-laboratory only

The paragraph above says the binary-to-source link is unclosed. This section
records what was measured when someone finally tried to close it, because the
result changes what may be claimed in a room.

**Before.** The strongest attack on the whole method was one shell command:

```console
$ strings -a target/release/trios-train | grep -cF "$HOME"
588
```

(The grep pattern is written as `"$HOME"` rather than as the original builder's
literal home directory so that the command is runnable on any machine; on the
host that produced the `588` the two are the same string.)

588 absolute paths of the original builder were baked into `.rodata`, 504 of
them under `.cargo/registry` and the remainder this crate's own source paths,
arriving mostly as panic location strings. The repository contained no
`.cargo/config.toml`, no `--remap-path-prefix` and no `SOURCE_DATE_EPOCH`.

The consequence is not cosmetic. `ckpt_replay` refuses to execute a trainer
whose SHA-256 differs from the record's `trainer.sha256`. So an auditor with
identical sources, an identical `Cargo.lock --locked` and the pinned `rustc`
from `rust-toolchain.toml`, building under a different `$HOME`, got a different
hash and therefore **no verdict at all** -- not a failure, an inability to
start. **L3 as defined at the top of this document was obtainable on exactly
one machine.**

**After.** `scripts/repro_build.sh` exports `--remap-path-prefix` for the
checkout, the cargo registry and the rustup toolchain, and writes the same
expanded flags into `.cargo/config.toml` so a plain `cargo build` in the
checkout cannot silently disagree with it.

| measurement | before | after |
|---|---|---|
| `strings -a target/release/trios-train \| grep -c "$HOME"` | 588 | **0** |
| of which `.cargo/registry` | 504 | 0 |

Two rebuilds of the working tree on the development host, 25 minutes apart on
2026-08-03, hashed to:

```
sha256  873166e82453ed7dd005adefcc5a30c94a3969cd6b3280cd077874ee864d1317   05:47
sha256  5a59f44eaa993e242da18fa55d682ba34b7d7934e9cbd417a62f8a8999d44225   06:12
```

Both measured 0 embedded builder paths. Neither is a stable identifier for this
tree and neither may be quoted as one: `src/` was under concurrent edit between
the two runs, so these are digests of two different source states, and the
working tree did not even compile at one point in between. **The reproducible
quantity established here is the 0, not the digest.** A citable
`trainer.sha256` can only be minted from a committed tree -- which is why the
two-laboratory trial below was run against `git archive HEAD` rather than
against the working directory.

**The two-laboratory trial, and its honest negative result.** Remapping the
paths is necessary, and it is not sufficient. Measured on the committed tree
`3c1f751`, macOS aarch64, `rustc 1.96.0`, two builds differing only in `$HOME`
and `$CARGO_HOME`:

| build | sha256 |
|---|---|
| lab ALPHA, run 1 | `f37d724f468d663aa59abd77b6a55c9b53db645a1bd23150db59336b8ac73e06` |
| lab ALPHA, run 2 | `f37d724f468d663aa59abd77b6a55c9b53db645a1bd23150db59336b8ac73e06` |
| lab BETA | `d5982a2701cf5097bb685cb4c4af668e6fb6947bc2281fa428435023d9e107f8` |

Read in order, those three rows say:

- the build **is** deterministic -- ALPHA repeated itself byte for byte;
- ALPHA and BETA nonetheless **differ**, so `trainer.sha256` still does not
  cross a laboratory boundary on macOS;
- but the gap is now 48 bytes out of 11 137 968. `cmp -l` locates all 48 in two
  runs: the 16-byte Mach-O `LC_UUID` at offset 2025, and the 32-byte ad-hoc
  code-signature hash computed over it. Every byte of code and data is
  identical, and the two string tables are identical.

`-Wl,-no_uuid` was tried as a fix and **rejected**: `dyld` refuses to execute a
Mach-O with no `LC_UUID`, so cargo cannot even run its own build scripts. ELF
has no equivalent load command and its build-id is content-derived, so the
residual gap is plausibly macOS-specific -- *plausibly*, because it has not
been measured on Linux. The `two-lab-repro` job added to
`.github/workflows/cross-arch-repro.yml` is what would measure it, and like the
cross-arch job above it is written to fail loudly and print the differing byte
count rather than to be relaxed into a pass.

**So the claim that may be made, and no more:** as of 2026-08-03 the L3 verdict
is demonstrated **intra-laboratory only**. The identified blocker was 588
absolute builder paths in `.rodata`; the standard fix is `--remap-path-prefix`
(now applied) plus dependency vendoring (**not done**); and **no third-party
rebuild has yet matched this binary.**

One further consequence, stated because an auditor will hit it within a minute:
**the currently published headline record names a `trainer.sha256` that no
longer exists in the tree.** Re-minting it is deliberately deferred until this
round's `src/` changes settle, so the reproduction command printed in
`README.md` returns `TRAINER MISMATCH` today. That is the verifier behaving
correctly on a stale record, not a new defect.

### (e) Verification costs what training cost

`ckpt_replay` re-executes the entire run. There is no shortcut, no segment
check, no spot audit of a window of steps, no statistical acceptance tier and
no tolerance band anywhere in the verifier: the only comparison it makes is
SHA-256 equality on the final artifact. Grading the 12 000-step headline took
9m30s (`t=569.9s` in the transcript above) because it performed the same
12 000 training steps the original run performed. Verification time equals
training time by construction, not by inefficiency.

At this fixture's scale that is a rounding error. At the scale 243-FZ is aimed
at, it is the whole problem: **verifying a foundation model this way costs a
second training run**, which is why L2 with a declared tolerance, not L3, is
the level any workable conformity scheme is likely to land on -- with
[L2.1](#l21----what-metric-equivalence-does-not-attest) attached, because that
rung is a portability claim and not an integrity one. This repository
demonstrates L3 on a fixture whose serialized artifact carries 212 992
parameters, of which 196 608 (196.6K) are effective and 16 384 are the frozen
layer-2 block, and makes no claim that the method scales as-is.

### (f) What this is not new relative to

Stated so the prior art is in the document rather than in an objection:

- **Artifact hashing and model signing are solved and standardised elsewhere.**
  OpenSSF Model Signing v1 and the sigstore `model-transparency` project cover
  signing model artifacts, and are adopted by NVIDIA NGC and Kaggle. Nothing in
  L1 here is novel.
- **The niche is occupied.** Proof-of-Training-Data (arXiv 2307.00682, NeurIPS
  2023) and zkPoT (CCS 2025) attack training-provenance verification directly,
  and EQTY Lab sells provenance attestation commercially.
- **The grading scale itself is prior art, and this document used to claim it.**
  `[SECONDARY SOURCE 2026-08-06]` Gundersen and Kjensmo, *State of the Art:
  Reproducibility in Artificial Intelligence* (AAAI 2018), defines a three-level
  scale -- R1 / R2 / R3, reported as experiment, data and method reproducibility
  -- and is reported to have been revised by the same author into four
  reproducibility types in 2021. ACM's *Artifact Review and Badging* policy,
  version 1.1, defines Artifacts Available / Artifacts Evaluated / Results
  Validated, with the *Reproduced* and *Replicated* labels swapped relative to
  v1.0 on 15.05.2020. A tiered reproducibility scale is therefore a solved and
  standardised thing, not a contribution.

What this repository contributes is narrower than this document previously
claimed, and must be described as the narrower thing: **a deciding command per
level -- a program that returns `VERIFIED` or `MISMATCH` without human
judgement** -- applied to a trainer whose own history contains a documented
L0, 1 851 runs behind a `checkpoint::save` that returned `Ok(())` and wrote
nothing.

`[RETRACTED 2026-08-06 -- until this date, this paragraph ended by naming the
grading scale itself as the contribution, alongside the refusal to round a
verdict up and against the cryptography. The first of those three is withdrawn
outright and the retired claim is deliberately not reproduced verbatim here, so
that no future reader or grep finds this document still asserting it. A tiered
reproducibility scale is Gundersen and Kjensmo 2018 and ACM badging v1.1. The
retraction is recorded rather than the change being made silently.]`

The surviving differentiator, stated as exactly that and no wider: **every
prior scale grades documentation completeness or a social process -- were the
artifacts posted, did a committee look at them, is the method described well
enough to re-implement. None of them has a deciding command.** No reviewer
badge is computed. `ckpt_replay` is, and it returns an exit code.

---

## What this does not attest

The section above states the scope of a `VERIFIED` verdict. This one states the
questions the method does not answer at all. They are written here, in the
repository's own words, so that a counterparty reads them from us first. Each
gap is stated plainly and then followed by the honest counter -- the counter is
never that the gap is small.

### (i) There is no defence against a dishonest submitter

Every digest in this repository is **self-produced, unsigned and
untimestamped**. Nothing signs a checkpoint, a sidecar or a build; there is no
detached signature, no transparency log entry and no trusted timestamp. This is
checkable rather than asserted:

```console
$ grep -rn 'sigstore\|cosign\|in-toto\|SLSA\|RFC 3161' --include='*.rs' src/ | wc -l
0
```

The verifier therefore closes executor substitution and nothing else. Section
[(d)](#d-what-is-attested-is-an-executable-not-a-source-tree) already concedes
the sharper half of this: **a `VERIFIED` verdict attests a binary, not that the
binary came from the source in this repository.** An applicant who wants a
favourable record and controls the machine can produce one, and no arithmetic
in `ckpt_replay` will notice.

The only third-party witness anywhere in the evidence base is the public CI run
`30767491098`: GitHub-hosted runners, logs the author cannot edit, on
infrastructure the author does not control. That is one witness, and it is a
witness to a `MISMATCH`.

**The counter, and it is a product decision rather than an engineering one:**
this method cannot be shipped as self-attestation. The productisable claim is
that **conformity testing must be executed by the laboratory, from a submitted
commit, on the laboratory's own hardware -- never by the applicant.** What the
applicant submits is a source revision and a recipe; what the laboratory
returns is a verdict it produced itself. Under that arrangement the absence of
signing is not a hole, because nothing the applicant hands over is trusted in
the first place. Under self-attestation it is a fatal hole, and this document
does not claim otherwise. Adding signing (OpenSSF Model Signing, sigstore,
RFC 3161 timestamps) would raise the cost of a forged record; it would not
change who ran the computation, which is the question that actually decides
conformity.

### (ii) The record cannot express data legality

The provenance record describes what the bytes *are*. It has no field for what
the bytes are *allowed to be*:

```console
$ grep -nic 'license\|rights\|copyright\|consent\|source_url' src/checkpoint.rs
0
```

There is no rights holder, no licence identifier, no acquisition date, no
jurisdiction, no consent basis and no upstream URL. Hashing a corpus proves
which bytes were used. It says nothing about whether they were lawfully
obtained, and a hash cannot be turned into a licence by any amount of further
hashing.

Two concrete instances in this repository, named rather than left for an
auditor to find:

- `data/fineweb_train.bin` is **committed to git** and carries **no licence
  statement** in `data/README.md`. That file is a careful provenance manifest --
  bytes, SHA-256, what each file is, whether it is fit for eval -- and it
  contains zero occurrences of `licen`, `copyright`, `rights` or `terms`. The
  manifest answers identity and is silent on legality, which is exactly the gap
  described here, visible in the repository's most provenance-conscious file.
- The CI corpus is fetched from
  `raw.githubusercontent.com/karpathy/char-rnn/**master**/data/tinyshakespeare/input.txt`
  -- a **mutable branch pointer**, not a commit SHA. The workflow does gate the
  download on three SHA-256 checks (train, val, and the union hash that proves
  the split is a partition), so a silent content change becomes a loud CI
  failure rather than a corrupted number. That protects the *measurement*. It
  does not create a citable acquisition record, because the thing named in the
  recipe is a pointer that can move.

**The counter:** this is a scoping statement, not a defect to be patched away.
The method attests the *technical* reproducibility of a development cycle.
Data legality is a separate conformity axis, decided by documents and by
counsel, and any regulation that folds the two together will get neither. The
useful contribution is that a record with a rights block would be *checkable in
the same way* -- fields that must be present, refusal when absent -- and this
repository does not yet have that block. Stated as a gap, not as future work
already done.

### (iii) The scale is a fixture, not a foundation model

What has been demonstrated is **single-threaded scalar f32 arithmetic on
196 608 parameters on one machine**. No data parallelism, no reduction across
devices, no mixed precision, no non-deterministic kernel, no multi-node
scheduler -- none of the mechanisms that make large-scale training
irreproducible in practice are present in the thing that was tested. Section
[(e)](#e-verification-costs-what-training-cost) adds the cost argument: because
`ckpt_replay` re-executes every step, verification time equals training time,
so an L3 audit of a foundation model costs a second full training run.

**The claim, in one sentence, and it is deliberately weaker than the
demonstration:** *we demonstrate L3 on a fixture and claim only L2 at scale.*
What transfers is **the deciding commands and the refusal to round a verdict
up**; the bit identity does not, and this repository has not measured whether it
could. The scale itself is borrowed rather than transferred -- see
[(f)](#f-what-this-is-not-new-relative-to): tiered reproducibility scales are
prior art (Gundersen and Kjensmo 2018, ACM badging v1.1) and this document no
longer claims one as its contribution.

**The one exception, and it is the cheapest result on this page.** The *initial
coefficient matrix* -- the artifact 243-FZ names in its own words alongside
training -- does reproduce bit-for-bit across a change of instruction set, at
L1 cost rather than L3 cost, because deciding it requires one `shasum` and not a
second training run. That is the one place where an L3-strength claim survives a
platform change, and it is written up in
[INIT-MATRIX-CRITERION.md](INIT-MATRIX-CRITERION.md) together with the ceiling
that travels with it.

### (iv) Almost nothing in the repository lets a third party start

**`[PARTIALLY CLOSED 2026-08-05 -- this section used to say "Nothing in the
repository lets a third party start" and asserted that `checkpoints/` and
`evidence/` both ship zero files. That is now FALSE of `evidence/`, and the
original wording is kept visible here rather than quietly replaced.]`**

`checkpoints/` is gitignored (`.gitignore` line 13, `/checkpoints/`) and still
ships nothing. `evidence/` now ships files. **Measured 2026-08-05** on the
working tree at `ba272b9`:

```console
$ git ls-files checkpoints | wc -l
0
$ git ls-files evidence | wc -l
31
$ git ls-files evidence | grep -c '\.bin$'
12
```

**These two counts are moving, on purpose, and will be stale before they are
wrong.** Evidence directories are being added as fast as measurements are taken
(`evidence/xarch-local-isa/` landed the same day; `evidence/r6-portable/` is
queued). A reader should not check the figures below against the repository and
conclude the document lies -- they should re-derive them with the one command
above and expect a larger number. The count is context; the argument in this
section depends only on the re-hash, which is stated as a procedure and not as a
total.

**One of the two cross-architecture arms is now fetchable and re-hashable by a
reader**, which is the first digest in this document that resolves to something
other than the author's disk:

```console
$ shasum -a 256 evidence/xarch-run-30767491098/12000.bin
bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3
$ wc -c < evidence/xarch-run-30767491098/12000.bin
852272
```

That is byte-for-byte the x86_64 Linux artifact named in the CI run and quoted
throughout this document, and it now travels with the repository rather than
with a claim about it.

**What is still not fetchable, and it is the more important half.** The
`aarch64/macOS` reference `8a86fe69...` -- the *headline* artifact, the one every
`VERIFIED` verdict and every quoted `final_val_bpb` in this document rests on --
**is not shipped**. `git ls-files evidence` contains no file that hashes to it.
That was checked 2026-08-05 by re-hashing **every** tracked `.bin` under
`evidence/` -- 12 of them on that date -- and finding no match:

```console
$ git ls-files evidence | grep '\.bin$' | xargs shasum -a 256 | grep -c '^8a86fe69'
0
```

The count of `.bin` files will grow; the command will not, and it is the command
rather than the number that carries the claim. Nor is the trainer binary
the headline record pins by `trainer.sha256`. So a reader can today check the arm
that *failed* and cannot check the arm that *passed*, which is exactly the wrong
way round for a reader trying to falsify us.

**The counter, unchanged in substance:** the remainder is fixable and cheap, and
until it is fixed the headline claim is not independently checkable by a reader
who declines to run a training job. The 852 272-byte headline checkpoint, its
sidecar and the pinned trainer binary are small enough to publish as release
assets. Until they are, the honest description of the evidence base is: *one
public CI transcript anyone can re-run, one of its two artifacts now shipped and
re-hashable, and a headline artifact whose digest still resolves to exactly one
filesystem.*

### (v) The scale has one axis. Validity is the other, and L3 is blind to it

**Reproducibility and validity are independent requirements.** L0 to L3 are
assertions about *re-derivability*: whether the recorded recipe yields the same
artifact and the same metric again. They are **invariant** to whether the
procedure measures what it claims to measure. A conformity scheme that checks
only the first axis will certify reproducible nonsense -- and will certify it at
its **highest** grade, because a defect that is deterministic reproduces
perfectly.

This is not a hypothetical objection. This repository's own worst
methodological defect passes every level of this scale.

**The defect.** `gf16_floor()` rewrites `embed` / `proj` / `lm_head` / `ctx`
**in place**. Before the fix it fired on `step % args.eval_every == 0` past the
70% mark -- so the *eval cadence*, an observation parameter whose only job is to
select when the held-out corpus is read, gated an in-place mutation of the
weights. **Observing the model changed the model: an observer effect, in the
literal sense.** That is the canonical metrology sin, an instrument that
perturbs the quantity it reads, and it survived 1 851 experiments unnoticed.
The defect itself is documented in
[`README.md`](../README.md) at line 148 -- which names "gating `gf16_floor` on
`--eval-every`" as the mistake not to repeat -- and at line 362, "`gf16_floor()`
mutates the weights", where the recipe knob and the observation knob are finally
separated. What was **not** documented anywhere, until this section, is what the
defect does to the grading scale.

**Now walk it up the scale, level by level.**

| Level | Verdict on a gf16-contaminated run | Why |
|-------|-----------------------------------|-----|
| **L1** | **PASSES** | the run writes `{step}.bin`, the sidecar `sha256` is taken over the bytes re-read from disk, and the two agree. It is a perfectly correct hash of contaminated weights. |
| **L2** | **PASSES** | re-run at the same `eval_every` and `final_val_bpb` returns within any tolerance you care to declare. The metric is stable *because* the contamination is deterministic. |
| **L3** | **PASSES** | `ckpt_replay` re-executes the recorded recipe and obtains byte-identical bytes -- precisely *because* the replay faithfully re-executes the same contaminated recipe. |

**An L3 `VERIFIED` verdict is blind to an observer effect**, and not by
oversight. L3 is blind **by construction**. The verifier's entire job is to
re-run the recipe exactly. A recipe in which the eval cadence
mutates weights is re-run exactly, mutation and all, and the verdict is
`VERIFIED`.

And that is measured in this document rather than argued. In the cadence table
above, setting `TRIOS_GF16_FLOOR_EVERY=1000` recovered
`igla-honest-20260802/12000.bin` **byte for byte** -- an archived pre-fix
artifact whose floor cadence *was* its eval cadence. A contaminated run
reproduced to the byte. That is an L3-shaped success on a procedure that was
invalid.

**Stated as a rule, because it generalises past this repository:** bit-identity
is orthogonal to whether the procedure is sound. Every level of this scale, and
every conformity scheme built on one, answers *"did you do again what you say
you did"*. None of them answers *"was what you did a measurement"*. A regulator
who asks only the first question will receive reproducible nonsense, correctly
graded, and the grade will be honest.

**The constructive half: the defect is mechanically detectable, just not by
hashing.** Hashing an output cannot see a contaminated input. It is detectable
by an invariant on the observation path:

> **V1 -- observation is read-only.** Every weight tensor is bitwise unchanged
> across a call to `evaluate()`. Hash `embed`, `proj`, `lm_head` and `ctx`
> immediately before the eval and immediately after; any difference is a defect,
> reported by name, whatever the metric said.

That is a far stronger test than any metric comparison could be. The two BPB
readings the eval-cadence defect was originally spotted through, 2.6141 and
2.6169, differ by 0.08 sigma of the estimator's own noise and could never have
carried the finding -- see section 3 of
[EVAL-UNCERTAINTY.md](EVAL-UNCERTAINTY.md). A hash comparison over the tensors
is exact, needs no statistics, and catches a single flipped bit on the first
eval.

V1 is the designed **first rung of a validity axis**, and deliberately the
cheapest one: no ground truth, no second laboratory, no distribution. **It is
NOT implemented.** No code in this repository performs this check today. It is
written here as a specification so that it can be built and so that no reader
mistakes it for a shipped capability. Related checks do exist in the trainer --
`assert_train_val_disjoint` (train/val overlap), the eval-corpus entropy
precondition (degenerate held-out data), `guard_bpb` (sentinel and
impossible-metric refusal), all in `src/train_loop.rs` -- but they were written
one at a time against particular incidents and are **not** organised as an axis,
not graded, and not reported by the verifier. Naming the second axis is what is
being claimed here. Populating it is not done.

### Prior art this method must be measured against

Stated so the closest existing work is named by us rather than produced as an
objection. Section [(f)](#f-what-this-is-not-new-relative-to) lists the
artifact-signing and training-provenance neighbours; the works below are the
ones that share this document's exact problem statement.

**Reading status, stated before the content.** The first four were read **by
arXiv abstract only**, on 2026-08-03. None was read in full. Nothing below
asserts anything about their experimental sections, their threat models beyond
what the abstract states, or their measured results, because none of that was
verified here. Anyone quoting these in a room should read the papers first. The
last four rows were added on 2026-08-06 and their reading status is stated per
row; where a figure comes from reporting rather than from a page read here, it
is marked `[SECONDARY SOURCE 2026-08-06]` at the point of use.

| Work | Identifier | Venue per arXiv | Read |
|------|-----------|-----------------|------|
| Proof-of-Learning: Definitions and Practice (Jia, Yaghini, Choquette-Choo, Dullerud, Thudi, Chandrasekaran, Papernot) | arXiv:2103.05633 | "To appear in the 42nd IEEE Symposium on Security and Privacy" (2021) | abstract only |
| Proof-of-Learning is Currently More Broken Than You Think (Fang, Jia, Thudi, Yaghini, Choquette-Choo, Dullerud, Chandrasekaran, Papernot) | arXiv:2208.03567 | "Published in IEEE EuroS&P 2023" | abstract only |
| RepDL: Bit-level Reproducible Deep Learning Training and Inference (Xie, Zhang, Chen) | arXiv:2510.09180 | listed comment "Originally drafted in 2023" | abstract only |
| Optimistic Verifiable Training by Controlling Hardware Nondeterminism (Srivastava, Arora, Boneh) | arXiv:2403.09603 | NeurIPS 2024 | abstract only, not verified against the PDF |
| State of the Art: Reproducibility in Artificial Intelligence (Gundersen, Kjensmo) | AAAI 2018 | AAAI | **not read here** `[SECONDARY SOURCE 2026-08-06]` |
| Artifact Review and Badging, version 1.1 | ACM policy | ACM | **not read here** `[SECONDARY SOURCE 2026-08-06]` |
| Verde: Verification via Refereed Delegation for Machine Learning Programs, and its `RepOps` library (Arun, St. Arnaud, Titov, Wilcox, Kolobaric, Brinkmann, Ersoy, Fielding, Bonneau -- Gensyn) | arXiv:2502.19405 | arXiv | title, authors and abstract read from the arXiv landing page 2026-08-06; **body not read**; overhead figures `[SECONDARY SOURCE 2026-08-06]` |
| EQTY Lab Verifiable Compute; ISP RAS Trusted AI Platform | vendor / institute pages | n/a | **not read here** `[SECONDARY SOURCE 2026-08-06]` |

**Proof-of-Learning** is the five-year-old academic protocol with this
document's problem statement. Its abstract states the gap directly: once final
parameters are released "there is currently no mechanism for the entity which
trained the model to prove that these parameters were indeed the result of this
optimization procedure." Its mechanism is logging training checkpoints so that
the work can be re-derived. This repository's L3 is, structurally, a
single-checkpoint special case of that idea with a deciding command attached.
Anyone selling reproducibility methodology into a regulatory process will be
asked why this is not PoL. The answer must not be that we had not heard of it.

**The 2023 break.** The follow-up, by an overlapping author set including
PoL's own authors, is titled *Proof-of-Learning is Currently More Broken Than
You Think*. Its abstract states that PoL verification "is not robust to
adversaries" and that prior work "largely underestimated this lack of
robustness", introduces reproducible spoofing strategies across verification
configurations at reduced cost, and concludes that one cannot build a provably
robust PoL verifier "without further understanding of optimization in deep
learning". That is the sharpest threat that can be brought into a meeting about
this method, and it deserves to be conceded before it is raised.

**Why it is also the wedge.** PoL needs a *tolerance band*: the verifier
re-executes a segment and accepts if the recomputed weights fall within some
distance of the logged ones. That band exists because the verifier is not on
the submitter's platform, so exact equality is unattainable and a threshold has
to be invented. A band is a decision boundary, and a decision boundary is a
thing to be gamed -- spoofing is done by landing inside it. The position this
repository takes is therefore:

> **A declared-and-hashed platform collapses the tolerance band to zero.** If
> the recipe pins the platform triple and the record hashes it, the verifier's
> acceptance test is SHA-256 equality, not a distance. There is nothing to tune
> and nothing to land inside.

That is a position, not a proof. Two honest riders attach to it. First, it is
only available when the verifier can *be* the declared platform -- which is the
same conclusion (i) reached from the other direction: the laboratory must run
the computation. Second, this repository has itself measured that the platform
triple is load-bearing and currently under-specified: run `30767491098` came
back `MISMATCH` across CPU architectures, and section
[(c)](#c-a-named-mechanism-that-predicts-exactly-that) names `expf`/`logf`
resolved from an unpinned host libc as a mechanism that predicts it. Collapsing
the band requires pinning strictly more than this repository pins today.

**RepDL** is the third corner and the one that would make the wedge unnecessary
if it holds. Its abstract claims an open-source library ensuring "deterministic
and bitwise-reproducible deep learning training and inference across diverse
computing environments", achieved "by enforcing correct rounding and order
invariance in floating-point computation" -- that is, by constraining exactly
the two mechanisms (rounding and reduction order) that this repository's
cross-architecture `MISMATCH` is attributed to. It is commonly attributed to
Microsoft Research; the arXiv abstract page consulted here does not state
affiliations, so that attribution is **unverified** in this document. Whether
its claim survives contact with the platform pairs that broke this trainer was
**not tested** and must not be assumed either way. If it holds, the correct
engineering move is to adopt correct-rounding primitives rather than to declare
a platform; if it holds only under conditions this trainer cannot meet, the
declared-platform position stands. Determining which is open work.

**Srivastava, Arora and Boneh** is the fourth, and it is the one that decides
how the cross-architecture boundary in section [(b)](#b-the-cross-laboratory-result-exists-and-it-is-negative)
may be described. Its abstract identifies the same obstacle this repository
measured -- "nondeterminism between GPU types during training prevents exact
replication of the training process, resulting in schemes that are non-robust"
-- and then reports having removed it: "Across three different NVIDIA GPUs (A40,
Titan XP, RTX 2080 Ti), we achieve exact training replication at FP32 precision
for both full-training and fine-tuning of ResNet-50 (23M) and GPT-2 (117M)
models." The mechanism is stated in the same abstract: train in a higher
precision than the target, round after intermediate computations, and share the
rounding decisions through an adaptive thresholding procedure. Verbatim
quotation is used here because the claim is load-bearing and paraphrase would
soften it.

**Therefore, and this must be said plainly: cross-architecture bit identity at
FP32 is achievable by constraining the arithmetic, at a documented cost, and
this trainer does not do it.** The cost is visible in the abstract's own
description -- carrying extra precision through training, and storing and
transmitting rounding decisions alongside the artifact -- so it is a price, not
a free lunch; and the result is reported across three GPUs of one vendor, not
across the CPU/OS/libc pairs that broke this trainer, so it is not a proof that
the same recipe closes *this* gap. What it does establish is the shape of the
argument that may not be made. **The defence "every framework fails at this" is
false and must not be used.** The published counter-example is exactly the
regime this repository operates in: FP32, replication across hardware, an
auditor re-running the training. A room that knows this paper will know it.

The position that survives contact with it is narrower and stronger: this
trainer's boundary is **declared, not eliminated**, and it is an engineering gap
with published fixes rather than a law of nature. That reading is now backed by
this repository's own measurement rather than by deference to the literature.
`docs/DIVERGENCE-MECHANISM.md` probes each floating-point primitive of the
training path in isolation and finds that on the aarch64/x86_64 pair the
divergence is carried by three libm functions -- `expf`, `powf`, `cosf`, each
differing by exactly one unit in the last place on about 1% of inputs -- while
every reduction, including two deliberately different summation orders over the
same 4096-element dot product, is byte-identical, and suppressing
auto-vectorisation changes no output byte. Naming three functions is a costed
fix; "floating point is like that" is not. Whether removing them closes the
native x86_64 Linux gap is **not established** and is stated as open in that
document.

#### Gundersen and Kjensmo, and ACM badging: the scale is theirs

`[SECONDARY SOURCE 2026-08-06 -- both are reported here from secondary
reporting, not read against the AAAI proceedings or the ACM policy page. Read
them before quoting either in a room.]`

Gundersen and Kjensmo (AAAI 2018) is reported to define a three-level scale --
R1 / R2 / R3, reported as experiment, data and method reproducibility -- and to
have been revised by the same author into four reproducibility types in 2021.
ACM's *Artifact Review and Badging* v1.1 defines Artifacts Available /
Artifacts Evaluated / Results Validated, with *Reproduced* and *Replicated*
swapped relative to v1.0 on 15.05.2020. Section
[(f)](#f-what-this-is-not-new-relative-to) now credits them and withdraws this
document's earlier claim that the scale was its contribution.

**And ACM states the counter-position to this whole document as policy.**
`[SECONDARY SOURCE 2026-08-06]` The badging policy is reported to say that
exact reproduction is *not* required and that results must agree **within a
tolerance**. That is the tolerance-band argument this repository rejects,
stated as standing policy by the largest computing society since 2020. It is
the strongest published objection available, it is not a fringe view, and it
must be conceded as such before it is answered.

**The answer is a measurement this repository already owns, and it is one line
long.** Two artifacts sharing **no common hash** -- the step-10 cross-ISA pair,
`efef1cba128a8c96e23124d1f139f73c11f8e00261b6148fcfb8cc427aaa0cac` on aarch64
against `5913542eb613abc3780ac959a0262af059f7b11ea6ed61b23bd0d62b9c8897ab` on
x86_64 -- report `final_val_bpb = 6.257606029510498` **to all sixteen digits**,
read out of `evidence/xarch-local-isa/isa-probe-arm64-10.json` and
`isa-probe-x86_64-10.json`. No tolerance band, however tight, separates them.
A tolerance of zero would accept them. So the objection and this document are
not disagreeing about how tight a band should be; they are answering different
questions, and [L2.1](#l21----what-metric-equivalence-does-not-attest) names
which is which: a metric band is a **portability** test and can never be an
**integrity** test, because agreement of the number is exactly what a modified
artifact also produces. ACM is right about reproducing a *result*. It is not
addressing the question a conformity statute asks about an *artifact*.

That is the concession and the answer together, and neither half may be quoted
without the other.

#### Gensyn Verde / RepOps: cross-hardware bit-identity now has a price tag

The arXiv landing page for **arXiv:2502.19405** was read on 2026-08-06 and
gives the title *Verde: Verification via Refereed Delegation for Machine
Learning Programs*, authors Arun, St. Arnaud, Titov, Wilcox, Kolobaric,
Brinkmann, Ersoy, Fielding and Bonneau (Gensyn), and an abstract that adapts
refereed delegation to machine learning and names a library, **RepOps**, for
bitwise-reproducing ML programs across different hardware setups. **The body of
the paper was not read here.**

`[SECONDARY SOURCE 2026-08-06 -- the three overhead figures below are from
reporting of the paper, NOT from the arXiv landing page, which carries no
overhead numbers. They are quoted as reported figures and must be checked
against the PDF before being used with a counterparty.]` RepOps is reported to
cost **258-312% overhead on DistilBERT training, 67-374% on Llama-1B, and 126%
on an 8B LoRA fine-tune**, evaluated on **four NVIDIA GPUs and no CPUs**.

Two consequences, and the first is the one that matters commercially.

**1. It makes the declared-platform position CHEAPER rather than ONLY
POSSIBLE, which is a strictly stronger argument.** Until this figure existed,
the honest framing of this repository's boundary was defensive: we declare the
platform because we cannot eliminate it. With a published price attached to
elimination, the framing becomes a trade: eliminating the platform dependency
is *available* and costs somewhere between a two-thirds and a four-fold
increase in training time; declaring and hashing the platform costs a sidecar
field. A regulator choosing between them is choosing on price, not on
feasibility, and that is a conversation this repository wins on its own terms.

**2. It finishes retiring the forbidden line.** Together with Srivastava, Arora
and Boneh above, "every framework fails at this" is now doubly false and must
never be used. Two independent published systems achieve cross-hardware bit
identity. The correct sentence is that both do it **by constraining the
arithmetic at a measured cost**, and that this trainer has not paid that cost
and says so.

The scope rider, which may not be dropped: **the reported evaluation is four
NVIDIA GPUs and no CPUs.** The pairs that broke this trainer are CPU / OS /
libc pairs. RepOps is therefore not demonstrated on this repository's failure
mode, and no claim is made here that adopting it would close the native x86_64
Linux gap.

#### EQTY Lab Verifiable Compute: the direct substitute, and why a sovereignty statute cannot buy it

`[SECONDARY SOURCE 2026-08-06 -- vendor material, not read here; described from
reporting.]`

EQTY Lab's Verifiable Compute is the sharpest commercial substitute for
everything on this page, because it answers the question **"did this code run
on this machine"** by hardware attestation -- and it answers it **without
bit-reproducibility at all**. If the silicon signs a statement that a specific
binary executed on it over specific inputs, no one needs to re-derive the
artifact, and the entire L3 apparatus, the cost argument in
[(e)](#e-verification-costs-what-training-cost) and the cross-architecture
boundary all become irrelevant in one move. Anyone selling reproducibility
methodology must expect to be asked why the buyer should not simply purchase
attestation instead. The answer must not be that attestation is weak. It is
not weak; it is answering a different and in some ways easier question, and for
many buyers it is the better purchase.

**The answer that does survive is jurisdictional rather than technical.** The
root of trust is reported to be Intel and NVIDIA silicon. Under a statute whose
entire motivation is technological sovereignty, a conformity scheme whose
foundation is a signing key held by a US chip vendor is not a scheme that can
be adopted, whatever its technical merits. That is a policy argument, it is
stated as one, and it does not claim any technical deficiency in the product.

#### ISP RAS Trusted AI Platform: the Russian incumbent, and a distinction rather than a denial

`[SECONDARY SOURCE 2026-08-06 -- the institute's public page is described from
reporting; it was not read here. This paragraph makes a claim about what a page
says and must be re-checked before use.]`

The Institute for System Programming of the Russian Academy of Sciences (ISP
RAS) runs a Trusted AI Platform, and its public description is reported to
claim `obespechenie vosproizvodimosti rezultatov` -- ensuring reproducibility
of results. That is the incumbent, it is domestic, it is institutionally
positioned exactly where a 243-FZ conformity scheme will be written, and any
proposal that ignores it will be told about it.

**The answer is a distinction, not a denial.** On the reported description, the
mechanisms named are **versioning and experiment isolation** -- recording which
code, data and configuration produced which run, and keeping runs from
contaminating one another. Those are necessary, they are real engineering, and
this repository does not claim to do them better. They are also not
**re-execution to a digest**. Nothing in that description decides, by running a
command, whether a re-run produces the same bytes; the platform makes a run
*describable* and *repeatable in intent*, and the verdict remains a human
reading of a record. The gap between "the run is fully described" and "the
re-run is proved identical" is precisely the gap
[DECIDABILITY-AND-TRACEABILITY.md](DECIDABILITY-AND-TRACEABILITY.md) is about,
and it is the only gap this repository claims to close.

Stated as the sentence to use: **ISP RAS makes a development cycle
reproducible in the ordinary sense; what has no procedure anywhere is deciding,
mechanically, whether a given re-execution reproduced it.** That is a
complement to the incumbent, not a competitor to it, and it should be offered
that way.

---

## What the verifier still cannot decide

Stated so no reader has to discover them by being wrong. These are the
mechanical limits; the limits on what a passing verdict *means* are in
[Scope of the verdict](#scope-of-the-verdict).

- **It grades one checkpoint, not a run.** A `VERIFIED` verdict on step 300 says
  nothing about step 12 000 beyond what determinism implies.
- **It does not re-derive the source tree.** Schemas 3 and 4 record
  `source_sha256`; `ckpt_replay` prints it and does not recompute it. The
  binary it re-executes *is* re-hashed against the record's `trainer.sha256`,
  which is the closer question -- but it closes executor substitution, not the
  binary-to-source link. See [(d)](#d-what-is-attested-is-an-executable-not-a-source-tree).
- **It cannot see a compiler.** The trainer is invoked as an already-built
  binary. A record that names a toolchain is trusted about it.
- **It cannot see the maths library.** `expf` and `logf` are resolved at run
  time from the host's libc, which no pin in this repository covers. See
  [(c)](#c-a-named-mechanism-that-predicts-exactly-that).
- **Schemas 1 and 2 omit `lr`, `attn_scale` and `attn_seq`.** They are replayed
  at the trainer's defaults. That cannot corrupt a `VERIFIED` verdict -- bit
  identity could not arise if the values had differed -- but it makes a
  `MISMATCH` on an old record ambiguous, and the verifier says so in its output.
- **The replay costs what the original run cost.** A checkpoint at step S of a
  T-step run needs the full T steps, so `--max-steps` (default 2 000) refuses
  rather than silently spending an hour.
