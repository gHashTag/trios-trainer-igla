# Held-out protocol: a third split, pre-registered

This document is written BEFORE the runs it describes. Sections 1-4 were
committed to disk and the file was closed before the trainer was invoked a
single time against the new corpus. The RESULT section was appended afterwards
and reports whatever the first execution produced.

Why this exists: `docs/EVAL-UNCERTAINTY.md` observes that this repository has a
train flag and a val flag and "no third one", and then drafts a grading clause
(L2-8) that treats any metric reported on the same corpus it was selected on as
a training diagnostic rather than a result. Under that clause the repository's
own headline figure (2.63 val BPB, `checkpoints/r6-headline/12000.json`) is a
training diagnostic: `--eval-every 1000` means the val corpus was consulted
twelve times during the run, and every "min observed BPB" or "this config beats
that one" judgement in the repository's history was made by looking at it.

The corpus needed to fix that is already in the tree. No new data is downloaded
and `data/tiny_shakespeare.txt` is not rewritten.


## 1. The three-way partition

The canonical 1 115 394-byte tinyshakespeare corpus is cut into three
byte-disjoint pieces. `train_core` and `test` are slices of the existing
`data/tiny_shakespeare.txt`; `val` is the existing file, untouched.

| Role | File | Bytes | Source slice | SHA-256 |
|------|------|-------|--------------|---------|
| train | `data/tiny_shakespeare_train_core.txt` | 915394 | `tiny_shakespeare.txt[0, 915394)` | `21f0788a3b5ef3d8138047559f393233f1282403887e1bda2de458af64877da6` |
| select (val) | `data/tiny_shakespeare_val.txt` | 100000 | pre-existing, unmodified | `2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502` |
| held out (test) | `data/tiny_shakespeare_test.txt` | 100000 | `tiny_shakespeare.txt[915394, 1015394)` | `00365e8aa883ffe50d954234fe810d9b2413450c944a67f0a621d433464e9603` |

Produced by:

```
head -c 915394 data/tiny_shakespeare.txt > data/tiny_shakespeare_train_core.txt
tail -c 100000 data/tiny_shakespeare.txt > data/tiny_shakespeare_test.txt
```

### Partition proof

A sentence claiming a split is disjoint is not checkable. A hash of the union
is. Concatenated in the order train_core, test, val, the three files reconstruct
the canonical corpus byte for byte:

```
$ cat data/tiny_shakespeare_train_core.txt \
      data/tiny_shakespeare_test.txt \
      data/tiny_shakespeare_val.txt | shasum -a 256
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  -
```

`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` is the union
hash already recorded in `data/MANIFEST.sha256` for the two-way split. The three
pieces sum to 1 115 394 bytes (915394 + 100000 + 100000), which is the length of
the canonical corpus, so the reconstruction is a permutation-free partition and
not a coincidence of hashing overlapping bytes. The first two pieces alone
reconstruct `data/tiny_shakespeare.txt`
(`1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d`), which is
the separate statement that `test` was carved out of the old training stream and
that the new training stream no longer contains it.


## 2. The one-look rule

`data/tiny_shakespeare_test.txt` is read exactly ONCE, by the single command in
section 4 (Run A). Whatever number that first execution produces is the number
published in RESULT.

- No re-runs. If the run crashes for a reason unrelated to the measurement
  (machine reboot, disk full), the failure is recorded here and the split is
  regarded as spent until a differently-seeded test slice is pre-registered.
- No hyperparameter change after seeing the number. The hyperparameters below
  are the ones the repository already published for the headline run; they were
  chosen before this corpus existed.
- No second look. There is no "min observed test BPB": the test corpus is not
  passed to `--eval-every` selection in any other run, and no future config
  choice in this repository may cite it.

A held-out number survives exactly as long as nobody optimises against it. The
rule above is the whole value of section 1.


## 3. Scope limit: this number is NOT comparable to 2.63

The training corpus for these runs is `tiny_shakespeare_train_core.txt`, 915 394
bytes. The headline run trained on `tiny_shakespeare.txt`, 1 015 394 bytes. The
new training set is 100 000 bytes smaller - 9.85% less data - because the test
slice was carved out of it.

Therefore the held-out figure below MUST NOT be quoted as an improvement or a
regression against 2.63. It is a different specimen trained on a different
corpus. Any difference between the two numbers confounds the change of
evaluation corpus with the change of training corpus, and this protocol makes no
attempt to separate them. The comparison that IS legitimate is the one inside
this document: Run A against Run B, which share a training corpus and a seed and
differ only in which stream is used for evaluation.


## 4. The two commands

Both runs are executed by a frozen binary, not by `target/release/trios-train`.
Other agents are editing `src/` in this repository concurrently, so a rebuild
between Run A and Run B would silently make the pair incomparable - which is
precisely the class of defect `--eval-every` turned out to be.

```
cargo build --release --bin trios-train
mkdir -p evidence/heldout
cp target/release/trios-train evidence/heldout/trios-train.frozen
shasum -a 256 evidence/heldout/trios-train.frozen | tee evidence/heldout/trainer.sha256
```

Frozen instrument:

```
a380ed4722e2fb4674ad1b5f3fe2e810362fef3284751006b9d521e0b87ad1ad  evidence/heldout/trios-train.frozen
```

Platform: Darwin arm64 (aarch64 macOS), rustc 1.96.0 (ac68faa20 2026-05-25),
repository at git 3c1f751cf4376c13d26e247c2cd86357ab51dd20 with a dirty working
tree (other agents' edits are in flight; the binary hash above, not the git sha,
is what identifies the instrument). `TRIOS_EVAL_CHUNKS` is NOT set in the
environment, so `eval_chunks_target()` resolves to the crate default - the same
sampled evaluation the headline run used, and the reason a `val_bpb_stderr` is
reported alongside every BPB below.

Every path in the two commands below is **relative to the repository root**, so
run them from a checkout and nothing depends on one machine's directory layout.
`TRIOS_CHECKPOINT_DIR` was written as an absolute path when these runs were
taken; `src/checkpoint.rs` resolves it against the working directory (its own
default is the relative `checkpoints`), so `evidence/heldout` names the same
directory from the repository root and the artifacts recorded below are
unchanged.

### Run A - the held-out measurement

```
env -u DATABASE_URL -u NEON_DATABASE_URL -u TRIOS_DATABASE_URL \
    TRINITY_AUTOMIGRATE=0 \
    TRIOS_CHECKPOINT_DIR=evidence/heldout \
    TRIOS_CANON_NAME=r6-heldout-test \
    evidence/heldout/trios-train.frozen \
      --seed 47 --steps 12000 --hidden 384 --attn-layers 2 --lr 0.003 \
      --eval-every 1000 \
      --train-data data/tiny_shakespeare_train_core.txt \
      --val-data data/tiny_shakespeare_test.txt \
  2>&1 | tee evidence/heldout/A.log
```

### Run B - the selection corpus, observer control

```
env -u DATABASE_URL -u NEON_DATABASE_URL -u TRIOS_DATABASE_URL \
    TRINITY_AUTOMIGRATE=0 \
    TRIOS_CHECKPOINT_DIR=evidence/heldout \
    TRIOS_CANON_NAME=r6-heldout-select \
    evidence/heldout/trios-train.frozen \
      --seed 47 --steps 12000 --hidden 384 --attn-layers 2 --lr 0.003 \
      --eval-every 1000 \
      --train-data data/tiny_shakespeare_train_core.txt \
      --val-data data/tiny_shakespeare_val.txt \
  2>&1 | tee evidence/heldout/B.log
```

The two commands are identical except for `--val-data` and the checkpoint run
name. The database variables are unset rather than merely ignored: an earlier
agent this round landed twelve rows on shared state by leaving `DATABASE_URL`
in the ambient environment, and `TRINITY_AUTOMIGRATE=0` additionally forbids
schema DDL.

### What Run B is for

Run B is not a second opinion on the model. It is a control on the instrument.
The evaluation corpus is supposed to be read-only with respect to the weights:
`evaluate()` should observe the specimen without perturbing it. If that holds,
Run A and Run B - same seed, same training bytes, same step count, same binary -
must produce byte-identical `12000.bin`.

This repository has already found one place where an "observation parameter" was
not one: `--eval-every` gates `gf16_floor`, which mutates embed/proj/lm_head/ctx
in place, so two runs differing only in evaluation cadence produced different
weights and different BPB (2.6141 vs 2.6169). Nobody has ever checked whether
the evaluation *corpus* has the same defect. The hypothesis under test is that
it does not; the prediction is byte-identity; and a mismatch is the finding, to
be reported and not re-run away.


## 5. Honest-failure clause

If `assert_train_val_disjoint` rejects the new test corpus - too few tokens, too
few eval chunks, verbatim overlap with the training stream, or a degenerate
8-gram distinct-window ratio - the refusal message is recorded verbatim in the
RESULT section and this work item stops there. No guard is weakened, bypassed or
reconfigured to make a run finish. An evidenced refusal is a valid result: it
would mean the third split as cut is not fit to carry a published number, which
is a fact worth more than a number that is not fit to be published.


---

# RESULT

Appended 2026-08-03 after both runs finished. Sections 1-5 above were on disk
before the trainer touched `data/tiny_shakespeare_test.txt`. Both runs exited 0.
Every number below is read from the checkpoint sidecar JSON, not from stdout.

Artifacts:

```
evidence/heldout/trios-train.frozen        the instrument
evidence/heldout/trainer.sha256            its hash
evidence/heldout/A.log                     Run A, full stdout+stderr, 62 lines
evidence/heldout/B.log                     Run B, full stdout+stderr, 62 lines
evidence/heldout/r6-heldout-test/12000.bin    Run A weights, 852272 bytes
evidence/heldout/r6-heldout-test/12000.json   Run A sidecar
evidence/heldout/r6-heldout-select/12000.bin  Run B weights, 852272 bytes
evidence/heldout/r6-heldout-select/12000.json Run B sidecar
```

No guard refused. `assert_train_val_disjoint` passed on both corpora: the test
slice is not too short, yields 40 eval chunks at the configured coverage, has no
verbatim window overlap with `train_core`, and is not degenerate under the
8-gram distinct-window ratio. Section 5 did not fire.


## (i) The held-out measurement

From `evidence/heldout/r6-heldout-test/12000.json`:

```
final_val_bpb    2.638519525527954
val_bpb_stderr   0.04238646477460861
eval_chunks      40
eval_tokens      5160
eval_seq         129
corpus.val       data/tiny_shakespeare_test.txt
                 100000 bytes
                 00365e8aa883ffe50d954234fe810d9b2413450c944a67f0a621d433464e9603
corpus.train     data/tiny_shakespeare_train_core.txt
                 915394 bytes
                 21f0788a3b5ef3d8138047559f393233f1282403887e1bda2de458af64877da6
trainer.sha256   a380ed4722e2fb4674ad1b5f3fe2e810362fef3284751006b9d521e0b87ad1ad
platform         macos / aarch64, rustc 1.96.0 (ac68faa20 2026-05-25)
```

**Held-out BPB = 2.6385 +/- 0.0424 (stderr, 40 windows).** This is the first
number in this repository measured on a corpus that was not consulted during the
run that produced it, and under the L2-8 clause drafted in
`docs/EVAL-UNCERTAINTY.md` it is the only figure here that is a result rather
than a training diagnostic.

Read section 3 before quoting it: it is NOT comparable to the 2.63 headline,
because the training corpus is 100 000 bytes smaller.

Two limitations that belong next to the number rather than in a footnote:

1. It is a 5.16% sample. The pre-registered command used the crate default
   `eval_chunks=40`, so 5160 of the test corpus's 100000 tokens were read - the
   same sampled evaluation the headline used, and the reason a stderr is quoted
   at all. A full-coverage reading would be a better number and the one-look
   rule forbids taking it: re-running the test corpus at higher coverage after
   seeing 2.6385 is exactly the second look section 2 exists to prevent. If a
   full-coverage held-out figure is wanted, it needs a newly pre-registered
   split, not a second pass over this one.
2. The pre-registration binds the corpus and the command, not the model class.
   These hyperparameters were inherited from the headline run, which selected
   them by looking at `tiny_shakespeare_val.txt`. So the *architecture and
   hyperparameters* still carry selection history; what is held out is the
   evaluation corpus for this specific run.


## (ii) THE OBSERVER CONTROL - byte-identical

```
$ shasum -a 256 evidence/heldout/r6-heldout-test/12000.bin \
                evidence/heldout/r6-heldout-select/12000.bin
902cfb69a84bdcf4d6ab524576ff375d612fd8f485dcbc7e02f35f6379599146  evidence/heldout/r6-heldout-test/12000.bin
902cfb69a84bdcf4d6ab524576ff375d612fd8f485dcbc7e02f35f6379599146  evidence/heldout/r6-heldout-select/12000.bin

$ cmp evidence/heldout/r6-heldout-test/12000.bin evidence/heldout/r6-heldout-select/12000.bin
(no output; exit 0)
```

852272 bytes each, identical byte for byte.

**The evaluation corpus does not perturb the specimen.** This is the first
direct evidence in this repository for that proposition. Two runs that differed
only in which 100 000-byte stream `evaluate()` read produced the same weights,
so `evaluate()` is genuinely an observation here and not, as `--eval-every`
turned out to be, a hidden intervention. The check was worth taking precisely
because the analogous assumption had already failed once: `--eval-every` gates
`gf16_floor`, which mutates embed/proj/lm_head/ctx in place, and two runs
differing only in evaluation cadence produced different weights and BPB 2.6141
vs 2.6169.

Scope of the claim, stated so it is not over-read: this is one binary
(`a380ed47...`), one seed, one configuration, one platform, and two specific
corpora of equal length. It shows the eval corpus is not coupled to the weights
along the path these two runs exercised. It does not prove no such coupling can
exist - a corpus that provoked a NaN or a different eval-chunk count could still
reach the weights through some other route. It is a control that passed, not a
theorem.

One incidental honesty note. The two sidecars disagree on `source_sha256`
(A: `aa80a56b...`, B: `0df9561b...`) because other agents were editing `src/`
between the two runs. That field digests the repository working tree at run
time, not the instrument. The field that identifies the instrument,
`trainer.sha256`, is `a380ed4722e2fb4674ad1b5f3fe2e810362fef3284751006b9d521e0b87ad1ad`
in both - which is the whole reason section 4 froze the binary. Had the runs
used `target/release/trios-train`, the pair would have been silently
incomparable and the byte-identity above would have proved nothing.


## (iii) The gap between the two corpora

```
A  held out   data/tiny_shakespeare_test.txt   BPB 2.638520  stderr 0.042386
B  selection  data/tiny_shakespeare_val.txt    BPB 2.919297  stderr 0.046898

B - A = +0.280777
combined stderr sqrt(se_A^2 + se_B^2) = 0.063214
ratio = 4.44
```

The gap is 4.4x the combined standard error, so it is not sampling noise. But it
is not a selection effect either, and calling it one would be wrong in two
independent ways.

**The sign is backwards.** A selection effect makes the held-out corpus look
WORSE than the corpus that was optimised against. Here the held-out corpus is
0.28 bpb EASIER. Whatever this measures, it is not the model having been fitted
to `tiny_shakespeare_val.txt`.

**There was no selection to detect.** Because the weights are byte-identical
(section ii), nothing about run B's readings could have influenced run B's
model, and nothing about run A's could have influenced run A's. The two numbers
are the same weights read against two different corpora. The entire 0.28 is
therefore attributable to the corpora, with a contribution of exactly zero from
the model. That is a stronger statement than a p-value: the confound was not
controlled for statistically, it was eliminated by construction.

What the gap actually measures is that the last 100 000 bytes of
`tiny_shakespeare.txt` and the 100 000 bytes of `tiny_shakespeare_val.txt` have
different conditional entropy under this model. The per-step trajectories
support that reading - the gap is absent at initialisation and grows as the
model learns:

```
step      0    1000   2000   3000   4000   5000   6000   7000   8000   9000  10000  11000  12000
A     7.0002  3.3156 3.1077 2.9690 2.9283 2.9128 2.7512 2.7145 2.6716 2.6577 2.6461 2.6396 2.6385
B     7.0001  3.4048 3.2596 3.1648 3.0823 3.0850 2.9468 2.9745 2.9242 2.9391 2.9259 2.9205 2.9193
B-A   -0.0001 0.0892 0.1519 0.1958 0.1540 0.1722 0.1956 0.2600 0.2526 0.2814 0.2798 0.2809 0.2808
```

At step 0 the model is uniform over 128 symbols, so both corpora read 7.00 bits
and the gap is zero to four decimals. The gap appears only once there is a model
to be easier or harder for. This is what a corpus-difficulty difference looks
like; it is not what selection looks like.

The consequence for the methodology is a caution, not a win: two equal-length
slices of the same book, cut from the same file on the same day, differ by 0.28
bpb - about seven times the stderr of either. A BPB is a property of a
(model, corpus) pair, and quoting one without naming and hashing the corpus is
not a measurement. That is the same conclusion the cross-architecture checkpoint
mismatch reached from the other direction: the conditions are part of the
result.

Finally, this document does NOT license a comparison of 2.6385 against the 2.63
headline. Section 3 forbids it, and section (iii) adds the reason it would be
meaningless even if the training corpora had matched: the two figures were
measured against different 100 000-byte corpora whose difficulty differs by more
than either error bar.
