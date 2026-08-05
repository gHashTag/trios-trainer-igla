# data/ corpus manifest

Every file in `data/` is listed here with its byte count, its SHA-256, what it
actually is, and whether it is fit to be used as an evaluation corpus. A file
that is not listed here has no provenance and must not be used to produce a
reported number.

This manifest is the smallest possible instance of the corpus-provenance record
the reproducibility methodology argues for: a reported BPB is only meaningful if
the exact bytes it was measured against can be named and hashed.

Regenerate the hash column with:

```
shasum -a 256 data/*
```

## Manifest

| File | Bytes | SHA-256 | What it is | Fit for eval? |
|------|-------|---------|------------|---------------|
| `fineweb_train.bin` | 61030 | `bce85ff54f13c3d088f39aec4a37e34ee5a51a6d1c65b6317f2a9ae8bdc0732c` | Small FineWeb byte slice used as the training corpus. | NO - it is the training set. |
| `fineweb_train_duplicate.bin` | 61030 | `bce85ff54f13c3d088f39aec4a37e34ee5a51a6d1c65b6317f2a9ae8bdc0732c` | Byte-identical copy of `fineweb_train.bin`. Was named `fineweb_heldout.bin`. | NO - 100% overlap with train. |
| `pangram_fixture_160b.bin` | 160 | `07ea982f70dea0c0bd10a5d33f87c5a38f1e0801566057470b46cd83da66af40` | A 160-byte hand-written pangram string. Was named `fineweb_val.bin`. Not FineWeb. | NO - degenerate, single eval window. |
| `tiny_shakespeare.txt` | 1015394 | `1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d` | TinyShakespeare training split, byte-disjoint from the val split below. | Train split. |
| `tiny_shakespeare_train_core.txt` | 915394 | `21f0788a3b5ef3d8138047559f393233f1282403887e1bda2de458af64877da6` | `tiny_shakespeare.txt[0, 915394)`. Training split of the three-way partition. | NO - it is the training set. |
| `tiny_shakespeare_test.txt` | 100000 | `00365e8aa883ffe50d954234fe810d9b2413450c944a67f0a621d433464e9603` | `tiny_shakespeare.txt[915394, 1015394)`. Held-out test split, read once. | YES - and only once, see below. |
| `tiny_shakespeare_val.txt` | 100000 | `2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502` | TinyShakespeare validation split, byte-disjoint from the train split above. | YES. |

Four of these files are committed in git. Two of them - `fineweb_train.bin` and
`pangram_fixture_160b.bin` - are committed despite `.gitignore: /data/*`, because
an ignore rule does not untrack a file added before it. The other two -
`tiny_shakespeare.txt` and `tiny_shakespeare_val.txt` - are tracked on purpose,
by explicit `!` negations added on 2026-08-05; see the next section. The rest of
`data/` (`fineweb_train_duplicate.bin`, `tiny_shakespeare_train_core.txt`,
`tiny_shakespeare_test.txt`) is still untracked and is either a known-defective
fixture or a recut derivable from the tracked bytes. `git ls-files data/` is the
authority; the header of `data/MANIFEST.sha256` predates the 2026-08-05 change
and still describes only the first two.

## The tracked copy is NORMATIVE

`data/tiny_shakespeare.txt` and `data/tiny_shakespeare_val.txt` are **in git**.
The bytes delivered by `git clone` are the normative ones: they are what every
BPB, every checkpoint SHA-256 and every ledger row in this repository was
measured against, and they are what a re-check must be run against. Their blobs
in the index hash to the two values in the manifest above, and concatenated in
the order train-then-val they hash to `86c4e6aa...` over 1115394 bytes.

The download recipe in `README.md` is now a **FALLBACK**, for environments that
have the source tree without the git objects (a tarball export, a Docker build
context that excluded `data/`, a partial checkout). It is no longer the primary
way to obtain the corpus, and CI uses it only when the tracked file is absent.

Why 1.1 MB is worth carrying in git: a checksum detects **substitution** but it
cannot detect **deletion**. If the upstream path moves or disappears,
`MANIFEST.sha256` can still tell an auditor that what they are holding is wrong,
but it cannot tell them what the right bytes were, and a fresh clone can then run
none of the documented verification commands. A conformity dossier has to stay
re-checkable for as long as the certificate it supports is valid, which is longer
than the guaranteed lifetime of a branch pointer on someone else's host. A hash
whose preimage nobody holds is a receipt, not an archival record.

### Provenance of these exact bytes

Upstream (a mutable branch pointer, not a commit SHA - stated as the limitation
it is):

```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

That single file is the canonical 1115394-byte corpus, `86c4e6aa...`. The pair in
this repository is a head/tail cut of it:

```
SIZE=$(wc -c < input.txt)                 # 1115394
head -c $((SIZE - 100000)) input.txt > data/tiny_shakespeare.txt   # 1015394 bytes
tail -c 100000            input.txt > data/tiny_shakespeare_val.txt #  100000 bytes
```

head/tail and not `head -c 100000 train > val`: the latter makes val a prefix of
train, which is the leaking split that tainted the 2026-04-30 ledger. The
concatenation hash above is the checkable proof that this pair is a partition of
the upstream file - disjoint, complete, nothing added.

### What a hash does and does not attest

A SHA-256 attests **byte identity, and nothing else**. It says that two parties
hold the same bytes. It does not attest that those bytes were lawfully obtained,
that they may be redistributed, that they are what the upstream author intended,
or that they are fit for any particular measurement. Fitness is a separate
judgement, made in the table above (`fineweb_train_duplicate.bin` hashes fine and
is 100% overlapped with train). Legality is a separate axis again, and it is not
decidable by any check in this repository.

### Rights status (partly UNVERIFIED)

Recorded here because `docs/REPRODUCIBILITY-GRADING.md` names the absence of a
rights block in this file as an open gap. This section narrows that gap for the
TinyShakespeare pair only; it says nothing about `fineweb_train.bin`, whose
rights status remains unrecorded.

- **Underlying text.** Plays of William Shakespeare (d. 1616). The works
  themselves are long out of copyright in every jurisdiction relevant here.
  *Which* transcription or edition upstream used is **UNVERIFIED** - the bytes
  carry no edition statement, no front matter and no copyright notice (the file
  opens directly on `First Citizen:`), and a modern typographic edition can carry
  its own thin rights claim. Nothing in this repository establishes which one
  this is.
- **Upstream repository licence.** `karpathy/char-rnn` **UNVERIFIED**. No copy of
  its licence is vendored here, and establishing it requires a network call that
  this record deliberately does not depend on. If the rights status of the
  redistribution matters to a reader, they must check the upstream repository
  themselves; do not read the presence of these bytes in git as a licence claim.
- **This repository's own LICENSE (MIT)** covers the code in this repository. It
  is not a grant over third-party corpus bytes and must not be read as one.

Anything above marked UNVERIFIED is unverified as of 2026-08-05 and should be
treated as an open item, not as an assurance.

## fineweb_train_duplicate.bin - was `fineweb_heldout.bin`

Renamed on 2026-08-02. It is not held out. Its SHA-256 is
`bce85ff54f13c3d088f39aec4a37e34ee5a51a6d1c65b6317f2a9ae8bdc0732c`, which is the
SHA-256 of `data/fineweb_train.bin`. The two files are byte-identical, 61030
bytes each. A run that evaluated against `fineweb_heldout.bin` evaluated against
its own training set with 100% verbatim overlap.

It is kept rather than deleted because a file that documents its own defect is
more useful than a missing one: any historical row or config that names
`fineweb_heldout.bin` can be traced to these bytes and to this hash collision.

## pangram_fixture_160b.bin - was `fineweb_val.bin`

Renamed on 2026-08-02. It is 160 bytes and it is not FineWeb. Its first 64 bytes,
verbatim:

```
The brown fox jumped over the lazy dog. Boxing wizards jump quic
```

The full 160 bytes are a short, hand-written pangram paragraph with heavy word
repetition.

Why any BPB measured against it is a single-window reading: `evaluate()` walks
the eval token stream with stride `SEQ + 1 = 129` and needs a full 129-token
window, so `max_start = 160 - 129 = 31`. Only `start = 0` fits; the next start
would be 129, which is past 31. The corpus therefore yields **exactly one**
evaluation chunk. A number such as the observed `DONE: seed=47 bpb=6.4358` is one
window of 129 tokens, not a validation-set average, and it carries no variance
information at all.

Note that a corpus can be degenerate in this way and still pass an 8-gram entropy
precondition: this fixture has 152 distinct 8-grams out of 153, because that
guard measures diversity, not sample size. Sample-size adequacy is a separate
check (a minimum val length), and the two are not substitutes.

## Why near-zero BPB was never a leak signature

Measured on this architecture, a 100% verbatim train/val overlap moves BPB by
only about 0.12. So an implausibly low BPB does not indicate a leak; it indicates
a degenerate evaluation corpus like `pangram_fixture_160b.bin`. Both defects
matter, but they are different defects and they need different guards.

## TinyShakespeare split

`tiny_shakespeare.txt` (1015394 bytes) and `tiny_shakespeare_val.txt` (100000
bytes) are byte-disjoint. Concatenated in that order they reconstruct the
canonical 1115394-byte corpus:

```
cat data/tiny_shakespeare.txt data/tiny_shakespeare_val.txt | shasum -a 256
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  -
```

This is the pair behind the honest calibration figures (7.00 at init, ~3.31 at
step 1000, ~2.61 raw val BPB at step 12000; h=384, two allocated attention
blocks, one effective; 196,608 effective parameters of 212,992 serialized --
the layer-2 block is allocated and provably frozen, see the test
`run_single_emits_a_loadable_artifact_and_freezes_layer_two` in
`src/train_loop.rs`). Earlier documentation described the val split as
`head -c 100000 train > val`, which would have been a leaky split; that is not
how these two files were produced, and the description has been corrected.

## TinyShakespeare three-way split (2026-08-03)

A two-way split has a train flag and a val flag and no third one, so every
figure this repository ever published was measured on the same corpus it was
selected on: `--eval-every 1000` consults the val stream twelve times per run,
and "best_val_bpb" is a selection made by looking at it. That is a training
diagnostic, not a held-out result.

The third split is a recut of the same bytes - no new data:

```
train_core = tiny_shakespeare.txt[0, 915394)        -> tiny_shakespeare_train_core.txt
test       = tiny_shakespeare.txt[915394, 1015394)  -> tiny_shakespeare_test.txt
val        = tiny_shakespeare_val.txt                  unchanged
```

Union proof - the same canonical hash as the two-way split, from three pieces:

```
cat data/tiny_shakespeare_train_core.txt \
    data/tiny_shakespeare_test.txt \
    data/tiny_shakespeare_val.txt | shasum -a 256
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  -
```

And `train_core` + `test` reproduce `tiny_shakespeare.txt`
(`1a5aead1...`), which is the separate proof that the test slice was removed
from the training stream rather than copied out of it.

`tiny_shakespeare_test.txt` is governed by a one-look rule: it is read by the
single pre-registered command in `docs/HELD-OUT-PROTOCOL.md` and by nothing
else. No config in this repository may be chosen by consulting it, and any run
that trains on `tiny_shakespeare.txt` (the full 1015394-byte stream) has that
corpus inside its training set and cannot use it as held-out data at all.

Because `train_core` is 100000 bytes smaller than `tiny_shakespeare.txt`, a BPB
measured under the three-way split is not comparable to the two-way headline
figure of 2.63. Different training corpus, different specimen.
