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

Two of these files are committed in git despite `.gitignore: /data/*` - an
ignore rule does not untrack a file added before it. `git ls-files data/` lists
`fineweb_train.bin` and `pangram_fixture_160b.bin`; the rest of `data/` really
is untracked. See the header of `data/MANIFEST.sha256`.

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
step 1000, ~2.61 raw val BPB at step 12000; h=384, 2 attention layers, ~196.6K
parameters). Earlier documentation described the val split as
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
