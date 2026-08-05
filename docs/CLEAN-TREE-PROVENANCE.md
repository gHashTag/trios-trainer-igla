# Provenance from a named source tree

**What this document is.** The first checkpoint in this repository's local
history produced from a source tree a counterparty can obtain by name, and the
census that shows why that had never happened before.

**What it is not.** It is not a headline record and must not be quoted as one.
It is a 2 000-step run from commit `3c1f751`, which is one commit BEHIND the
uncommitted honesty fixes described in `docs/audit/HONEST_FINDINGS.md`. It
demonstrates provenance discipline. It measures nothing new.

---

## 1. The census

Every sidecar under `checkpoints/`, grouped by the `git_dirty` field:

```bash
for f in checkpoints/*/*.json checkpoints/*/*/*.json; do
  [ -f "$f" ] && python3 -c \
    "import json,sys;print(json.load(open(sys.argv[1])).get('git_dirty'))" "$f"
done 2>/dev/null | sort | uniq -c
```

Output on 2026-08-03:

```
   4 None
  47 True
```

51 records. **47 say the tree was dirty, 4 say no tree was inspected, and not
one says `false`.**

Re-run on 2026-08-05, after two further days of experiments in this working
tree:

```
   4 None
 105 True
```

109 records. The population grew by 58 and the finding did not move: the count
of `git_dirty: false` under `checkpoints/` is still **zero**. That is the
expected result, not a regression - the artifact this document reports lives in
a throwaway clone at `/tmp/cleantree`, deliberately outside `checkpoints/`, and
every run performed in the repository itself is by construction a run on a dirty
tree. The census is worth re-running rather than quoting, which is why the
command is printed above and the number is dated wherever it appears.

`git_dirty: true` at `git_sha X` means: the recorded commit is *not* the code
that ran. `git_dirty: null` is weaker still - schema 1 reported an uninspected
tree and a clean tree identically, which is the defect
`Option<bool>` was introduced to fix. Neither value names a tree a second party
can check out.

### Why that means L3 had not been demonstrated

L3 asks for every recipe input to be in the record. The source tree is a recipe
input - the largest one. A record whose only tree identifier is a commit hash
the tree did not match has not stated that input; it has stated a *neighbouring*
input and left the difference unbounded. `source_sha256` narrows this (it
digests the tree that was actually underfoot), but it is **recorded by the
trainer and never re-derived by the verifier**, so it is an attestation rather
than a proof, and until this document nothing had ever produced a record where
the commit and the tree agreed.

So the honest statement was not "L3 is satisfied intra-laboratory". It was
**"intra-laboratory L3 has never been tested"**, and that is a four-minute check
for a hostile reviewer.

### One counterexample, and why it does not close the gap

`evidence/xarch-run-30767491098/12000.json` **does** record `git_dirty: false`.
It is excluded from the census above because it is not under `checkpoints/`: it
was produced on a GitHub Actions runner, where `actions/checkout` yields a clean
tree by construction, and downloaded here as a CI artifact.

That matters in both directions. It shows the clean-tree property was already
reachable - but on somebody else's machine, by a mechanism this laboratory did
not exercise. Of the records produced **on this machine**, the count of
`git_dirty: false` was, before the run below, exactly zero.

---

## 2. The clean-tree procedure

Verbatim, and reproducible without a commit. The point is that a throwaway clone
of the local repository has a real `.git`, so `git rev-parse` and
`git status --porcelain` answer honestly, and the checkout is a tree a
counterparty can obtain by name.

```bash
# 1. A real clone with a real .git, parked at a named commit.
#    Run from the repository root; `.` is the source checkout, so the procedure
#    does not name any one machine's directory layout.
git clone --local . /tmp/cleantree
git -C /tmp/cleantree checkout 3c1f751
git -C /tmp/cleantree status --porcelain     # MUST print nothing

# 2. The corpus is gitignored, so copy it in and re-verify all three hashes
#    BEFORE training. The third hash is the one that proves the split is a
#    partition and not a copy.
mkdir -p /tmp/cleantree/data
cp data/tiny_shakespeare.txt data/tiny_shakespeare_val.txt /tmp/cleantree/data/
cd /tmp/cleantree
shasum -a 256 data/tiny_shakespeare.txt data/tiny_shakespeare_val.txt
cat data/tiny_shakespeare.txt data/tiny_shakespeare_val.txt | shasum -a 256
git status --porcelain                       # STILL nothing: /data/* is ignored

# 3. Build from the named tree.
cargo build --release --locked --bin trios-train

# 4. Train. `env -i` has nothing to enumerate, so no DSN alias can survive into
#    the child, and TRINITY_AUTOMIGRATE defaults to 1 when unset - pin it to 0.
env -i PATH="$PATH" HOME="$HOME" TRINITY_AUTOMIGRATE=0 \
  TRIOS_CANON_NAME=cleantree-2000 ./target/release/trios-train \
  --seed 47 --steps 2000 --hidden 384 --attn-layers 2 --eval-every 1000 \
  --lr 0.003 --train-data data/tiny_shakespeare.txt \
  --val-data data/tiny_shakespeare_val.txt
```

`git clone --local` printed `warning: source repository is shallow, ignoring
--local` and performed an ordinary clone. Recorded because it changes how the
clone was made and nothing about what it contains; the checkout and the
`status --porcelain` result below are what the claim rests on.

Corpus hashes, checked in `/tmp/cleantree` before any training:

```
1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d  data/tiny_shakespeare.txt      (1 015 394 bytes)
2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502  data/tiny_shakespeare_val.txt  (  100 000 bytes)
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  train ++ val                   (1 115 394 bytes)
```

All three match `data/MANIFEST.sha256`.

---

## 3. The hashes

On 2026-08-03 the command in step 4 was run twice, unchanged, on the same
machine. Both runs produced the same artifact:

```
run 1 : 7e567530acd2d265a08832dd845ac2d89945fee810f06bc428dbe63ef6774ec8   852 272 bytes
run 2 : 7e567530acd2d265a08832dd845ac2d89945fee810f06bc428dbe63ef6774ec8   852 272 bytes
```

`cmp` reports no difference. Same-machine determinism was already established
elsewhere in this repository; what is new is that it is now established **on a
tree that has a name**, so "identical inputs" is a checkable statement rather
than a description of a working directory that no longer exists.

### Repeated from a second clone, 2026-08-05

The original `/tmp/cleantree` did not survive; `/tmp` was reaped between
sessions. That turned out to be worth more than keeping it. The whole procedure
in section 2 was executed again from scratch two days later - a **new** clone, a
**new** `cargo build --release --locked`, on the same host - and every hash in
this document came back unchanged:

| Quantity | 2026-08-03 clone | 2026-08-05 clone |
|---|---|---|
| corpus: train / val / union | `1a5aead1...` / `2088af36...` / `86c4e6aa...` | identical, all three re-checked before training |
| `git status --porcelain` at `3c1f751` | empty | empty |
| `source_sha256` | `19aa22fb...` | `19aa22fb...` |
| `trainer.sha256` (the release binary) | `bbafbe51...` | `bbafbe51...` |
| checkpoint `sha256` | `7e567530...` | `7e567530...` |

The trainer row is the one that was not planned. A `cargo build --release
--locked` run on a different day, in a different directory, from a fresh clone,
produced a **byte-identical executable** - so on this host the chain
*named commit -> source digest -> binary -> checkpoint* reproduces end to end,
and not merely the last link of it. This is still one host: it says nothing
about the crossing documented in `docs/CROSS-ARCH-DIVERGENCE.md`.

### What `PATH` changes, and what it does not

Three runs were made in the 2026-08-05 clone. Two used a pinned minimal
`PATH=/usr/bin:/bin:/usr/sbin:/sbin`; the third used the ambient `PATH` that the
recipe in section 2 passes through. All three wrote the same
`7e567530...` checkpoint, so the artifact is invariant to `PATH`.

**The record is not.** `rustc` is not under `/usr/bin` on this host, so the two
minimal-`PATH` runs recorded:

```
platform.toolchain            = "unknown"
platform.toolchain_provenance = "unavailable"
```

and the ambient-`PATH` run recorded:

```
platform.toolchain            = "rustc 1.96.0 (ac68faa20 2026-05-25)"
platform.toolchain_provenance = "runtime-path-query"
```

This is the `TOOLCHAIN_PROVENANCE_NONE` branch of `resolve_toolchain` behaving
as designed: the crate has no `build.rs`, so there is no compiler-injected
version constant, and when the run-time query cannot be made the field says
`unavailable` rather than filling in the compiler that probably built it. The
practical rule for anyone repeating this procedure is to keep `PATH="$PATH"` as
section 2 has it - **not** because the bytes depend on it, but because
stripping it silently costs you a provenance field. A recipe that is more
hermetic than the record can describe buys nothing.

### What the record says

From `/tmp/cleantree/checkpoints/cleantree-2000/2000.json`:

| Field | Value |
|---|---|
| `git_dirty` | **`false`** |
| `git_sha` | `3c1f751cf4376c13d26e247c2cd86357ab51dd20` |
| `git_provenance` | `verified-local` (git was actually queried, not asserted by env) |
| `source_sha256` | `19aa22fb7cd187774b71cbde7cb89b664aff7260b57f55afd865dd447707108b` |
| `sha256` | `7e567530acd2d265a08832dd845ac2d89945fee810f06bc428dbe63ef6774ec8` |
| `trainer.sha256` | `bbafbe51c47ae8f871250cdb1a21c24e7805035dd1791a6b9e790027ee3c1074` |
| `platform` | `macos/aarch64`, `pointer_width 64`, `libc undetermined`; `toolchain unknown` / `toolchain_provenance unavailable` -- this is the minimal-`PATH` run, see above. The ambient-`PATH` run of the same command, `checkpoints/cleantree-2000-toolchain/`, records `rustc 1.96.0 (ac68faa20 2026-05-25)` / `runtime-path-query` and the same checkpoint `sha256` |
| `ledger` | `skipped-no-dsn` (nothing outside this checkout supplied a number) |
| `schema` | `trios-checkpoint-record/4` -- the tag *this* record carries, not the tag a record written today would carry; see "Gap now closed" below |
| `final_val_bpb` | `2.9743857383728027` (an `f32` expansion - see below) |

The four assertions that were checked mechanically, not read off by eye:
`git_dirty is False`; `git_sha` starts with `3c1f751`; `source_sha256` matches
`[0-9a-f]{64}`; and `hashlib.sha256` of the `.bin` on disk equals the `sha256`
the sidecar records.

### An incidental corroboration

`source_sha256 = 19aa22fb...` is **byte-identical to the digest both arms of the
cross-architecture experiment recorded** (`docs/CROSS-ARCH-DIVERGENCE.md`, table
in section 1). That document argued that the aarch64 arm's `git_dirty: true` was
harmless because the two arms' `source_sha256` agreed. This run strengthens the
argument by one step: the digest of a *genuinely clean* `3c1f751` is that same
value, so the dirt in that experiment provably did not reach any file the digest
covers - `src/**/*.rs`, `migration/src/**/*.rs`, `Cargo.toml`, `Cargo.lock`,
`rust-toolchain.toml` and the compiled feature set. The dirt was in prose and
configuration.

This is corroboration of an existing argument, not a new measurement.

---

## 4. What this artifact is NOT

Stated here so it cannot be quoted past its scope.

* **It is commit `3c1f751`, so it does not contain the uncommitted honesty
  fixes.** The checkpoint-save rewrite, the `load_data` synthetic-corpus
  refusal, the `Option`-returning `loss_on_seq`/`evaluate`, the full-coverage
  `assert_train_val_disjoint` and the schema 6 eval-plan fields are all in the
  working tree and none of them is in this artifact. Its sidecar is
  `trios-checkpoint-record/4` and states no sampling plan at all: no
  `eval_chunks`, no `eval_seq`, no `val_bpb_stderr`. By this repository's own
  argument in `docs/EVAL-UNCERTAINTY.md`, its BPB is therefore a reading and not
  a measurement result.
* **2 000 steps, not 12 000.** `final_val_bpb 2.9744` is an undertrained
  waypoint. It is not comparable to the 12 000-step headline, and nothing here
  invites that comparison.
* **`2.9743857383728027` is an `f32`'s decimal expansion.** Fourteen of those
  digits describe the float. What the run supports is `2.97 +/- 0.04` bpb, and
  the `+/- 0.04` is the unpaired band from `docs/EVAL-UNCERTAINTY.md` because
  this record cannot state the grid it was measured on.
* **One machine.** `macos/aarch64` only. The cross-architecture boundary
  documented in `docs/CROSS-ARCH-DIVERGENCE.md` is untouched by this run.

What it demonstrates is exactly one thing: a record in which the commit and the
tree agree, produced by a procedure a counterparty can repeat.

---

## 5. Reading `source_sha256` correctly after this change

This run exposed a false sentence in `src/checkpoint.rs`, now removed. The
documentation on `resolve_source_digest` claimed that a run launched outside its
source tree records `SOURCE_DIGEST_NOT_COMPUTED` "rather than a digest of
whatever tree happened to be underfoot". It does not. The walk resolves the
RELATIVE paths `src` and `Cargo.toml` against the process working directory, so
any directory that happens to contain a Rust source tree yields a well-formed
64-hex digest of *that* tree. The sentinel appears only when there is no
`src/**/*.rs` or no `Cargo.toml` at all.

The clean-tree run above is precisely the case the deleted sentence denied: it
ran in `/tmp/cleantree`, not in the repository, and recorded a well-formed
digest. The paragraph immediately above the deleted sentence already stated the
true limit ("swapping the whole source tree under a pre-compiled binary changes
this digest without changing a single instruction that executes"), so nothing
was lost by removing it.

`PlatformProvenance::source_digest_scope` now records WHICH tree the walk ranged
over, so `/tmp/cleantree` is legible in the record instead of being
indistinguishable from a repository run.

**It records a classification, not a path**, and this paragraph used to say the
opposite. Schema 6 wrote the absolute directory, which answered the question at
the price of publishing `/Users/<user>/...` in every locally produced sidecar -
in the same artifact set that reports those paths being removed from the binary,
and those sidecars are uploaded as CI artifacts. Schema 7 replaced it with
`repository-root`, or `other:<sha256 of the absolute path>`, or the sentinel
`not-computed`. The hash is one-way, so it still separates a throwaway clone
from the repository and one throwaway clone from another - which is the entire
job of the field - while naming no directory. Read
`resolve_source_digest_scope` in `src/checkpoint.rs` for the current definition
rather than trusting this sentence.

Note what that means for the artifact in section 3: it was written by the
`3c1f751` binary, which predates the field, so its sidecar carries **no** scope
at all. An absent scope is defined as *silent*, not as "the repository root" -
so the fact that this particular run happened in `/tmp/cleantree` is established
here by the procedure and by `git_sha` + `git_dirty`, not by that field. The
field is what makes the NEXT such run self-describing.

**Gap now closed, with a residue that is not.** This paragraph used to disclose
that the schema tag had not been bumped when `source_digest_scope` was added, so
a record could carry a field its tag did not name. The bump has since landed:
the tag is defined in exactly one place, `CHECKPOINT_RECORD_SCHEMA` in
`src/checkpoint.rs`, and a freshly written sidecar carries whatever that constant
says. Do not hardcode the number here or anywhere else in the docs - read the
constant, or read the `schema` field of the record in front of you.

**The rule, stated so it survives the next bump.** The tag is
`trios-checkpoint-record/N`, `N` is a single monotonically increasing integer,
and it is bumped whenever a field is added to the record. No document in this
repository should assert a value for `N`; it should say how to obtain it:

```bash
grep -oE 'trios-checkpoint-record/[0-9]+' src/checkpoint.rs | head -1
```

Run on 2026-08-05 that printed `trios-checkpoint-record/8`, and a bump to `/9`
was already in flight as this was written - which is exactly why the number is
quoted as a dated observation and not as a fact about the format. A reader who
gets a different `N` has not caught an error; they have caught a later version,
and the correct response is to decide by field presence, as below.

The residue is the records already on disk. A sidecar written before the bump
keeps its old tag forever, so a reader holding a mixed set of checkpoints must
still decide by field presence, which is the rule every schema block in
`src/checkpoint.rs` already states, and why an absent scope is defined as
"silent" rather than as "the repository root". The sidecar shown in section 3
predates the field entirely: it was written by the `3c1f751` binary.

**phi^2 + phi^-2 = 3 | TRINITY**
