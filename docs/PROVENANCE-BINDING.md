# Provenance binding: sealing the half of a record that no digest covered

Status: implemented 2026-08-05; sealed set corrected and every digest reminted
2026-08-06 (section 2.1). Reference implementation `src/provenance_seal.rs`,
second implementation `provenance_seal()` in `interop/triosckp_reader.py`,
tests `tests/provenance_seal_tamper.rs`, published digests
`evidence/SEALS.txt`.

## 0. What this is, before what it does

This is a **seal, not a signature**.

It binds a record's declaration to a digest that has to be **published out of
band**. An adversary who controls both the record and the channel the digest is
published on simply recomputes the seal over the forgery, and this document
stops them exactly nowhere. Closing that requires key material - a signature
over the seal by a party that is not the vendor - which this project does not
have and does not claim.

What a published seal buys is narrower and still worth having: a forged
declaration moves from **undetectable by anyone** to **detectable by anyone who
read the published digest**. That is the whole claim. Any sentence in a pitch
that upgrades it is false.

## 1. Why this file exists

The claim under audit is: *same-machine determinism is proven, the platform is
declared and hashed in every artifact, and the boundary is publicly checkable.*

The middle clause was false. The container digest (`sha256` in the record,
re-hashed by `--integrity-only` and by `interop/triosckp_reader.py`) covers the
**weights**. The platform block, the corpus digests, the trainer digest,
`git_sha`, `git_dirty`, `steps_total`, `eval_every`, `final_val_bpb` and the
`schema` tag live in the sidecar and were covered by **nothing**. They are free
text written by the party being audited - and a conformity scheme's adversary is
a vendor with a certificate to obtain.

That was demonstrated, not argued. The headline cross-architecture record
`evidence/xarch-run-30767491098/12000.json` - produced by GitHub Actions run
30767491098 on **x86_64 Linux** - was copied to a scratch directory and its
sidecar rewritten to claim `aarch64` / `macos`, a rustc string that never
existed, and `final_val_bpb = 1.5492`, the exact number this project publicly
retracted. The container bytes were not touched. Both verifiers passed it.

Verifier 1, verbatim (last line of a 60-line report; the `[...]` is a cut for
length and nothing else):

```text
$ python3 interop/triosckp_reader.py <tmp>/12000.bin --sidecar <tmp>/12000.json
...
scope           ECHOED from the sidecar, NOT verified against the container (23): canon_name, path, ema_bpb, git_sha, git_dirty, corpus, run_id, ledger, ts, steps_total, gf16_floor_every, eval_every, final_val_bpb, git_provenance, source_sha256, trainer, best_val_bpb, platform.os, platform.arch, platform.pointer_width, platform.libc, platform.toolchain, platform.toolchain_provenance
RESULT PASS  container ok, sha256 bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3; derived attention layers declared 2 / effective 1, [...]
reader exit=0
```

Verifier 2, verbatim and complete:

```text
$ ./target/release/ckpt_replay --record <tmp>/12000.json --integrity-only
INTEGRITY OK: "<tmp>/12000.json" (schema trios-checkpoint-record/4)
  artifact: "<tmp>/12000.bin" (852272 bytes)
  resolved: published beside the record (<stem>.bin)
  sha256:   bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3 - re-hashed from disk, matches the record
  NOT A REPRODUCTION: nothing was re-derived and no trainer was executed. This says the published bytes are unchanged, and says nothing about whether the recipe in this record produces them; run ckpt_replay without --integrity-only to ask that, at the cost of the run.
exit=0
```

Neither verifier is wrong. The bytes were unaltered and both said so. Both were
**silent about the half of the record the claim actually rests on**, and the
Python reader's own honest "ECHOED, unverifiable here" line was, in practice,
printed one line above `RESULT PASS`.

## 2. What changed

1. `provenance_seal(record)` - one digest over the declaration (spec in section 3).
2. `ckpt_replay --provenance-seal` prints it; exits 0; verifies nothing.
3. `ckpt_replay --expect-provenance-seal sha256:...` makes it a check.
   A disagreement is `SEAL MISMATCH`, exit 1, and nothing else is run.
4. Every `INTEGRITY OK` and every `VERIFIED` now carries an adjacent,
   unconditional line naming the sealed fields and calling them an
   **UNAUTHENTICATED DECLARATION** unless `--expect-provenance-seal` was
   supplied and matched. `INTEGRITY OK` never appears alone again.
5. `interop/triosckp_reader.py` prints the same seal on every run, from an
   independent implementation. The two agreeing is the interop check.

The forgery above, after the change (verbatim, run 2026-08-06):

```text
$ ./target/release/ckpt_replay --record <tmp>/12000.json --integrity-only --expect-provenance-seal sha256:fbd96023...
SEAL MISMATCH: record "<tmp>/12000.json" declares sha256:d217a185cf3dc93afe4c15969e23ba857bfcd6846e99aa9c93409741c295980c, you expected sha256:fbd960235a100ac77ee3283cb29b869fbf1aaba1e460b6cb2a20c2a5a31e748d
  the declaration - platform block, corpus and trainer digests, git_sha, git_dirty, seed, step, steps_total, eval_every, gf16_floor_every, final_val_bpb, schema, and the artifact this record names (sha256, bytes) - is not the one that seal was published for. Nothing was graded. [...]
exit=1
```

and without the flag, the byte verdict is still a pass - it is the same bytes -
but it can no longer be quoted alone:

```text
INTEGRITY OK: "evidence/xarch-run-30767491098/12000.json" (schema trios-checkpoint-record/4)
  ...
UNAUTHENTICATED DECLARATION: the verdict above grades BYTES. These 27 field(s) are written by the party being audited and NOTHING above checked them - pass --expect-provenance-seal sha256:fbd96023... to make this line a check instead of a quote: bytes, corpus.train.bytes, ..., platform.arch, ..., sha256, ..., trainer.sha256
```

### 2.1 Round 7: the seal did not name the artifact

The first version of this seal left `sha256` and `bytes` OUT, and section 5
defended that with the sentence *"sealed by the container digest already"*. It
is circular. `sha256` **is** the container digest; a digest cannot seal itself.
The consequence was not theoretical: because the seal named no artifact, one
authenticated declaration fitted **every** container. Take the genuine x86_64
Linux record, change nothing in its declaration, and point it at the aarch64
macOS `.bin` - both are 852272 bytes - by rewriting `sha256` and `bytes` to
describe the file it now sits beside. Both shipped verifiers passed, the record
was internally consistent, and the seal still equalled the digest published for
the Linux run. That forgery manufactures the exact claim this project rests on
- **cross-architecture bit identity** - out of an authenticated declaration.

`sha256`, `bytes`, `seed`, `step` and `gf16_floor_every` are therefore inside
the seal as of 2026-08-06, and **every digest published before that date is
superseded**. The same swap now:

```text
$ ./target/release/ckpt_replay --record <tmp>/12000.json --integrity-only \
    --expect-provenance-seal "$(grep xarch-run-30767491098 evidence/SEALS.txt | cut -f2)"
SEAL MISMATCH: record "<tmp>/12000.json" declares sha256:5be5ed6a949bb60f7201bff366cc7250575dfe8febfa649bc139218ee3b55ad3, you expected sha256:fbd960235a100ac77ee3283cb29b869fbf1aaba1e460b6cb2a20c2a5a31e748d
exit=1
```

Two adjacent holes closed with it:

* `gf16_floor_every` is present in every schema/4 record, demonstrably changes
  the weights (`src/train_loop.rs` applies the floor on an eval-cadence gate),
  and was in neither the container header nor the sealed set. Section 5 named
  it "the sharpest remaining hole in this list" and it is now sealed.
* `ckpt_replay --integrity-only` never compared the record's `bytes` against
  the file length, while `interop/triosckp_reader.py` always did. A record
  edited to say `"bytes": 1` printed `INTEGRITY OK` in Rust and
  `SIDECAR_MISMATCH` in Python. Two verifiers disagreeing about one record is
  the defect - and the one reporting SUCCESS was the weaker. Rust now prints
  `ARTIFACT ALTERED`, exit 1.

## 3. The seal, normatively

The normative text is the module documentation of `src/provenance_seal.rs`;
this is a summary and the module wins any disagreement.

**Sealed fields.** `platform.os`, `platform.arch`, `platform.toolchain`,
`platform.libc`, every other key present under `platform`, plus `bytes`,
`corpus`, `eval_every`, `final_val_bpb`, `gf16_floor_every`, `git_dirty`,
`git_sha`, `schema`, `seed`, `sha256`, `source_sha256`, `step`, `steps_total`,
`trainer`.

**Rendering.** Sorted `dotted.key=value` lines, one per scalar, each terminated
by `\n`. Objects and arrays are flattened (`corpus.train.sha256=...`), so there
is no canonical-JSON question to argue about. Strings verbatim with `\`, LF and
CR escaped; booleans `true`/`false`; integers decimal; other numbers via Rust
`{}` on `f64`. A field the record does not carry renders `key=<absent>`, so
**deleting a field changes the seal**.

**Seal.** `sha256:` plus lowercase hex SHA-256 of those bytes.

Worked example - the complete listing of the headline record, whose seal is
`sha256:fbd960235a100ac77ee3283cb29b869fbf1aaba1e460b6cb2a20c2a5a31e748d`:

```text
bytes=852272
corpus.train.bytes=1015394
corpus.train.path=data/tiny_shakespeare.txt
corpus.train.sha256=1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d
corpus.val.bytes=100000
corpus.val.path=data/tiny_shakespeare_val.txt
corpus.val.sha256=2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502
eval_every=1000
final_val_bpb=2.637763500213623
gf16_floor_every=1
git_dirty=false
git_sha=3c1f751cf4376c13d26e247c2cd86357ab51dd20
platform.arch=x86_64
platform.libc=gnu
platform.os=linux
platform.pointer_width=64
platform.toolchain=rustc 1.96.0 (ac68faa20 2026-05-25)
platform.toolchain_provenance=runtime-path-query
schema=trios-checkpoint-record/4
seed=47
sha256=bb14ab18f2c8e7a9a4c19f452471018f3a72cc18765175db44858c2c4e5c03f3
source_sha256=19aa22fb7cd187774b71cbde7cb89b664aff7260b57f55afd865dd447707108b
step=12000
steps_total=12000
trainer.path=/home/runner/work/trios-trainer-igla/trios-trainer-igla/target/release/trios-train
trainer.provenance=self-hashed
trainer.sha256=8528b1fee99a411fcbf6db3510593b966f86175ce658faaa681024be236eecdd
```

## 4. The published seals

**The machine-readable table is `evidence/SEALS.txt`**; the prose below renders
from it and the file wins any disagreement. One `<record path><TAB>sha256:<seal>`
line per tracked sidecar under `evidence/`, which is exactly the record list
`.github/workflows/ckpt-replay-audit.yml` iterates - so a record that is
published without a seal fails the audit rather than being graded quietly.

Every tracked sidecar under `evidence/`. Both implementations agree on all
fourteen (see section 6 for the command that proves it). These digests are what
a later forgery is detectable against - and only for a reader who obtained them
from somewhere the forger does not control.

**Reminted 2026-08-06.** `sha256`, `bytes`, `seed`, `step` and
`gf16_floor_every` entered the sealed set (section 2.1), so every seal published
before that date is superseded. If you are holding an older digest, it will not
match, and that is the change working rather than a tamper.

| record | schema | declared platform | steps | sealed fields | seal |
|---|---|---|---|---|---|
| `evidence/heldout/r6-heldout-select/12000.json` | trios-checkpoint-record/8 | macos/aarch64 | 12000 | 34 | `sha256:ecbb311ff04d238b64ec4218dfde0b00e205d9eb1186f3575c9ab485eb87ef96` |
| `evidence/heldout/r6-heldout-test/12000.json` | trios-checkpoint-record/8 | macos/aarch64 | 12000 | 34 | `sha256:9edeb4bde2bc30d0e673073c23b640d90586944d8d769cab7ff139f0eb6a902c` |
| `evidence/xarch-aarch64-reference/12000.json` | trios-checkpoint-record/4 | macos/aarch64 | 12000 | 27 | `sha256:ba83ebc2b6585da1cdb29f57836ac1d681674cda942a464f2ba59cb37a652a76` |
| `evidence/xarch-local-isa/isa-probe-arm64-0.json` | trios-checkpoint-record/8 | macos/aarch64 | 10 | 34 | `sha256:1dacd92ce0a9fd62cf233634b7ebe3e587d689395138f8f859a179b6d2d32988` |
| `evidence/xarch-local-isa/isa-probe-arm64-10.json` | trios-checkpoint-record/8 | macos/aarch64 | 10 | 34 | `sha256:a149633516924f7716fdd4841e6ce0cc22b9cc1665f70fa6704a1fd3bb7229c5` |
| `evidence/xarch-local-isa/isa-probe-x86_64-0.json` | trios-checkpoint-record/8 | macos/x86_64 | 10 | 34 | `sha256:8304ec46248cadcbf873bd1d1b6f7470b648e07b3034c6dc43260fc561cf06c9` |
| `evidence/xarch-local-isa/isa-probe-x86_64-10.json` | trios-checkpoint-record/8 | macos/x86_64 | 10 | 34 | `sha256:2d082292e5d26b191945879b4548b9ddb3c9b9b695307f6c377c6e5b7592296b` |
| `evidence/xarch-local-isa/probe.json` | trios-local-isa-probe/1 | no `platform` block | (not a checkpoint record) | 19 | `sha256:b42eb2377b25c4143511761b3667fa0cd8500029b311842539e06621a811ac07` |
| `evidence/xarch-run-30767491098/12000.json` | trios-checkpoint-record/4 | linux/x86_64 | 12000 | 27 | `sha256:fbd960235a100ac77ee3283cb29b869fbf1aaba1e460b6cb2a20c2a5a31e748d` |
| `evidence/xarch-rustc/linux-rustc191/lin191/300.json` | trios-checkpoint-record/2 | no `platform` block | 300 | 23 | `sha256:91889a74c1d947ff8892fd4a3c5638bf421687b196e91faaf3c6d7d9532c9d18` |
| `evidence/xarch-rustc/linux-rustc191/lin3k/3000.json` | trios-checkpoint-record/2 | no `platform` block | 3000 | 23 | `sha256:7122f7557fe4672e5866e14803c7769652af1eb5862e002febe66f493ac88039` |
| `evidence/xarch-rustc/linux-rustc196/lin196/300.json` | trios-checkpoint-record/2 | no `platform` block | 300 | 23 | `sha256:91889a74c1d947ff8892fd4a3c5638bf421687b196e91faaf3c6d7d9532c9d18` |
| `evidence/xarch-rustc/macos/mac/xarch-mac/300.json` | trios-checkpoint-record/2 | no `platform` block | 300 | 23 | `sha256:2c0a560cb1c0e1d68023de1eb17f5252d8b2d40b9ca15f4a9b3bd272f51c4c55` |
| `evidence/xarch-rustc/macos/mac3k/mac3k/3000.json` | trios-checkpoint-record/2 | no `platform` block | 3000 | 23 | `sha256:061278bc1a055638e77cf9d53bc06cef719124c4cff7a9bdb586f3235a3a6c9c` |

`probe.json` is a `trios-local-isa-probe/1` document, not a checkpoint record;
it publishes no `.bin` and `--integrity-only` SKIPs it. It is sealed anyway,
because the seal is defined over any record object and a document that is
allowed to have no seal is a document that can be edited for free.

### 4.1 Two records seal identically, and that is a finding

`linux-rustc191/lin191/300.json` and `linux-rustc196/lin196/300.json` both seal
to `sha256:91889a74...`. That is correct, not a collision: the two records
differ in exactly three fields - `canon_name`, `path` and `ts` - none of which
is sealed, and the two experiments produced a byte-identical checkpoint
(`9866cb2a...`). The round-7 additions did not separate them and could not:
`sha256`, `bytes`, `seed` and `step` are equal across the pair precisely
because the checkpoint is the same.

The interesting part is what the seal makes visible. Those two runs exist to
compare **rustc 1.91 against rustc 1.96**, and neither record states a rustc
version at all: both carry `"platform": null`, so the seal renders
`platform.toolchain=<absent>` for each. The variable the experiment was about
is recorded **only in the directory name**. Nothing in either artifact
distinguishes them. Any citation of that pair has to say so.

## 5. What is NOT sealed

Stated in full, because a seal that is quoted as covering the record when it
covers a subset is the same defect this file exists to close.

**Sealed here AND cross-checked against the container**: `sha256`, `bytes`,
`seed`, `step`. The earlier text put these in a "sealed by the container digest
already" bucket and left them out of the seal. That was wrong twice over.

1. It is circular for `sha256`: that field **is** the container digest, and a
   digest does not seal itself. Re-hashing the `.bin` proves the record's
   `sha256` describes the file beside it, and says nothing about whether that
   file is the one the declaration was published for.
2. A seal that names no artifact binds its declaration to **no particular
   `.bin`**. Section 2.1 shows the swap that follows: the honest x86_64 Linux
   declaration attached to the aarch64 macOS weights, both verifiers passing,
   the seal still matching the published digest for the Linux run.

`seed` and `step` join them because they name which artifact the recipe was
meant to produce. They are cross-checked against the decoded header by
`interop/triosckp_reader.py`, which catches a record that disagrees with the
bytes beside it - and did not catch a record that had been moved to sit beside
different bytes.

**Checked against the container header, and NOT sealed**: `format_version`,
`hidden`, `d_model`, `num_attn_layers`, `optimizer`, `fake_quant_format`,
`data_synthetic` (`SIDECAR_CHECKS`), plus `lr`, `attn_scale`, `attn_seq`,
`vocab` and `gf16_enabled` when the record's schema carries them
(`SCHEMA_HEADER_CHECKS`). These are compared
field by field against the decoded header by `interop/triosckp_reader.py`, so
editing one is detectable by that reader without any published digest. That is
a weaker guarantee than the seal (it holds only for whoever runs the Python
reader, and only for fields the container actually carries), and it is stated
here rather than folded into the sealed set so the two mechanisms are not
confused for one.

**Neither sealed nor container-checked - still forgeable in silence:**

| field | why it matters |
|---|---|
| `ema_bpb`, `best_val_bpb` | metrics. `final_val_bpb` is sealed; these two are not, so a record can still be re-labelled on a secondary number. |
| `canon_name`, `path`, `run_id`, `ts` | bookkeeping. Unsealed on purpose (they are local to the training machine), which is what section 4.1 turns on. |
| `ledger` | checked against a closed vocabulary by the Python reader, so an unknown value fails there - but its VALUE is not sealed. |
| `git_provenance`, `git_untracked` | statements about the tree, not sealed. `git_sha` and `git_dirty` are. |
| `eval_chunks`, `eval_tokens`, `eval_seq`, `val_bpb_stderr`, `optimizer_params`, `min_observed_val_bpb`, `format_faithful` | schema 6/9 fields present on the newer records and outside the sealed set. |

Extending the sealed set is a one-line change in two tables
(`SEALED_TOP_LEVEL` in `src/provenance_seal.rs` and in
`interop/triosckp_reader.py`) - and it invalidates every digest in
`evidence/SEALS.txt` and section 4, which is why it is a deliberate act and not
a drive-by edit. It was done once, on 2026-08-06, for the reasons in section
2.1; the old digests are superseded, not tampered with.

**And the boundary again:** the seal is not a signature. `evidence/SEALS.txt`
and section 4 are files in a git repository the vendor controls. Their value
comes entirely from a reader having seen them before the record they are
auditing.

### 5.1 `platform.rustflags_sha256` gates nothing

`ckpt_replay` prints `platform.rustflags_source` and
`platform.rustflags_sha256` under a `TRAINER MISMATCH` verdict. It never
**compares** them, and it could not: that digest hashes the `.cargo/config.toml`
that `scripts/repro_build.sh` writes, which is gitignored and carries the
training host's absolute paths (`/Users/playra/...` for the local records). A
fresh clone cannot produce that file, so no third party can ever match the
digest. Measured, not assumed: the `rustflags_sha256` of the heldout records is
`b18ac6a3a1eaa5f2126007da415a13b27913ec13f0412df94a9baaa7a8e8ac58`, and
`shasum -a 256 .cargo/config.toml` on the machine that wrote them returns the
same digest for a file that is 1040 bytes, matched by `.gitignore` line 34
(`.cargo/`), and contains three literal `/Users/playra/...` paths. It is now
printed under a heading that says so. It is a hint about
which flags to rebuild with, and nothing else.

## 6. How to check this yourself

```bash
cargo build --release --bin ckpt_replay
cargo test  --release --test provenance_seal_tamper

# Both implementations AND the published table, every tracked record. A
# disagreement between the two implementations means the SPECIFICATION is
# wrong, not the record; a disagreement with evidence/SEALS.txt means the
# record or the table moved.
for j in $(git ls-files evidence | grep '\.json$'); do
  R=$(./target/release/ckpt_replay --record $j --provenance-seal \
      | grep -o 'sha256:[0-9a-f]*' | head -1)
  P=$(python3 interop/triosckp_reader.py ${j%.json}.bin --sidecar $j 2>/dev/null \
      | grep -o 'sha256:[0-9a-f]*' | tail -1)
  T=$(awk -F'\t' -v p="$j" '$1 == p {print $2}' evidence/SEALS.txt)
  test -n "$T"      || { echo "UNPUBLISHED $j (no line in evidence/SEALS.txt)"; exit 1; }
  test "$R" = "$P"  || { echo "DISAGREE $j rust=$R python=$P"; exit 1; }
  test "$R" = "$T"  || { echo "UNSEALED  $j declares=$R published=$T"; exit 1; }
  echo "AGREE $j $R"
done

# The declaration, checked against the published table rather than a digest
# copied by hand out of prose.
./target/release/ckpt_replay --record evidence/xarch-run-30767491098/12000.json \
  --integrity-only \
  --expect-provenance-seal \
    "$(awk -F'\t' '$1 ~ /xarch-run-30767491098/ {print $2}' evidence/SEALS.txt)"
```

## 7. Related

- `src/canonical_digest.rs` - a digest over WEIGHTS under a signed-zero
  normalisation rule. A different object; deliberately not merged with this one.
- `docs/CROSS-ARCH-DIVERGENCE.md` - the finding that makes the platform
  declaration load-bearing in the first place: a checkpoint is not byte-portable
  across CPU architectures, so a verdict is only meaningful with its platform
  stated.
- `docs/REPRODUCIBILITY-GRADING.md` - the L2/L3 ladder the verdicts use.
