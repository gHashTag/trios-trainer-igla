# Interlaboratory comparison, TRIOSCKP format version 1

This directory holds a second, independent implementation of the TRIOSCKP
checkpoint container and the record of what comparing it against the first
implementation did and did not establish.

- `SPEC-SNAPSHOT.txt` - the frozen specification text (lines 1-60 of
  `src/checkpoint.rs`), sha256
  `d9e99c298296a1741f36d0e3ae009db3d1bfce6601dcb3c0a4f3ebb6a4fd12fe`.
- `triosckp_reader.py` - a Python 3 standard-library reader written from that
  text alone.

Method rule, and the only reason this exercise means anything: the Rust encoder
and decoder were not read while writing the Python. Specifically,
`to_checkpoint_bytes` / `from_checkpoint_bytes` in `src/train_loop.rs` and the
bodies of `save` / `load` in `src/checkpoint.rs` were not opened. Where the
specification was ambiguous, the ambiguity was resolved by choosing an
interpretation and recording it below, never by consulting the Rust. The
sidecar JSON records were read, because they are the artifacts under test.

Environment: Darwin 25.5.0 arm64, Python 3.14.6, repository at
`77eba48aac933c067e87b7fc7ed27e8865926bd8` (working tree dirty).

---

## 1. Repeatability is not reproducibility

These are distinct terms of art in metrology, and 243-FZ does not distinguish
them. It says "reproducibility of the development cycle." Anyone reading that
sentence as a metrologist will ask which of the two is meant, and the honest
answer today is: only the weaker one has data.

**Repeatability** - same operator, same implementation, same equipment, same
laboratory, short interval. What has been demonstrated: running the trainer
twice with identical parameters produces byte-identical checkpoint files, and
the reported BPB is identical between those runs. In this session that claim was
re-derived from the outside: `checkpoints/det-a/4000.bin` and
`checkpoints/det-b/4000.bin` are two independent runs and this Python reader,
hashing each file as read back from disk, computed the same digest
`ef6f08875a008373c254d8e33e0474732db5a0165a05a026cb41b6d841a86926` for both.

**Reproducibility** - different implementation, different operator, different
equipment, different laboratory, agreement within a stated tolerance. This has
**not** been demonstrated for the trainer. There is one trainer, one author, one
machine, and no stated tolerance, because with bit-determinism the tolerance has
never had to be stated. A measurement procedure with a single implementation and
no interlaboratory comparison has no established uncertainty. That is not a
rhetorical objection; it is the definition of an unvalidated method.

This directory closes a strictly smaller gap: it provides a second, independent
implementation of the **artifact format**, and an interlaboratory comparison at
the level of decoding. It is the first data point of any kind on the
reproducibility side of the line, and it should be presented as exactly that
size and no larger.

## 2. What this reader establishes, and what it does not

Establishes:

- The TRIOSCKP container is specified well enough that a second implementer,
  working from the prose and forbidden from reading the encoder, produced a
  reader that decodes every field of every existing checkpoint and agrees with
  the first implementation's own recorded metadata on all cross-checked fields.
  Ambiguities were found (section 4); none of them blocked the decode.
- The checkpoint files on disk are internally consistent: magic, declared
  version, declared header length, declared tensor count, the reserved-byte
  rule, the equal-count rule for `attn_down` / `attn_up`, and the length
  identity `file_len = 304 + 4 * sum(directory counts)` all hold, computed
  independently.
- The SHA-256 digests recorded in the sidecar records are correct digests of the
  files as they exist on disk right now, computed by a different hash
  implementation in a different language.

Does **not** establish:

- Anything about training. This is an independent *decode of an artifact*, not
  an independent *re-execution of the training run*. No weights were compared
  against weights produced by a second trainer, because no second trainer
  exists. Nothing here shows that running the pipeline again elsewhere yields
  these bytes.
- Anything about the numbers the sidecars carry that are not derivable from the
  container. The sidecar for `checkpoints/igla-honest-provenance/12000.json`
  records `bpb` 2.616914749145508 and `ema_bpb` 2.920900821685791; the sidecar
  for `checkpoints/det-a/4000.json` records `bpb` 2.7727296352386475. This
  reader does **not** verify those - it cannot, because BPB is not stored in the
  container and recomputing it would require an independent forward pass. They
  are quoted here only to be explicit that they are outside the scope of this
  instrument.
- Payload integrity below the length check. There is no digest inside the
  container itself (ambiguity A7). A single flipped bit inside the payload would
  pass every check this reader performs, and would only be caught by comparing
  against the externally recorded SHA-256.

## 3. Measured result

Command 1, full decode and sidecar cross-check of the honest-provenance final
checkpoint:

```
$ python3 interop/triosckp_reader.py checkpoints/igla-honest-provenance/12000.bin \
      --sidecar checkpoints/igla-honest-provenance/12000.json
```

Decoded header (exit 0): `format_version` 1, `header_len` 152, `bytes` 852272,
`sha256` `c1b1a12b914924ac7d14303227124326501d61b94bf338621670ec79489b6f0e`,
`vocab` 128, `dim` 64, `num_ctx` 6, `hidden` 384, `d_model` 64, `num_heads` 4,
`attn_cfg_seq_len` 8, `num_attn_layers` 2, `ngram` 8, `tensor_count` 19,
`qk_gain` 2.618033988749895, `attn_cfg_lr` 0.0035, `train_lr`
0.003000000026077032, `attn_scale` 0.10000000149011612, `attn_seq` 8,
`ctx_weights` [0.699999988079071, 0.44999998807907104, 0.30000001192092896,
0.20000000298023224, 0.12999999523162842, 0.07999999821186066], `seed` 47,
`step` 12000, `gf16_enabled` 1, `data_synthetic` 0, `optimizer` `adamw`,
`fake_quant_format` `f32`. Directory total 212992 elements, payload 851968
bytes, container prefix 304 bytes. All eleven cross-checked fields agree with
the sidecar.

Command 2, every pair in the tree:

```
$ python3 interop/triosckp_reader.py --verify-all checkpoints/
```

9 pairs, 9 passed, 0 failed, exit 0:

| file | sha256 (computed by this reader) |
| --- | --- |
| `checkpoints/det-a/4000.bin` | `ef6f08875a008373c254d8e33e0474732db5a0165a05a026cb41b6d841a86926` |
| `checkpoints/det-b/4000.bin` | `ef6f08875a008373c254d8e33e0474732db5a0165a05a026cb41b6d841a86926` |
| `checkpoints/igla-honest-20260802/3000.bin` | `de2595a964da65df292251cf9addca68724c104216ccc33268fedbd34ecb7fd9` |
| `checkpoints/igla-honest-20260802/6000.bin` | `060ac8dc118d6adb31318936323a4a363a7e36da3c43363014b27e0e96a6906c` |
| `checkpoints/igla-honest-20260802/9000.bin` | `63e8597a3232d5f21741dc90562b8bd15422d98a58ab18809dc94cfbb81a144f` |
| `checkpoints/igla-honest-20260802/12000.bin` | `b2fe52b0760dc36923fc54b41b9e3bbaed50bd2169b9c73095a0d8f89ab91d5e` |
| `checkpoints/igla-honest-provenance/4000.bin` | `9eb162017b53edc143e938ee8ca51dc7a356ce9aa569b7a096db0ddb6efe9ffa` |
| `checkpoints/igla-honest-provenance/8000.bin` | `c27ed8fc8569282356ba4e70cad1f6e8f971217d1d6874700af4af20984b3328` |
| `checkpoints/igla-honest-provenance/12000.bin` | `c1b1a12b914924ac7d14303227124326501d61b94bf338621670ec79489b6f0e` |

The det-a / det-b identity in rows 1 and 2 is the bit-determinism result,
re-derived here by a different tool: two independent runs, one digest.

Negative controls. Every one was constructed in this session from
`checkpoints/det-a/4000.bin` and every one exited 1 with a named error:

| fixture | result |
| --- | --- |
| first 1000 bytes only | `TRUNCATED_PAYLOAD: directory declares 212992 elements, so file_len must be 852272; file is 1000 bytes (851272 missing)` |
| four extra zero bytes appended | `TRAILING_GARBAGE: ... file is 852276 bytes (4 extra)` |
| byte at offset 126 set to 1 | `RESERVED_NONZERO: the 2 reserved bytes at offset 126 must be zero, found b'\x01\x00'` |
| magic overwritten | `BAD_MAGIC: expected b'TRIOSCKP' at offset 0, found b'NOTACKPT'` |
| format_version set to 2 | `UNSUPPORTED_FORMAT_VERSION: this reader implements version 1, file declares 2` |
| non-NUL byte in optimizer padding | `NONZERO_STRING_PADDING: optimizer padding after NUL is not all zero` |
| `det-a/4000.bin` paired with `igla-honest-provenance/12000.json` | `SIDECAR_MISMATCH: sha256 ...; step: sidecar=12000 decoded=4000` |

Truncation and trailing garbage are reported as separate named errors, as
required: they are different failure modes and a container that conflates them
cannot tell a partial write from an appended payload.

## 4. Specification ambiguities found

These are the deliverable. A format specification that a second implementer
cannot follow unaided is defective, and finding out where is the entire purpose
of an interlaboratory comparison. Quotations are verbatim from
`SPEC-SNAPSHOT.txt`.

**A1. Ten header integers have no declared type.** The spec writes explicit
types for some fields and not others. It declares `80   4  attn_seq: u32`,
`108   8  seed: u64`, `124   1  gf16_enabled: u8`, `56   8  qk_gain: f64 bits`,
`72   4  train_lr: f32 bits`. But offsets 16 through 55 are written as bare
names with widths only:

> ```
>   16   4  vocab      20   4  dim         24   4  num_ctx
>   28   4  hidden     32   4  d_model     36   4  num_heads
>   40   4  attn_cfg_seq_len                44   4  num_attn_layers
>   48   4  ngram      52   4  tensor_count = 19
> ```

Signedness is unstated for all ten. This reader assumed unsigned
little-endian u32. The assumption is unfalsifiable from the corpus, since every
observed value is small and positive, but it is a real fork: an `i32` of -1 and
a `u32` of 4294967295 are the same four bytes, and a reader that guesses wrong
reports a plausible-looking absurd dimension instead of an error.

**A2. `header_len` and the directory length are both 152, and the file_len
formula hard-codes 304.** Line 16 draws the layout as
`header (152 bytes) | tensor directory (152 bytes) | payload`, line 24 defines
`header_len: u32 = 152 (absolute offset of the tensor directory)`, and line 57
states `file_len = 304 + 4 * sum(directory counts)`. The two 152s are a
coincidence of version 1 (19 * 8 = 152). A reader that treats the declared
`header_len` as also giving the directory size, or that keeps 304 as a literal,
works today and breaks silently the moment the header grows. The spec never says
that 304 is `header_len + 8 * tensor_count`. This reader rejects any
`header_len` other than 152 rather than honoring the declared value, because the
spec fixes it at 152 for version 1 and gives no rule for interpreting any other
value.

**A3. The payload encoding is never stated.** The directory is described as
"19 consecutive `u64` element counts (f32 counts, not byte counts)" and the
length identity multiplies the sum by 4, which together *imply* contiguous
4-byte little-endian f32 elements in canonical directory order with no
per-tensor framing. The spec never says so. This reader does not decode tensor
values, so the assumption costs nothing here, but a second implementer writing a
*writer*, or a reader that compares weights, would be guessing at the one thing
the format exists to carry.

**A4. The two u8 flags have no declared value domain.** The spec is emphatic
about the reserved bytes - `126   2  reserved (must be zero, rejected if
nonzero)` - and silent about whether `gf16_enabled = 7` is a legal encoding of
true or a corrupt file. This reader accepts any u8 and reports the raw value.
Observed across all nine files: `gf16_enabled` 1, `data_synthetic` 0.

**A5. The directory's redundancy cannot actually be checked.** The spec says the
directory

> is purely redundant with the header scalars; it exists so that a load which
> would produce a differently shaped model fails loudly instead of reshaping
> silently.

but gives no formula relating any entry to any scalar. A second implementer
therefore cannot perform the exact check the directory was added for. The only
cross-check derivable from the text alone is the one lines 53-55 state
outright - that `attn_down` (8) and `attn_up` (9) have identical element counts -
and this reader implements it; it holds on all nine files. The observed counts
(`embed` 8192, `proj` 24576, `lm_head` 49152, `wq` 4096, with `vocab` 128,
`d_model` 64, `hidden` 384) are clearly products of the header scalars, but
which product belongs to which tensor is guesswork, and guessing is precisely
what this exercise must not do. **This is the most consequential gap: the
integrity feature the spec advertises is not specified.**

**A6. The sidecar record is not in the specification at all.** The frozen text
describes the binary container and nothing else. It never mentions the sidecar,
its schema, the `<step>.json` beside `<step>.bin` naming convention, which of
its fields mirror header fields, or - critically - that its `sha256` covers the
whole file rather than the payload or the weights. All of that was inferred
from the artifacts. The sidecar is the object that actually carries the
provenance claim to a third party, and it is the undocumented half of the
format.

The record has since been versioned. The tag lives in the record's own `schema`
field and is currently `trios-checkpoint-record/3`; the container's
`format_version` is still 1 and is deliberately NOT bumped in step, because the
sidecar is unhashed evidence and rewriting the container spec for a JSON field
would invalidate every archived artifact hash for nothing.

| schema | what it added |
| --- | --- |
| `/1` | the original record: identity (`canon_name`, `seed`, `step`, `path`), integrity (`sha256`, `bytes`, `format_version`), architecture (`hidden`, `d_model`, `num_attn_layers`), `optimizer`, `fake_quant_format`, `data_synthetic`, `bpb`, `ema_bpb`, `git_sha`, `git_dirty`, `corpus`, `run_id`, `ledger`, `ts` |
| `/2` | `steps_total`, `gf16_floor_every`, `eval_every`, `git_provenance`; renamed `bpb` -> `final_val_bpb` and added `best_val_bpb`; widened `git_dirty` to nullable |
| `/3` | `lr`, `attn_scale`, `attn_seq`, `platform` (`os`, `arch`, `pointer_width`, `libc`, `toolchain`, `toolchain_provenance`), `source_sha256` |

**The nine archived checkpoints under `checkpoints/` are all schema `/1`.** They
carry none of the `/2` and none of the `/3` fields. For an auditor holding only
those artifacts, that costs, precisely:

- **No learning rate.** Two runs an order of magnitude apart in `lr` produce
  archived records that differ only in `sha256` and in the BPB. The value is
  recoverable here only because the CONTAINER happens to carry `train_lr` at
  offset 72 - the record itself never claimed it, and an auditor reading the
  record alone cannot reconstruct the run.
- **No declaration of the architecture env vars.** `attn_scale` and `attn_seq`
  are set from `TRIOS_ATTN_SCALE` and `TRIOS_ATTN_SEQ`, defaults 0.1 and 8. The
  same remark applies: offsets 76 and 80 preserve them, the schema `/1` record
  does not, so nothing in the evidence file says an environment variable was
  able to change the architecture.
- **No platform at all.** This is the expensive one. The cross-libc experiment
  showed the same seed, the same corpus and the same source tree producing
  DIFFERENT checkpoint hashes on macOS versus glibc. An auditor who fails to
  reproduce `ef6f0887...` on their own machine cannot tell, from a schema `/1`
  record, whether the method is broken or whether they are simply on the other
  libc - and neither can we.
- **No `source_sha256`.** Every archived record has `git_dirty: true`. A commit
  hash over a tree that did not match it is honest but not reconstructive: it
  names code that is not the code that ran. Schema `/3` hashes the tracked and
  dirty source together, so the record self-describes its own code; the
  archived records cannot.
- **No `steps_total` / `gf16_floor_every` / `eval_every`** (the `/2` gap, stated
  here for completeness): `gf16_floor()` rewrites weights in place past the 70%
  mark, so its cadence is part of the recipe, and until `/2` nothing recorded
  it.

`interop/triosckp_reader.py` therefore dispatches on FIELD PRESENCE rather than
on matching the `schema` string, accepts `/1`, `/2` and `/3` alike, and prints
for every record which of the five schema `/3` provenance fields are absent. On
the nine archived pairs it prints all five as absent, which is the correct and
unflattering answer.

**A7. There is no digest inside the container.** The integrity of a checkpoint
rests entirely on an external assertion in a separate file. A `.bin` handed over
on its own is unverifiable beyond its length identity: the negative controls
above were caught only because they landed in fields with declared constraints,
and a single flipped bit inside the 851968-byte payload would pass every check
in this reader.

**A8. `attn_cfg_seq_len` (offset 40) and `attn_seq` (offset 80) are two fields
whose relationship is undeclared.** The spec disambiguates exactly one such
collision, on the adjacent field - `attn_cfg_lr: f64 bits (HybridAttnConfig::lr,
NOT the training lr)` - and does nothing similar for the two sequence lengths.
Observed: both are 8 in all nine files, which is consistent with them being the
same quantity stored twice and equally consistent with them being different
quantities that happen to coincide.

**A9. `f32` and `f64` bit patterns are declared to round-trip NaN payloads and
signed zeros** (lines 10-12), which is correct and worth keeping, but no field is
declared to have a valid range. A `train_lr` of NaN decodes without complaint.
This reader reports raw values and does not range-check, since the spec gives no
ranges to check against.

None of A1-A9 prevented a successful decode. That is the finding: the container
half of the format survived independent implementation, and the specification
should be amended on all nine points before anyone is asked to rely on it.

## 5. Scope limits that must be stated whenever this result is cited

- **Single node, single platform, single thread of execution.** Everything above
  was produced on one machine (Darwin 25.5.0, arm64). Cross-platform
  reproducibility is untested. Multi-threaded reproducibility is untested. GPU
  reproducibility is untested - there is no GPU path in evidence here.
  Multi-node reproducibility is untested.
- **Single-node bit-determinism is a necessary precondition, not the property
  being claimed.** A pipeline that is not deterministic on one machine cannot
  possibly be reproducible across laboratories. Being deterministic on one
  machine implies nothing about the other direction: floating-point reduction
  order, library versions, CPU instruction selection and thread scheduling all
  change results, and none of them have been varied.
- **This is a decode comparison, not a re-execution comparison.** A second
  implementation of the *reader* is not a second implementation of the
  *trainer*. The stronger claim - two independent trainers, agreement within a
  stated tolerance - has no data behind it and should not be implied.
- **Observation parameters are not all inert.** `--eval-every` is known to
  change the resulting weights on this codebase, so runs that differ only in
  eval cadence are not comparable. Any reproducibility protocol built on this
  format has to pin the full invocation, not just the seed - and the container
  does not record the invocation.

## 6. Usage

```
python3 interop/triosckp_reader.py <path.bin>                 # decode and report
python3 interop/triosckp_reader.py <path.bin> --sidecar <p.json>
python3 interop/triosckp_reader.py --verify-all <dir>
```

Exit 0 when every requested check passed, 1 on any container defect or sidecar
disagreement, 2 on a usage error. Standard library only; no third-party
dependency, by design - an independent implementation that needs the first
implementation's toolchain is not independent.

Whenever a sidecar is read, the reader also prints the record's `schema` tag,
the highest schema version whose fields are actually all present, and the list
of schema `/3` provenance fields the record lacks (see A6). A record whose tag
claims a version it does not carry is a failure, not a warning. For a schema
`/3` record the reader additionally cross-checks `lr`, `attn_scale` and
`attn_seq` against header offsets 72, 76 and 80, taking the count of compared
fields from 11 to 14.
