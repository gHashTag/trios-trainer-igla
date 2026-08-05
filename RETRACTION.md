# RETRACTION - the tracked BPB ledger and the champion numbers it published

Date: 2026-08-03
Scope: `assertions/seed_results.jsonl` (now
[`assertions/RETRACTED-seed_results.jsonl.txt`](assertions/RETRACTED-seed_results.jsonl.txt)),
`assertions/champion_lock.txt`, and the five BPB figures that coexisted in this
tree as if each were the champion.

This note lives at the repository root on purpose. An earlier retraction was
written to `.trinity/results/RETRACTION.md`, but `.trinity/` is gitignored, so
that note could never ship next to the evidence it retracts. This one ships.

Nothing has been deleted. The ledger rows are preserved verbatim under the
`RETRACTED-` name, behind a header that states, per row family, what is missing.

---

## 1. What was retracted

`assertions/seed_results.jsonl` was tracked in git and declared the source of
truth for BPB evidence by five documents: `SOURCE_OF_TRUTH.md`, `MIGRATION.md`,
`docs/TRAINING_FLOW_V2.md`, `docs/preregistration/asymlogit_ngram_v1.md` and
`igla-dash/README.md`. It held 11 rows. None of them is citable.

The defects are recorded in full in the header of the renamed file. In summary,
for every row:

| Missing | Evidence |
|---|---|
| Artifact | `checkpoint::save` is `pub fn save(_run, _step, _bytes) -> Result<()> { Ok(()) }` at both referenced commits (`git show 4c0b04c:src/checkpoint.rs`, line 116; same at `cd91c45`). No weights were written. No row can be re-measured. |
| Corpus hash | No row names a corpus file, byte count or digest. At both commits `load_data` substituted `b"The quick brown fox jumps over the lazy dog. ".repeat(2500)` for a missing file with only a stderr line (`train_loop.rs:39-51`). A row cannot distinguish a real corpus from that 112 500-byte pangram, and in the fallback case train and val are the same bytes. |
| Trainer hash | `sha` is a 7-character abbreviation of repository HEAD. It pins no binary, no toolchain, no config digest and no data. |
| Eval coverage | No val token count, no eval window count, no eval cadence. `--eval-every` is not an observation-only parameter (it gates an in-place weight transform late in training), so rows without a recorded cadence are not comparable to each other. |
| Canon-legal seed | Every seed is 42, 43 or 44. `src/seed_canon.rs` declares {42, 43, 44, 45} FORBIDDEN under Canon #93; the allowed set is {47, 89, 123, 144}. Zero rows use a canon-allowed seed. |

Two further facts about the rows themselves:

- **They did not come from the validated emit path.** The in-repo `LedgerRow` at
  `cd91c45` is `{agent, bpb, step, seed, sha, jsonl_row, gate_status, ts}`
  (`src/ledger.rs:22-31`). No row in the file carries `agent`, `jsonl_row` or
  `ts`. The R7 triplet validation that `SOURCE_OF_TRUTH.md` calls mandatory
  never ran on any of them.
- **Six of the eleven violate R8** (`step >= 4000`); they declare `steps: 3000`.
  Today's `emit_row` rejects them outright.

`assertions/champion_lock.txt` published `champion@2446855 BPB=2.2393 seed=43
step=27000`. `2446855` does not resolve to an object in this repository
(`git cat-file -t 2446855` -> `Not a valid object name`); it is a reference into
`gHashTag/trios`. The lock file's contents have been replaced with a pointer
here.

---

## 2. The five coexisting numbers

| BPB | Declared in | Citable? |
|---|---|---|
| **2.5193** | `src/invariants.rs::BPB_CHAMPION` | **No.** Already documented as retracted in its own doc comment; `docs/audit/HONEST_FINDINGS.md` calls it a stale placeholder. It pre-dates `checkpoint::save` doing any work, so no artifact exists. It survives only as an internal upper bound asserted against `ASHA_PRUNE_THRESHOLD`; changing the constant would break call sites without making any claim more honest. Not edited here. |
| **2.2393** | `assertions/champion_lock.txt`, `configs/champion.toml`, `tests/champion_reproduction.rs`, `docs/TRAINING_FLOW_V2.md`, `MIGRATION.md` | **No.** Attributed to `gHashTag/trios@2446855`, a commit that does not exist in this repository. No artifact, no corpus digest, no trainer hash. `tests/champion_reproduction.rs` asserted a reproduction tolerance of `+/- 0.01` around it while `train_loop::run()` discards most of the config those assertions checked (see section 3). |
| **2.2111** | `DEPLOYMENT_BLOCKER.md` ("NEW CHAMPION"), `.trinity/dashboard.md`, `.trinity/STATUS.md`, `.trinity/experience/trios_20260427_pt2.trinity` | **No.** Retracted here and already graded RETRACTED in `docs/audit/HONEST_FINDINGS.md`. Same withdrawn family as 2.2393 and 2.1919: seed 43 (forbidden under Canon #93), no artifact, no corpus digest, no recorded eval cadence. It is the weakest of the five - it has **no ledger row at all**. `grep -rn 2.2111 assertions/` returns nothing; the only row for the same run family (seed 43, step 81000) reads `bpb: 2.1919, hidden: 828`, while `DEPLOYMENT_BLOCKER.md` reports the run as `hidden=384`. The two descriptions disagree and neither is checkable. The Gate-1 pass claimed from it (`2.2111 < 2.22`) is retracted with it. |
| **2.1919** | `assertions/seed_results.jsonl` (last-but-one row), `igla-dash/README.md` ("Champion baseline"), `docs/audit/HONEST_FINDINGS.md` ("Honest champion (Wave-8)") | **No.** Retracted with the ledger. Written outside the emit path, forbidden seed 43, no artifact, no corpus digest. Its `gate_status: "above_target"` was read as a pass; the Gate-2 target in force was BPB < 1.85, and 2.1919 is above it. It is also *better* than `BPB_CHAMPION = 2.5193` and than 2.2393, which is what let it displace both in the documents that cited it. |
| **2.6348** | `README.md` (calibration reference) | **Yes, with its protocol stated.** This is the only one of the four with an artifact behind it. `final_val_bpb = 2.6347548961639404` at step 12 000, seed 47, `hidden=384`, `d_model=64`, 2 attention layers, `lr=0.003`, `eval_every=1000`, `data_synthetic=false`, over `data/tiny_shakespeare.txt` (1 015 394 B, sha256 `1a5aead1...`) and `data/tiny_shakespeare_val.txt` (100 000 B, sha256 `2088af36...`), which are byte-disjoint and sum to the canonical corpus length. Four independent runs (`r4-docs-ckpt`, `r4-docs-repro`, `r5-adv-recheck`, `trios-train-rng47`) wrote byte-identical checkpoints, sha256 `8a86fe69...`, 852 272 B. |

### The caveat that belongs to 2.6348

`src/invariants.rs` names a different figure - raw val_bpb **2.6169** at 12 000
steps, sidecar `checkpoints/igla-honest-provenance/12000.json` - as "the only
checkpoint-backed figure in this repo", and `checkpoints/igla-honest-20260802/`
carries a third, **2.6141**, at the same step, same seed, same corpus digests.

These are not contradictions to be resolved by picking one. They are the eval-
cadence effect: `src/train_loop.rs` gates `gf16_floor`, which mutates
`embed`/`proj`/`lm_head`/`ctx` **in place**, on `step % eval_every == 0` after
70 % of training. Runs that differ only in `--eval-every` end with different
weights. Any of these three numbers may be cited only together with its cadence
and its checkpoint sha256. None of them may be compared to a number whose
cadence is unrecorded - which is every row in the retracted ledger.

---

## 3. What the test suite was certifying

- `tests/preregistration_seed_lock_final.rs` hardcoded `GATE_FINAL_ALLOWED_SEEDS
  = [42, 43, 44]` - exactly the set `src/seed_canon.rs` forbids. The suite did
  not merely miss the violation; it asserted the violation was the only legal
  state. The test now derives its verdict from `seed_canon::parse_seed`, so the
  source law wins.
- Both seed-lock falsifiers `return`ed early when their evidence file was
  absent, so they passed vacuously on any checkout without the ledger. Missing
  evidence is now a `panic!`.
- `tests/champion_reproduction.rs` asserted `d_model = 256`, `vocab_size =
  32000`, `seq_len = 1024`, both AdamW betas, `weight_decay`, `schedule`,
  `warmup_steps` and the whole `[objective]` block. `train_loop::run()` builds
  `TrainArgs` from six fields only - `seed`, `steps`, `optimizer.lr`,
  `model.hybrid_attn` (as a 1-or-2 layer count), `data.train_path`,
  `data.val_path` - and hardcodes `hidden: 828` and `eval_every: 1000`
  (`src/train_loop.rs:2182-2192`). Every other asserted value is parsed and
  thrown away. Those assertions were replaced by one test that documents the gap
  and fails if it silently closes or widens.
- `tests/ledger_seaorm.rs` ran DDL migrations plus an insert against whatever
  `DATABASE_URL` happened to be exported. A `cargo test` on a machine with the
  production DSN in its environment would have migrated the live ledger. A
  survey run had already inserted `canon_name='ledger_seaorm_smoke_test'` into
  the local trios database by accident. The test now requires an explicit
  `TRIOS_ALLOW_LIVE_LEDGER_TESTS=1` plus a localhost-only host allowlist, and is
  `#[ignore]`d so a skip shows as ignored rather than counted as a pass.

---

## 4. What is NOT retracted here

- `src/invariants.rs` is not edited. `BPB_CHAMPION = 2.5193` stays as an
  internal threshold and is already documented as retracted in place.
- `README.md`'s calibration section is not edited; its number is the citable one,
  subject to the cadence caveat in section 2.
- `.trinity/results/RETRACTION.md` (the train-set-as-val retraction of
  `igla_trigram.rs` / `trinity_pr1722.rs`) stands on its own grounds and is
  unaffected by this one.
- The retracted rows are preserved, unmodified, in
  `assertions/RETRACTED-seed_results.jsonl.txt`.

Anchor: `phi^2 + phi^-2 = 3` - [Zenodo 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877).
