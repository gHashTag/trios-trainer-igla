# Pre-Registration · Wave-9 ASYMLOGIT-NGRAM port

**Anchor:** φ² + φ⁻² = 3 · DOI [10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)
**Date frozen:** 2026-05-06 (Bangkok)
**SOT:** `gHashTag/trios-trainer-igla/assertions/seed_results.jsonl`
**Tracker:** `gHashTag/trios#508` · Active matrix: `trios#446` comment 4370442020
**Source lineage:**
- `openai/parameter-golf#2135` — codemath3000 leader (BPB 1.05651 @ 3 seeds, GPTQ_CALIB_BATCHES=32 single-knob delta vs #2130).
- `openai/parameter-golf#1514` (merged) — Token-Only N-gram tilt legality precedent.
- `openai/parameter-golf#1923` — AsymLogit Rescale precedent.

## H1 (primary hypothesis)
Adding **AsymLogit Rescale** + **Token-Only N-gram Tilt** (closed-form, char-level `TOKEN_ORDER=8`) on top of the existing trainer-igla pipeline reduces best-cell BPB by ≥ 0.01 nats on at least one (format, algo) pair without raising any cell above its 2026-05-04 baseline by more than +0.005 nats (3 σ measurement noise band).

## H0 (null) and falsification rule
H0: median Δ-BPB ≥ 0 across all 38 cells.
**Falsification:** If, after Wave-9 completes, the per-cell paired delta (5-seed mean Wave-9 minus 5-seed mean baseline-replay) is **non-negative for ≥ 30 / 38 cells** at p < 0.05 (Wilcoxon signed-rank), the port is rejected and rolled back. No "best-of-N seed" cherry-picking. No re-runs after unblinding.

## Forbidden values (R7) — must appear verbatim in every job log
```
WITHIN_TAU=99.0
WITHIN_BOOST=0.0
WORD_TAU=99.0
WORD_BOOST=0.0
AGREE_ADD_BOOST=0.0
```
Every Wave-9 seed log MUST emit `within_gate=0 word_gate=0 agree2plus=0`. CI grep gate fails the wave if any log line matches `within_gate=[1-9]|word_gate=[1-9]|agree2plus=[1-9]`.

## Compliance gates

| Gate | Statement | Enforcement |
|---|---|---|
| **C1 causality** | within-word + word-start channels disabled; only token_hint may fire | grep gate above + assertion `assert!(within_boost == 0.0 && word_boost == 0.0)` in Rust |
| **C2 normalization** | `logit += boost; softmax` preserves probability mass | property test: Σ p_i = 1.0 ± 1e-6 over 1000 random vocab vectors |
| **C3 no-TTT** | TTT path NOT ported in Wave-9 | feature flag `--ttt off` is the only allowed mode; CI rejects `--ttt on` |
| **C4 single-pass** | each validation char contributes exactly 1 BPB term | counter `bpb_terms == val_chars` assertion |
| **C5 no-val-leak** | n-gram stats from training corpus only | precompute hash logged + diff against val hash MUST differ |
| **C6 floor** | architectural floor BPB = 2.19; cull only after 5-tick plateau in 0.005 band, step ≥ 50_000 | gardener_runs.cull_reason audit |

## R8 falsification witnesses (Coq-anchored)
- `coq/AsymLogitRescale.v` — proves `softmax_renorm_invariant_under_logit_shift` (port from PR #1923 algebra).
- `coq/TokenNgramTilt.v` — proves `closed_form_tilt_strict_causal` (no future-token access).
- `assertions/asymlogit_ngram_v1.json` — single-source-of-truth for runtime guards (mirrors Coq).

## Frozen knobs (bit-for-bit reproducibility)
```
asymlogit.softcap_pos_init = 30.0
asymlogit.softcap_neg_init = 30.0
ngram.token_order          = 8     # char-level (alphabet ~65); paramgolf used 16 for SP8192
ngram.token_threshold      = 0.80
ngram.token_boost          = 2.625
hidden                     = 828
lr                         = 0.003
steps                      = 81000
seeds                      = [1597, 2584, 4181, 6765, 10946]   # Fibonacci, frozen
queue_id_range             = [19367, 19556]                    # 190 jobs
image_pin                  = ghcr.io/ghashtag/trios-train@sha256:<TBD-Wave9>
```

## Acceptance gates (Wave-9 ⇒ DONE)
- **G1** 190/190 jobs report success with non-empty `bpb_samples` row in `igla-dash` SQLite.
- **G2** 0/190 logs contain forbidden-channel signatures (C1).
- **G3** Coq files compile in CI; both theorems Qed-closed.
- **G4** Aggregated paired Δ-BPB table appended to trios#446 (matrix comment) with Wilcoxon p-values.

## Out-of-scope (explicit)
- GPTQ int6/int7, LQER asym rank-4, per-group compress, TTT — all artifact-cap optimizations, not portable to char-level IGLA.
- Hyperparameter sweep on `TOKEN_ORDER ∈ {4,6,8,10,12}` — deferred to L-NG2.
- Any architecture change (recurrence frac, SmearGate window) — frozen at champion config.

## R12 dead-man switch
If a CLAIMED lane has no heartbeat for > 4 h, Queen Watchdog cron (`ad62640a`) auto-releases it. This pre-reg is not invalidated by lane reassignment.
