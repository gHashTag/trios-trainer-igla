# Coder-Loop+7 -- generator-axis control, first real pass@1, and the main-track audit

Branch: `feat/igla-coder-v1` (PR #186, epic #181). HEAD before loop: `5d6a7d2`
(Loop+6 optimizer fix). All runs CPU-only. User picked A + B + C.

> Negative-first summary: (A) on the fixed optimizer the four generator axes
> rank STRICTLY by the learning-rate multiplier each prescribes -- phi is not
> uniquely bad, it just prescribes a smaller lr; (B) a genuinely-learning
> checkpoint (val BPB 2.25) still scores pass@1 = 0/4 and compile@1 = 0/4 --
> honest and expected at this scale, and now a REAL measurement after fixing a
> harness-crashing bug; (C) the main-track `f2-methodology` WD-mediation sweep
> ran the SAME unscaled-decay bug live, so those WD findings are now [Retr]/[Risk].

---

## A. Generator-axis ablation on the FIXED optimizer

Question: "is phi special as an axis?" Each axis injects its config_prior
(beta1, weight_decay, lr_mult) at equal budget (hidden=64, 800 steps, 3 seeds,
fim_loss=all, lr = 0.002 x lr_mult). `standard` is the null; phi must BEAT it.

| axis | lr_mult | mean code_val_bpb | std | ci95 |
|---|---|---|---|---|
| standard | 1.000 | **2.6605** | 0.0944 | +/-0.1068 |
| phi | 0.236 | 3.2269 | 0.0167 | +/-0.0189 |
| dyadic (g=2) | 0.125 | 3.4859 | 0.0172 | +/-0.0195 |
| e | 0.050 | 4.1151 | 0.0483 | +/-0.0547 |

Provenance: sha256(raw)[:16] = `44ba6167e9cbd132`. Saved to
`coder_ablation/coder_generator_axis.csv`.

**VERDICT: phi WORSE than standard by +0.566 BPB (CIs disjoint) -- phi falsified
on this axis at this budget.** But the honest nuance a referee would demand:
the ranking is **monotone in lr_mult** (1.0 > 0.236 > 0.125 > 0.050 maps exactly
to 2.66 < 3.23 < 3.49 < 4.12). The axes differ almost entirely by the learning
rate their prior prescribes, and at a fixed 800-step budget a smaller lr simply
descends less. phi is not a pathological axis -- it is a slow-lr axis. The lower
variance of phi/dyadic/e (std ~0.02 vs standard's 0.09) is the flip side: small
lr = less seed-to-seed spread. "The method survives, phi does not (yet)."
Status: **[Efit]** (single budget; the lr_mult confound is identified, not
controlled -- an equal-effective-lr re-test is the clean follow-up).

## B. First execution pass@1 from a genuinely-learning checkpoint

Trained one real checkpoint: `generate --wd 0 --steps 1500 --hidden 64 --save`.
train_bpb descended 3.41 -> 1.37; trained val BPB = **2.2544**; checkpoint
593 KB / 148224 params (`/tmp/coder_real.bin`). Then scored with
`eval_pass1.py --mode load`.

**Result: compile@1 = 0/4, pass@1 = 0/4** (add_u32, clamp_u8, popcount8,
mod_add). Status **[Verified] negative** -- this is the expected near-term
outcome for a 148K-param CPU model, and it is the honest number we want to
watch move off zero. The model emits structured-but-wrong fragments
(`bit3_bit_b_va; = = = ...`), i.e. it has learned token statistics but not
compilable C.

### B-bug found and fixed: load-generate loaded the training corpus

The first pass@1 attempt returned `gen_rc=101` (Rust panic) for every spec --
NOT a model result but a real bug. `igla_coder` loaded `data/code_train.bin` +
`data/code_val.bin` **unconditionally for every subcommand**, including
`load-generate`, which only samples from a checkpoint and needs no corpus. When
`eval_pass1.py` ran from `coder_ablation/`, the relative `data/` path did not
resolve -> `panic: open .bin: NotFound`. Fix (`src/bin/igla_coder.rs`, +6/-2):
skip the corpus load when `cmd == "load-generate"`. After the fix `gen_rc=0` and
the harness measures the model honestly. ASCII-clean; gradcheck 23/0 unchanged.

## C. Audit: the optimizer bug reaches the main `f2-methodology` track

Fetched `f2-methodology` (read-only, no commits). Traced the WD-mediation chain:

- `f2_ablation_sweep::run_wd_sweep` sweeps `wd in {0.0, 0.005, 0.01, 0.03, 0.1,
  0.3}` through `run_multi_seed` -> `src/race/multi_seed.rs`.
- `multi_seed.rs` constructs `AdamWCpu::with_params(n, lr, 0.9, 0.999, wd)` and
  calls `optimizer.step(...)` LIVE inside `for step in 0..config.steps`
  (lines 2234/2237/2274). This is real training, not synthetic numbers.
- `src/optimizer.rs:121` on `f2-methodology` is **the identical bug**:
  `params[i] -= self.weight_decay as f32 * params[i];` (decay NOT scaled by lr).

So at `wd=0.3` the main sweep multiplied params by ~0.7 every step (severe
collapse); at `wd=0.0` no collapse. **The measured "WD mediates BPB" gradient is
dominated by the weight-collapse artifact, the same mechanism as the retracted
coder +3.544.** The wd=0 point is the only bug-free sample in that sweep.

### Retroactive claim-status (locked Loop+7)

| Main-track finding | New status | Reason |
|---|---|---|
| NIE_M1 via WD = +4.99 BPB; the whole WD-sweep slope | **[Retr]** | confounded by unscaled-decay collapse; any wd>0 point mixes regularization with collapse |
| `wd0` stratum results (weight_decay pinned to 0) | **[Risk] -> re-confirm** | wd=0 is the one bug-free point; safe in isolation but its cross-stratum CONTRASTS used buggy wd>0 arms |
| rms-mediated NIE = -0.75 (Loop 50) and other non-WD PSEs | **[Risk]** | did not vary WD, but baseline ran at wd=0.1 collapsing weights every step -> needs a fixed-optimizer re-run before re-confirming |

The F2 *methodology* (stratified CDE, Lambda-sweep, provenance, 12 binaries, the
tests) is sound and survives. The *numbers* from any sweep that trained with the
buggy decay must be re-generated. This does not invalidate the publishable
contribution -- it sharpens it: the methodology now has a documented worked
example of catching a confound (the bug) that a naive BPB comparison would miss.

---

## Verification

- `cargo build --release --bin igla_coder` -- OK.
- `gradcheck` -- `checks=23 fails=0 PASS`.
- `cargo test --release --lib` -- **559 passed** (locked count held).
- `cargo test --release --bin igla_coder` -- **6 passed**.
- ASCII-clean on all added lines (igla_coder.rs, ablate_generator_axis.py, CSVs).

## Files touched

- `src/bin/igla_coder.rs` -- load-generate no longer loads the training corpus.
- `coder_ablation/ablate_generator_axis.py` -- stale "+3.544 falsified" comment
  replaced with the Loop+6 retraction + Loop+7 lr_mult-confound framing.
- `coder_ablation/coder_generator_axis.csv` -- A results (provenance-hashed).
- `coder_ablation/eval_pass1_loop7.csv` -- B results (compile@1 / pass@1 = 0/4).
- `coder_ablation/LOOP7_REPORT.md` -- this report.

## Research applied

- Loshchilov & Hutter 2019 (decoupled AdamW) -- the decay-must-be-lr-scaled rule
  that both the coder track (Loop+6) and now the main track (Loop+7 audit) violated.
- HumanEval (Chen et al. 2021) -- the fixed-reference-test pass@1 pattern used by
  eval_pass1.py.

---

## Three collaboration options for Coder-Loop+8

**A. Equal-effective-lr generator-axis re-test (control the lr_mult confound).**
Direction: re-run the four axes but hold the EFFECTIVE lr equal across arms (drop
lr_mult, or budget-match steps to compensate), isolating beta1/wd from lr. Turns
the Loop+7 [Efit] "phi is slow-lr" into a clean "phi axis at iso-lr is / is not
competitive" statement.
Cost/Risk: ~15 min CPU foreground; low risk; the cleanest phi-as-axis claim yet.

**B. Re-generate the main-track WD-sweep on the FIXED optimizer.**
Direction: port the optimizer fix to a clone of `f2-methodology` (its own commit,
confirm_action), re-run `f2_ablation_sweep --mode wd_stratified`, and re-compute
the WD-mediation PSEs. Resolves whether NIE_M1-via-WD survives the fix or stays
[Retr]. This is the highest-value honesty work for the publishable methodology.
Cost/Risk: cross-branch change + a real sweep; medium effort; protects the paper.

**C. Push the coder model toward non-zero compile@1 (capacity / budget).**
Direction: scale the real checkpoint (hidden=128, 5-10k steps, or curriculum on
the shortest single-language functions) and re-score eval_pass1; report the first
compile@1 > 0 honestly if it appears, or document the budget where it does not.
Cost/Risk: longer CPU runs (checkpoint-and-resume); moves the headline pass@1
number, which is the user's stated goal ("make the code corpus learn").

STOP. Pick A / B / C (or a combination) for Loop+8.
