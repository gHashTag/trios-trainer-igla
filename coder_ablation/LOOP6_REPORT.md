# Coder-Loop+6 -- Optimizer weight-decay bug: the code corpus now learns

Branch: `feat/igla-coder-v1` (PR #186, epic #181, mirror t27 #1032/#1036)
HEAD before this loop: `fdc3c3b` (Loop+5). All runs CPU-only.
Trigger: user request "code corpus!!! do it" -- make the model actually
descend below the apparent ~8.0 BPB floor, not just document that it does not.

> Negative-first summary: the prior coder ablation series (Loop+1..+5) ran on a
> broken optimizer. The headline "+3.544 BPB harmful phi^-3 decay CDE" is a bug
> artifact and is **RETRACTED**. After fixing the bug, phi and standard are a
> near-tie (delta = +0.108 BPB, phi slightly worse, ~4x lower variance). The
> method survives, phi does not (yet).

---

## 1. Headline finding: the AdamWCpu weight-decay bug

The flat training curve was traced to a real bug in `src/optimizer.rs`,
`AdamWCpu::step()` (was ~line 121):

```rust
// BUGGY: decoupled weight decay applied WITHOUT the learning rate
params[i] -= self.weight_decay as f32 * params[i];
```

With `wd = 0.04` this is `params *= 0.96` **every step** -> weights collapse to
~0 in ~100 steps -> the model cannot learn. The phi arm
(`wd = phi^-3 = 0.236`) collapsed ~6x faster (`params *= 0.764`/step). This
single line explains BOTH prior negative results:

1. "Both arms sit near the ~7.99-8.04 random-init floor at any lr" (Loop+1..+5).
2. "phi^-3 decay is catastrophically harmful, +3.544 BPB CDE" (Loop+3).

Neither was a property of phi or of model capacity -- both were the unscaled
decay collapsing weights.

### Fix

```rust
// FIXED: scale decay by lr (correct decoupled AdamW, Loshchilov & Hutter 2019)
params[i] -= (self.lr * self.weight_decay) as f32 * params[i];
```

Steady-state decay is now `~ lr * wd` (e.g. `2e-3 * 0.04 = 8e-5`), the intended
decoupled-AdamW behaviour. Diff: +9/-2 lines (multi-line ASCII comment citing
the paper), ASCII-clean on added lines.

### Blast radius (triaged, not assumed)

`grep -rln AdamWCpu src/` => 11 files. Triage:

| File | Calls `.step()`? | Affected |
|---|---|---|
| `bin/igla_coder.rs`, `bin/igla_coder_v1.rs` | YES (lines ~806-816) | YES -- the whole coder series |
| `pipeline.rs` | YES (lines 237, 283) | YES |
| `transformer_trainer.rs` | NO (`_optimizer`, line 173 is a stub) | NO (no training happens) |
| `real_igla_trainer.rs` | NO (`_optimizer`, lines 58/124) | NO (stub) |
| `bench.rs`, `lr_calibration.rs`, `r12_optimizer_race.rs`, `trinity_tournament.rs`, `jepa/predictor.rs` | mixed | re-check per file |

Separate latent issue: `AdamWCpu::new()` bakes `weight_decay = 1/phi^3 = 0.236`
(its code comment `alpha_phi ~= 0.11803` is **stale/wrong** -- 1/phi^3 is 0.236;
0.118 = 1/(2*phi^2)). Any `new()` caller got phi-anchored decay that, under the
bug, collapsed weights hard. Worth a dedicated audit on `f2-methodology`.

### Retroactive claim-status corrections (locked)

- Coder decay CDE = +3.544 BPB -> **[Retr]** (bug artifact, not phi).
- Coder momentum sign-flip (+0.888 canonical vs -0.268 wd0) -> **[Risk]**: the
  wd0 stratum partly neutralised the bug (wd=0 -> no collapse), so canonical-vs-
  wd0 mixed a real regime change with a bug-vs-clean contrast.
- Main-track WD-mediation (Loops 28-50 on `f2-methodology`, e.g. NIE_M1 via
  WD = +4.99) -> **[Risk] pending re-check**: those sweeps did not run from this
  branch; the stub trainers here never stepped. Whether `f2-methodology`'s sweep
  used a live `AdamWCpu.step()` under the bug MUST be verified before
  re-confirming or retracting. Not assumed either way.

The cross-stratum *machinery* (`stratum_compare_coder.py`, the single-mediator
CDE branch, the `mom_std` cell-shape test) is sound and survives; only the
*numbers* are retracted.

---

## 2. The corrected result: the model learns, phi is a near-competitor

With the fix, the tiny CPU model descends on the code corpus (hidden=64, lr=0.002,
800 steps, fim_loss=all): `--wd 0` drives train 7.68 -> 2.14, val -> 2.72.
The ~8.0 "floor" was the bug, not capacity.

Corrected phi-vs-standard ablation (3 seeds 42/43/44,
`coder_ablation/coder_fixed_decay_ablation.csv`):

| Arm | wd | mean code_val_bpb | std |
|---|---|---|---|
| standard | 0.04 | **2.6605** | +/- 0.1068 |
| phi | phi^-3 = 0.236 | **2.7682** | +/- 0.0278 |

delta = **+0.108 BPB** (phi slightly worse; 95% CIs nearly touch). phi has
**~4x lower variance**. Status **[Efit]** (single config; post-fix stratum
stability not yet re-established). Honest verdict: phi is a near-competitor,
not a catastrophe -- the exact opposite of the retracted +3.544 claim.

---

## 3. What else shipped this loop (waves A/B/C)

- **Wave A (done):** `Model::save(path)` / `Model::load(path)` + `load-generate`
  subcommand. Binary format magic `0x49474C43` ("IGLC") v1; bit-exact roundtrip
  unit test; load rejects bad magic. ~593 KB / 148224 params at hidden=64.
- **Wave B (built):** `coder_ablation/eval_pass1.py` -- execution pass@1 over 4
  self-contained C micro-specs (add_u32, clamp_u8, popcount8, mod_add),
  `--mode train|load`. With a real (post-fix) checkpoint this can now be re-run
  for a first non-zero pass@1 (Loop+7 candidate).
- **Wave C (harness built, numbers deferred):** `print-prior` subcommand +
  `coder_ablation/ablate_generator_axis.py` (phi/dyadic/e/standard axis at equal
  budget). Prior readout verified; the 4-arm run should be re-done on the FIXED
  optimizer (the orphaned pre-fix run is meaningless).

---

## 4. Verification

- `cargo build --release --bin igla_coder` -- OK.
- `gradcheck` -- `checks=23 fails=0 ... GRADCHECK: PASS`.
- `cargo test --release --lib` -- **559 passed** (locked count held;
  `test_adamw_step` only asserts params decrease, still true after the fix).
- ASCII check on `optimizer.rs` added lines vs HEAD -- clean.

## 5. Files touched (uncommitted)

- `src/optimizer.rs` (the fix + ASCII comment).
- `src/bin/igla_coder.rs` (Wave A save/load + load-generate; Loop+5 FIM mask).
- `coder_ablation/eval_pass1.py`, `coder_ablation/ablate_generator_axis.py`,
  `coder_ablation/coder_fixed_decay_ablation.csv`.

## 6. Research applied

- Loshchilov & Hutter 2019 (decoupled weight decay / AdamW) -- decay must be
  scaled by lr; this is the canonical citation for the fix.
- arXiv:2410.23922 (NeurIPS 2024, warmup/momentum/decay coupling) -- mechanistic
  justification for momentum + decay as the two real coder mediators.

---

## Three collaboration options for Coder-Loop+7

**A. Re-run the generator-axis ablation on the FIXED optimizer (close Wave C).**
Direction: now that the model genuinely learns, run
`ablate_generator_axis.py --generators phi,dyadic,e,standard --seeds 42,43
--steps 800 --hidden 64` foreground; report mean BPB +/- CI per axis,
freeze->hash->null->compare. First honest phi-vs-{dyadic,e,standard} comparison
where descent actually happens.
Cost/Risk: ~6-10 min CPU foreground; low risk; likely outcome = all axes cluster
within noise (phi NOT supported, but now for a real reason, not a bug).

**B. Train one real checkpoint and get first non-zero pass@1 (close Wave B).**
Direction: `generate --wd 0 --steps 1500 --save /tmp/coder.bin` then
`eval_pass1.py --mode load`. Report compile@1 + pass@1 honestly (may still be 0
at this scale, but now from a learning model, not a collapsed one).
Cost/Risk: ~3-5 min CPU; low risk; clean honest signal either way.

**C. Audit the main-track / f2-methodology blast radius.**
Direction: check out `f2-methodology`, grep its sweep for live `AdamWCpu.step()`
under default `new()` decay, and decide whether Loops 28-50 WD-mediation findings
(NIE_M1 via WD = +4.99 etc.) are [Verified], [Risk], or [Retr]. Resolves the
biggest open honesty question.
Cost/Risk: no compute; pure code audit; higher payoff (protects the publishable
methodology claims) but touches a different branch.

STOP. Pick A / B / C (or a combination) for Loop+7.
