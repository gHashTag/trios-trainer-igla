# Coder-Loop+10 -- beta2 isolation (C), confound-ladder figure (A), and the C-only resume curriculum (B)

Branch (coder): `feat/igla-coder-v1`, HEAD before loop `430f71b` (Loop+9).
All CPU-only. User picked ALL THREE Loop+9 options (A + B + C).

Anchor: `phi^2 + phi^-2 = 3`. Position: phi is a coordinate axis we measure
from, not a claim. The method survives, phi does not (yet).

## NEGATIVE-FIRST HEADLINE

1. **(C) The second-moment knob is the same story as the first.** With lr, wd,
   AND beta1 all pinned to standard, varying only Adam `beta2` shows NO
   phi-native value that beats standard on phi grounds. A phi-native `beta2 ~
   1 - phi^-7 = 0.966` did test -0.164 BPB BETTER than `beta2 = 0.999` -- but a
   non-phi control sweep (0.98 / 0.95 / 0.92) beats the null by an IDENTICAL
   margin (-0.146 to -0.166 BPB). **The gain is a generic "0.999 is too high for
   a tiny noisy run" effect, not a phi effect.** The two aggressive phi-native
   values (`phi^-1 = 0.618` and `1/phi^2 = 0.382`) are catastrophically WORSE
   (+2.3 and +32 BPB). `grad_clip` is reported INERT (it is printed in the prior
   but never applied in `opt_step`), so it is NOT swept -- reporting it inert
   rather than fabricating a mediator.
2. **(A) The Loop+7->8->9 sequence is published as a confound ladder.** A single
   figure shows the phi-standard delta collapsing +0.566 -> +0.092 -> -0.027 BPB
   as the lr then the decay confound are each controlled. The "phi advantage"
   was a confound at every rung; the honest endpoint is neutrality.
3. **(B) First resumable C-only curriculum at hidden=128.** A new `--resume`
   flag warm-starts weights across short CPU sessions, lifting the hidden=64
   capacity ceiling (296K params -> 493K). val BPB descended cleanly to **1.038**
   (the lowest the coder track has recorded) -- but **compile@1 is STILL 0/4 and
   pass@1 STILL 0/4**. The doubled capacity bought a much better likelihood model
   yet did NOT cross the threshold where syntactically valid C emerges. The
   capacity ceiling is real and sits above 493K params at this corpus/budget.

The one-line position after Loop+10: across EVERY isolated AdamW knob the coder
track can vary -- lr (Loop+7/8), weight_decay (F2 mediation), beta1 (Loop+9),
beta2 (this loop) -- phi structure is either neutral or harmful, never a clean
benefit. The per-knob phi optimizer-prior audit is now CLOSED. Only the
weight_decay = phi^-3 (0.236, ~6x standard) knob was ever robustly harmful; the
rest are non-events. Only `phi^2 + phi^-2 = 3` is `[Verified]`.

## C -- beta2 (second-moment) isolation sweep

New harness `coder_ablation/ablate_beta2_sweep.py`: symmetric counterpart to
the Loop+9 beta1 harness. lr (0.002), weight_decay (0.04), AND beta1 (0.9) are
ALL pinned to standard for every arm; the ONLY across-arm difference is
`--beta2` (the new Loop+10 flag -- beta2 was hardcoded 0.999 before this loop).
hidden=64, 800 steps, 3 seeds (42/43/44), fim_loss=all. `beta2=0.999` is the
null any phi-native beta2 must beat.

### Primary sweep (phi-native grid)

| beta2 | phi label | mean code_val_bpb | ci95 | delta vs null |
|---|---|---|---|---|
| 0.999 | standard (null) | 2.3855 | +/-0.0414 | (null) |
| 0.966 | ~1 - phi^-7 | **2.2212** | +/-0.0179 | **-0.164** (BETTER, CIs disjoint) |
| 0.764 | ~1 - phi^-3 | 2.3441 | +/-0.0300 | -0.041 (NEUTRAL, CIs overlap) |
| 0.618 | phi^-1 | 4.7129 | +/-0.5714 | +2.327 (WORSE) |
| 0.382 | 1/phi^2 | 34.5641 | +/-10.08 | +32.18 (WORSE, diverging) |

Provenance sha256[:16] = `d434d548fd0bfbb0`. Saved
`coder_ablation/coder_beta2_sweep.csv`.

### Control sweep (non-phi neighbours of 0.966) -- the decisive test

The naive read of the primary sweep ("phi-native 0.966 beats standard!") is a
trap. To test whether 0.966 wins BECAUSE it is `1 - phi^-7` or simply because
0.999 is a poor beta2 for this regime, a control sweep runs four NON-phi values
bracketing 0.966:

| beta2 | phi label | mean code_val_bpb | ci95 | delta vs null |
|---|---|---|---|---|
| 0.999 | standard (null) | 2.3855 | +/-0.0414 | (null) |
| 0.980 | (none) | 2.2193 | +/-0.0581 | -0.166 (BETTER) |
| 0.966 | ~1 - phi^-7 | 2.2212 | +/-0.0179 | -0.164 (BETTER) |
| 0.950 | (none) | 2.2317 | +/-0.0284 | -0.154 (BETTER) |
| 0.920 | (none) | 2.2395 | +/-0.0475 | -0.146 (BETTER) |

Provenance sha256[:16] = `0b1efb5b8d19aad7`. Saved
`coder_ablation/coder_beta2_control.csv`.

**Verdict [Efit]:** the phi-native 0.966 (-0.164) is statistically
indistinguishable from the non-phi 0.980 (-0.166), 0.950 (-0.154), and 0.920
(-0.146). The entire 0.92-0.98 neighbourhood beats 0.999 by the same ~0.15 BPB.
This is a generic second-moment-tuning effect: **0.999 is simply too high for an
800-step, sub-1M-param CPU run; any moderately lower beta2 helps equally.** The
`~1 - phi^-7` label is post-hoc numerology applied to a value that happens to
fall in the good range. **There is no phi-specific beta2 benefit.** The two
genuinely phi-native aggressive values (0.618, 0.382) are catastrophically
worse -- second-moment forgetting that fast destabilises the RMS denominator.

This is exactly the confound the critical-honesty discipline exists to catch:
had the control sweep been skipped, the primary sweep would have read as a phi
win. It is not.

### grad_clip -- reported INERT, not swept

The four generator axes prescribe distinct `grad_clip` values (printed by
`print-prior`), but `grad_clip` is **never applied in `opt_step`** in this
trainer -- it is dead metadata. Sweeping it would be theatre. Per the
f2-mediation rule (do NOT pin/vary a non-existent mediator -- the rule that
dropped the fictitious "warmup0" in Loop+8), `grad_clip` is reported inert and
left unswept. If a future loop wires gradient clipping into `opt_step`, it
becomes a real knob and can be swept honestly then.

## A -- the confound ladder (figure + write-up)

`coder_ablation/plot_confound_ladder.py` -> `coder_ablation/confound_ladder.png`.
A single line-with-error-bars figure plotting the phi-standard delta in
code_val_bpb across the three control stages:

| stage | control | phi-standard delta | cause of the gap |
|---|---|---|---|
| Loop+7 | raw (lr confounded) | +0.566 BPB | phi was a low-lr axis, not a bad prior |
| Loop+8 | iso-lr (decay free) | +0.092 BPB | residual = phi^-3 decay (~6x standard wd) |
| Loop+9 | iso-everything (lr+wd pinned) | -0.027 BPB | momentum NEUTRAL; CIs overlap |

The figure's title states the finding plainly ("the phi advantage was a
confound; as controls tighten, the phi-standard gap collapses to neutral") and
the footer carries the anchor and the negative-first disclaimer. This is the
P8.3 write-up artifact -- the methodology's honesty headline: a +0.57 BPB
apparent phi advantage that fully dissolved to noise once two confounds were
removed, with each rung independently reproducible from a committed CSV.

### P8.3 write-up subsection (drop-in)

> **8.3 A worked confound ladder.** The clearest demonstration that the
> generator-axis method is honest is that it falsified its own most favourable
> early reading. A naive axis comparison (Loop+7) reported phi as +0.566 BPB
> worse than a standard AdamW control. That gap was a learning-rate sweep in
> disguise: the phi axis prescribed a 0.236x lr multiplier, so it simply
> descended more slowly. Pinning the learning rate equal across arms (Loop+8)
> shrank the gap to +0.092 BPB. The residual was traced -- via the F2 mediation
> analysis, which showed weight-decay dominates BPB roughly 4:1 over momentum --
> to the phi axis's heavy phi^-3 = 0.236 weight decay, about six times the
> standard 0.04. Pinning weight decay equal too (Loop+9) collapsed the gap to
> -0.027 BPB, with overlapping 95% confidence intervals: a clean tie. The
> apparent phi advantage at every stage was a confound, not a signal. We report
> this ladder not as a failure but as the method's central validity check: a
> generator axis is only as trustworthy as the controls that surround it, and
> this one survives precisely because it dissolves its own false positives.

## B -- C-only resume curriculum (hidden=128)

Trainer change: new `--resume <ckpt>` flag on the `generate` subcommand
(`train_from_model`, a warm-start-weights / fresh-optimizer-state continuation;
the checkpoint format stores weights only, not the Adam moment buffers, and the
code is honest about that -- it does NOT claim bit-exact optimizer-trajectory
continuation). `cfg.steps` becomes the number of ADDITIONAL steps. A shape
guard rejects a checkpoint whose (d, heads, layers) mismatch the args. Combined
with `--save`, a long run can be accumulated across many short CPU sessions.

Driver `coder_ablation/run_cB_curriculum.sh`: hidden=128 (493K params, vs the
296K hidden=64 ceiling that pinned compile@1=0 in Loop+7/8), lang_id=1 (C),
fim_loss=middle (infill, the natural completion objective), standard optimizer
(phi is neutral per Loop+9, so the capacity probe is not handicapped by it).
Accumulates ~12000 steps in 3000-step resumable chunks on top of a 200-step
warmup checkpoint. After training, `coder_ablation/eval_pass1.py --mode load`
scores compile@1 / pass@1 on the four C micro-specs (add_u32, clamp_u8,
popcount8, mod_add).

### Result -- the curriculum descended BPB but compile@1 stayed at zero

The resume chain ran end-to-end across four 3000-step chunks. Each chunk's
`pre_resume_val_bpb` matched the prior chunk's saved BPB exactly, confirming the
checkpoint weight load is bit-exact:

| cumulative steps | code_val_bpb |
|---|---|
| 200 (warmup) | 3.0340 |
| 3200 | 1.3281 |
| 6200 | 1.1372 |
| 9200 | 1.0756 |
| 12200 | **1.0378** |

This 1.038 BPB is the lowest code_val_bpb the coder track has produced, and the
doubled capacity (493K vs 296K params) is clearly doing real work. But the
execution eval is unambiguous and negative:

```
compile@1 = 0/4 = 0.000
pass@1    = 0/4 = 0.000   (add_u32, clamp_u8, popcount8, mod_add)
```

Saved `coder_ablation/eval_pass1_loop10_cB.csv`. A sample greedy completion of
`uint32_t add_u32(uint32_t a, uint32_t b) {` is token-salad: it stays in the C
lexical neighbourhood (`int32_t`, plausible identifiers, operators) but emits
degenerate repetition (`_srex_srex_copop...`) and never a valid statement or
`return`. The model has learned local byte statistics, not C grammar.

**Verdict [Efit, negative]:** at hidden=128 / 493K params / 12200 steps on this
corpus, the model reaches a record-low BPB but cannot synthesise a single
compiling C function. **compile@1 = 0 is a capacity-ceiling result, not a
beta1/beta2/lr/wd tuning result** -- consistent with Loop+7/8 (296K ceiling),
now pushed up to a confirmed-still-failing 493K. The `--resume` machinery itself
is VERIFIED working (BPB descended monotonically across resumed chunks); the
bottleneck is parameter count, not the training loop. This sets up Loop+11
Option B (capacity scan) with a clean lower bound: 493K is not enough.

## Verification

- `cargo build --release --bin igla_coder` OK (incremental; threads `--beta2`
  through `arm_hparams`/`make_arm`/`make_opt` and adds `--resume` to `generate`).
- `gradcheck --hidden 32 --heads 4 --layers 2`: `checks=23 fails=0 PASS`
  (analytic vs f64 central-difference, gate abs<2e-3 OR rel<5%) -- unchanged by
  the new flags.
- `--beta2` and `--resume` smoke-tested: resume loaded a checkpoint
  bit-exactly (`pre_resume_val_bpb` matched the saved arm's final BPB) and
  continued training.
- All 30 beta2 sweep runs (primary + control) returned a valid `code_val_bpb`;
  no NaN, no corpus-load failure.
- New files ASCII-clean.

## Claim-status summary (updated)

- [Verified] `phi^2 + phi^-2 = 3` (Lucas L2, 1878 -- not original).
- [Efit] (Loop+10, NEW) phi-native Adam beta2 gives NO phi-specific benefit:
  the apparent 0.966 win is a generic beta2-tuning effect (non-phi 0.92-0.98
  win identically); aggressive phi-native beta2 (0.618, 0.382) is harmful.
- [Efit] (Loop+9, carried) pure phi-momentum (beta1=phi^-1) is indistinguishable
  from standard once lr AND weight_decay are controlled (delta -0.027 BPB).
- [Verified] (carried) weight_decay = phi^-3 (0.236) is the dominant harmful
  knob; the only AdamW knob where phi structure is robustly bad.
- [Inert] (Loop+10, NEW) grad_clip is dead metadata in this trainer (never
  applied in opt_step); not a knob until wired into opt_step.
- [Efit, negative] (Loop+10 B, NEW) at hidden=128 (493K params), 12200 steps,
  C-only: code_val_bpb reaches a record-low 1.038 but compile@1 = pass@1 = 0/4.
  The capacity ceiling for compiling C is above 493K params; the --resume
  curriculum machinery is verified working (monotonic BPB descent across
  bit-exact resumed chunks).
- [Retr] "phi-momentum beats standard"; "phi axis worst" (Loop+8 framing
  attributed to decay confound); "phi-native beta2 0.966 beats standard"
  (this loop's control sweep retracts the phi attribution).

## Files touched (coder branch `feat/igla-coder-v1`)

- MOD `src/bin/igla_coder.rs` (NEW `--beta2` flag threaded through the optimizer
  builders; NEW `--resume` flag + `train_from_model` on `generate`)
- NEW `coder_ablation/ablate_beta2_sweep.py` (beta2 isolation harness)
- NEW `coder_ablation/coder_beta2_sweep.csv` (C primary sweep result)
- NEW `coder_ablation/coder_beta2_control.csv` (C non-phi control result)
- NEW `coder_ablation/plot_confound_ladder.py` (A figure generator)
- NEW `coder_ablation/confound_ladder.png` (A figure)
- NEW `coder_ablation/run_cB_curriculum.sh` (B curriculum driver)
- NEW `coder_ablation/eval_pass1_loop10_cB.csv` (B compile@1/pass@1 = 0/4)
- NEW `coder_ablation/LOOP10_REPORT.md` (this report)

NOT committed (gitignored, regenerable): `ckpt_cB_h128.bin` (1.9 MB checkpoint),
`data/*.bin`, `loop10_*.log`.

## Three options for Loop+11

| Option | Header | Direction | Cost / Risk |
|---|---|---|---|
| A | wire grad_clip into opt_step, then sweep it | grad_clip is currently inert dead metadata. Implement actual gradient clipping in opt_step (global-norm or per-tensor), re-run gradcheck, then run the symmetric isolation sweep (pin lr+wd+beta1+beta2, vary only grad_clip) to finally test the LAST AdamW knob honestly. Completes the per-knob audit with a real, not phantom, mediator. | Medium. Trainer change + gradcheck + ~12 min CPU sweep. |
| B | scale the capacity probe (B continued) | If the hidden=128 curriculum still yields compile@1=0, the honest next step is a capacity scan (hidden=192/256, layers=3) with the resume machinery, to locate the first non-zero compile@1 budget -- or to document the CPU ceiling below which C synthesis cannot emerge, as a real negative bound. | High compute, multi-session. The P4/P5 capacity question. |
| C | external replication packet | Bundle the confound ladder + the beta2 control logic into a one-page methods note for an external reader (the per-knob isolation + control-sweep pattern is the transferable artifact). Pairs with the existing P2 external-audit drafts. No new compute; turns the closed phi-optimizer-prior audit into a citable methodology. | Low. Doc only. |

STOP. Pick A / B / C (or a combination) for Loop+11.
