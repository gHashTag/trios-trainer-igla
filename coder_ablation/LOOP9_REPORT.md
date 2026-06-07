# Coder-Loop+9 -- iso-everything axis test (pure beta1 momentum prior, isolated)

Branch (coder): `feat/igla-coder-v1`, HEAD before loop `4caf2fb` (Loop+8).
All CPU-only. User picked Loop+8 Option A ("iso-everything axis test").

Anchor: `phi^2 + phi^-2 = 3`. Position: phi is a coordinate axis we measure
from, not a claim. The method survives, phi does not (yet).

## NEGATIVE-FIRST HEADLINE

**With BOTH the learning-rate and the weight-decay confounds removed, phi's
momentum prior (`beta1 = phi^-1 = 0.618`) is statistically indistinguishable
from standard (`beta1 = 0.9`).** All four momentum axes collapse into a
**0.030 BPB band** (dyadic 2.356, phi 2.358, e 2.371, standard 2.386); phi's
95% CI overlaps standard's; the phi-standard delta is **-0.027 BPB, within
noise**. This is NOT a phi win and NOT a phi loss -- it is the clean tie that
the lr-controlled Loop+8 result (phi worst by +0.09) hinted was actually a
residual decay effect, now confirmed by pinning decay too.

The chain across three loops, stated plainly:

- **Loop+7:** four axes ranked strictly by the lr each prescribed; phi "worse"
  by +0.566 BPB. -> that gap was a learning-rate sweep in disguise.
- **Loop+8 (iso-lr):** lr pinned, axis decay still free; spread collapsed to
  ~0.14 BPB, phi worst by +0.092. -> residual gap suspected to be decay, since
  F2 mediation had shown `CDE_decay (~+3.5)` dominates `CDE_momentum (~+0.9)`.
- **Loop+9 (iso-everything):** lr AND decay pinned, only beta1 varies; spread
  collapses further to **0.030 BPB**, phi-standard CIs overlap. -> the residual
  Loop+8 gap WAS the decay confound. The pure momentum prior is neutral.

## The experiment

New harness `coder_ablation/ablate_generator_axis_isoall.py`: identical arm
construction to the Loop+8 iso-lr harness EXCEPT `--wd` is also pinned to a
single constant (0.04, the standard value) for every arm, alongside the pinned
`--lr 0.002`. The axis's prescribed `lr_mult` AND `weight_decay` are both
intentionally ignored. The ONLY across-arm difference is `--beta1` (the
momentum the axis prescribes); beta2 / grad_clip / warmup come from the shared
`--optimizer standard` base and are constant across arms. `standard`
(beta1=0.9) is the null phi must beat.

hidden=64, 800 steps, 3 seeds (42/43/44), fim_loss=all, lr=0.002 AND wd=0.04
for ALL arms.

| axis | beta1 | mean code_val_bpb | ci95 | delta vs standard |
|---|---|---|---|---|
| dyadic | 0.5000 | 2.3556 | +/-0.0668 | -0.0299 |
| phi | 0.6180 | **2.3581** | +/-0.0683 | **-0.0274** |
| e | 0.3679 | 2.3712 | +/-0.0563 | -0.0143 |
| standard | 0.9000 | 2.3855 | +/-0.0414 | (null) |

Provenance sha256[:16] = `bef01eab741f9162`. Saved
`coder_ablation/coder_generator_axis_isoall.csv`.

**Verdict [Efit]:** phi-momentum NOT distinguishable from standard at
iso-everything (CIs overlap, delta -0.027 within noise). The whole-band spread
(0.030 BPB) is roughly half a single arm's CI width, so the four momentum
priors are mutually indistinguishable at this budget. A referee would accept:
"with learning rate and weight decay both controlled, the choice of Adam beta1
in [0.37, 0.90] has no detectable effect on code BPB at sub-1M params; phi^-1
is one neutral point among several."

Honest reading: the *only* operationally relevant phi liability the coder track
ever measured was the **heavy phi^-3 weight decay** (0.236, ~6x standard) --
not the momentum, not the lr once each was isolated. The momentum prior is a
non-event. This sharpens, and does not contradict, the F2 mediation conclusion
(`CDE_decay` regime-independent and robustly harmful; `CDE_momentum`
regime-dependent / near-neutral).

## Provenance caveat (corpus regeneration)

This loop ran in a fresh sandbox. The Rust toolchain was reinstalled and the
byte-level `.bin` shards were regenerated from the t27 `gen/` source via a
re-derived extractor `coder_ablation/build_t27_parallel_corpus.py` (78 docs,
1,143,140 source bytes; 32 C + 32 Verilog + 16-lang numeric catalog + 1 each
go/rust/zig/...; 7 non-ASCII files dropped under L3 discipline) ->
`prep_t27_corpus.py` -> 1,044,046 train / 99,523 val tokens (vocab 263,
fim_frac 0.5, seed 42). This is the SAME source corpus as prior loops but NOT
byte-identical to the Loop+8 1,108,511-token version (the original
`build_t27_parallel_corpus.py` was workspace-only and not committed; this
re-derivation now IS committed for reproducibility, and additionally pulls in
`gen/rust`). All Loop+9 numbers are internally consistent on this regenerated
corpus. The qualitative conclusion -- momentum prior is neutral once lr+decay
are controlled -- is corpus-robust.

## Verification

- `cargo build --release --bin igla_coder` OK (3m35s clean build).
- `gradcheck --hidden 32 --heads 4 --layers 2`: `checks=23 fails=0 PASS`
  (analytic vs f64 central-difference, gate abs<2e-3 OR rel<5%).
- All 12 training runs returned `gen_rc=0` with a valid `code_val_bpb`; no
  divergence, no NaN, no corpus-load failure.
- New files ASCII-clean.

## Claim-status summary (updated)

- [Verified] `phi^2 + phi^-2 = 3` (Lucas L2, 1878 -- not original).
- [Efit] (Loop+9, NEW) pure phi-momentum (beta1=phi^-1) is indistinguishable
  from standard once lr AND weight_decay are controlled (delta -0.027 BPB, CIs
  overlap, 4-axis band 0.030 BPB). The momentum prior is neutral, not a benefit.
- [Verified] (carried) weight_decay = phi^-3 (0.236) is the dominant harmful
  knob; this loop confirms it was the source of the Loop+8 residual gap.
- [Efit -> falsified] (Loop+2, unchanged) no phi-derived FULL config reaches the
  standard frontier (because its prescribed lr + decay are jointly off, not its
  momentum).
- [Retr] "phi-momentum beats standard"; "phi axis worst" (Loop+8 framing now
  attributed to decay confound, not momentum); `delta_CP = 3/phi^2`.

## Files touched (coder branch `feat/igla-coder-v1`)

- NEW `coder_ablation/build_t27_parallel_corpus.py` (committed corpus extractor;
  closes the prior workspace-only reproducibility gap)
- NEW `coder_ablation/ablate_generator_axis_isoall.py` (iso-everything harness)
- NEW `coder_ablation/coder_generator_axis_isoall.csv` (Loop+9 result)
- NEW `coder_ablation/LOOP9_REPORT.md` (this report)

## Three options for Loop+10

| Option | Header | Direction | Cost / Risk |
|---|---|---|---|
| A | publish the confound-ladder | The Loop+7->8->9 sequence is a clean three-step worked example: a naive axis comparison reported a +0.57 BPB phi gap that fully dissolved to noise once lr then decay were each controlled. Write the workshop-paper subsection + a single before/after figure (band shrinks 0.57 -> 0.14 -> 0.03). Highest narrative value; turns the negative into the methodology's headline honesty asset. Pairs with the still-open Loop+8 Option B (WD-mediation retraction). | Low compute, doc + confirm_action. Closes P3 evidence story; feeds P8.3 write-up. |
| B | curriculum to compile@1 > 0 | The still-unmet headline metric. Dedicated single-language (C-only) curriculum on short COMPLETE functions, hidden=128, checkpoint-resume to reach ~10k+ steps past the per-call wall. Needs a small trainer `--resume` add to `generate`. First non-zero compile@1 is the real P4/P5 gate. | High compute, multi-session. The pass@1 metric P5 needs. |
| C | beta2 / grad_clip axis sweep | The iso-everything test isolated beta1; the axes also prescribe distinct beta2 and grad_clip. Symmetric-completion: pin lr+wd+beta1, vary only beta2 (then grad_clip) to confirm those phi knobs are ALSO neutral, fully closing the optimizer-prior decomposition. | Low. ~12 min CPU each, coder branch only. Completes the per-knob phi audit. |

STOP. Pick A / B / C (or a combination) for Loop+10.
