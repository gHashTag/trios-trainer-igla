# Coder-Loop+8 -- iso-lr control, main-track WD re-baseline, scaled pass@1

Branch (coder): `feat/igla-coder-v1`, HEAD before loop `a30ddb6` (Loop+7).
Branch (main): `f2-methodology`, HEAD `f2a4f6d` (Loop 137), fix applied locally.
All CPU-only. User picked A + B + C ("vse").

Anchor: `phi^2 + phi^-2 = 3`. Position: phi is a coordinate axis we measure
from, not a claim. The method survives, phi does not (yet).

## NEGATIVE-FIRST HEADLINE

- **A: phi is NOT special once lr is controlled.** At equal effective lr the
  four generator axes collapse into a ~0.14 BPB band (e 2.612, dyadic 2.647,
  standard 2.656, phi 2.748). phi is the WORST of the four, by +0.092 BPB vs
  standard. The clean monotone-in-lr ranking from Loop+7 disappeared, which is
  itself the finding: the Loop+7 "+0.566 BPB phi gap" was almost entirely a
  learning-rate confound, not a property of the (beta1, weight_decay) prior.
- **B: the main-track WD-mediation finding is empirically RETRACTED.** On the
  fixed optimizer the entire WD sweep is flat: bpb(wd=0.3) - bpb(wd=0.0) drops
  from +2.97 BPB (buggy) to +0.11 BPB (fixed, within noise). ~96% of the
  measured WD effect was the weight-collapse artifact. The frozen regression
  baseline (phi 5.93 / zoo 5.83) was itself a bug artifact and is re-based to
  the honest post-fix values (phi 2.78 / zoo 2.29).
- **C: scaling 2x did NOT buy a non-zero compile@1.** A 296K-param checkpoint
  (hidden=96, val BPB 2.069, better than Loop+7's 148K / 2.254) still scores
  compile@1 = 0/4, pass@1 = 0/4. The model learned C-header surface tokens
  (`#include`, `#ifndef`/`#define`, GF identifiers) but emits degenerate
  repetitive non-compilable text. This is a capacity ceiling, not a bug
  (`gen_rc=0` throughout -- the Loop+7 corpus-load fix holds).

## A -- generator-axis ablation, ISO-LR control

New harness `coder_ablation/ablate_generator_axis_isolr.py`: identical to the
Loop+7 harness EXCEPT every arm trains at the same effective lr (`--lr 0.002`),
with `lr_mult` intentionally ignored. The only across-arm difference is the
(beta1, weight_decay) the axis prescribes. `standard` is the null phi must beat.

hidden=64, 800 steps, 3 seeds (42/43/44), fim_loss=all, lr=0.002 for ALL arms.

| axis | beta1 | wd | mean code_val_bpb | ci95 |
|---|---|---|---|---|
| e | 0.3679 | 0.0498 | **2.6124** | +/-0.0309 |
| dyadic | 0.5000 | 0.1250 | 2.6468 | +/-0.0359 |
| standard | 0.9000 | 0.0400 | 2.6560 | +/-0.0448 |
| phi | 0.6180 | 0.2361 | 2.7482 | +/-0.0366 |

Provenance sha256[:16] = `0ec2a1838b5eab52`. Saved
`coder_ablation/coder_generator_axis_isolr.csv`.

**Verdict [Efit]:** phi is the worst arm (+0.092 vs standard, CIs barely
disjoint), but the spread across ALL four axes is only 0.136 BPB and the Loop+7
lr-monotone ranking is gone. Controlling lr removed the large gap; what remains
is a small, second-order (beta1, wd) effect in which phi's heavy decay
(wd=phi^-3=0.236) is mildly unhelpful at this budget. Honest reading: phi-as-
axis is a near-competitor whose only real disadvantage is the slow lr its prior
prescribes; the momentum/decay prior itself is roughly neutral. NOT a phi win,
NOT a catastrophe. A referee would accept "lr-controlled, phi indistinguishable
from the format zoo to within ~0.1 BPB at sub-1M params."

Caveat (provenance): the sandbox reset mid-loop wiped `/tmp`; `data/*.bin` were
regenerated from the persistent `t27_corpus/corpus.jsonl` and now total
1,108,511 train tokens (vs the original 1,094,761 -- a slightly newer corpus
version). All Loop+8 numbers are on this regenerated corpus and are internally
consistent; they are NOT byte-identical to pre-Loop+8 runs. The qualitative
conclusion is unchanged from the pre-reset run (phi worst by ~0.09 BPB).

## B -- main-track WD-mediation re-generated on the fixed optimizer

Cloned `f2-methodology` read-write, applied the same lr-scaled decay fix the
coder track shipped in 5d6a7d2 to `src/optimizer.rs:121`
(`params -= (lr*wd)*params`, citing Loshchilov & Hutter 2019). Re-ran
`f2_ablation_sweep --mode wd_sweep` (6 WD values x 5 seeds = 30 trainings,
synthetic corpus, steps=50) on BOTH the buggy and the fixed optimizer for a
direct contrast.

| wd | BUGGY mean BPB | FIXED mean BPB |
|---|---|---|
| 0.000 | 2.878 | 2.878 |
| 0.005 | 3.446 | 2.899 |
| 0.010 | 3.828 | 2.871 |
| 0.030 | 5.010 | 2.815 |
| 0.100 | 5.587 | 2.871 |
| 0.300 | 5.844 | 2.987 |

**WD effect bpb(0.3) - bpb(0.0): BUGGY +2.97 BPB -> FIXED +0.11 BPB.** The
buggy sweep's near-monotone BPB rise with WD (the basis of the "WD strongly
mediates BPB / NIE_M1 via WD = +4.99" finding) is almost entirely the
weight-collapse artifact: at wd=0 (the one bug-free point) buggy and fixed are
byte-identical (2.878), and the gap grows with wd exactly as `params *= (1-wd)`
collapse predicts. After the fix the sweep is flat to within the +/-0.19..0.30
seed CIs -- WD has NO statistically detectable BPB effect at this scale.

Saved `f2m_wd_sweep_buggy.csv` + `f2m_wd_sweep_fixed.csv` (workspace).

**Claim-status resolution (closes the Loop+7 "[Risk] pending re-check"):**
- Main-track WD-mediation slope / `NIE_M1 via WD = +4.99` -> **[Retr]**
  (confirmed: ~96% artifact, residual +0.11 within noise).
- `wd0` stratum results -> **[Risk]** unchanged (wd=0 is the bug-free point and
  is sound in isolation, but every cross-stratum CONTRAST it fed used buggy
  wd>0 arms).
- The F2 methodology (stratified CDE, Lambda-sweep, provenance, 12 binaries,
  703 tests) is SOUND and survives; only the WD-sweep NUMBERS were artifacts.
  Loop+8 turns this into a documented worked example of the methodology
  catching a confound a naive BPB comparison would miss.

**Honest side-finding (re-baseline).** The frozen regression test
`regression_phi_zoo_baseline_2026_06_01` pinned phi BPB ~= 5.93 / zoo ~= 5.83.
Those values were THEMSELVES the bug artifact (collapse at WD=0.1). The fix
drops them to phi 2.78 / zoo 2.29; the test is re-based to the honest post-fix
values with a comment, and the full suite returns to green (702+1-fail ->
703 pass). This is a clean example of a "frozen-baseline" test correctly
firing when underlying math is corrected.

## C -- scale toward non-zero compile@1

Trained a 2x-larger checkpoint: hidden=96 (296,064 params vs Loop+7's
148,224), 1500 steps, wd=0, lr=0.002, fim_loss=all, seed=42. Train BPB
8.25 -> 1.77; trained val BPB **2.069** (better than Loop+7's 2.254 -- the
extra capacity does reduce BPB). Saved `/tmp/coder_scaled_h96.bin`.

Scored with `coder_ablation/eval_pass1.py --mode load` on the 4 C micro-specs
(add_u32, clamp_u8, popcount8, mod_add):

**compile@1 = 0/4, pass@1 = 0/4** [Verified negative]. `gen_rc=0` on all 4
(no path/corpus bug -- the Loop+7 load-generate fix holds). Saved
`coder_ablation/eval_pass1_loop8.csv`.

Raw-generation diagnostic (temp=0.2): given `#include <stdint.h>` the model
emits header-guard surface structure (`#ifndef`-like, `#define`, MASK/GF
identifier fragments from the corpus) but degenerates into repetitive
non-syntactic text. It has learned the TOKEN DISTRIBUTION of C headers, not
syntactic validity. Honest conclusion: compile@1 > 0 needs more than a 2x
param bump -- likely a single-language curriculum on short complete functions
plus a much larger step budget, which exceeds the per-call CPU wall limit and
should be a dedicated Loop (see Loop+9 Option C).

## Verification

- Coder branch: `cargo build --release --bin igla_coder` OK; gradcheck
  `checks=23 fails=0 PASS`; `cargo test --release --lib` = **559 passed**
  (unchanged). 3 new files, all ASCII-clean, no unintended diffs.
- f2-methodology branch: `cargo build --release --bin f2_ablation_sweep` OK;
  `cargo test --release --lib` = **703 passed** (after re-baseline);
  `optimizer.rs` + `multi_seed.rs` added lines ASCII-clean (0/0).

## Files touched

Coder branch (`feat/igla-coder-v1`):
- NEW `coder_ablation/ablate_generator_axis_isolr.py` (iso-lr harness)
- NEW `coder_ablation/coder_generator_axis_isolr.csv` (wave A result)
- NEW `coder_ablation/eval_pass1_loop8.csv` (wave C result)
- NEW `coder_ablation/LOOP8_REPORT.md` (this report)

Main branch (`f2-methodology`), SEPARATE commit:
- `src/optimizer.rs` (lr-scaled decoupled weight decay, +9/-2)
- `src/race/multi_seed.rs` (re-baseline regression test to post-fix values)

## Three collaboration options for Loop+9

| Option | Header | Direction | Cost / Risk |
|---|---|---|---|
| A | iso-everything axis test | Re-run wave A controlling BOTH lr AND wd (pin wd=0.04 for every arm, vary only beta1) to isolate the pure momentum prior -- the last confound. Settles whether phi's beta1=phi^-1 alone helps/hurts. | Low. ~12 min CPU, coder branch only. Cleanest possible phi-as-axis statement. |
| B | publish the confound-catch | Write the F2 WD-mediation retraction up as the methodology's worked example: stratified-CDE caught a confound (the decay bug) that a naive BPB comparison reported as a +4.99 effect. Draft the workshop-paper section + a clean before/after figure from `f2m_wd_sweep_{buggy,fixed}.csv`. | Low compute, high narrative value. Cross-branch doc work + confirm_action. Turns a negative into the paper's strongest honesty asset. |
| C | curriculum to compile@1 > 0 | Dedicated scaling loop: single-language (C only) curriculum on short COMPLETE functions, hidden=128, multi-call checkpoint-resume to reach ~10k+ steps despite the wall limit. Goal: first compile@1 > 0. | High compute, multi-session. The headline pass@1 metric, but needs a resume path (generate has --save but no --resume yet -- a small trainer change). |

STOP. Pick A / B / C (or a combination) for Loop+9.
