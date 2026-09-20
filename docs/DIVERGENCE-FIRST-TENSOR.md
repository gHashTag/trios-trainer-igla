# The first tensor that diverges across the ISA boundary

Measured 2026-08-06 on one macOS host, branch `fix/509-qat-v2`.
Instrument: `TRIOS_TRACE_STAGE=1` (src/train_loop.rs, `STAGE TRACE` block).
Driver: `scripts/stage_trace_isa.py`. Guard: `tests/stage_trace_inert.rs`.
Evidence: `evidence/stage-trace-isa/`.

## Why bisect in space

`scripts/local_isa_probe.py` established that `0.bin` is byte-identical across
the ISA change and `10.bin` is not. That is a bisection in TIME, and it had
narrowed the window to ten steps. Narrowing it further costs one run per step
and ends at "step 1", which is still a whole training step and names nothing.

Bisecting in SPACE ends at a NAME. Hash every intermediate tensor of one
forward/backward pair at step 1, on both arms, and report the first stage whose
hash differs. One run per arm.

Two candidate explanations were already excluded before this pass:

- **reduction order** - `dot_sequential_4096` and `dot_split8_4096` hash
  identically ACROSS the arms while disagreeing WITH EACH OTHER on each arm, so
  the two instruction sets associate a long sum the same way, twice over.
  `-force-vector-width=1` moved not one output byte.
- **FMA contraction** - excluded by disassembly; neither binary contains one.

What survived was libm (`expf`, called ~13,300 times per step in two different
softmaxes; `cosf`, once per step) and the discontinuous NCA amplifier at
`src/train_loop.rs:2986`, where `.round()` buckets a value so that one ULP can
flip a bucket and multiply every element of `gp` by a different scale.

## The measurement

```
FIRST DIVERGENCE: l1_scores_post_softmax
  aarch64 f93988580b7ae44679209792bfdc9d8d9d88296fa68e3eef72330785dd6da07c
  x86_64  095bec698a30d2eb959d47b31f5d27e4c2771cafe46933d4e6829aa7267b4302
```

18 of 53 traced stages differ. Within-arm control PASSED (n=2 per arm, all 53
stages self-matching on both). Both traced runs reproduced the step-10
checkpoints measured WITHOUT the trace (`efef1cba...` / `5913542e...`), so the
instrument observed the divergence and did not cause it.

`l1_scores_pre_softmax` is SAME. Between it and `l1_scores_post_softmax` there
is exactly one function, `softmax_inplace` (src/model_hybrid_attn.rs:1383):
subtract the max (exact), `expf`, sum (a sequential f32 accumulation in source
order, identical on both arms), divide (exact). **The divergence is `expf`.**

## The reading table

Written before the run, so the result could not be fitted to it afterwards.

| First DIFF at | with | reads as |
|---|---|---|
| `l1_scores_post_softmax` | `l1_scores_pre_softmax` SAME | **`expf` in `softmax_inplace`** (src/model_hybrid_attn.rs:1383) |
| `logits_post_softmax` | all attention stages SAME | `expf` in `train_loop::softmax` (src/train_loop.rs:800) only |
| nothing, but `lr_bits` DIFF | all gradients SAME | `cosf` in `cosine_lr`. Fix costs nothing - **and DOES NOT TRANSFER.** See the trap below. |
| `bc1_bits` / `bc2_bits` | - | `__powisf2` / the inlined square-and-multiply behind `f32::powi`. A **toolchain** finding, not an ISA finding. |
| `gp_post_nca_scale` | `gp_pre_nca_scale` SAME **and** `nca_ent_bits` differing by MORE than 1 ULP | the `.round()` bucket flip at src/train_loop.rs:2986. Would qualify the "1 ULP amplified over 12000 steps" narrative in `docs/CROSS-ARCH-DIVERGENCE.md`. |
| any PRE-libm stage: `ln`, `l1_q`, `l1_k`, `l1_v`, `l1_scores_pre_softmax`, `hidden_raw`, `logits_pre_softmax` | - | reduction order after all, in a compiled unit the primitive probe does not cover. Would CONTRADICT `docs/DIVERGENCE-MECHANISM.md` and is the most valuable outcome available. |

The measured answer is row 1. Every pre-libm stage is SAME, so row 6 did not
happen and `docs/DIVERGENCE-MECHANISM.md` is corroborated, not contradicted.
(That file is owned by another work item and is not edited here.)

## The trap: `cosf` is cheap to fix and does not transfer

`warmup = args.steps / 10` (src/train_loop.rs:2786). In the **12000-step
headline regime** warmup is 1200, so steps 1-1199 take the linear branch of
`cosine_lr` and **`cosf` is not called at all** for the first 1,199 steps. In
the **10-step probe regime** warmup is 1, so step 1 takes the cosine branch.

`expf` fires from step 1 in BOTH regimes. A fix aimed at `cosf` would therefore
cost nothing, look like progress, and change nothing about the artifact that is
actually cited.

## Three stages that look like evidence and are not

Named because a green "SAME" here would otherwise be read as an acquittal.

- **`lr_bits` SAME does not exonerate `cosf`.** At step 1 of a 10-step run
  `p = (1 - 1)/(10 - 1) = 0`, so the call is `cos(0.0)`, which is exactly 1.0 in
  any implementation. The stage is vacuous for its intended subject.
- **`bc1_bits` / `bc2_bits` SAME does not exonerate `powi`.** At step 1 the
  exponent is 1, so `beta.powi(1)` returns the base unchanged. Also vacuous.
- **Every layer-2 stage is vacuous.** Layer 2 is frozen at zero weights on this
  path (see `run_single_emits_a_loadable_artifact_and_freezes_layer_two`), so
  `l2_q`, `l2_k`, `l2_v` and `l2_attn_out` are 512 zeros - their hash is the
  hash of a zero buffer - `l2_scores_pre_softmax` is 144 zeros, and
  `l2_scores_post_softmax` is `exp(0)` normalised, which is exact. Likewise
  `post_update_wq2` ... `post_update_wo2` stay all-zero, and
  `post_update_attn_down` cannot differ because `compute_grads` never writes
  `g_attn_down` at all - `attn_down` moves only under weight decay, from an
  identical initial value.

At least 13 of the 53 stages are structurally incapable of showing a difference
at this configuration. "18 of 53 differ" must be read against 40 stages that
could differ, not 53.

## What the trace found that the brief did not anticipate

Four results the reading table had no row for. They are measurements, and they
constrain the story more tightly than the headline does.

**1. The first divergence is washed out inside the same forward pass.**
`l1_scores_post_softmax` DIFF and `l1_attn_out` DIFF (inherited), and then
`attn_out` - the final attention output for that position - is SAME again, as
are `attn_up_out`, `hidden_raw`, `hidden`, `logits_pre_softmax` and
`logits_post_softmax`. The residual add and `layer_norm_rows` renormalised the
difference below the f32 rounding grid for this position. Divergence in this
trainer is not monotone: a stage returning to SAME is not evidence that an
earlier DIFF was noise.

**2. It therefore does not reach the artifact through the traced position.**
All six gradient stages of the traced pair - `d_hidden`, `g_head`,
`g_attn_up`, `g_proj`, `g_embed`, `g_ctx` - are SAME. But
`gp_pre_nca_scale`, which is the `g_proj` accumulator over all 32 accumulation
forwards after `/= tp`, DIFFERS. The divergence reaches the step-1 weight update
through one or more of the OTHER 31 sampled positions, where it survived the
layer-norm wash-out. The mechanism is `expf`; the path is the accumulation.

**3. On this host the artifact does not depend on the optimisation level.**
`tests/stage_trace_inert.rs` was run under both `cargo test --release` and the
default `test` profile. The unoptimised `trios-train` produced the same
`efef1cba...` as the release build. Opt-level is one more variable that does NOT
move this checkpoint on macos/aarch64 - measured on that host only, and not a
claim about any other.

**4. The NCA amplifier did not fire differently at step 1.** `nca_ent_bits` and
`nca_loss_bits` are byte-identical across the arms, so no `.round()` bucket
flipped and `gp_post_nca_scale` differs only by inheritance from
`gp_pre_nca_scale`. The amplifier is live at step 1 (`gp_pre` and `gp_post`
differ from each other, so a non-zero penalty was applied) - it simply applied
the SAME scale on both arms. This does not show the bucket flip cannot happen
later; it shows it is not the step-1 mechanism.

## Scope, and what it does not say

- One macOS host, one pinned `rustc 1.96.0`, one working tree, `--locked`
  dependencies. The ISA target is the only variable.
- **The x86_64 arm runs under Rosetta 2 binary translation, not native x86_64
  silicon.** A third measurement exists at fixed architecture and different OS -
  `a32e9b2a...` on x86_64 Linux, glibc 2.39, CI run 31004703001 - and it differs
  from the Rosetta x86_64 value, so the OS and its libm also move the bytes.
  This trace localises the divergence WITHIN the macOS pair; it does not carry
  to the Linux arm without being re-run there.
- The `expf` attribution is by elimination inside a four-operation function, not
  by direct comparison of `expf` outputs across the arms. Nothing here measured
  how many of the score values differ, or by how many ULP.
- `l1_attn_out` is the attention-weighted value sum BEFORE the `W_o` projection.
  Three reductions between it and `l2_q` are untraced: the `W_o` matmul, the
  residual add and `layer_norm_rows`. A first DIFF at `l2_q` would have had
  three candidate producers. It did not occur, so this gap did not bite here.

## Follow-ups this pass did not do

- `evidence/SEALS.txt` has no line for `evidence/stage-trace-isa/`. That file is
  owned by another work item; sealing this directory is outstanding.
- Measure `expf` agreement across the arms directly, at the exact score
  arguments this trace saw, and count the differing values and their ULP
  distance. That converts an elimination into a direct observation.
- `compute_grads` never writes `g_attn_down`, so `attn_down` receives an
  all-zero gradient on every step and changes only under weight decay. Found in
  passing here; it is a defect in the trainer, not in the trace, and belongs to
  whoever owns that gradient.
- Re-run the trace on native x86_64 Linux, where the third checkpoint hash was
  measured, to see whether the first divergent stage is the same one.
