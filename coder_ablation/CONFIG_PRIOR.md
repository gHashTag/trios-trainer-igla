# Config-Prior: a portable hyperparameter "exoskeleton" (Loop+4)

> Status discipline per `igla-phi-architecture`. Only `phi^2 + phi^-2 = 3` is
> [Verified]. phi is a falsifiable DESIGN PRIOR used as a coordinate axis, NOT a
> proven optimum. No hype words. English+ASCII only. CPU-only.

## Idea in one line

Derive an entire optimizer/attention/schedule config from ONE generator constant
`g` and a self-similar filtration `{g^-1, g^-2, g^-3, ...}`, instead of tuning each
hyperparameter independently. This is a minimum-description-length (MDL) prior on
the CONFIG -- a compact, reproducible "exoskeleton" that can wrap any trainer.

The generator `g` is a free choice. Setting `g = phi` is the IGLA prior. Setting
`g = 2` (dyadic) or a tuned-standard config are the CONTROL axes. The framework's
value is that it makes "phi vs not-phi" a one-line, measurable swap -- phi is the
ORIGIN of the coordinate system we measure from, never an assumed winner.

## The generated config (single generator g, default g = phi)

| Knob | Rule | g = phi value | Standard control |
|---|---|---|---|
| AdamW beta1 | `g^-1` | 0.6180339887 | 0.9 |
| AdamW beta2 | `1 - g^-5` (slow second moment) | 0.9098... | 0.999 |
| weight_decay | `g^-3` | 0.2360679775 | 0.04 |
| grad_clip | `g^-1` | 0.6180339887 | 1.0 |
| warmup (steps) | Fibonacci(k): 21, 34, 55 ... | 34 | linear/const |
| lr (relative) | base * `g^-3` | base * 0.236 | base * 1.0 |

Only the ARITHMETIC of these (e.g. `g^-3 = 0.2360679775` for g = phi) is
[Verified]. That any of them is BETTER than the standard control is [Open
conjecture] with a falsification path: a non-phi generator of equal cost reaching
equal BPB falsifies phi-specificity.

## The measurement scaffold (freeze -> hash -> null -> compare)

Carried from the Catalog42 protocol (igla-phi-architecture):

1. FREEZE the config, serialize it, take a SHA-256 hash before any run.
2. NULL baseline: same budget with the standard control config (g-free).
3. COMPARE arms at EQUAL tuning budget and EQUAL lr; report BPB mean +/- 95% CI
   over >= 3 seeds. If CIs overlap, phi is NOT supported -- say so first.
4. Robustness: Gamma_tip (E-value) for any non-zero effect; >= 2.0 == robust.

This is exactly the harness already wired in `igla_coder ablate` (arms
standard / phi / phi_b1 / phi_wd) and the F2 mediation pipeline. The config-prior
module just makes the generator pluggable so the SAME harness can test g = phi vs
g = 2 vs g = e vs random-rational.

## How it wraps ANY model

The prior is generator-only: it emits (beta1, beta2, wd, grad_clip, warmup, lr_mult)
as plain numbers. Any optimizer that accepts those (AdamW, Lion, AdamWCpu here)
can consume them. Porting to a new trainer = one function call returning a struct;
the model architecture is untouched. That is the "exoskeleton": bolt-on, removable,
measurable, falsifiable.

## Realized in-repo evidence (negative-first, Coder-Loop+1..+3)

- phi^-3 weight_decay is robustly HARMFUL on the coder: decay CDE = +3.544 BPB
  [3.420, 3.643], Gamma_tip ~65, stable across canonical + mom_std strata. [Verified]
- No phi config reaches the standard frontier (BPB 3.775). [Verified-falsification]
- phi_b1 (beta1 = phi^-1) TIES standard with 2.7x higher variance -- no benefit.
- The method (the exoskeleton + harness + F2 mediation) survives; phi does not (yet).

## Files

- `src/config_prior.rs` -- the pluggable generator module (this Loop+4).
- `src/bin/igla_coder.rs` -- `arm_hparams` consumes phi arms; `ablate` is the harness.
- `coder_ablation/CONFIG_PRIOR.md` -- this spec.
