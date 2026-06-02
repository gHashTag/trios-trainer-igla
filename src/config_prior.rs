//! Config-Prior: a portable hyperparameter "exoskeleton" (Coder-Loop+4).
//!
//! Derive a whole optimizer config from ONE generator constant `g` and the
//! self-similar filtration {g^-1, g^-2, g^-3, ...}, instead of tuning each
//! hyperparameter independently. This is a minimum-description-length prior on
//! the CONFIG. It bolts onto any AdamW-style trainer and is removable.
//!
//! Honesty (igla-phi-architecture): only `phi^2 + phi^-2 = 3` is [Verified].
//! `g = phi` is the IGLA prior; `g = 2` / standard are the CONTROL axes. phi is
//! the ORIGIN we measure from, NOT an assumed optimum. The realized coder
//! ablation FALSIFIES the phi arms (phi^-3 decay is robustly harmful); the
//! method survives, phi does not (yet). No hype. CPU-only.
//!
//! Anchor: phi^2 + phi^-2 = 3.

/// The golden ratio phi = (1 + sqrt 5) / 2.
pub const PHI: f64 = 1.618_033_988_749_895_f64;

/// A generated optimizer config. Plain numbers so ANY optimizer can consume it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PriorConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub warmup_steps: usize,
    /// learning-rate multiplier relative to a caller-supplied base lr.
    pub lr_mult: f64,
}

/// The choice of generator. `Phi` is the IGLA prior; the rest are control axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Generator {
    /// g = phi: the IGLA design prior (origin of the coordinate system).
    Phi,
    /// g = 2: dyadic control (self-similar but phi-free).
    Dyadic,
    /// Standard tuned AdamW control (NOT generator-derived; the null baseline).
    Standard,
    /// Arbitrary generator g (for sweeping the axis, e.g. g = e or random).
    Custom(f64),
}

impl Generator {
    /// The numeric generator value, where defined. `Standard` has no single g.
    pub fn value(self) -> Option<f64> {
        match self {
            Generator::Phi => Some(PHI),
            Generator::Dyadic => Some(2.0),
            Generator::Custom(g) => Some(g),
            Generator::Standard => None,
        }
    }
}

/// Nearest Fibonacci number >= a hint, used for the warmup schedule (>= 1).
fn fib_warmup(hint: usize) -> usize {
    let (mut a, mut b) = (1usize, 1usize);
    while b < hint {
        let c = a + b;
        a = b;
        b = c;
    }
    b.max(1)
}

/// Build a config from a generator. `warmup_hint` seeds the Fibonacci warmup.
///
/// For a generator g the rules are:
///   beta1 = g^-1,  beta2 = 1 - g^-5,  weight_decay = g^-3,
///   grad_clip = g^-1,  lr_mult = g^-3,  warmup = nearest Fib >= hint.
/// `Standard` returns the tuned null baseline (0.9, 0.999, 0.04, 1.0, hint, 1.0).
pub fn build(generator: Generator, warmup_hint: usize) -> PriorConfig {
    match generator.value() {
        None => PriorConfig {
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.04,
            grad_clip: 1.0,
            warmup_steps: warmup_hint,
            lr_mult: 1.0,
        },
        Some(g) => {
            let inv = 1.0 / g;
            PriorConfig {
                beta1: inv,
                beta2: 1.0 - inv.powi(5),
                weight_decay: inv.powi(3),
                grad_clip: inv,
                warmup_steps: fib_warmup(warmup_hint),
                lr_mult: inv.powi(3),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // [Verified] arithmetic: the ONLY verified phi fact, used as a runtime anchor.
    #[test]
    fn trinity_anchor_holds() {
        let lhs = PHI * PHI + 1.0 / (PHI * PHI);
        assert!((lhs - 3.0).abs() < 1e-12, "phi^2 + phi^-2 must equal 3");
    }

    // [Verified] arithmetic of the phi-generated knobs (matches arm_hparams).
    #[test]
    fn phi_generated_values_match_arm_hparams() {
        let c = build(Generator::Phi, 21);
        assert!((c.beta1 - 0.618_033_988_749_895).abs() < 1e-12);
        assert!((c.weight_decay - 0.236_067_977_499_79).abs() < 1e-12);
        assert!((c.grad_clip - c.beta1).abs() < 1e-12);
    }

    #[test]
    fn standard_is_the_null_baseline() {
        let c = build(Generator::Standard, 50);
        assert_eq!(c.beta1, 0.9);
        assert_eq!(c.weight_decay, 0.04);
        assert_eq!(c.warmup_steps, 50);
    }

    #[test]
    fn dyadic_control_is_phi_free() {
        let c = build(Generator::Dyadic, 8);
        assert!((c.beta1 - 0.5).abs() < 1e-12); // 2^-1
        assert!((c.weight_decay - 0.125).abs() < 1e-12); // 2^-3
    }

    #[test]
    fn fib_warmup_rounds_up() {
        assert_eq!(fib_warmup(20), 21);
        assert_eq!(fib_warmup(34), 34);
        assert_eq!(fib_warmup(35), 55);
    }
}
