//! Multi-seed Welch two-sample comparison of the phi-ladder vs the zoo arm.
//!
//! STATUS DISCIPLINE (skill `goldenfloat-ladder` / `igla-phi-architecture`).
//! "the method survives, phi does not (yet)." This module produces a
//! statistically honest verdict over multiple seeds, and the verdict is allowed
//! to be ZooWins or Tie. A PhiWins on this in-sandbox PROXY is NOT a Verdict and
//! never promotes the breadth moat (FL-002) to [Verified]; the per-rung accuracy
//! axis is explicitly NOT the moat. The moat is breadth / fewer lossy
//! cross-format conversions, reported separately as a mechanical count.
//!
//! Statistics: Welch's two-sample t-test (unequal variances), with the
//! Welch-Satterthwaite degrees of freedom and a two-sided p-value from a
//! Student-t survival function. We compare proxy_bits across seeds (LOWER is
//! better), so a NEGATIVE mean difference (phi - zoo) favours phi.
//!
//! DEGENERATE INPUT IS AN ERROR, NOT A RESULT. `welch` used to answer a sample
//! with no spread by returning `t = inf, p = 0.0` when the means differed and
//! `p = 1.0` when they did not - a manufactured "p < 0.001" from a comparison
//! that has no standard error to divide by, and a manufactured "perfect tie"
//! from the same absence of information. `p = 0` printed under the heading
//! "two-sided p" is indistinguishable from a real significance claim, which is
//! precisely the failure `src/race/victory.rs` names `DegenerateSample`. Both
//! paths now return [`WelchError`], the verdict becomes `None`, and the caller
//! is required to say "undefined" instead of quoting a number.

use crate::format_ladder::{run_arm, ConversionCounter, Family};

/// Per-arm aggregate over seeds.
#[derive(Clone, Debug)]
pub struct ArmSamples {
    /// proxy_bits for each seed (lower is better).
    pub bits: Vec<f64>,
    /// Total lossy cross-format conversions summed across seeds.
    pub total_lossy: u64,
    /// Total coherent in-family widenings summed across seeds.
    pub total_coherent: u64,
}

impl ArmSamples {
    fn mean(&self) -> f64 {
        self.bits.iter().sum::<f64>() / self.bits.len() as f64
    }
    /// Unbiased (n-1) sample variance, or `None` when `n < 2`.
    ///
    /// `None` rather than `0.0`: one observation has no dispersion to report,
    /// and returning zero made a single-seed run look like a perfectly precise
    /// one instead of an uninformative one.
    fn var(&self) -> Option<f64> {
        let n = self.bits.len();
        if n < 2 {
            return None;
        }
        let m = self.mean();
        Some(self.bits.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (n as f64 - 1.0))
    }
}

/// Why a Welch comparison could not be computed.
///
/// Each variant is a refusal to publish a p-value, not a p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WelchError {
    /// An arm has fewer than two seeds, so its variance is undefined.
    InsufficientSamples { n_phi: usize, n_zoo: usize },
    /// The pooled standard error is at or below the floating-point resolution
    /// of the values themselves: the samples carry no resolvable spread. The
    /// test is not `se == 0.0`, for the same reason as in
    /// `race::victory::run_ttest` - three bit-identical readings can sum to a
    /// mean whose round-off produces a nonzero-but-meaningless `se`.
    DegenerateSample {
        pooled_se: f64,
        noise_floor: f64,
        n_phi: usize,
        n_zoo: usize,
    },
    /// The t statistic came out non-finite (NaN or infinite) despite a nonzero
    /// standard error - a non-finite input reached the arms.
    NonFiniteStatistic { t_stat: f64 },
    /// The Student-t tail could not be evaluated at this `(t, df)`.
    UndefinedTailProbability { t_stat: f64, df: f64 },
}

impl std::fmt::Display for WelchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientSamples { n_phi, n_zoo } => write!(
                f,
                "fewer than 2 seeds in an arm: variance undefined (n_phi={n_phi}, n_zoo={n_zoo})"
            ),
            Self::DegenerateSample {
                pooled_se,
                noise_floor,
                n_phi,
                n_zoo,
            } => write!(
                f,
                "zero pooled variance: pooled_se={pooled_se:.3e} <= noise_floor={noise_floor:.3e} \
                 (n_phi={n_phi}, n_zoo={n_zoo})"
            ),
            Self::NonFiniteStatistic { t_stat } => {
                write!(f, "non-finite Welch t statistic (t={t_stat})")
            }
            Self::UndefinedTailProbability { t_stat, df } => write!(
                f,
                "Student-t tail undefined at t={t_stat}, df={df}"
            ),
        }
    }
}

impl std::error::Error for WelchError {}

/// The computable part of a Welch comparison.
#[derive(Clone, Copy, Debug)]
pub struct WelchStats {
    /// Welch t-statistic.
    pub t_stat: f64,
    /// Welch-Satterthwaite degrees of freedom.
    pub df: f64,
    /// Two-sided p-value.
    pub p_two_sided: f64,
}

/// The three possible breadth-harness verdicts on the accuracy axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F2Verdict {
    /// phi proxy loss significantly LOWER than zoo (p2 < alpha, mean_diff < 0).
    PhiWins,
    /// No significant difference at level alpha.
    Tie,
    /// zoo proxy loss significantly lower than phi (p2 < alpha, mean_diff > 0).
    ZooWins,
}

/// Full multi-seed report.
#[derive(Clone, Debug)]
pub struct MultiSeedReport {
    pub phi: ArmSamples,
    pub zoo: ArmSamples,
    /// mean(phi) - mean(zoo). Negative favours phi. `NaN` if an arm is empty.
    pub mean_diff: f64,
    /// The Welch statistics, or the reason there are none.
    pub welch: Result<WelchStats, WelchError>,
    /// Significance level used.
    pub alpha: f64,
    /// Accuracy-axis verdict, or `None` when `welch` is an error.
    ///
    /// A verdict without a p-value is not a weaker verdict, it is not a
    /// verdict; callers must print the refusal rather than defaulting to `Tie`.
    pub verdict: Option<F2Verdict>,
}

/// Run both arms over `seeds.len()` seeds and compute the Welch comparison.
///
/// `warmup_steps_unquantized` is the number of leading steps each arm runs in
/// full precision before quantization kicks in (continual-QAT warmup); this
/// prevents an early-saturation strawman against the phi arm.
pub fn run_multi_seed(
    seeds: &[u64],
    steps: usize,
    warmup_steps_unquantized: usize,
    dim: usize,
    alpha: f64,
) -> MultiSeedReport {
    let mut phi_bits = Vec::with_capacity(seeds.len());
    let mut zoo_bits = Vec::with_capacity(seeds.len());
    let mut phi_conv = ConversionCounter::default();
    let mut zoo_conv = ConversionCounter::default();

    for &s in seeds {
        let p = run_arm(Family::Phi, s, steps, warmup_steps_unquantized, dim);
        let z = run_arm(Family::Zoo, s, steps, warmup_steps_unquantized, dim);
        phi_bits.push(p.proxy_bits);
        zoo_bits.push(z.proxy_bits);
        phi_conv.lossy += p.conversions.lossy;
        phi_conv.coherent += p.conversions.coherent;
        zoo_conv.lossy += z.conversions.lossy;
        zoo_conv.coherent += z.conversions.coherent;
    }

    let phi = ArmSamples {
        bits: phi_bits,
        total_lossy: phi_conv.lossy,
        total_coherent: phi_conv.coherent,
    };
    let zoo = ArmSamples {
        bits: zoo_bits,
        total_lossy: zoo_conv.lossy,
        total_coherent: zoo_conv.coherent,
    };

    let (mean_diff, welch) = welch(&phi, &zoo);

    let verdict = welch.as_ref().ok().map(|w| {
        if w.p_two_sided < alpha {
            if mean_diff < 0.0 {
                F2Verdict::PhiWins
            } else {
                F2Verdict::ZooWins
            }
        } else {
            F2Verdict::Tie
        }
    });

    MultiSeedReport {
        phi,
        zoo,
        mean_diff,
        welch,
        alpha,
        verdict,
    }
}

/// Welch two-sample t-test.
///
/// Returns `(mean_diff = mean(a) - mean(b), stats)`. The mean difference is
/// reported even when the test cannot be computed: the difference of two means
/// is an arithmetic fact about the samples, while a p-value is an inference
/// that needs resolvable spread to exist at all.
fn welch(a: &ArmSamples, b: &ArmSamples) -> (f64, Result<WelchStats, WelchError>) {
    let n_phi = a.bits.len();
    let n_zoo = b.bits.len();
    let ma = a.mean();
    let mb = b.mean();
    let mean_diff = ma - mb;

    // `ArmSamples::var` is `None` for n < 2. Substituting `0.0` here would have
    // turned an arm with no measurable dispersion into a noiseless one, which
    // is exactly the manufactured-certainty bug this module exists to refuse.
    let (va, vb) = match (a.var(), b.var()) {
        (Some(va), Some(vb)) => (va, vb),
        _ => {
            return (
                mean_diff,
                Err(WelchError::InsufficientSamples { n_phi, n_zoo }),
            )
        }
    };

    let na = n_phi as f64;
    let nb = n_zoo as f64;
    let sa = va / na;
    let sb = vb / nb;
    let pooled_se = (sa + sb).sqrt();

    // Degenerate input is an error, not a result. The old code answered a
    // spreadless sample with `t = inf, p = 0.0` (or `p = 1.0` when the means
    // also matched) and printed it under the heading "two-sided p". The test is
    // not `pooled_se == 0.0`, for the reason spelled out in
    // `race::victory::run_ttest`: bit-identical readings can still leave
    // round-off in the mean, producing a nonzero but meaningless standard
    // error. The bound is n ulps of the larger mean.
    let noise_floor = ma.abs().max(mb.abs()) * f64::EPSILON * (na + nb);
    if !(pooled_se > noise_floor) {
        return (
            mean_diff,
            Err(WelchError::DegenerateSample {
                pooled_se,
                noise_floor,
                n_phi,
                n_zoo,
            }),
        );
    }

    let t_stat = mean_diff / pooled_se;
    if !t_stat.is_finite() {
        return (mean_diff, Err(WelchError::NonFiniteStatistic { t_stat }));
    }

    // Welch-Satterthwaite df.
    let df_num = (sa + sb) * (sa + sb);
    let df_den = (sa * sa) / (na - 1.0) + (sb * sb) / (nb - 1.0);
    let df = if df_den > 0.0 {
        df_num / df_den
    } else {
        na + nb - 2.0
    };

    match two_sided_p_from_t(t_stat.abs(), df) {
        Some(p_two_sided) => (
            mean_diff,
            Ok(WelchStats {
                t_stat,
                df,
                p_two_sided,
            }),
        ),
        None => (
            mean_diff,
            Err(WelchError::UndefinedTailProbability { t_stat, df }),
        ),
    }
}

/// Two-sided p-value for |t| under a Student-t with `df` degrees of freedom,
/// or `None` when the tail is not evaluable at this `(t, df)`.
///
/// Uses the regularized incomplete beta function via a continued fraction. The
/// non-finite case used to `return 0.0`, i.e. report the most significant
/// p-value representable for an input that carries no information at all.
fn two_sided_p_from_t(t_abs: f64, df: f64) -> Option<f64> {
    if !t_abs.is_finite() || !df.is_finite() || df <= 0.0 {
        return None;
    }
    // p = I_{df/(df+t^2)}(df/2, 1/2)  (this is the two-sided tail probability).
    let x = df / (df + t_abs * t_abs);
    let p = betai(df / 2.0, 0.5, x);
    if !p.is_finite() {
        return None;
    }
    Some(p.clamp(0.0, 1.0))
}

/// Regularized incomplete beta function I_x(a, b).
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let front = (a * x.ln() + b * (1.0 - x).ln() + ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for the incomplete beta function (Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3.0e-12_f64;
    let fpmin = 1.0e-300_f64;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        // even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;
        // odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

/// Lanczos approximation to ln(Gamma(z)) for z > 0.
fn ln_gamma(z: f64) -> f64 {
    const G: [f64; 8] = [
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        // reflection
        std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - ln_gamma(1.0 - z)
    } else {
        let z = z - 1.0;
        let mut x = 0.99999999999980993;
        for (i, &g) in G.iter().enumerate() {
            x += g / (z + i as f64 + 1.0);
        }
        let t = z + G.len() as f64 - 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials() {
        // ln Gamma(n) = ln((n-1)!)
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-9); // 4! = 24
        assert!((ln_gamma(6.0) - 120.0_f64.ln()).abs() < 1e-9); // 5! = 120
    }

    fn arm(bits: Vec<f64>) -> ArmSamples {
        ArmSamples {
            bits,
            total_lossy: 0,
            total_coherent: 0,
        }
    }

    #[test]
    fn two_sided_p_known_values() {
        // t=0 -> p=1 regardless of df.
        assert!((two_sided_p_from_t(0.0, 10.0).unwrap() - 1.0).abs() < 1e-9);
        // Large |t| with moderate df -> very small p.
        assert!(two_sided_p_from_t(10.0, 8.0).unwrap() < 1e-4);
        // Classic: t=2.228, df=10 -> two-sided p ~= 0.05 (t_0.025,10 = 2.228).
        let p = two_sided_p_from_t(2.228, 10.0).unwrap();
        assert!((p - 0.05).abs() < 0.005, "p={p}");
    }

    #[test]
    fn two_sided_p_refuses_non_finite_input() {
        // The old code returned 0.0 here, i.e. "p < 0.001" from a t that does
        // not exist.
        assert_eq!(two_sided_p_from_t(f64::INFINITY, 10.0), None);
        assert_eq!(two_sided_p_from_t(f64::NAN, 10.0), None);
        assert_eq!(two_sided_p_from_t(2.0, 0.0), None);
    }

    #[test]
    fn welch_identical_samples_is_tie() {
        let a = arm(vec![1.0, 2.0, 3.0, 4.0]);
        let b = arm(vec![1.0, 2.0, 3.0, 4.0]);
        let (md, w) = welch(&a, &b);
        assert_eq!(md, 0.0);
        let w = w.expect("spread is present, so the test is computable");
        assert!(w.p_two_sided > 0.99, "p={}", w.p_two_sided);
    }

    #[test]
    fn welch_clear_separation_is_significant() {
        let a = arm(vec![1.0, 1.1, 0.9, 1.05]);
        let b = arm(vec![5.0, 5.1, 4.9, 5.05]);
        let (md, w) = welch(&a, &b);
        assert!(md < 0.0);
        let w = w.expect("spread is present, so the test is computable");
        assert!(w.p_two_sided < 0.01, "p={}", w.p_two_sided);
    }

    #[test]
    fn welch_zero_variance_is_an_error_not_a_verdict() {
        // Two constant arms with DIFFERENT means: the old code answered
        // t = inf, p = 0.0, which reads as overwhelming significance.
        let a = arm(vec![1.0, 1.0, 1.0]);
        let b = arm(vec![2.0, 2.0, 2.0]);
        let (md, w) = welch(&a, &b);
        assert_eq!(md, -1.0);
        assert!(
            matches!(w, Err(WelchError::DegenerateSample { .. })),
            "expected DegenerateSample, got {w:?}"
        );

        // Two constant arms with the SAME mean: the old code answered p = 1.0,
        // a manufactured perfect tie from the same absence of information.
        let c = arm(vec![3.0, 3.0, 3.0]);
        let d = arm(vec![3.0, 3.0, 3.0]);
        let (md, w) = welch(&c, &d);
        assert_eq!(md, 0.0);
        assert!(
            matches!(w, Err(WelchError::DegenerateSample { .. })),
            "expected DegenerateSample, got {w:?}"
        );
    }

    #[test]
    fn welch_single_seed_arm_is_insufficient_not_precise() {
        let a = arm(vec![1.0]);
        let b = arm(vec![2.0, 3.0, 4.0]);
        let (_md, w) = welch(&a, &b);
        assert!(
            matches!(
                w,
                Err(WelchError::InsufficientSamples { n_phi: 1, n_zoo: 3 })
            ),
            "expected InsufficientSamples, got {w:?}"
        );
    }

    #[test]
    fn multi_seed_runs_and_reports_conversions() {
        // Canon #93 allowed seeds only ({42, 43, 44, 45} are forbidden).
        let seeds = [47, 89, 123, 144];
        let r = run_multi_seed(&seeds, 20, 4, 64, 0.05);
        // The mechanical breadth signature must hold regardless of the accuracy
        // verdict: phi incurs zero lossy conversions, zoo incurs many.
        assert_eq!(r.phi.total_lossy, 0);
        assert!(r.zoo.total_lossy > r.phi.total_lossy);
        // The verdict is `Option`: `None` is the honest answer when the Welch
        // comparison refused. We do NOT assert PhiWins (honesty), and we do NOT
        // let a refusal silently pass as a Tie.
        match (&r.welch, r.verdict) {
            (Ok(_), Some(F2Verdict::PhiWins | F2Verdict::Tie | F2Verdict::ZooWins)) => {}
            (Err(e), None) => panic!("welch refused on this proxy: {e}"),
            (w, v) => panic!("welch/verdict disagree: welch={w:?}, verdict={v:?}"),
        }
    }
}
