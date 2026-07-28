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
    /// Unbiased (n-1) sample variance.
    fn var(&self) -> f64 {
        let n = self.bits.len();
        if n < 2 {
            return 0.0;
        }
        let m = self.mean();
        self.bits.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (n as f64 - 1.0)
    }
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
    /// mean(phi) - mean(zoo). Negative favours phi.
    pub mean_diff: f64,
    /// Welch t-statistic.
    pub t_stat: f64,
    /// Welch-Satterthwaite degrees of freedom.
    pub df: f64,
    /// Two-sided p-value.
    pub p_two_sided: f64,
    /// Significance level used.
    pub alpha: f64,
    /// Accuracy-axis verdict.
    pub verdict: F2Verdict,
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

    let (mean_diff, t_stat, df, p_two_sided) = welch(&phi, &zoo);

    let verdict = if p_two_sided < alpha {
        if mean_diff < 0.0 {
            F2Verdict::PhiWins
        } else {
            F2Verdict::ZooWins
        }
    } else {
        F2Verdict::Tie
    };

    MultiSeedReport {
        phi,
        zoo,
        mean_diff,
        t_stat,
        df,
        p_two_sided,
        alpha,
        verdict,
    }
}

/// Welch two-sample t-test. Returns (mean_diff = m_a - m_b, t, df, two_sided_p).
fn welch(a: &ArmSamples, b: &ArmSamples) -> (f64, f64, f64, f64) {
    let na = a.bits.len() as f64;
    let nb = b.bits.len() as f64;
    let ma = a.mean();
    let mb = b.mean();
    let va = a.var();
    let vb = b.var();
    let mean_diff = ma - mb;

    let sa = va / na;
    let sb = vb / nb;
    let denom = (sa + sb).sqrt();

    // Degenerate: zero pooled variance -> infinite separation if means differ,
    // else a perfect tie.
    if denom == 0.0 {
        if mean_diff == 0.0 {
            return (0.0, 0.0, na + nb - 2.0, 1.0);
        }
        return (mean_diff, f64::INFINITY, na + nb - 2.0, 0.0);
    }

    let t = mean_diff / denom;
    // Welch-Satterthwaite df.
    let df_num = (sa + sb) * (sa + sb);
    let df_den = (sa * sa) / (na - 1.0) + (sb * sb) / (nb - 1.0);
    let df = if df_den > 0.0 {
        df_num / df_den
    } else {
        na + nb - 2.0
    };

    let p = two_sided_p_from_t(t.abs(), df);
    (mean_diff, t, df, p)
}

/// Two-sided p-value for |t| under a Student-t with `df` degrees of freedom.
/// Uses the regularized incomplete beta function via a continued fraction.
fn two_sided_p_from_t(t_abs: f64, df: f64) -> f64 {
    if !t_abs.is_finite() {
        return 0.0;
    }
    // p = I_{df/(df+t^2)}(df/2, 1/2)  (this is the two-sided tail probability).
    let x = df / (df + t_abs * t_abs);
    betai(df / 2.0, 0.5, x).clamp(0.0, 1.0)
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

    #[test]
    fn two_sided_p_known_values() {
        // t=0 -> p=1 regardless of df.
        assert!((two_sided_p_from_t(0.0, 10.0) - 1.0).abs() < 1e-9);
        // Large |t| with moderate df -> very small p.
        assert!(two_sided_p_from_t(10.0, 8.0) < 1e-4);
        // Classic: t=2.228, df=10 -> two-sided p ~= 0.05 (t_0.025,10 = 2.228).
        let p = two_sided_p_from_t(2.228, 10.0);
        assert!((p - 0.05).abs() < 0.005, "p={p}");
    }

    #[test]
    fn welch_identical_samples_is_tie() {
        let a = ArmSamples {
            bits: vec![1.0, 2.0, 3.0, 4.0],
            total_lossy: 0,
            total_coherent: 0,
        };
        let b = ArmSamples {
            bits: vec![1.0, 2.0, 3.0, 4.0],
            total_lossy: 0,
            total_coherent: 0,
        };
        let (md, _t, _df, p) = welch(&a, &b);
        assert_eq!(md, 0.0);
        assert!(p > 0.99);
    }

    #[test]
    fn welch_clear_separation_is_significant() {
        let a = ArmSamples {
            bits: vec![1.0, 1.1, 0.9, 1.05],
            total_lossy: 0,
            total_coherent: 0,
        };
        let b = ArmSamples {
            bits: vec![5.0, 5.1, 4.9, 5.05],
            total_lossy: 0,
            total_coherent: 0,
        };
        let (md, _t, _df, p) = welch(&a, &b);
        assert!(md < 0.0);
        assert!(p < 0.01, "p={p}");
    }

    #[test]
    fn multi_seed_runs_and_reports_conversions() {
        let seeds = [43, 44, 45, 46, 47];
        let r = run_multi_seed(&seeds, 20, 4, 64, 0.05);
        // The mechanical breadth signature must hold regardless of the accuracy
        // verdict: phi incurs zero lossy conversions, zoo incurs many.
        assert_eq!(r.phi.total_lossy, 0);
        assert!(r.zoo.total_lossy > r.phi.total_lossy);
        // Verdict is one of the three; we do NOT assert PhiWins (honesty).
        assert!(matches!(
            r.verdict,
            F2Verdict::PhiWins | F2Verdict::Tie | F2Verdict::ZooWins
        ));
    }
}
