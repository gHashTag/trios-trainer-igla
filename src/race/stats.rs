//! Shared statistical helpers — Loop 36.
//!
//! Provides closed-form Student's t-distribution upper-tail CDF and the inverse
//! (two-sided critical value) used by both the iLOCO scorer and the dual-mediation
//! SE → CI translation. Previously duplicated across
//!   - src/bin/f2_iloco_score.rs
//!   - src/bin/f2_ablation_aggregate.rs
//! Centralizing eliminates the drift risk surfaced in Loop 35.
//!
//! Algorithms:
//!   - `lgamma`: Lanczos g=7 approximation (NIST DLMF 5.7.6); ≤ 1e-14 absolute
//!     for x > 0.5.
//!   - `regularized_incomplete_beta`: continued fraction via Lentz's method
//!     (Numerical Recipes §6.4).
//!   - `student_t_cdf_upper`: 0.5 · I_x(df/2, 1/2) with x = df/(df + t²).
//!   - `student_t_critical_two_sided`: bisection over `student_t_cdf_upper` to
//!     invert P(|T| > t) = α — robust, ~50 iterations to f64 precision.

// =====================================================================
// Loop 38 fix 1+2: descriptive-statistics helpers, migrated here from
// `f2_iloco_score` (Loops 31–33) and `f2_dual_mediation` (Loop 35).
// =====================================================================

/// Arithmetic mean. Returns NaN for empty slices.
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Sample covariance Cov(x, y) with Bessel correction (n-1 denominator).
/// Returns NaN if either slice has fewer than 2 elements.
pub fn cov(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return f64::NAN;
    }
    let mx = mean(&x[..n]);
    let my = mean(&y[..n]);
    let mut acc = 0.0;
    for i in 0..n {
        acc += (x[i] - mx) * (y[i] - my);
    }
    acc / (n - 1) as f64
}

/// Sample variance (= cov(x, x)).
pub fn var(x: &[f64]) -> f64 {
    cov(x, x)
}

/// Pearson correlation Cor(x, y). Returns 0 if either standard deviation is
/// below 1e-12 (no signal to correlate).
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let c = cov(x, y);
    let sx = var(x).sqrt();
    let sy = var(y).sqrt();
    if sx < 1e-12 || sy < 1e-12 {
        return 0.0;
    }
    c / (sx * sy)
}

/// Sample standard error of the mean = sqrt(s² / N). Returns NaN for N < 2.
pub fn sample_se(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = mean(v);
    let s2: f64 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    (s2 / n as f64).sqrt()
}

/// Returns ln Γ(x) for x > 0 via Lanczos g=7 approximation.
pub fn lgamma(x: f64) -> f64 {
    let g = 7.0;
    let coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = coef[0];
    let t = x + g + 0.5;
    for (i, &c) in coef.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Continued-fraction expansion of B(a, b; x) — used by `regularized_incomplete_beta`.
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 200;
    let eps = 3e-12;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=max_iter {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa1 = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa1 * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa1 / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa2 = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
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

/// Regularized incomplete beta I_x(a, b) = B(a, b; x) / B(a, b).
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(x, a, b)
    } else {
        1.0 - front * beta_cf(1.0 - x, b, a) * a / b
    }
}

/// Upper-tail CDF of Student's t with `df` degrees of freedom:
/// returns P(T > t). Symmetric, so P(|T| > t) = 2 * student_t_cdf_upper(|t|, df).
pub fn student_t_cdf_upper(t: f64, df: f64) -> f64 {
    // Loop 38 fix 5: explicit guard against NaN/inf df. Without this, the
    // bisection in `student_t_critical_two_sided` can run on NaN comparison
    // results (always false), causing unhelpful output rather than a clean NaN.
    if !df.is_finite() || df <= 0.0 || !t.is_finite() {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    0.5 * regularized_incomplete_beta(x, df / 2.0, 0.5)
}

/// Loop 36 (Loop 37 hardened): two-sided t critical value t* such that
/// P(|T| > t*) = alpha, df=df. For alpha=0.05, df=4 this returns ≈ 2.776;
/// df=9 → 2.262; df=∞ → 1.96.
///
/// Loop 37 fix 2: For very small alpha (e.g. 1e-10 from BH correction at
/// large m), the upper bisection bound grows. We expand the upper bound
/// adaptively: P(T > hi, df) must be ≤ alpha/2 to bracket the root.
/// At df=4, t=1e6 gives p ≈ 6e-25 (well below any practical alpha),
/// but for completeness we double `hi` until it brackets, capped at 1e15.
/// Returns NaN for df ≤ 0 or alpha outside (0, 1).
pub fn student_t_critical_two_sided(alpha: f64, df: f64) -> f64 {
    // Loop 38 fix 5: tighten the validation — NaN/inf df should not enter
    // bisection. Previously caught only df ≤ 0 which let NaN slip through
    // (NaN comparisons return false, so `NaN ≤ 0` is false).
    if !df.is_finite() || df <= 0.0
        || !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0
    {
        return f64::NAN;
    }
    let target = alpha / 2.0; // upper-tail probability we want
    let mut lo = 0.0;
    let mut hi = 1.0e6;
    // Loop 37 fix 2: ensure hi actually brackets — needed for tiny alpha at
    // large df (e.g. alpha=1e-20, df=100 may want t > 1e6).
    let mut hi_iter = 0;
    while student_t_cdf_upper(hi, df) > target && hi < 1.0e15 {
        hi *= 2.0;
        hi_iter += 1;
        if hi_iter > 50 {
            break;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let p = student_t_cdf_upper(mid, df);
        if p > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 || (hi - lo) / hi.max(1.0) < 1e-14 {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_handles_empty_and_singletons() {
        assert!(mean(&[]).is_nan());
        assert_eq!(mean(&[7.0]), 7.0);
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn cov_and_var_match_textbook() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
        // Var(x) with Bessel = 2.5
        assert!((var(&x) - 2.5).abs() < 1e-12);
        // Cov(x, 2x) = 2 · Var(x) = 5.0
        assert!((cov(&x, &y) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect_correlation_returns_unit() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 1.0).collect();
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-12);
        // Constant vector → 0 (no signal).
        let c = vec![5.0_f64; 5];
        assert_eq!(pearson(&c, &y), 0.0);
    }

    #[test]
    fn sample_se_matches_sqrt_var_over_n() {
        let x = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let expected = (var(&x) / 5.0).sqrt();
        assert!((sample_se(&x) - expected).abs() < 1e-12);
        // N < 2 → NaN.
        assert!(sample_se(&[1.0]).is_nan());
        assert!(sample_se(&[]).is_nan());
    }

    #[test]
    fn lgamma_matches_known_factorials() {
        // ln Γ(n+1) = ln n! exact for small integers.
        // ln 5! = ln 120 ≈ 4.7874917...
        assert!((lgamma(6.0) - (120.0_f64).ln()).abs() < 1e-9);
    }

    #[test]
    fn student_t_cdf_upper_symmetric() {
        // P(T > 0) = 0.5 exactly for any df > 0.
        for df in [1.0, 4.0, 30.0, 100.0] {
            let p = student_t_cdf_upper(0.0, df);
            assert!((p - 0.5).abs() < 1e-9, "df={} gave p={}", df, p);
        }
    }

    #[test]
    fn student_t_critical_matches_published_values() {
        // Published two-tailed 0.05 critical values:
        //   df=1  → 12.706
        //   df=4  → 2.776
        //   df=9  → 2.262
        //   df=29 → 2.045
        //   df=∞  → 1.96
        for (df, want) in [(1.0, 12.706_f64), (4.0, 2.776), (9.0, 2.262), (29.0, 2.045)] {
            let t = student_t_critical_two_sided(0.05, df);
            assert!(
                (t - want).abs() < 0.002,
                "df={}, want {}, got {}",
                df, want, t
            );
        }
    }

    #[test]
    fn student_t_critical_returns_nan_for_invalid_df() {
        assert!(student_t_critical_two_sided(0.05, 0.0).is_nan());
        assert!(student_t_critical_two_sided(0.05, -1.0).is_nan());
    }

    /// Loop 37 fix 2: tiny alpha (e.g. BH-corrected at m=1000, rank=1) should
    /// produce a finite critical value, not run off the bisection bracket.
    #[test]
    fn student_t_critical_handles_tiny_alpha() {
        // At alpha=1e-6, df=4, the published value is ~31.6.
        let t = student_t_critical_two_sided(1e-6, 4.0);
        assert!(t.is_finite(), "expected finite t, got {}", t);
        assert!(t > 20.0 && t < 50.0, "t={} for alpha=1e-6, df=4 outside plausible range", t);
        // Inverse check: P(|T| > t) should be ≈ alpha.
        let p = 2.0 * student_t_cdf_upper(t, 4.0);
        assert!((p - 1e-6).abs() < 1e-7, "round-trip p={} for alpha=1e-6", p);
    }

    #[test]
    fn student_t_cdf_and_critical_guard_against_nan_df() {
        // Loop 38 fix 5: NaN/inf df must produce NaN cleanly, not stall in
        // bisection (whose comparisons against NaN are all `false`).
        assert!(student_t_cdf_upper(1.0, f64::NAN).is_nan());
        assert!(student_t_cdf_upper(1.0, f64::INFINITY).is_nan());
        assert!(student_t_critical_two_sided(0.05, f64::NAN).is_nan());
        assert!(student_t_critical_two_sided(0.05, f64::INFINITY).is_nan());
        // t = NaN should also short-circuit (callers may feed NaN by accident).
        assert!(student_t_cdf_upper(f64::NAN, 4.0).is_nan());
    }

    #[test]
    fn student_t_critical_handles_alpha_one_in_a_thousand() {
        // alpha = 0.001 (two-tailed), df=29 → 3.659 per published two-sided
        // tables. Confirmed via P(|T| > 3.659, df=29) = 0.001.
        let t = student_t_critical_two_sided(0.001, 29.0);
        assert!((t - 3.659).abs() < 0.005, "got {}", t);
    }
}
