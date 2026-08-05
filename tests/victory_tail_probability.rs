//! The victory gate's Student-t lower tail, against reference values.
//!
//! `src/race/victory.rs` used to carry its own `incomplete_beta`, whose doc
//! comment claimed a Lentz continued fraction it did not implement. It omitted
//! the `1/B(a, b)` factor of the REGULARIZED incomplete beta, so the p it
//! reported was not a probability at all: it SHRANK as the sample grew, and it
//! tended to 0 as `t -> 0`. The gate was therefore most "significant" exactly
//! when the measured effect was smallest.
//!
//! Since `df = n - 1` and the closed form was special-cased on `df == 2`, the
//! broken branch was taken for EVERY sample except exactly `n = 3` -- and every
//! t-test unit test in `victory.rs` built exactly three `SeedResult`s. The
//! branch that decided every other verdict had zero coverage, and a green suite
//! was not evidence about it. Measured before the fix (`t_cdf_lower_tail`,
//! lower tail):
//!
//! | df | t       | gate p   | true p   |
//! |----|---------|----------|----------|
//! |  6 | -0.05   | 0.008103 | 0.480873 |
//! | 10 | -0.05   | 0.003941 | 0.480553 |
//! | 20 | -0.50   | 0.012592 | 0.311266 |
//! |  4 | -4.0605 | 0.010003 | 0.007671 |
//!
//! The first three are false positives (a null sample declared significant at
//! alpha = 0.01); the fourth is the same bug failing the other way, rejecting a
//! genuinely significant sample.
//!
//! Reference values below are the exact regularized incomplete beta,
//! `I_{df/(df+t^2)}(df/2, 1/2) / 2` for `t < 0`, evaluated at 30 decimal digits
//! and rounded to 9. They are computed independently of this crate, so a shared
//! error in the crate's own beta implementation cannot make the test agree with
//! itself.

use std::process::Command;

use trios_trainer::race::victory::{t_cdf_lower_tail, TTEST_ALPHA};

/// Tolerance for agreement with the reference table.  Loose enough not to pin
/// the continued fraction's last bits, tight enough that 0.0081-vs-0.4809 is
/// four orders of magnitude outside it.
const TOL: f64 = 1e-4;

/// `(df, t, reference lower tail P(T <= t))`.
const REFERENCE: &[(f64, f64, f64)] = &[
    // df = 1 (Cauchy): the closed form is 1/2 + atan(t)/pi, so -1.0 -> 0.25
    // exactly, and the whole row is checkable by hand.
    (1.0, -6.314, 0.049_998_064_5),
    (1.0, -1.0, 0.25),
    (1.0, -0.05, 0.484_097_749),
    (1.0, 0.0, 0.5),
    (1.0, 1.0, 0.75),
    // df = 2: the branch that used to be the ONLY correct one.
    (2.0, -2.92, 0.049_999_577_8),
    (2.0, -0.05, 0.482_333_369),
    (2.0, 0.0, 0.5),
    (2.0, 2.0, 0.908_248_29),
    (3.0, -4.541, 0.009_998_238_06),
    (3.0, -1.0, 0.195_501_109),
    (3.0, -0.05, 0.481_632_572),
    // df = 4: the gate used to REJECT this sample (gate p = 0.010003 >= alpha)
    // although the true p = 0.007671 clears alpha = 0.01.
    (4.0, -4.0605, 0.007_671_196_16),
    (4.0, -3.747, 0.009_999_543_24),
    (4.0, -0.5, 0.321_664_982),
    // df = 6 / 10 / 20: the false-positive rows of the table above.
    (6.0, -3.585, 0.005_786_630_1),
    (6.0, -1.943, 0.050_012_498_9),
    (6.0, -0.05, 0.480_872_659),
    (10.0, -2.228, 0.025_005_885_9),
    (10.0, -0.05, 0.480_553_494),
    (10.0, 2.228, 0.974_994_114),
    (20.0, -0.5, 0.311_265_921),
    (30.0, -2.457, 0.010_006_032_8),
    (30.0, -0.05, 0.480_226_904),
    (30.0, 3.0, 0.997_305_018),
];

#[test]
fn lower_tail_matches_reference_values() {
    for &(df, t, expected) in REFERENCE {
        let got = t_cdf_lower_tail(t, df)
            .unwrap_or_else(|| panic!("tail must be evaluable at df={df}, t={t}"));
        assert!(
            (got - expected).abs() < TOL,
            "df={df}, t={t}: got {got}, reference {expected} (delta {})",
            (got - expected).abs()
        );
    }
}

/// The signature of the bug, stated as a property rather than a table.
///
/// A t-statistic of -0.05 is no effect at all: whatever the sample size, the
/// lower tail must stay just under 1/2.  The old implementation returned
/// 0.480873 at df=2 (the closed form) and then FELL to 0.008103 at df=6 and
/// 0.003941 at df=10 -- p decreasing as evidence accumulated, driven below
/// alpha by nothing but a larger n.
#[test]
fn a_null_effect_is_never_significant_at_any_df() {
    for df in 1..=60 {
        let df = df as f64;
        let p = t_cdf_lower_tail(-0.05, df).expect("tail must be evaluable");
        assert!(
            (0.40..0.50).contains(&p),
            "df={df}: t=-0.05 is no effect, but the gate reports p={p}"
        );
        assert!(
            p > TTEST_ALPHA,
            "df={df}: a null sample must never clear alpha={TTEST_ALPHA} (p={p})"
        );
    }
}

/// For a fixed df the lower tail is a CDF, so it must be non-decreasing in `t`.
#[test]
fn lower_tail_is_monotone_in_t() {
    for df in [1.0_f64, 2.0, 3.0, 4.0, 6.0, 10.0, 30.0] {
        let mut prev = 0.0_f64;
        for step in -60..=60 {
            let t = step as f64 / 10.0;
            let p = t_cdf_lower_tail(t, df).expect("tail must be evaluable");
            assert!(
                p >= prev - 1e-12,
                "df={df}: CDF decreased at t={t} ({prev} -> {p})"
            );
            assert!(
                (0.0..=1.0).contains(&p),
                "df={df}, t={t}: p={p} not in [0,1]"
            );
            prev = p;
        }
    }
}

/// The symmetry the Student-t has by construction: `P(T <= -t) = 1 - P(T <= t)`.
#[test]
fn lower_tail_is_symmetric_about_zero() {
    for df in [1.0_f64, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0] {
        for t in [0.05_f64, 0.5, 1.0, 2.228, 4.0605] {
            let lo = t_cdf_lower_tail(-t, df).expect("tail must be evaluable");
            let hi = t_cdf_lower_tail(t, df).expect("tail must be evaluable");
            assert!(
                (lo - (1.0 - hi)).abs() < 1e-12,
                "df={df}, t={t}: P(T<=-t)={lo} but 1-P(T<=t)={}",
                1.0 - hi
            );
        }
    }
}

/// The `df == 2` closed form is kept as a CROSS-CHECK, not as the production
/// path.
///
/// `t_cdf_lower_tail` now routes every df -- including 2 -- through the same
/// regularized-incomplete-beta implementation in `multi_seed`, so there is no
/// special case that can be right while the general case is wrong.  This test
/// is what the closed form is still good for: an independent formula that must
/// agree with the production path to 1e-12.
#[test]
fn df_two_closed_form_agrees_with_the_beta_path() {
    // Exact for df = 2: P(T <= t) = 1/2 + t / (2 * sqrt(2 + t^2)).
    fn closed_form(t: f64) -> f64 {
        0.5 + t / (2.0 * (2.0 + t * t).sqrt())
    }
    for step in -100..=100 {
        let t = step as f64 / 5.0;
        let via_beta = t_cdf_lower_tail(t, 2.0).expect("tail must be evaluable");
        let exact = closed_form(t);
        assert!(
            (via_beta - exact).abs() < 1e-12,
            "t={t}: beta path {via_beta} vs closed form {exact} (delta {})",
            (via_beta - exact).abs()
        );
    }
}

/// Inputs on which no tail exists get a refusal, not a number.
#[test]
fn unevaluable_inputs_return_none_not_a_p_value() {
    assert_eq!(t_cdf_lower_tail(f64::NAN, 6.0), None);
    assert_eq!(t_cdf_lower_tail(f64::INFINITY, 6.0), None);
    assert_eq!(t_cdf_lower_tail(f64::NEG_INFINITY, 6.0), None);
    assert_eq!(t_cdf_lower_tail(-1.0, 0.0), None);
    assert_eq!(t_cdf_lower_tail(-1.0, -3.0), None);
    assert_eq!(t_cdf_lower_tail(-1.0, f64::NAN), None);
}

// ----------------------------------------------------------------------
// End to end: the null ledger that used to print "IGLA FOUND"
// ----------------------------------------------------------------------

/// Seven rows: six at BPB 0.5 and one outlier at 7.5.
///
/// Chosen so the arithmetic is exact in binary floating point and can be
/// checked by hand: mean = (6 * 0.5 + 7.5) / 7 = 1.50 exactly; the squared
/// deviations are 6 * 1.0 + 36.0 = 42.0, so s^2 = 42/6 = 7 and
/// s = sqrt(7); the standard error is sqrt(7)/sqrt(7) = 1, hence
///
///     t  = (1.50 - 1.55) / 1 = -0.05,  df = n - 1 = 6.
///
/// The true one-tailed p is 0.4809 -- a sample with no effect whatsoever. The
/// old gate computed 0.008103, cleared alpha = 0.01, cleared the effect-size
/// floor (1.50 <= 1.55 - 0.05, exactly), and printed `VERDICT: IGLA FOUND`.
fn null_ledger() -> String {
    let mut rows = vec![
        r#"{"_schema":"trios.assertions.seed_results.v1","_target":1.5,"_warmup":4000}"#
            .to_string(),
    ];
    for seed in 101..=106 {
        rows.push(format!(
            r#"{{"seed":{seed},"bpb":0.5,"step":5000,"sha":"null-fixture"}}"#
        ));
    }
    rows.push(r#"{"seed":107,"bpb":7.5,"step":5000,"sha":"null-fixture"}"#.to_string());
    rows.join("\n")
}

fn run_ledger_check(path: &std::path::Path, extra: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_ledger_check"))
        .arg("--ledger")
        .arg(path)
        .args(extra)
        .output()
        .expect("ledger_check must be runnable");
    let code = out.status.code().expect("ledger_check must exit normally");
    (code, String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn a_null_seven_row_ledger_is_not_a_victory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("seed_results.jsonl");
    std::fs::write(&path, null_ledger()).expect("write ledger fixture");

    let (code, human) = run_ledger_check(&path, &[]);
    assert!(
        !human.contains("IGLA FOUND"),
        "a sample with t = -0.05 (true p = 0.481) must not be a victory:\n{human}"
    );
    assert_ne!(code, 0, "victory exit code on a null sample:\n{human}");

    // The refusal is the t-test, not a structural gate check: six of the seven
    // rows are below the target, so `check_victory` accepts and the verdict
    // turns entirely on the tail probability.
    assert_eq!(
        code, 3,
        "expected GateOkStatError (exit 3), got {code}:\n{human}"
    );
    assert!(
        human.contains("STAT-ERROR") && human.contains("TtestFailed"),
        "the printed reason must name the failing t-test:\n{human}"
    );

    let (json_code, json) = run_ledger_check(&path, &["--json"]);
    assert_eq!(json_code, 3);
    let v: serde_json::Value = serde_json::from_str(&json).expect("--json must emit valid JSON");
    assert_eq!(v["kind"], "gate_ok_stat_error");

    // The winning slice is the three LOWEST BPBs, and min_bpb is a real
    // minimum of the seeds actually listed -- the two used to be selected
    // independently, so the printed min_bpb belonged to a different set of
    // seeds than the printed seed list.
    let seeds: Vec<u64> = serde_json::from_value(v["detail"]["winning_seeds"].clone())
        .expect("winning_seeds must be a seed list");
    assert_eq!(seeds.len(), 3);
    assert!(
        seeds.iter().all(|s| (101..=106).contains(s)),
        "the outlier row must not be in the winning slice: {seeds:?}"
    );
    assert_eq!(v["detail"]["min_bpb"], 0.5);
    assert_eq!(v["detail"]["mean_bpb"], 0.5);
}

/// The same fixture through the library, so the exact p is asserted rather
/// than inferred from an exit code.
#[test]
fn the_null_ledger_p_value_is_the_true_one() {
    let t = -0.05_f64;
    let df = 6.0_f64;
    let p = t_cdf_lower_tail(t, df).expect("tail must be evaluable");
    assert!(
        (p - 0.480_872_659).abs() < TOL,
        "the null fixture's one-tailed p must be 0.4809, got {p}"
    );
    assert!(
        p >= TTEST_ALPHA,
        "0.4809 must not clear alpha = {TTEST_ALPHA}"
    );
    // What the deleted `incomplete_beta` returned for this same (t, df).
    let retracted = 0.008_103;
    assert!(
        retracted < TTEST_ALPHA && p > 50.0 * retracted,
        "the old value cleared alpha and was ~59x too small; this is the regression under test"
    );
}
