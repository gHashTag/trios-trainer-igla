//! F2 mediation sensitivity envelope — Loop 40.
//!
//! Reads a `f2_dual_mediation` CSV (Loop 35-36 schema: per-fix PSEs + t-CIs)
//! and computes the **additive bridge-score envelope** (Ohnishi & Li 2026,
//! arXiv:2605.18724 Theorem 2) on each NDE/NIE_M1/NIE_M2/NIE_chain estimate.
//!
//! ## What's the envelope?
//!
//! Sequential ignorability (SI) — the identifying assumption behind every
//! natural direct/indirect effect — can be violated by unobserved
//! mediator-outcome confounders. Theorem 2 of the bridge-score paper gives
//! a sharp additive envelope on each PSE parameterized by two interpretable
//! quantities:
//!
//!   * Γ ≥ 1 — residual *selection ratio*: how strongly an unobserved
//!     confounder shifts treatment/exposure probability at fixed observed X.
//!     Γ = 1 means no unmeasured confounding; Γ = 2 means a doubling of the
//!     odds.
//!   * Λ ≥ 0 — *outcome scale residual*: the maximum gap (in outcome units)
//!     that the unobserved confounder can induce in Y between strata of the
//!     mediator. For BPB this is bounded above by `max(BPB) − min(BPB)` in
//!     our sweep — Loop 30 data has `Λ ≤ 5` BPB.
//!
//! The additive envelope is:
//!
//!     PSE_lower(Γ, Λ) = PSE_hat − Γ · Λ · K     (K = (Γ−1)/Γ, see Theorem 2)
//!     PSE_upper(Γ, Λ) = PSE_hat + Γ · Λ · K
//!
//! This is the **mean-difference** analog of the Ding-VanderWeele E-value bound
//! (which is on the risk-ratio scale). BPB is on the additive scale so this is
//! the natural fit.
//!
//! ## Per-PSE output
//!
//! For each (fix_X, PSE) row, we report:
//!   * the original t-CI from the input
//!   * envelope expansion ΓΛ·(Γ−1)/Γ at user-supplied (Γ, Λ)
//!   * worst-case (envelope_lo, envelope_hi) = (CI_lo − expand, CI_hi + expand)
//!   * a `survives_at_zero` boolean: does the envelope still exclude 0?
//!
//! A PSE whose envelope **still excludes zero** at plausible (Γ, Λ) is robust
//! to that level of unobserved confounding — strengthening the Loop 36
//! "all rms PSEs exclude zero" conclusion.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Loop 40: column indices into f2_dual_mediation CSV (Loop 36 schema).
/// rank,fix_x,n,delta_x,nde,se_nde,ci95_nde_lo,ci95_nde_hi,nie_m1,se_nie_m1,
/// ci95_nie_m1_lo,ci95_nie_m1_hi,nie_m2,se_nie_m2,ci95_nie_m2_lo,ci95_nie_m2_hi,
/// nie_chain,se_nie_chain,ci95_nie_chain_lo,ci95_nie_chain_hi,sum,residual,
/// pct_nde,pct_m1,pct_m2,pct_chain
const COL_FIX_X: usize = 1;
const COL_NDE: usize = 4;
const COL_CI95_NDE_LO: usize = 6;
const COL_CI95_NDE_HI: usize = 7;
const COL_NIE_M1: usize = 8;
const COL_CI95_NIE_M1_LO: usize = 10;
const COL_CI95_NIE_M1_HI: usize = 11;
const COL_NIE_M2: usize = 12;
const COL_CI95_NIE_M2_LO: usize = 14;
const COL_CI95_NIE_M2_HI: usize = 15;
const COL_NIE_CHAIN: usize = 16;
const COL_CI95_NIE_CHAIN_LO: usize = 18;
const COL_CI95_NIE_CHAIN_HI: usize = 19;

#[derive(Debug, Clone)]
struct PseRow {
    fix_x: String,
    pse_name: &'static str,
    estimate: f64,
    ci_lo: f64,
    ci_hi: f64,
}

fn parse_csv(path: &Path) -> Vec<PseRow> {
    let f = File::open(path).expect("open input CSV");
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for line in r.lines() {
        let line = line.expect("read");
        if line.is_empty() || line.starts_with('#') || line.starts_with("rank,") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 20 {
            continue;
        }
        let fx = parts[COL_FIX_X].to_string();
        let push = |out: &mut Vec<PseRow>, name: &'static str, ec: usize, lo: usize, hi: usize| {
            let est: f64 = parts[ec].parse().unwrap_or(f64::NAN);
            let l: f64 = parts[lo].parse().unwrap_or(f64::NAN);
            let h: f64 = parts[hi].parse().unwrap_or(f64::NAN);
            if est.is_finite() && l.is_finite() && h.is_finite() {
                out.push(PseRow { fix_x: fx.clone(), pse_name: name, estimate: est, ci_lo: l, ci_hi: h });
            }
        };
        push(&mut out, "NDE", COL_NDE, COL_CI95_NDE_LO, COL_CI95_NDE_HI);
        push(&mut out, "NIE_M1", COL_NIE_M1, COL_CI95_NIE_M1_LO, COL_CI95_NIE_M1_HI);
        push(&mut out, "NIE_M2", COL_NIE_M2, COL_CI95_NIE_M2_LO, COL_CI95_NIE_M2_HI);
        push(&mut out, "NIE_chain", COL_NIE_CHAIN, COL_CI95_NIE_CHAIN_LO, COL_CI95_NIE_CHAIN_HI);
    }
    out
}

/// Additive envelope expansion = Γ · Λ · (Γ−1)/Γ per Theorem 2 of arXiv:2605.18724.
/// Γ = 1 (no unmeasured confounding) → expansion = 0 → envelope ≡ original CI.
fn envelope_expansion(gamma: f64, lambda: f64) -> f64 {
    if gamma <= 1.0 || lambda <= 0.0 {
        return 0.0;
    }
    gamma * lambda * (gamma - 1.0) / gamma
}

#[derive(Debug, Clone)]
struct SensitivityRow {
    fix_x: String,
    pse_name: &'static str,
    estimate: f64,
    ci_lo: f64,
    ci_hi: f64,
    expansion: f64,
    env_lo: f64,
    env_hi: f64,
    survives_at_zero: bool,
}

fn compute_envelope(rows: &[PseRow], gamma: f64, lambda: f64) -> Vec<SensitivityRow> {
    let expansion = envelope_expansion(gamma, lambda);
    rows.iter()
        .map(|r| {
            let env_lo = r.ci_lo - expansion;
            let env_hi = r.ci_hi + expansion;
            // Survives iff the envelope still excludes 0 (strict, matches Loop 36
            // "CI excludes zero" verdict semantics).
            let survives_at_zero = (env_lo > 0.0 && env_hi > 0.0)
                || (env_lo < 0.0 && env_hi < 0.0);
            SensitivityRow {
                fix_x: r.fix_x.clone(),
                pse_name: r.pse_name,
                estimate: r.estimate,
                ci_lo: r.ci_lo,
                ci_hi: r.ci_hi,
                expansion,
                env_lo,
                env_hi,
                survives_at_zero,
            }
        })
        .collect()
}

fn emit<W: Write>(w: &mut W, rows: &[SensitivityRow], gamma: f64, lambda: f64) -> std::io::Result<()> {
    writeln!(
        w,
        "# Bridge-score sensitivity envelope (Ohnishi & Li 2026 arXiv:2605.18724 Theorem 2)"
    )?;
    writeln!(w, "# Gamma = {:.3} (residual selection ratio); 1.0 = no unmeasured confounding", gamma)?;
    writeln!(w, "# Lambda = {:.3} (outcome scale residual, BPB units)", lambda)?;
    writeln!(w, "# Envelope expansion ΓΛ(Γ−1)/Γ = {:.6} BPB", envelope_expansion(gamma, lambda))?;
    writeln!(
        w,
        "fix_x,pse_name,estimate,ci95_lo,ci95_hi,envelope_expansion,env_lo,env_hi,survives_at_zero"
    )?;
    for r in rows {
        writeln!(
            w,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            r.fix_x,
            r.pse_name,
            r.estimate,
            r.ci_lo,
            r.ci_hi,
            r.expansion,
            r.env_lo,
            r.env_hi,
            r.survives_at_zero
        )?;
    }
    Ok(())
}

fn print_help() {
    println!("f2_mediation_sensitivity — Loop 40: additive bridge-score envelope on dual-mediation PSEs");
    println!();
    println!("USAGE: f2_mediation_sensitivity [FLAGS] DUAL_MEDIATION_CSV");
    println!();
    println!("FLAGS:");
    println!("  --help, -h          Print this help and exit");
    println!("  --gamma F           Residual selection ratio Γ ≥ 1 (default 1.5)");
    println!("                      Γ = 1 → no unmeasured confounding (no envelope)");
    println!("                      Γ = 1.25 → mild; Γ = 2.0 → strong (E-value scale)");
    println!("  --lambda F          Outcome scale residual Λ (BPB units; default 1.0)");
    println!("  --out PATH          Write sensitivity CSV to file (default stdout)");
    println!();
    println!("Refs: arXiv:2605.18724 (Ohnishi & Li 2026, Theorem 2 — additive envelope).");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let mut input: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut gamma: f64 = 1.5;
    let mut lambda: f64 = 1.0;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if a == "--out" {
            out_path = args.get(i + 1).cloned();
            i += 2;
        } else if a == "--gamma" {
            gamma = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1.5);
            i += 2;
        } else if a == "--lambda" {
            lambda = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            i += 2;
        } else if a.starts_with("--") {
            eprintln!("# ERROR: unknown flag {}", a);
            std::process::exit(2);
        } else {
            input = Some(a.clone());
            i += 1;
        }
    }
    let Some(input) = input else {
        eprintln!("# ERROR: no input CSV. See --help.");
        std::process::exit(2);
    };
    if gamma < 1.0 {
        eprintln!("# ERROR: --gamma must be ≥ 1.0 (got {}).", gamma);
        std::process::exit(2);
    }
    let pse_rows = parse_csv(Path::new(&input));
    if pse_rows.is_empty() {
        eprintln!("# ERROR: no PSE rows parsed from {} (is it a f2_dual_mediation output?)", input);
        std::process::exit(1);
    }
    let env_rows = compute_envelope(&pse_rows, gamma, lambda);
    let n_survives = env_rows.iter().filter(|r| r.survives_at_zero).count();
    eprintln!(
        "# {} of {} PSEs survive at zero under (Γ={}, Λ={}) — Loop 36 robustness check",
        n_survives, env_rows.len(), gamma, lambda
    );
    if let Some(p) = out_path {
        let mut f = File::create(&p).expect("create out CSV");
        emit(&mut f, &env_rows, gamma, lambda).expect("write");
        eprintln!("# Wrote {} rows to {}", env_rows.len(), p);
    } else {
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        emit(&mut h, &env_rows, gamma, lambda).expect("write stdout");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_zero_when_gamma_is_one() {
        assert_eq!(envelope_expansion(1.0, 5.0), 0.0);
        assert_eq!(envelope_expansion(1.5, 0.0), 0.0);
        // Strict (Γ ≤ 1) → 0.
        assert_eq!(envelope_expansion(0.99, 5.0), 0.0);
    }

    #[test]
    fn envelope_grows_with_gamma_and_lambda() {
        let a = envelope_expansion(2.0, 1.0);
        let b = envelope_expansion(3.0, 1.0);
        assert!(b > a);
        let c = envelope_expansion(2.0, 2.0);
        assert!(c > a);
    }

    #[test]
    fn envelope_at_gamma_2_lambda_1_is_half() {
        // Γ=2, Λ=1: expansion = 2·1·(1/2) = 1.0
        assert!((envelope_expansion(2.0, 1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn compute_envelope_preserves_estimate_and_extends_ci() {
        let pse = vec![PseRow {
            fix_x: "rms".into(),
            pse_name: "NDE",
            estimate: -4.12,
            ci_lo: -4.68,
            ci_hi: -3.55,
        }];
        let out = compute_envelope(&pse, 1.5, 1.0);
        let r = &out[0];
        // Expansion = 1.5·1·(0.5/1.5) = 0.5
        assert!((r.expansion - 0.5).abs() < 1e-9);
        assert!((r.env_lo - (-5.18)).abs() < 1e-9, "env_lo={}", r.env_lo);
        assert!((r.env_hi - (-3.05)).abs() < 1e-9, "env_hi={}", r.env_hi);
        // Both endpoints < 0 → survives at zero.
        assert!(r.survives_at_zero);
    }

    #[test]
    fn survives_flips_when_envelope_crosses_zero() {
        // Marginal NIE_M2 from Loop 36: estimate +1.27, CI [0.51, 2.02].
        // Expansion of 1.0 (Γ=2, Λ=1) pushes env_lo to −0.49 → crosses zero.
        let pse = vec![PseRow {
            fix_x: "rms".into(),
            pse_name: "NIE_M2",
            estimate: 1.27,
            ci_lo: 0.51,
            ci_hi: 2.02,
        }];
        let out = compute_envelope(&pse, 2.0, 1.0);
        assert!(!out[0].survives_at_zero, "expected zero-crossing at Γ=2, Λ=1");
    }
}
