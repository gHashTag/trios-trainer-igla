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

// Loop 41 fix 2: column lookup now goes by header *name*, not position.
// The hardcoded indices used in Loop 40 silently broke when f2_dual_mediation
// added/reordered columns; the name-based lookup tolerates schema drift.
//
// Per BurntSushi/rust-csv issue #169: the idiomatic pattern is to build a
// name→index map once from the header row, then index parts in the hot loop.

#[derive(Debug, Clone)]
struct PseRow {
    fix_x: String,
    pse_name: &'static str,
    estimate: f64,
    ci_lo: f64,
    ci_hi: f64,
}

/// Loop 41 fix 2 + Loop 49: build a header-name → column-index map AND extract
/// the `# INPUT STRATUM = <name>` line if the CSV came from `f2_dual_mediation`
/// (Loop 47 audit fix 2). Skips other `#` preamble + empty lines. Returns
/// `None` if no header found. Caller gets `(stratum, header_map, reader)`.
fn read_header_map(
    path: &Path,
) -> Option<(
    String,
    std::collections::HashMap<String, usize>,
    BufReader<File>,
)> {
    let f = File::open(path).expect("open input CSV");
    let mut r = BufReader::new(f);
    let mut header_line = String::new();
    let mut stratum = String::from("canonical");
    loop {
        header_line.clear();
        let n = r.read_line(&mut header_line).unwrap_or(0);
        if n == 0 {
            return None;
        }
        let trimmed = header_line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            // Loop 49: capture stratum if present. Expected form:
            //   `# INPUT STRATUM = warmup0` or `# INPUT STRATUM = warmup0 (Loop 47 audit fix 2)`
            if let Some(rest) = trimmed.strip_prefix("# INPUT STRATUM") {
                let after_eq = rest.splitn(2, '=').nth(1).map(|s| s.trim()).unwrap_or("");
                // Take the first whitespace-delimited token (drop trailing
                // " (Loop 47 audit fix 2)" annotations).
                if let Some(tok) = after_eq.split_whitespace().next() {
                    if !tok.is_empty() {
                        stratum = tok.to_string();
                    }
                }
            }
            continue;
        }
        let mut map = std::collections::HashMap::new();
        for (i, name) in trimmed.split(',').enumerate() {
            map.insert(name.trim().to_string(), i);
        }
        return Some((stratum, map, r));
    }
}

/// Loop 49: returns `(stratum, pse_rows)` so emit callers can propagate the
/// `# INPUT STRATUM = ...` banner from `f2_dual_mediation` into sensitivity output.
fn parse_csv(path: &Path) -> (String, Vec<PseRow>) {
    let Some((stratum, header, mut r)) = read_header_map(path) else {
        eprintln!("# ERROR: no header row found in {}", path.display());
        return ("canonical".into(), Vec::new());
    };
    let mut out = Vec::new();
    // Loop 41 fix 2: resolve every needed column by name. Missing column → skip
    // that PSE entirely (don't try to fabricate NaN — emit nothing rather than
    // confuse downstream).
    let col = |name: &str| header.get(name).copied();
    let fx_col = col("fix_x").expect("CSV missing required column 'fix_x'");
    let pse_specs: &[(&'static str, &str, &str, &str)] = &[
        ("NDE", "nde", "ci95_nde_lo", "ci95_nde_hi"),
        ("NIE_M1", "nie_m1", "ci95_nie_m1_lo", "ci95_nie_m1_hi"),
        ("NIE_M2", "nie_m2", "ci95_nie_m2_lo", "ci95_nie_m2_hi"),
        (
            "NIE_chain",
            "nie_chain",
            "ci95_nie_chain_lo",
            "ci95_nie_chain_hi",
        ),
    ];
    let mut line = String::new();
    loop {
        line.clear();
        let n = r.read_line(&mut line).unwrap_or(0);
        if n == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Loop 42 fix 2: trim each cell to match the header-name trim. Defensive
        // against `f2_dual_mediation` printf-padding or hand-edited CSVs with
        // leading/trailing whitespace.
        let parts: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
        if parts.len() <= fx_col {
            continue;
        }
        let fx = parts[fx_col].to_string();
        for (pse_name, est_col, lo_col, hi_col) in pse_specs {
            let (Some(ec), Some(lc), Some(hc)) = (col(est_col), col(lo_col), col(hi_col)) else {
                eprintln!(
                    "# WARN: input CSV missing one of '{}','{}','{}' — skipping {} PSE.",
                    est_col, lo_col, hi_col, pse_name
                );
                continue;
            };
            if parts.len() <= ec.max(lc).max(hc) {
                continue;
            }
            let est: f64 = parts[ec].parse().unwrap_or(f64::NAN);
            let l: f64 = parts[lc].parse().unwrap_or(f64::NAN);
            let h: f64 = parts[hc].parse().unwrap_or(f64::NAN);
            if est.is_finite() && l.is_finite() && h.is_finite() {
                out.push(PseRow {
                    fix_x: fx.clone(),
                    pse_name,
                    estimate: est,
                    ci_lo: l,
                    ci_hi: h,
                });
            }
        }
    }
    (stratum, out)
}

/// Loop 49: write a stratum banner identical in shape to f2_dual_mediation's
/// banner (Loop 47), so downstream tooling treats both binaries' outputs the
/// same way and `f2_stratum_compare` can chain CSVs without losing context.
fn write_stratum_banner<W: Write>(w: &mut W, stratum: &str) -> std::io::Result<()> {
    writeln!(w, "# INPUT STRATUM = {} (Loop 49 propagated)", stratum)?;
    if stratum != "canonical" {
        writeln!(
            w,
            "# NOTE: sensitivity envelope/tipping curve computed over Pearl CDE PSEs,"
        )?;
        writeln!(
            w,
            "# not marginal NDE/NIE; (Γ, Λ) interpretation is conditional on the stratum."
        )?;
    }
    Ok(())
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

/// Loop 41 fix 1: tipping-point Γ at fixed Λ.
///
/// For a single PSE (CI lo, hi), the envelope expansion that just touches zero
/// from the closer-to-zero endpoint is `min(|lo|, |hi|)`. Solving
/// `Γ·Λ·(Γ−1)/Γ = expansion` for Γ at fixed Λ gives `Γ = 1 + expansion / Λ`.
///
/// Returns `Γ_tip ∈ [1.0, +∞)`. By convention:
///   * `Γ_tip ≤ 1.25` → fragile (small bias of unmeasured confounding flips the verdict)
///   * `Γ_tip ≥ 2.0`  → robust (E-value-style threshold)
///
/// Returns NaN when the original CI brackets zero (no tipping point — already
/// fails at Γ=1).
pub fn tipping_point_gamma(ci_lo: f64, ci_hi: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 || !lambda.is_finite() {
        return f64::NAN;
    }
    // If the CI already brackets zero, there's no tipping point — already "doesn't survive".
    if (ci_lo <= 0.0 && ci_hi >= 0.0) || !ci_lo.is_finite() || !ci_hi.is_finite() {
        return f64::NAN;
    }
    let closer = ci_lo.abs().min(ci_hi.abs());
    1.0 + closer / lambda
}

/// Robustness label per E-value convention (Loop 41).
pub fn robustness_label(gamma_tip: f64) -> &'static str {
    if !gamma_tip.is_finite() {
        "n/a (CI brackets 0)"
    } else if gamma_tip < 1.25 {
        "fragile"
    } else if gamma_tip < 2.0 {
        "moderate"
    } else {
        "robust"
    }
}

fn compute_envelope(rows: &[PseRow], gamma: f64, lambda: f64) -> Vec<SensitivityRow> {
    let expansion = envelope_expansion(gamma, lambda);
    rows.iter()
        .map(|r| {
            let env_lo = r.ci_lo - expansion;
            let env_hi = r.ci_hi + expansion;
            // Survives iff the envelope still excludes 0 (strict, matches Loop 36
            // "CI excludes zero" verdict semantics).
            let survives_at_zero = (env_lo > 0.0 && env_hi > 0.0) || (env_lo < 0.0 && env_hi < 0.0);
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

fn emit<W: Write>(
    w: &mut W,
    rows: &[SensitivityRow],
    gamma: f64,
    lambda: f64,
) -> std::io::Result<()> {
    writeln!(
        w,
        "# Bridge-score sensitivity envelope (Ohnishi & Li 2026 arXiv:2605.18724 Theorem 2)"
    )?;
    writeln!(
        w,
        "# Gamma = {:.3} (residual selection ratio); 1.0 = no unmeasured confounding",
        gamma
    )?;
    writeln!(
        w,
        "# Lambda = {:.3} (outcome scale residual, BPB units)",
        lambda
    )?;
    writeln!(
        w,
        "# Envelope expansion ΓΛ(Γ−1)/Γ = {:.6} BPB",
        envelope_expansion(gamma, lambda)
    )?;
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

/// Loop 42 fix 5: 2D Λ-sweep tipping curve. For each PSE × Λ in the grid,
/// emit Γ_tip(Λ). Long-form (one row per PSE × Λ) per CMAverse/`cmsens` style;
/// downstream grouping by `pse_id` recovers the hyperbola for plotting.
///
/// The default Λ grid covers BPB outcome residuals from very small (0.1) to
/// large (5.0), spanning the empirical Loop 30-40 range.
const DEFAULT_LAMBDA_GRID: &[f64] = &[0.1, 0.25, 0.5, 1.0, 2.0, 5.0];

/// Loop 43 fix 3: parse a comma-separated `--lambda-grid "0.1,0.5,1,2"` value.
/// Per Rust CLI Book idiom (rust-cli.github.io/book/tutorial/cli-args.html):
/// split → trim → parse → collect. Rejects Λ ≤ 0 (log/exp blow up); allows
/// non-monotone grids (legitimate for probing known phase transitions) but
/// dedupes + sorts ascending so downstream consumers see a canonical order.
/// Loop 44 fix 3: pathological inputs (10K+ entries) would explode `--wide-form`
/// output (one CSV column per Λ). Cap at MAX_GRID_SIZE for predictable resource
/// usage. 64 is generous — Frauen et al. (arXiv:2305.16988) typically use ≤20.
const MAX_LAMBDA_GRID_SIZE: usize = 64;

fn parse_lambda_grid(arg: &str) -> Result<Vec<f64>, String> {
    let mut out: Vec<f64> = Vec::new();
    for tok in arg.split(',') {
        let s = tok.trim();
        if s.is_empty() {
            continue;
        }
        let v: f64 = s
            .parse()
            .map_err(|e| format!("'{}' is not a number: {}", s, e))?;
        if !v.is_finite() || v <= 0.0 {
            return Err(format!(
                "Λ value '{}' must be positive and finite (got {})",
                s, v
            ));
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err("--lambda-grid must contain at least one positive Λ".into());
    }
    if out.len() > MAX_LAMBDA_GRID_SIZE {
        return Err(format!(
            "--lambda-grid has {} entries; max is {} (per Loop 44 fix 3 — prevents wide-form column explosion)",
            out.len(), MAX_LAMBDA_GRID_SIZE
        ));
    }
    // Dedupe + sort ascending — canonical order regardless of input order.
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    Ok(out)
}

fn emit_lambda_sweep<W: Write>(
    w: &mut W,
    rows: &[PseRow],
    lambda_grid: &[f64],
) -> std::io::Result<()> {
    writeln!(
        w,
        "# Λ-sweep tipping curve (Ohnishi & Li 2026 Theorem 2 + Loop 41 inverse)"
    )?;
    writeln!(
        w,
        "# Long-form: one row per (PSE × Λ). Group by fix_x,pse_name to recover the hyperbola."
    )?;
    writeln!(
        w,
        "# Tipping region = (Γ ≤ Γ_tip(Λ)): unmeasured confounding strength below this curve preserves the verdict."
    )?;
    writeln!(
        w,
        "fix_x,pse_name,estimate,ci95_lo,ci95_hi,lambda,gamma_tip,robustness"
    )?;
    for r in rows {
        for &lambda in lambda_grid {
            let g_tip = tipping_point_gamma(r.ci_lo, r.ci_hi, lambda);
            let label = robustness_label(g_tip);
            writeln!(
                w,
                "{},{},{:.6},{:.6},{:.6},{:.4},{:.4},{}",
                r.fix_x, r.pse_name, r.estimate, r.ci_lo, r.ci_hi, lambda, g_tip, label
            )?;
        }
    }
    Ok(())
}

/// Loop 45 fix 6 + Loop 46 fix 2: emit a Λ value as a short header suffix.
///
/// Strategy:
///   * for Λ in the typical sensitivity range [1e-3, 1e3], use 4-decimal
///     fixed format and trim trailing zeros (e.g. `0.1000` → `0.1`);
///   * for Λ ≥ 1e4 or Λ < 1e-3, defer to Rust's `Display` impl (Ryu
///     shortest-roundtrip), which yields e.g. `10000` for 1e4 and
///     `0.0001` for 1e-4 (Rust never adds scientific notation for normal
///     finite f64 via `{}`). This avoids the audit-flagged `{:.4}`
///     scientific-notation glitch at Λ ≥ 1e4.
fn format_lambda_for_header(lambda: f64) -> String {
    if !(1e-3..1e4).contains(&lambda) {
        // Rust's Display impl is Grisu/Ryu — shortest roundtrip, no
        // scientific notation in this domain.
        return format!("{}", lambda);
    }
    let s = format!("{:.4}", lambda);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Loop 43 fix 5: wide-form pivot of the Λ-sweep.
///
/// One row per PSE; columns are `gamma_tip_lambda_<Λ>`. Useful for spreadsheet /
/// notebook consumption where the long-form requires a pivot step. Per Wickham
/// "Tidy Data" the long-form is the canonical/tidy view (emit_lambda_sweep
/// remains the source of truth); this wide-form is a *derived* view explicitly
/// for human-friendly inspection.
fn emit_lambda_sweep_wide<W: Write>(
    w: &mut W,
    rows: &[PseRow],
    lambda_grid: &[f64],
) -> std::io::Result<()> {
    writeln!(
        w,
        "# Λ-sweep tipping curve (wide form; derived from long-form per Wickham tidy-data caveat)"
    )?;
    writeln!(
        w,
        "# Columns gamma_tip_lambda_<Λ> hold Γ_tip(Λ) = 1 + min(|ci_lo|, |ci_hi|) / Λ."
    )?;
    // Header: fix_x,pse_name,estimate,ci95_lo,ci95_hi,gamma_tip_lambda_<L1>,...
    // Loop 45 fix 6: `{:g}` format drops trailing zeros (0.1 not 0.1000)
    // so spreadsheet headers stay short and readable.
    write!(w, "fix_x,pse_name,estimate,ci95_lo,ci95_hi")?;
    for &lambda in lambda_grid {
        write!(w, ",gamma_tip_lambda_{}", format_lambda_for_header(lambda))?;
    }
    writeln!(w)?;
    for r in rows {
        write!(
            w,
            "{},{},{:.6},{:.6},{:.6}",
            r.fix_x, r.pse_name, r.estimate, r.ci_lo, r.ci_hi
        )?;
        for &lambda in lambda_grid {
            let g_tip = tipping_point_gamma(r.ci_lo, r.ci_hi, lambda);
            if g_tip.is_finite() {
                write!(w, ",{:.4}", g_tip)?;
            } else {
                write!(w, ",NaN")?;
            }
        }
        writeln!(w)?;
    }
    Ok(())
}

/// Loop 41 fix 1: emit per-PSE tipping-point Γ at fixed Λ.
fn emit_tipping<W: Write>(w: &mut W, rows: &[PseRow], lambda: f64) -> std::io::Result<()> {
    writeln!(w, "# Tipping-point Γ at fixed Λ = {:.3} BPB", lambda)?;
    writeln!(
        w,
        "# Convention (VanderWeele-Ding / Alvarez-Bartolo & MacKinnon 2025):"
    )?;
    writeln!(
        w,
        "#   Γ_tip < 1.25 → fragile;  1.25 ≤ Γ_tip < 2.0 → moderate;  Γ_tip ≥ 2.0 → robust"
    )?;
    writeln!(
        w,
        "fix_x,pse_name,estimate,ci95_lo,ci95_hi,gamma_tip,robustness"
    )?;
    for r in rows {
        let g_tip = tipping_point_gamma(r.ci_lo, r.ci_hi, lambda);
        let label = robustness_label(g_tip);
        writeln!(
            w,
            "{},{},{:.6},{:.6},{:.6},{:.4},{}",
            r.fix_x, r.pse_name, r.estimate, r.ci_lo, r.ci_hi, g_tip, label
        )?;
    }
    Ok(())
}

fn print_help() {
    println!(
        "f2_mediation_sensitivity — Loop 40: additive bridge-score envelope on dual-mediation PSEs"
    );
    println!();
    println!("USAGE: f2_mediation_sensitivity [FLAGS] DUAL_MEDIATION_CSV");
    println!();
    println!("FLAGS:");
    println!("  --help, -h          Print this help and exit");
    println!("  --gamma F           Residual selection ratio Γ ≥ 1 (default 1.5)");
    println!("                      Γ = 1 → no unmeasured confounding (no envelope)");
    println!("                      Γ = 1.25 → mild; Γ = 2.0 → strong (E-value scale)");
    println!("  --lambda F          Outcome scale residual Λ (BPB units; default 1.0)");
    println!("  --tipping-point     Loop 41: emit per-PSE minimum Γ at fixed Λ that flips");
    println!("                      'survives at zero' (Alvarez-Bartolo & MacKinnon 2025).");
    println!("  --lambda-sweep      Loop 42: emit 2D tipping curve — per PSE × Λ ∈");
    println!("                      {{0.1, 0.25, 0.5, 1.0, 2.0, 5.0}} BPB. Long-form CSV");
    println!("                      (CMAverse cmsens style; group by fix_x,pse_name).");
    println!("  --lambda-grid LIST  Loop 43: override DEFAULT_LAMBDA_GRID with a");
    println!("                      comma-separated list (e.g. \"0.1,0.5,1,2,5\"). Values");
    println!("                      must be positive; duplicates removed, sorted ascending.");
    println!("  --wide-form         Loop 43: pivot --lambda-sweep output to wide form");
    println!("                      (one row per PSE, columns = Λ values). Spreadsheet-");
    println!("                      friendly derived view; long-form remains canonical.");
    println!("  --out PATH          Write sensitivity CSV to file (default stdout)");
    println!();
    println!("Refs: arXiv:2605.18724 (Ohnishi & Li 2026, Theorem 2 — additive envelope);");
    println!("      Smith & VanderWeele 2019 Epidemiology 30(6):835 (Mediational E-values).");
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
    let mut tipping_point_mode = false;
    let mut lambda_sweep_mode = false;
    let mut lambda_grid: Vec<f64> = DEFAULT_LAMBDA_GRID.to_vec();
    let mut wide_form = false;
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
        } else if a == "--tipping-point" {
            tipping_point_mode = true;
            i += 1;
        } else if a == "--lambda-sweep" {
            lambda_sweep_mode = true;
            i += 1;
        } else if a == "--lambda-grid" {
            let Some(g) = args.get(i + 1) else {
                eprintln!("# ERROR: --lambda-grid requires a comma-separated value");
                std::process::exit(2);
            };
            match parse_lambda_grid(g) {
                Ok(v) => lambda_grid = v,
                Err(e) => {
                    eprintln!("# ERROR: --lambda-grid: {}", e);
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if a == "--wide-form" {
            wide_form = true;
            i += 1;
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
    // Loop 45 fix 3: --wide-form is only meaningful inside --lambda-sweep.
    // Reject the combination explicitly instead of silently ignoring the flag
    // (which produces an envelope CSV the user didn't ask for).
    if wide_form && !lambda_sweep_mode {
        eprintln!(
            "# ERROR: --wide-form requires --lambda-sweep (it pivots the Λ × PSE table). \
             Add --lambda-sweep or drop --wide-form."
        );
        std::process::exit(2);
    }
    let (stratum, pse_rows) = parse_csv(Path::new(&input));
    if pse_rows.is_empty() {
        eprintln!(
            "# ERROR: no PSE rows parsed from {} (is it a f2_dual_mediation output?)",
            input
        );
        std::process::exit(1);
    }
    eprintln!(
        "# INPUT STRATUM = {} (propagated from f2_dual_mediation preamble)",
        stratum
    );
    if lambda_sweep_mode {
        // Loop 42 fix 5 + Loop 43 fix 3 (CLI override) + Loop 43 fix 5 (wide form).
        let kind = if wide_form { "wide" } else { "long" };
        eprintln!(
            "# Λ-sweep ({}): {} PSEs × {} Λ values",
            kind,
            pse_rows.len(),
            lambda_grid.len()
        );
        let writer: Box<dyn Write> = if let Some(p) = out_path.as_deref() {
            Box::new(File::create(p).expect("create out CSV"))
        } else {
            Box::new(std::io::stdout().lock())
        };
        let mut writer = writer;
        // Loop 49: emit stratum banner first so downstream tools can chain-parse.
        write_stratum_banner(&mut writer, &stratum).expect("write banner");
        if wide_form {
            emit_lambda_sweep_wide(&mut writer, &pse_rows, &lambda_grid).expect("write wide");
        } else {
            emit_lambda_sweep(&mut writer, &pse_rows, &lambda_grid).expect("write long");
        }
        if let Some(p) = out_path.as_deref() {
            eprintln!("# Wrote Λ-sweep CSV to {}", p);
        }
        return;
    }
    if tipping_point_mode {
        // Loop 41 fix 1: per-PSE tipping-point Γ at fixed Λ.
        let n_robust = pse_rows
            .iter()
            .filter(|r| tipping_point_gamma(r.ci_lo, r.ci_hi, lambda) >= 2.0)
            .count();
        eprintln!(
            "# {} of {} PSEs are robust (Γ_tip ≥ 2.0) at Λ = {}",
            n_robust,
            pse_rows.len(),
            lambda
        );
        if let Some(p) = out_path {
            let mut f = File::create(&p).expect("create out CSV");
            write_stratum_banner(&mut f, &stratum).expect("write banner");
            emit_tipping(&mut f, &pse_rows, lambda).expect("write");
            eprintln!("# Wrote tipping-point CSV to {}", p);
        } else {
            let stdout = std::io::stdout();
            let mut h = stdout.lock();
            write_stratum_banner(&mut h, &stratum).expect("write banner");
            emit_tipping(&mut h, &pse_rows, lambda).expect("write stdout");
        }
        return;
    }
    let env_rows = compute_envelope(&pse_rows, gamma, lambda);
    let n_survives = env_rows.iter().filter(|r| r.survives_at_zero).count();
    eprintln!(
        "# {} of {} PSEs survive at zero under (Γ={}, Λ={}) — Loop 36 robustness check",
        n_survives,
        env_rows.len(),
        gamma,
        lambda
    );
    if let Some(p) = out_path {
        let mut f = File::create(&p).expect("create out CSV");
        write_stratum_banner(&mut f, &stratum).expect("write banner");
        emit(&mut f, &env_rows, gamma, lambda).expect("write");
        eprintln!("# Wrote {} rows to {}", env_rows.len(), p);
    } else {
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        write_stratum_banner(&mut h, &stratum).expect("write banner");
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
    fn tipping_point_matches_closer_endpoint_over_lambda() {
        // Loop 41 fix 1: CI [-4.68, -3.55], Λ=1.0 → closer = 3.55 → Γ_tip = 1 + 3.55 = 4.55.
        let g = tipping_point_gamma(-4.68, -3.55, 1.0);
        assert!((g - 4.55).abs() < 1e-9, "got {}", g);
        assert_eq!(robustness_label(g), "robust");

        // CI [0.51, 2.02], Λ=1.0 → closer = 0.51 → Γ_tip = 1.51 → moderate.
        let g = tipping_point_gamma(0.51, 2.02, 1.0);
        assert!((g - 1.51).abs() < 1e-9);
        assert_eq!(robustness_label(g), "moderate");

        // Fragile case: CI [0.05, 2.0], Λ=1.0 → Γ_tip = 1.05.
        assert_eq!(
            robustness_label(tipping_point_gamma(0.05, 2.0, 1.0)),
            "fragile"
        );

        // CI brackets zero → NaN tipping point → n/a label.
        let g = tipping_point_gamma(-0.5, 0.5, 1.0);
        assert!(g.is_nan());
        assert_eq!(robustness_label(g), "n/a (CI brackets 0)");
    }

    #[test]
    fn parse_lambda_grid_accepts_csv_and_normalizes() {
        // Loop 43 fix 3: accepts comma-separated, trims, dedupes, sorts.
        let g = parse_lambda_grid(" 1.0, 0.1 , 0.5 , 1.0 ").unwrap();
        assert_eq!(g, vec![0.1, 0.5, 1.0]); // sorted + deduped
    }

    #[test]
    fn parse_lambda_grid_rejects_oversize() {
        // Loop 44 fix 3: > MAX_LAMBDA_GRID_SIZE rejected.
        let big = (1..=(MAX_LAMBDA_GRID_SIZE + 1))
            .map(|i| (i as f64) * 0.1)
            .map(|f| format!("{}", f))
            .collect::<Vec<_>>()
            .join(",");
        let err = parse_lambda_grid(&big).unwrap_err();
        assert!(
            err.contains("max is"),
            "expected max-size error, got: {}",
            err
        );
    }

    #[test]
    fn parse_lambda_grid_rejects_non_positive() {
        assert!(parse_lambda_grid("0.5,0,1").is_err());
        assert!(parse_lambda_grid("-0.1").is_err());
        assert!(parse_lambda_grid("").is_err()); // no values
        assert!(parse_lambda_grid("abc").is_err()); // garbage
    }

    #[test]
    fn read_header_map_captures_input_stratum_banner() {
        // Loop 49: parse_csv must extract `# INPUT STRATUM = warmup0` from
        // f2_dual_mediation preamble (Loop 47 banner format) so emit calls can
        // re-propagate it to downstream tooling.
        use std::io::Write;
        let tmp = std::env::temp_dir().join("f2_sens_stratum_banner.csv");
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "# Dual-mediator decomposition (Zhao-Luo 2020)").unwrap();
        writeln!(f, "# INPUT STRATUM = warmup0 (Loop 47 audit fix 2)").unwrap();
        writeln!(f, "# M1 = wd, M2 = rms").unwrap();
        writeln!(f, "rank,fix_x,n,delta_x,nde,se_nde,ci95_nde_lo,ci95_nde_hi,nie_m1,se_nie_m1,ci95_nie_m1_lo,ci95_nie_m1_hi,nie_m2,se_nie_m2,ci95_nie_m2_lo,ci95_nie_m2_hi,nie_chain,se_nie_chain,ci95_nie_chain_lo,ci95_nie_chain_hi").unwrap();
        writeln!(
            f,
            "1,rms,5,1.0,-4.0,0.2,-4.5,-3.5,5.0,0.2,4.5,5.5,1.0,0.2,0.5,1.5,-1.0,0.2,-1.5,-0.5"
        )
        .unwrap();
        drop(f);
        let (stratum, rows) = parse_csv(&tmp);
        assert_eq!(stratum, "warmup0", "stratum should be captured from banner");
        assert!(!rows.is_empty(), "PSE rows should still parse");
    }

    #[test]
    fn read_header_map_defaults_to_canonical_when_no_banner() {
        // CSV without `# INPUT STRATUM` line → defaults to "canonical".
        use std::io::Write;
        let tmp = std::env::temp_dir().join("f2_sens_no_banner.csv");
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "rank,fix_x,n,delta_x,nde,se_nde,ci95_nde_lo,ci95_nde_hi,nie_m1,se_nie_m1,ci95_nie_m1_lo,ci95_nie_m1_hi,nie_m2,se_nie_m2,ci95_nie_m2_lo,ci95_nie_m2_hi,nie_chain,se_nie_chain,ci95_nie_chain_lo,ci95_nie_chain_hi").unwrap();
        writeln!(
            f,
            "1,rms,5,1.0,-4.0,0.2,-4.5,-3.5,5.0,0.2,4.5,5.5,1.0,0.2,0.5,1.5,-1.0,0.2,-1.5,-0.5"
        )
        .unwrap();
        drop(f);
        let (stratum, _) = parse_csv(&tmp);
        assert_eq!(stratum, "canonical");
    }

    #[test]
    fn write_stratum_banner_includes_cde_note_for_non_canonical() {
        let mut buf = Vec::new();
        write_stratum_banner(&mut buf, "warmup0").unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("INPUT STRATUM = warmup0"));
        assert!(
            s.contains("Pearl CDE"),
            "non-canonical banner must include CDE note"
        );

        let mut buf2 = Vec::new();
        write_stratum_banner(&mut buf2, "canonical").unwrap();
        let s2 = String::from_utf8(buf2).unwrap();
        assert!(s2.contains("INPUT STRATUM = canonical"));
        assert!(
            !s2.contains("Pearl CDE"),
            "canonical banner should not include CDE note"
        );
    }

    #[test]
    fn format_lambda_for_header_drops_trailing_zeros() {
        // Loop 45 fix 6: spreadsheet-friendly headers per ESS-DIVE convention.
        assert_eq!(format_lambda_for_header(0.1), "0.1");
        assert_eq!(format_lambda_for_header(1.0), "1");
        assert_eq!(format_lambda_for_header(0.005), "0.005");
        assert_eq!(format_lambda_for_header(2.5), "2.5");
    }

    #[test]
    fn format_lambda_for_header_avoids_scientific_notation_at_extremes() {
        // Loop 46 fix 2: Λ ≥ 1e4 must not produce "1.0000e4" headers.
        let s = format_lambda_for_header(10000.0);
        assert!(
            !s.contains('e'),
            "Λ=1e4 produced scientific notation: '{}'",
            s
        );
        assert_eq!(s, "10000");
        // Λ at the threshold (exactly 1e4) goes through the Display branch.
        assert!(!format_lambda_for_header(1e6).contains('e'));
        // Tiny Λ → Display impl gives short form (no scientific until ≤ ~1e-5).
        let small = format_lambda_for_header(1e-4);
        assert!(
            !small.contains('e'),
            "Λ=1e-4 became scientific: '{}'",
            small
        );
        assert_eq!(small, "0.0001");
    }

    #[test]
    fn wide_form_one_row_per_pse_with_lambda_columns() {
        // Loop 43 fix 5: wide-form pivot.
        let rows = vec![
            PseRow {
                fix_x: "rms".into(),
                pse_name: "NDE",
                estimate: -4.0,
                ci_lo: -4.5,
                ci_hi: -3.5,
            },
            PseRow {
                fix_x: "rms".into(),
                pse_name: "NIE_M1",
                estimate: 5.0,
                ci_lo: 4.5,
                ci_hi: 5.5,
            },
        ];
        let lambdas = &[0.5_f64, 1.0, 2.0];
        let mut buf = Vec::new();
        emit_lambda_sweep_wide(&mut buf, &rows, lambdas).expect("write");
        let s = String::from_utf8(buf).unwrap();
        let data_rows: Vec<&str> = s
            .lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("fix_x,"))
            .collect();
        assert_eq!(data_rows.len(), 2, "expected 1 row per PSE = 2 rows");
        // Header should include three gamma_tip_lambda_* columns.
        let header = s.lines().find(|l| l.starts_with("fix_x,")).unwrap();
        assert_eq!(header.matches("gamma_tip_lambda_").count(), 3);
    }

    #[test]
    fn lambda_sweep_emits_one_row_per_pse_times_lambda() {
        // Loop 42 fix 5: ensure the long-form sweep emits |PSE| × |Λ_grid| data rows.
        let rows = vec![
            PseRow {
                fix_x: "rms".into(),
                pse_name: "NDE",
                estimate: -4.0,
                ci_lo: -4.5,
                ci_hi: -3.5,
            },
            PseRow {
                fix_x: "rms".into(),
                pse_name: "NIE_M1",
                estimate: 5.0,
                ci_lo: 4.5,
                ci_hi: 5.5,
            },
        ];
        let lambdas = &[0.5_f64, 1.0, 2.0];
        let mut buf = Vec::new();
        emit_lambda_sweep(&mut buf, &rows, lambdas).expect("write");
        let s = String::from_utf8(buf).unwrap();
        let data_rows: Vec<&str> = s
            .lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("fix_x,"))
            .collect();
        assert_eq!(
            data_rows.len(),
            2 * 3,
            "expected 2 PSEs × 3 Λ = 6 rows, got {}",
            data_rows.len()
        );
    }

    #[test]
    fn lambda_sweep_gamma_tip_decreases_as_lambda_grows() {
        // Theory: Γ_tip = 1 + closer / Λ — strictly decreasing in Λ.
        let lo = -4.5_f64;
        let hi = -3.5_f64;
        let small = tipping_point_gamma(lo, hi, 0.1);
        let med = tipping_point_gamma(lo, hi, 1.0);
        let big = tipping_point_gamma(lo, hi, 5.0);
        assert!(
            small > med && med > big,
            "γ_tip not monotone decreasing: {} {} {}",
            small,
            med,
            big
        );
    }

    #[test]
    fn tipping_point_inverse_of_envelope_expansion() {
        // At Γ = Γ_tip, the envelope expansion should equal the closer endpoint
        // distance from zero. Verify via round-trip.
        let (lo, hi) = (-4.68_f64, -3.55_f64);
        let lambda = 1.0;
        let g_tip = tipping_point_gamma(lo, hi, lambda);
        let exp = envelope_expansion(g_tip, lambda);
        // Expansion at Γ_tip should equal min(|lo|, |hi|) = 3.55.
        assert!((exp - lo.abs().min(hi.abs())).abs() < 1e-9, "exp={}", exp);
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
        assert!(
            !out[0].survives_at_zero,
            "expected zero-crossing at Γ=2, Λ=1"
        );
    }
}
