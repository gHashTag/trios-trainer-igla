//! f2_dual_mediation (coder-branch minimal port).
//!
//! Reads an F2 long-form CSV of a 2x2 optimizer factorial (columns include
//! m1_momentum, m2_decay, code_val_bpb) and computes path-specific effects of
//! the phi-prior on code BPB: the Pearl controlled direct effect (CDE) of each
//! mediator, the total effect, and the mediated-interaction surplus, each with
//! a percentile bootstrap 95% CI. Writes a dual-mediation long-form CSV and
//! re-emits the `# INPUT STRATUM = <s>` banner (single source of truth for
//! marginal-NDE vs Pearl-CDE framing downstream).
//!
//! This is the minimal real-Rust bridge requested in Loop+1 Option B: the full
//! upstream f2_dual_mediation (path-specific NDE/NIE on a generic sweep) is not
//! on this branch; this port targets the coder 2x2 factorial specifically.
//!
//! HONESTY (skills igla-phi-architecture, f2-mediation-loop): only
//! phi^2 + phi^-2 = 3 is [Verified]. The CDEs computed here describe phi's
//! measured HARM, not any benefit. No phi-superiority claim is made.
//! References: Pearl 2001; VanderWeele 2015; VanderWeele & Ding 2017.

use std::collections::BTreeMap;
use std::env;
use std::fs;

const BOOT: usize = 20000;

/// Tiny deterministic SplitMix64 RNG (no external crate; reproducible).
struct SplitMix64 {
    s: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { s: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// uniform usize in [0, n)
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

/// Parsed CSV: cells keyed by (m1_momentum, m2_decay) -> Vec<bpb>, plus stratum.
struct Parsed {
    cells: BTreeMap<(u8, u8), Vec<f64>>,
    stratum: String,
    source: String,
}

fn parse_csv(text: &str) -> Parsed {
    let mut cells: BTreeMap<(u8, u8), Vec<f64>> = BTreeMap::new();
    let mut stratum = String::from("canonical");
    let mut source = String::from("unknown");
    let mut header: Option<Vec<String>> = None;
    for line in text.lines() {
        if line.starts_with('#') {
            if let Some(rest) = line.strip_prefix("# INPUT STRATUM") {
                // form: "# INPUT STRATUM = <s>"
                if let Some(eq) = rest.find('=') {
                    stratum = rest[eq + 1..].trim().to_string();
                }
            }
            if line.starts_with("# W3C-PROV:") {
                if let Some(idx) = line.find("source =") {
                    let tail = &line[idx + "source =".len()..];
                    source = tail.split(',').next().unwrap_or("").trim().to_string();
                }
            }
            continue;
        }
        let cols: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        if header.is_none() {
            header = Some(cols);
            continue;
        }
        let h = header.as_ref().unwrap();
        let get = |name: &str| -> Option<String> {
            h.iter().position(|c| c == name).map(|i| cols[i].clone())
        };
        let m1: u8 = get("m1_momentum").and_then(|v| v.parse().ok()).unwrap_or(0);
        let m2: u8 = get("m2_decay").and_then(|v| v.parse().ok()).unwrap_or(0);
        if let Some(bpb) = get("code_val_bpb").and_then(|v| v.parse::<f64>().ok()) {
            cells.entry((m1, m2)).or_default().push(bpb);
        }
    }
    Parsed {
        cells,
        stratum,
        source,
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Bootstrap CI for a contrast over the cell map; returns (point, lo, hi, p).
fn boot_contrast<F>(cells: &BTreeMap<(u8, u8), Vec<f64>>, f: F, seed: u64) -> (f64, f64, f64, f64)
where
    F: Fn(&BTreeMap<(u8, u8), Vec<f64>>) -> f64,
{
    let point = f(cells);
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(BOOT);
    for _ in 0..BOOT {
        let mut rs: BTreeMap<(u8, u8), Vec<f64>> = BTreeMap::new();
        for (k, v) in cells.iter() {
            let n = v.len();
            let mut resamp = Vec::with_capacity(n);
            for _ in 0..n {
                resamp.push(v[rng.below(n)]);
            }
            rs.insert(*k, resamp);
        }
        samples.push(f(&rs));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = samples[(0.025 * BOOT as f64) as usize];
    let hi = samples[((0.975 * BOOT as f64) as usize).min(BOOT - 1)];
    let le = samples.iter().filter(|&&x| x <= 0.0).count() as f64 / BOOT as f64;
    let ge = samples.iter().filter(|&&x| x >= 0.0).count() as f64 / BOOT as f64;
    let p = (2.0 * le.min(ge)).min(1.0);
    (point, lo, hi, p)
}

/// E-value-style Gamma_tip from point and CI half-width (matches the Loop+1
/// numpy stand-in: VanderWeele-Ding monotone transform of signal/CI ratio).
fn gamma_tip(effect: f64, lo: f64, hi: f64) -> f64 {
    let half = (hi - lo) / 2.0;
    if half <= 0.0 {
        return f64::INFINITY;
    }
    let rr = effect.abs() / half;
    if rr <= 0.0 {
        return 1.0;
    }
    let big = 1.0 + rr;
    big + (big * (big - 1.0)).sqrt()
}

fn status(effect: f64, lo: f64, hi: f64, g: f64) -> &'static str {
    let spans0 = lo <= 0.0 && 0.0 <= hi;
    let _ = effect;
    if g >= 2.0 && !spans0 {
        "Verified"
    } else if (1.25..2.0).contains(&g) && !spans0 {
        "Efit"
    } else {
        "Conj"
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .iter()
        .position(|a| a == "--csv")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "coder_ablation_f2.csv".to_string());
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let parsed = parse_csv(&text);
    let c = &parsed.cells;

    // Single-mediator strata: one of the two mediators is pinned so only one
    // varying cell exists alongside the (0,0) baseline. Two symmetric cases:
    //   decay-pinned (wd0):       cells (0,0),(1,0) -> momentum CDE (non-decay paths)
    //   momentum-pinned (mom_std): cells (0,0),(0,1) -> decay CDE  (non-momentum paths)
    // Detect by which off-baseline cell is present. Handle before the 2x2 path.
    let full_2x2 = c.contains_key(&(1, 0))
        && c.contains_key(&(0, 1))
        && c.contains_key(&(1, 1))
        && c.contains_key(&(0, 0));
    if !full_2x2 {
        let momentum_pinned = c.contains_key(&(0, 1)) && !c.contains_key(&(1, 0));
        let (pinned_name, free_name, row_name, cell): (&str, &str, &str, (u8, u8)) =
            if momentum_pinned {
                ("momentum", "decay", "cde_decay_momentum_pinned", (0, 1))
            } else {
                ("decay", "momentum", "cde_momentum_decay_pinned", (1, 0))
            };
        assert!(
            c.contains_key(&(0, 0)) && c.contains_key(&cell),
            "single-mediator stratum needs baseline (0,0) and the free-mediator cell"
        );
        println!(
            "# W3C-PROV: source = {}, mode = dual_mediation",
            parsed.source
        );
        println!("# INPUT STRATUM = {}", parsed.stratum);
        println!(
            "# CDE-framing: {} -- single-mediator ({}) Pearl CDE along \
             non-{} paths",
            parsed.stratum, free_name, pinned_name
        );
        println!("pse,effect_bpb,ci95_lo,ci95_hi,p,gamma_tip,status");
        let cde = move |c: &BTreeMap<(u8, u8), Vec<f64>>| mean(&c[&cell]) - mean(&c[&(0, 0)]);
        let (point, lo, hi, p) = boot_contrast(c, cde, 0xF2A);
        let g = gamma_tip(point, lo, hi);
        let st = status(point, lo, hi, g);
        println!("{row_name},{point:.4},{lo:.4},{hi:.4},{p:.3},{g:.2},{st}");
        return;
    }

    // Pearl controlled direct effects on the 2x2 grid (positive = worse BPB).
    let cde_mom = |c: &BTreeMap<(u8, u8), Vec<f64>>| mean(&c[&(1, 0)]) - mean(&c[&(0, 0)]);
    let cde_dec = |c: &BTreeMap<(u8, u8), Vec<f64>>| mean(&c[&(0, 1)]) - mean(&c[&(0, 0)]);
    let te = |c: &BTreeMap<(u8, u8), Vec<f64>>| mean(&c[&(1, 1)]) - mean(&c[&(0, 0)]);
    let inter = |c: &BTreeMap<(u8, u8), Vec<f64>>| {
        (mean(&c[&(1, 1)]) - mean(&c[&(0, 0)]))
            - (mean(&c[&(1, 0)]) - mean(&c[&(0, 0)]))
            - (mean(&c[&(0, 1)]) - mean(&c[&(0, 0)]))
    };

    let effects: [(&str, fn(&BTreeMap<(u8, u8), Vec<f64>>) -> f64); 4] = [
        ("cde_momentum_decay0", cde_mom),
        ("cde_decay_momentum0", cde_dec),
        ("total_effect", te),
        ("mediated_interaction", inter),
    ];

    // Re-emit preamble (single source of truth for stratum framing).
    println!(
        "# W3C-PROV: source = {}, mode = dual_mediation",
        parsed.source
    );
    println!("# INPUT STRATUM = {}", parsed.stratum);
    let framing = if parsed.stratum == "canonical" {
        "marginal NDE/NIE (no nuisance pinned)"
    } else {
        "Pearl CDE along non-pinned paths"
    };
    println!("# CDE-framing: {} -- {}", parsed.stratum, framing);
    println!("pse,effect_bpb,ci95_lo,ci95_hi,p,gamma_tip,status");
    for (name, f) in effects.iter() {
        let seed = 0xF2_u64
            .wrapping_mul(name.bytes().map(|b| b as u64).sum::<u64>())
            .wrapping_add(1);
        let (point, lo, hi, p) = boot_contrast(c, f, seed);
        let g = gamma_tip(point, lo, hi);
        let st = status(point, lo, hi, g);
        println!(
            "{name},{point:.4},{lo:.4},{hi:.4},{p:.3},{g:.2},{st}",
            name = name,
            point = point,
            lo = lo,
            hi = hi,
            p = p,
            g = g,
            st = st
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# W3C-PROV: source = trios-trainer-igla@deadbee, mode = optimizer_factorial_2x2
# INPUT STRATUM = canonical
# CDE-framing: canonical stratum -- PSEs are marginal NDE/NIE
mode,arm,seed,x_phi_prior,m1_momentum,m2_decay,beta1,wd,hidden,n_steps,lr,code_val_bpb
optimizer_factorial,standard,42,0,0,0,0.9,0.04,64,300,0.03,4.0
optimizer_factorial,standard,43,0,0,0,0.9,0.04,64,300,0.03,4.4
optimizer_factorial,phi_b1,42,1,1,0,0.618,0.04,64,300,0.03,5.0
optimizer_factorial,phi_b1,43,1,1,0,0.618,0.04,64,300,0.03,5.2
optimizer_factorial,phi_wd,42,1,0,1,0.9,0.236,64,300,0.03,8.0
optimizer_factorial,phi_wd,43,1,0,1,0.9,0.236,64,300,0.03,7.8
optimizer_factorial,phi,42,1,1,1,0.618,0.236,64,300,0.03,7.0
optimizer_factorial,phi,43,1,1,1,0.618,0.236,64,300,0.03,7.4
";

    #[test]
    fn reads_csv_with_preamble_and_stratum_banner() {
        let p = parse_csv(SAMPLE);
        assert_eq!(p.stratum, "canonical");
        assert!(p.source.contains("trios-trainer-igla@deadbee"));
        assert_eq!(p.cells.len(), 4);
        assert_eq!(p.cells[&(0, 0)], vec![4.0, 4.4]);
        assert_eq!(p.cells[&(1, 1)], vec![7.0, 7.4]);
    }

    #[test]
    fn cde_decay_is_dominant_and_decomposition_holds() {
        let p = parse_csv(SAMPLE);
        let c = &p.cells;
        let m = |k: (u8, u8)| mean(&c[&k]);
        let cde_mom = m((1, 0)) - m((0, 0));
        let cde_dec = m((0, 1)) - m((0, 0));
        let te = m((1, 1)) - m((0, 0));
        let inter = te - cde_mom - cde_dec;
        // decay path dominates momentum path
        assert!(cde_dec > cde_mom);
        // 4-way decomposition identity is exact
        assert!((cde_dec + cde_mom + inter - te).abs() < 1e-9);
    }

    #[test]
    fn wd0_stratum_is_detected() {
        let s = SAMPLE.replace("INPUT STRATUM = canonical", "INPUT STRATUM = wd0");
        let p = parse_csv(&s);
        assert_eq!(p.stratum, "wd0");
    }

    #[test]
    fn wd0_single_mediator_csv_has_only_decay0_cells() {
        let wd0 = "\
# W3C-PROV: source = trios-trainer-igla@deadbee, mode = optimizer_factorial_wd0
# INPUT STRATUM = wd0
mode,arm,seed,x_phi_prior,m1_momentum,m2_decay,beta1,wd,hidden,n_steps,lr,code_val_bpb
w,standard,42,0,0,0,0.9,0.0,64,300,0.03,3.78
w,phi_b1,42,1,1,0,0.618,0.0,64,300,0.03,3.45
";
        let p = parse_csv(wd0);
        assert_eq!(p.stratum, "wd0");
        assert!(p.cells.contains_key(&(0, 0)));
        assert!(p.cells.contains_key(&(1, 0)));
        assert!(!p.cells.contains_key(&(0, 1)));
        assert!(!p.cells.contains_key(&(1, 1)));
    }

    #[test]
    fn mom_std_single_mediator_csv_has_only_momentum0_cells() {
        // Momentum-pinned stratum: beta1 fixed at 0.9, only decay varies.
        // Cells are baseline (0,0) and decay-on (0,1); (1,0) and (1,1) absent.
        let mom_std = "\
# W3C-PROV: source = trios-trainer-igla@deadbee, mode = optimizer_factorial_mom_std
# INPUT STRATUM = mom_std
mode,arm,seed,x_phi_prior,m1_momentum,m2_decay,beta1,wd,hidden,n_steps,lr,code_val_bpb
m,standard,42,0,0,0,0.9,0.04,64,300,0.03,4.31
m,phi_wd,42,1,0,1,0.9,0.2360679775,64,300,0.03,7.95
";
        let p = parse_csv(mom_std);
        assert_eq!(p.stratum, "mom_std");
        assert!(p.cells.contains_key(&(0, 0)));
        assert!(p.cells.contains_key(&(0, 1)));
        assert!(!p.cells.contains_key(&(1, 0)));
        assert!(!p.cells.contains_key(&(1, 1)));
    }
}
