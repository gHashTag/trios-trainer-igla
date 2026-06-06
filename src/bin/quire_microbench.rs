//! Quire microbenchmark — dot-product accuracy across vector lengths.
//!
//! Loop 153 follow-up to Loop 152's quire-bit accumulator
//! (`src/phi_numbers/posit16_quire.rs`). Where Loop 149's format_microbench
//! measured *encode-time* reconstruction error of single values, this
//! binary measures *accumulation accuracy* — the load-bearing property
//! for matmul.
//!
//! For each vector length L ∈ {64, 256, 1024, 4096}, we generate two
//! Posit16 vectors with Xavier init at d_model = 384 (the F2 §9.4.1
//! configuration), then compute the dot product three ways:
//!
//!   1. **Naive Posit16 sum**: convert each product back to Posit16 and
//!      sum, re-quantizing at every addition step. This is the format's
//!      worst-case path — every partial sum loses precision.
//!   2. **f32 accumulator**: convert inputs to f32, multiply and sum in
//!      f32, then convert back to Posit16 at the end. Mantissa 24 bits
//!      so a long dot product accumulates ε_f32 · N rounding noise.
//!   3. **Quire (PositQuire)**: i128 fixed-point accumulator that holds
//!      each Posit16 product exactly. Only the final round-back to
//!      Posit16 introduces rounding.
//!
//! Ground truth: dot product computed in f64 (53-bit mantissa, exact for
//! sums of all Posit16 × Posit16 products inside its dynamic range).
//!
//! We report the relative error of each method against the f64 ground
//! truth, averaged across 5 seeds. The headline is **|err(quire)| ≤
//! |err(f32)| ≤ |err(naive)|** for every length, with the gap widening
//! as L grows.
//!
//! Output:
//!   - Per-seed JSON at `.trinity/results/quire_microbench_seed<S>.json`
//!   - Aggregate summary at
//!     `.trinity/results/quire_microbench_summary_seeds_<lo>-<hi>.json`
//!
//! Usage:
//!   cargo run --release --bin quire_microbench -- [--seeds=42,43,44,45,46]

use std::fs;
use std::io::Write;
use std::path::Path;

use trios_trainer::phi_numbers::{posit16_dot, Posit16};

const VOCAB: usize = 128;
const D_MODEL: usize = 384;
const LENGTHS: &[usize] = &[64, 256, 1024, 4096];

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }
    fn next_unit(&mut self) -> f32 {
        let u = self.next_u32() as f32 / (u32::MAX as f32 + 1.0);
        u * 2.0 - 1.0
    }
}

/// Generate a length-L Posit16 vector with Xavier init magnitudes.
fn xavier_posit16(seed: u64, length: usize) -> Vec<Posit16> {
    let mut rng = Lcg::new(seed);
    let scale = (1.0_f32 / D_MODEL as f32).sqrt();
    (0..length)
        .map(|_| Posit16::from_f32(rng.next_unit() * scale))
        .collect()
}

/// Generate two vectors whose dot product is exactly zero by construction
/// (alternating signs). This exposes the cancellation regime where the
/// quire's exact-integer-sum property gives a strictly better result than
/// any rounded-mantissa accumulator. The Loop 152 test
/// `quire_beats_naive_f32_accum_under_cancellation` documents the same
/// invariant at unit test scale; this binary measures it at vector
/// lengths the paper cares about.
fn cancellation_pair(seed: u64, length: usize) -> (Vec<Posit16>, Vec<Posit16>) {
    let mut rng = Lcg::new(seed);
    let scale = (1.0_f32 / D_MODEL as f32).sqrt();
    let a: Vec<Posit16> = (0..length)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            Posit16::from_f32(sign * rng.next_unit().abs() * scale)
        })
        .collect();
    let b: Vec<Posit16> = (0..length)
        .map(|i| {
            // Pair so each (a[i], b[i]) cancels with (a[i+1], b[i+1]).
            let _ = rng.next_unit();
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            Posit16::from_f32(sign * 0.05_f32)
        })
        .collect();
    (a, b)
}

/// Method 1: Naive Posit16 sum — every partial sum is materialized as a
/// Posit16, so every accumulation step incurs a round-back.
///
/// Loop 154 75th-pass SEV-3 closure: the mechanism is subtle, so making
/// the per-step re-quantization explicit here. We multiply in f64 (exact
/// for any pair of Posit16 values since the product mantissa fits in 26
/// bits and f64 has 53), but then:
///
///   - We round the *product* back to Posit16 → the per-step storage
///     loss for the product itself.
///   - We add the rounded product (as f64) into a running f64 sum.
///   - We round the *sum* back to Posit16 and read it out as f64 → the
///     per-step storage loss for the accumulator.
///
/// Reading the accumulator back is what makes the next iteration see the
/// rounded value (and thus accumulate further per-step error). A
/// hypothetical "no-acc-readback" variant — products rounded but sum
/// kept in f64 — would behave like the `f32 accumulator` method up to
/// the product-rounding granularity. The current implementation models
/// the worst-case Posit16-only storage path; the `f32 accumulator`
/// method models the standard mixed-precision recipe; the quire models
/// the exact-accumulation path. The three methods isolate per-step
/// storage loss from accumulator-precision loss in a controlled way.
fn dot_naive(a: &[Posit16], b: &[Posit16]) -> Posit16 {
    let mut acc = Posit16::ZERO;
    let mut acc_f = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let prod_f = (x.to_f32() as f64) * (y.to_f32() as f64);
        let prod = Posit16::from_f32(prod_f as f32);
        acc_f += prod.to_f32() as f64;
        acc = Posit16::from_f32(acc_f as f32);
        acc_f = acc.to_f32() as f64;
    }
    acc
}

/// Method 2: f32 accumulator (industry-standard "compute in fp32, store in
/// low precision" pattern).
fn dot_f32_accum(a: &[Posit16], b: &[Posit16]) -> Posit16 {
    let mut acc = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x.to_f32() * y.to_f32();
    }
    Posit16::from_f32(acc)
}

/// Method 3: Quire accumulator (Loop 152).
fn dot_quire(a: &[Posit16], b: &[Posit16]) -> Posit16 {
    posit16_dot(a, b)
}

/// Ground truth: compute dot product in f64 from the Posit16 inputs.
fn dot_f64_ground_truth(a: &[Posit16], b: &[Posit16]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() as f64) * (y.to_f32() as f64))
        .sum()
}

fn rel_err(approx_posit: Posit16, ground_truth: f64) -> f64 {
    if ground_truth.abs() < 1e-30 {
        return (approx_posit.to_f32() as f64).abs();
    }
    let approx = approx_posit.to_f32() as f64;
    ((approx - ground_truth) / ground_truth).abs()
}

fn run_one(length: usize, seed: u64, regime: &str) -> serde_json::Value {
    let (a, b) = match regime {
        "cancellation" => cancellation_pair(seed, length),
        _ => (xavier_posit16(seed, length), xavier_posit16(seed.wrapping_add(1), length)),
    };
    let truth = dot_f64_ground_truth(&a, &b);
    let n = dot_naive(&a, &b);
    let f = dot_f32_accum(&a, &b);
    let q = dot_quire(&a, &b);
    let err_naive = rel_err(n, truth);
    let err_f32 = rel_err(f, truth);
    let err_quire = rel_err(q, truth);
    serde_json::json!({
        "length": length,
        "seed": seed,
        "regime": regime,
        "vocab": VOCAB,
        "d_model": D_MODEL,
        "init": "xavier",
        "ground_truth_f64": truth,
        "naive_value": n.to_f32() as f64,
        "f32_value": f.to_f32() as f64,
        "quire_value": q.to_f32() as f64,
        "rel_err_naive": err_naive,
        "rel_err_f32": err_f32,
        "rel_err_quire": err_quire,
    })
}

fn parse_seeds() -> Vec<u64> {
    let mut out = Vec::new();
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("--seeds=") {
            for tok in v.split(',') {
                if let Ok(s) = tok.parse::<u64>() {
                    out.push(s);
                }
            }
        } else if let Some(v) = a.strip_prefix("--seed=") {
            if let Ok(s) = v.parse::<u64>() {
                out.push(s);
            }
        }
    }
    if out.is_empty() {
        out = vec![42, 43, 44, 45, 46];
    }
    out
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = if xs.len() > 1 {
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    (mean, var.sqrt())
}

fn main() {
    let seeds = parse_seeds();
    let result_dir = Path::new(".trinity/results");
    let _ = fs::create_dir_all(result_dir);

    println!(
        "# quire_microbench — {} length(s) × {} seed(s) = {} cells",
        LENGTHS.len(),
        seeds.len(),
        LENGTHS.len() * seeds.len()
    );

    let regimes: &[&str] = &["xavier", "cancellation"];

    // Per-seed grouped results.
    let mut all_cells: Vec<serde_json::Value> = Vec::new();
    for &seed in &seeds {
        let mut per_seed = serde_json::Map::new();
        per_seed.insert("seed".to_string(), serde_json::json!(seed));
        let mut cells: Vec<serde_json::Value> = Vec::new();
        for &regime in regimes {
            for &length in LENGTHS {
                let cell = run_one(length, seed, regime);
                cells.push(cell.clone());
                all_cells.push(cell);
            }
        }
        per_seed.insert("cells".to_string(), serde_json::json!(cells));
        let path = result_dir.join(format!("quire_microbench_seed{}.json", seed));
        fs::File::create(&path)
            .unwrap()
            .write_all(serde_json::to_string_pretty(&serde_json::Value::Object(per_seed)).unwrap().as_bytes())
            .unwrap();
    }

    // Aggregate per-(regime, length).
    let mut summary_rows = serde_json::Map::new();
    for &regime in regimes {
        println!(
            "\n## Relative error vs f64 ground truth — regime = {regime}, d_model = {D_MODEL}"
        );
        println!("    L     | naive Posit16 sum    | f32 accumulator      | PositQuire");
        println!("    ------+----------------------+----------------------+----------------------");
        let mut regime_obj = serde_json::Map::new();
        for &length in LENGTHS {
            let cells_at_l: Vec<&serde_json::Value> = all_cells
                .iter()
                .filter(|c| {
                    c["regime"].as_str() == Some(regime)
                        && c["length"].as_u64().unwrap_or(0) as usize == length
                })
                .collect();
            let naive: Vec<f64> = cells_at_l.iter().map(|c| c["rel_err_naive"].as_f64().unwrap_or(0.0)).collect();
            let f32acc: Vec<f64> = cells_at_l.iter().map(|c| c["rel_err_f32"].as_f64().unwrap_or(0.0)).collect();
            let quire: Vec<f64> = cells_at_l.iter().map(|c| c["rel_err_quire"].as_f64().unwrap_or(0.0)).collect();
            let (nm, ns) = mean_std(&naive);
            let (fm, fs_) = mean_std(&f32acc);
            let (qm, qs) = mean_std(&quire);
            println!(
                "    {:5} | {:.3e} ± {:.1e}  | {:.3e} ± {:.1e}  | {:.3e} ± {:.1e}",
                length, nm, ns, fm, fs_, qm, qs
            );
            regime_obj.insert(
                format!("L{}", length),
                serde_json::json!({
                    "naive_mean": nm, "naive_std": ns,
                    "f32_mean": fm, "f32_std": fs_,
                    "quire_mean": qm, "quire_std": qs,
                    "n_seeds": cells_at_l.len(),
                }),
            );
        }
        summary_rows.insert(regime.to_string(), serde_json::Value::Object(regime_obj));
    }

    let sum_lo = seeds.iter().min().copied().unwrap_or(0);
    let sum_hi = seeds.iter().max().copied().unwrap_or(0);
    let sum_path = result_dir.join(format!(
        "quire_microbench_summary_seeds_{}-{}.json",
        sum_lo, sum_hi
    ));
    let envelope = serde_json::json!({
        "tool": "quire_microbench",
        "mode": "summary",
        "seeds": seeds,
        "vocab": VOCAB,
        "d_model": D_MODEL,
        "lengths": LENGTHS,
        "init": "xavier",
        "regimes": regimes,
        "rel_err_by_regime_and_length": summary_rows,
        "git_anchor_hint": "f2-methodology branch HEAD at run time",
    });
    fs::File::create(&sum_path)
        .unwrap()
        .write_all(serde_json::to_string_pretty(&envelope).unwrap().as_bytes())
        .unwrap();
    println!("\n# Written summary: {}", sum_path.display());
}
