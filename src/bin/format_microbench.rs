//! Format quantization microbenchmark — GF16 vs Posit16 vs bf16 round-trip error.
//!
//! Produces the direct "format quality" numbers used in the F2 paper's
//! §9.4 format-zoo arm. For each format, we measure:
//!
//!   - mean / max absolute reconstruction error
//!   - relative L2 error
//!   - count of values that underflowed to zero (catastrophic loss)
//!   - count of values that saturated (overflowed)
//!
//! All formats are evaluated on the SAME data drawn from a deterministic seed,
//! so deltas are apples-to-apples. The data is:
//!   1. A Xavier-initialized "embedding matrix" of (vocab=128) × (d_model=384),
//!      magnitudes ~1/√384 ≈ 0.05 — the regime where ternary-style formats
//!      collapse to zero (the load-bearing F2 §9.4 catch).
//!   2. The byte histogram of tiny_shakespeare.txt as a "real-data" proxy for
//!      a 200-sample eval distribution.
//!
//! Usage:
//!   cargo run --release --bin format_microbench -- [--seed=42]
//!
//! Output: stdout summary + .trinity/results/format_microbench_seed<S>.json

use std::fs;
use std::io::Write;
use std::path::Path;

use trios_trainer::gf16::GF16;
use trios_trainer::phi_numbers::Posit16;

const VOCAB: usize = 128;
const D_MODEL: usize = 384;

/// Linear-congruential generator — deterministic, seedable, no external dependency.
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
        // Numerical Recipes LCG parameters.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }
    /// Uniform in [-1, +1).
    fn next_unit(&mut self) -> f32 {
        let u = self.next_u32() as f32 / (u32::MAX as f32 + 1.0);
        u * 2.0 - 1.0
    }
}

/// Xavier-init embedding matrix: U[-1, +1) × √(1 / d_model).
fn xavier_init(seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let scale = (1.0_f32 / D_MODEL as f32).sqrt();
    (0..VOCAB * D_MODEL).map(|_| rng.next_unit() * scale).collect()
}

#[derive(Default, Debug, Clone)]
struct FormatStats {
    name: &'static str,
    mean_abs_err: f64,
    max_abs_err: f64,
    rel_l2_err: f64,
    underflow_to_zero: usize,
    saturated_high: usize,
    n: usize,
}

impl FormatStats {
    fn measure<F: Fn(f32) -> f32>(name: &'static str, data: &[f32], roundtrip: F) -> Self {
        let mut sum_abs = 0.0_f64;
        let mut max_abs = 0.0_f64;
        let mut sq_err = 0.0_f64;
        let mut sq_sig = 0.0_f64;
        let mut uflow = 0;
        let mut sat = 0;
        let saturation_threshold = 0.99_f32;
        for &x in data {
            let q = roundtrip(x);
            let err = (q as f64) - (x as f64);
            let abs_err = err.abs();
            sum_abs += abs_err;
            if abs_err > max_abs {
                max_abs = abs_err;
            }
            sq_err += err * err;
            sq_sig += (x as f64) * (x as f64);
            if x != 0.0 && q == 0.0 {
                uflow += 1;
            }
            if x.abs() > saturation_threshold && (q.abs() / x.abs()) < 0.9 {
                sat += 1;
            }
        }
        let n = data.len();
        FormatStats {
            name,
            mean_abs_err: sum_abs / n as f64,
            max_abs_err: max_abs,
            rel_l2_err: (sq_err / sq_sig.max(f64::EPSILON)).sqrt(),
            underflow_to_zero: uflow,
            saturated_high: sat,
            n,
        }
    }
}

fn bf16_round_trip(x: f32) -> f32 {
    let bits = x.to_bits();
    let bf = (bits >> 16) as u16;
    f32::from_bits((bf as u32) << 16)
}

fn report(stats: &FormatStats) -> serde_json::Value {
    serde_json::json!({
        "name": stats.name,
        "n": stats.n,
        "mean_abs_err": stats.mean_abs_err,
        "max_abs_err": stats.max_abs_err,
        "rel_l2_err": stats.rel_l2_err,
        "underflow_to_zero": stats.underflow_to_zero,
        "saturated_high": stats.saturated_high,
    })
}

fn parse_seed_arg() -> u64 {
    let args: Vec<String> = std::env::args().collect();
    for a in args.iter().skip(1) {
        if let Some(v) = a.strip_prefix("--seed=") {
            if let Ok(s) = v.parse::<u64>() {
                return s;
            }
        }
    }
    42
}

fn load_real_data_bytes() -> Vec<f32> {
    // Map tiny_shakespeare bytes to a centered, normalized signal in [-1, +1)
    // so the format saturation/underflow logic is exercised on real-world
    // (non-Gaussian) value distribution rather than only on Xavier init.
    let path = Path::new("data/tiny_shakespeare.txt");
    let bytes = fs::read(path).unwrap_or_default();
    if bytes.is_empty() {
        return Vec::new();
    }
    bytes
        .iter()
        .map(|&b| (b as f32 - 128.0) / 128.0)
        .collect()
}

fn main() {
    let seed = parse_seed_arg();
    println!("# format_microbench — seed = {seed}, vocab = {VOCAB}, d_model = {D_MODEL}");

    // Dataset 1: Xavier-init embedding matrix (small-magnitude regime).
    let embed = xavier_init(seed);
    let embed_n = embed.len();

    // Dataset 2: real bytes from tiny_shakespeare (full byte range).
    let real = load_real_data_bytes();
    let real_n = real.len();
    println!(
        "# datasets: xavier_embed n={embed_n} (~|x|={:.4}), tiny_shakespeare_bytes n={real_n}",
        embed.iter().map(|x| x.abs()).sum::<f32>() / embed_n as f32
    );

    // Stats per format, on each dataset.
    let dsets: [(&str, &[f32]); 2] = [("xavier_embed", &embed), ("tiny_shakespeare", &real)];
    let mut out = serde_json::Map::new();

    for (dset_name, data) in &dsets {
        if data.is_empty() {
            continue;
        }
        let gf16 = FormatStats::measure("gf16", data, |x| GF16::from_f32(x).to_f32());
        let posit16 = FormatStats::measure("posit16", data, |x| Posit16::from_f32(x).to_f32());
        let bf16 = FormatStats::measure("bf16", data, bf16_round_trip);

        println!("\n## {dset_name} (n = {})", data.len());
        for s in [&gf16, &posit16, &bf16] {
            println!(
                "  {:8} mean_abs_err = {:.6e}  max = {:.4e}  rel_L2 = {:.4e}  uflow={}  sat={}",
                s.name, s.mean_abs_err, s.max_abs_err, s.rel_l2_err, s.underflow_to_zero, s.saturated_high
            );
        }

        out.insert(
            (*dset_name).to_string(),
            serde_json::json!({
                "gf16": report(&gf16),
                "posit16": report(&posit16),
                "bf16": report(&bf16),
            }),
        );
    }

    // Headline deltas (the F2-paper-relevant numbers).
    if let Some(xavier) = out.get("xavier_embed").cloned() {
        let gf16_err = xavier["gf16"]["rel_l2_err"].as_f64().unwrap_or(f64::NAN);
        let posit16_err = xavier["posit16"]["rel_l2_err"].as_f64().unwrap_or(f64::NAN);
        let bf16_err = xavier["bf16"]["rel_l2_err"].as_f64().unwrap_or(f64::NAN);
        let posit_vs_gf16 = (posit16_err - gf16_err) / gf16_err;
        let posit_vs_bf16 = (posit16_err - bf16_err) / bf16_err;
        println!("\n## Headline deltas (Xavier-init embed regime)");
        println!("  rel_L2(gf16)    = {gf16_err:.4e}");
        println!("  rel_L2(posit16) = {posit16_err:.4e}");
        println!("  rel_L2(bf16)    = {bf16_err:.4e}");
        println!("  Δ(posit16 vs gf16) = {:+.2}%", 100.0 * posit_vs_gf16);
        println!("  Δ(posit16 vs bf16) = {:+.2}%", 100.0 * posit_vs_bf16);
        out.insert(
            "headline".to_string(),
            serde_json::json!({
                "rel_l2_gf16": gf16_err,
                "rel_l2_posit16": posit16_err,
                "rel_l2_bf16": bf16_err,
                "delta_posit16_vs_gf16": posit_vs_gf16,
                "delta_posit16_vs_bf16": posit_vs_bf16,
            }),
        );
    }

    // Write JSON.
    let _ = fs::create_dir_all(".trinity/results");
    let result_path = format!(".trinity/results/format_microbench_seed{seed}.json");
    let envelope = serde_json::json!({
        "tool": "format_microbench",
        "seed": seed,
        "vocab": VOCAB,
        "d_model": D_MODEL,
        "datasets": out,
        "git_anchor_hint": "f2-methodology branch HEAD at run time",
    });
    fs::File::create(&result_path)
        .unwrap()
        .write_all(serde_json::to_string_pretty(&envelope).unwrap().as_bytes())
        .unwrap();
    println!("\n# Written: {result_path}");
}
