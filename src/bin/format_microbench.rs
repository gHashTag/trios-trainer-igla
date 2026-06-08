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
//! Two modes:
//!
//! 1. **Single-cell** (Loop 146, kept for backward compat):
//!    - Xavier-init at d_model=384, plus tiny_shakespeare bytes.
//!    - Output: `.trinity/results/format_microbench_seed<S>.json`.
//!    - Invocation: `--seed=<N>` only.
//!
//! 2. **Grid** (Loop 149, new):
//!    - {d_model: 128, 384, 768, 1024} × {init: xavier, he, normal_002}
//!      = 12 cells per seed × N seeds.
//!    - Output: `.trinity/results/format_microbench_grid/d<D>_<init>_seed<S>.json`
//!      + a summary `format_microbench_grid_summary_seeds_<lo>-<hi>.json`.
//!    - Invocation: `--grid --seeds=<csv>`.
//!
//! Usage:
//!   cargo run --release --bin format_microbench -- [--seed=42]
//!   cargo run --release --bin format_microbench -- --grid [--seeds=42,43,44,45,46]
//!
//! In grid mode, omitting `--seeds=` uses the seeds listed in the shared
//! config (`papers/scripts/format_microbench_grid_config.json`). An explicit
//! `--seed=N` or `--seeds=...` overrides for ad-hoc subset runs (e.g.,
//! `--grid --seed=42` runs only seed 42 — useful for quick smoke-tests
//! but does NOT regenerate the full committed grid). 73rd-pass SEV-5
//! documentation.

use std::fs;
use std::io::Write;
use std::path::Path;

use trios_trainer::gf16::GF16;
use trios_trainer::phi_numbers::Posit16;

const VOCAB: usize = 128;
const DEFAULT_D_MODEL: usize = 384;

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
    /// Uniform in [0, 1).
    fn next_uniform(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }
    /// Standard normal via Box-Muller transform. Returns one sample per call;
    /// the second sample is discarded (NOT buffered) to keep per-cell RNG state
    /// deterministic across grid-cell boundaries — we don't want one cell's
    /// residual to bleed into the next.
    ///
    /// u1 is clamped to [1e-30, 1) to guard the ln(u1) singularity at exactly
    /// zero. The LCG can in principle emit u1==0 on a pathological sequence
    /// (≈1 in 2^32 per draw); the clamp keeps `r` finite. Loop 149 71st-pass
    /// SEV-1 catch — documentation, not algorithmic change.
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_uniform().max(1e-30);
        let u2 = self.next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

/// Initialization scheme for the embedding matrix. Loop 149 71st-pass SEV-2
/// disambiguation: the `He` variant uses σ = √(2/d_model), NOT the strict
/// He-2015 σ = √(2/fan_in). For an embedding matrix, fan_in is the vocabulary
/// dimension (here 128) and fan_out is d_model; "He-fan-in" would be
/// regime-independent of d_model, defeating the purpose of the d_model sweep.
/// We use the d_model-scaled form as a *regime label* for "He-style normal
/// init scaled with output dimension", which is the form most relevant to
/// the post-LayerNorm activations that follow an embedding lookup. The
/// `xavier` variant uses σ = √(1/d_model) (uniform) for the same regime-
/// labeling reason.
#[derive(Clone, Copy, Debug)]
enum Init {
    Xavier,
    He,
    Normal002,
}

impl Init {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "xavier" => Some(Self::Xavier),
            "he" => Some(Self::He),
            "normal_002" | "normal" => Some(Self::Normal002),
            _ => None,
        }
    }
    fn slug(self) -> &'static str {
        match self {
            Self::Xavier => "xavier",
            Self::He => "he",
            Self::Normal002 => "normal_002",
        }
    }
    fn typical_abs_mean(self, d_model: usize) -> f32 {
        // Documentation-only: not used for measurement, just for the report.
        let d = d_model as f32;
        match self {
            // Xavier U[-a,a] with a = √(1/d) → E[|X|] = a/2
            Self::Xavier => (1.0 / d).sqrt() / 2.0,
            // He normal σ = √(2/d) → E[|X|] = σ·√(2/π)
            Self::He => (2.0 / d).sqrt() * (2.0 / std::f32::consts::PI).sqrt(),
            // GPT-style fixed σ = 0.02 → E[|X|] = 0.02·√(2/π)
            Self::Normal002 => 0.02 * (2.0 / std::f32::consts::PI).sqrt(),
        }
    }
}

/// Generate the d_model-wide embedding matrix for a given (init, seed, d_model)
/// triple. Deterministic — re-running with the same inputs reproduces bit-exact
/// values.
fn embed_init(init: Init, seed: u64, d_model: usize) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let n = VOCAB * d_model;
    let mut out = Vec::with_capacity(n);
    match init {
        Init::Xavier => {
            let scale = (1.0_f32 / d_model as f32).sqrt();
            for _ in 0..n {
                out.push(rng.next_unit() * scale);
            }
        }
        Init::He => {
            let sigma = (2.0_f32 / d_model as f32).sqrt();
            for _ in 0..n {
                out.push(rng.next_normal() * sigma);
            }
        }
        Init::Normal002 => {
            let sigma = 0.02_f32;
            for _ in 0..n {
                out.push(rng.next_normal() * sigma);
            }
        }
    }
    out
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

fn load_real_data_bytes() -> Vec<f32> {
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

#[derive(Default)]
struct Args {
    grid: bool,
    seeds: Vec<u64>,
    seeds_explicit: bool,
}

fn parse_args() -> Args {
    let mut out = Args::default();
    let argv: Vec<String> = std::env::args().collect();
    for a in argv.iter().skip(1) {
        if a == "--grid" {
            out.grid = true;
        } else if let Some(v) = a.strip_prefix("--seed=") {
            if let Ok(s) = v.parse::<u64>() {
                out.seeds.push(s);
                out.seeds_explicit = true;
            }
        } else if let Some(v) = a.strip_prefix("--seeds=") {
            for tok in v.split(',') {
                if let Ok(s) = tok.parse::<u64>() {
                    out.seeds.push(s);
                }
            }
            out.seeds_explicit = true;
        }
    }
    out
}

fn run_one(init: Init, seed: u64, d_model: usize, data: &[f32]) -> serde_json::Value {
    let gf16 = FormatStats::measure("gf16", data, |x| GF16::from_f32(x).to_f32());
    let posit16 = FormatStats::measure("posit16", data, |x| Posit16::from_f32(x).to_f32());
    let bf16 = FormatStats::measure("bf16", data, bf16_round_trip);

    let gf16_err = gf16.rel_l2_err;
    let posit16_err = posit16.rel_l2_err;
    let bf16_err = bf16.rel_l2_err;
    let posit_vs_gf16 = if gf16_err > 0.0 { (posit16_err - gf16_err) / gf16_err } else { 0.0 };
    let posit_vs_bf16 = if bf16_err > 0.0 { (posit16_err - bf16_err) / bf16_err } else { 0.0 };

    serde_json::json!({
        "init": init.slug(),
        "d_model": d_model,
        "seed": seed,
        "vocab": VOCAB,
        "n_values": data.len(),
        "expected_abs_mean": init.typical_abs_mean(d_model),
        "gf16": report(&gf16),
        "posit16": report(&posit16),
        "bf16": report(&bf16),
        "headline": {
            "rel_l2_gf16": gf16_err,
            "rel_l2_posit16": posit16_err,
            "rel_l2_bf16": bf16_err,
            "delta_posit16_vs_gf16": posit_vs_gf16,
            "delta_posit16_vs_bf16": posit_vs_bf16,
        },
    })
}

fn single_mode(seed: u64) {
    println!("# format_microbench — seed = {seed}, vocab = {VOCAB}, d_model = {DEFAULT_D_MODEL}");
    let embed = embed_init(Init::Xavier, seed, DEFAULT_D_MODEL);
    let real = load_real_data_bytes();
    let real_n = real.len();
    println!(
        "# datasets: xavier_embed n={} (~|x|={:.4}), tiny_shakespeare_bytes n={real_n}",
        embed.len(),
        embed.iter().map(|x| x.abs()).sum::<f32>() / embed.len() as f32
    );

    let xavier_cell = run_one(Init::Xavier, seed, DEFAULT_D_MODEL, &embed);
    let real_cell = if !real.is_empty() {
        Some(run_one(Init::Xavier, seed, DEFAULT_D_MODEL, &real))
    } else {
        None
    };

    for (name, cell) in [("xavier_embed", &xavier_cell)]
        .into_iter()
        .chain(real_cell.as_ref().map(|c| ("tiny_shakespeare", c)))
    {
        println!("\n## {name}");
        for fmt_key in ["gf16", "posit16", "bf16"] {
            let s = &cell[fmt_key];
            println!(
                "  {:8} mean_abs_err = {:.6e}  max = {:.4e}  rel_L2 = {:.4e}  uflow={}  sat={}",
                fmt_key,
                s["mean_abs_err"].as_f64().unwrap_or(0.0),
                s["max_abs_err"].as_f64().unwrap_or(0.0),
                s["rel_l2_err"].as_f64().unwrap_or(0.0),
                s["underflow_to_zero"].as_i64().unwrap_or(0),
                s["saturated_high"].as_i64().unwrap_or(0),
            );
        }
    }

    let h = &xavier_cell["headline"];
    println!("\n## Headline deltas (Xavier-init embed regime)");
    println!("  rel_L2(gf16)    = {:.4e}", h["rel_l2_gf16"].as_f64().unwrap_or(0.0));
    println!("  rel_L2(posit16) = {:.4e}", h["rel_l2_posit16"].as_f64().unwrap_or(0.0));
    println!("  rel_L2(bf16)    = {:.4e}", h["rel_l2_bf16"].as_f64().unwrap_or(0.0));
    println!(
        "  Δ(posit16 vs gf16) = {:+.2}%",
        100.0 * h["delta_posit16_vs_gf16"].as_f64().unwrap_or(0.0)
    );
    println!(
        "  Δ(posit16 vs bf16) = {:+.2}%",
        100.0 * h["delta_posit16_vs_bf16"].as_f64().unwrap_or(0.0)
    );

    let _ = fs::create_dir_all(".trinity/results");
    let result_path = format!(".trinity/results/format_microbench_seed{seed}.json");
    let envelope = serde_json::json!({
        "tool": "format_microbench",
        "mode": "single",
        "seed": seed,
        "vocab": VOCAB,
        "d_model": DEFAULT_D_MODEL,
        "datasets": {
            "xavier_embed": xavier_cell,
            "tiny_shakespeare": real_cell,
        },
        "git_anchor_hint": "f2-methodology branch HEAD at run time",
    });
    fs::File::create(&result_path)
        .unwrap()
        .write_all(serde_json::to_string_pretty(&envelope).unwrap().as_bytes())
        .unwrap();
    println!("\n# Written: {result_path}");
}

/// Load the shared grid config from
/// `papers/scripts/format_microbench_grid_config.json` and return
/// `(d_models, inits, seeds_from_config)`. If the file is missing or
/// malformed, fall back to the documented defaults so the binary still
/// produces a valid grid (Loop 151 A: the config is the *contract* between
/// the binary and the freshness gate, not a hard runtime dependency).
fn load_grid_config(
    seeds_override: &[u64],
) -> (Vec<usize>, Vec<Init>, Vec<u64>) {
    let config_path = Path::new("papers/scripts/format_microbench_grid_config.json");
    let default_d_models: Vec<usize> = vec![128, 384, 768, 1024];
    let default_inits: Vec<Init> = vec![Init::Xavier, Init::He, Init::Normal002];
    let default_seeds: Vec<u64> = vec![42, 43, 44, 45, 46];

    let text = match fs::read_to_string(config_path) {
        Ok(t) => t,
        Err(_) => {
            eprintln!(
                "# WARN  {} missing — using defaults",
                config_path.display()
            );
            return (
                default_d_models,
                default_inits,
                if seeds_override.is_empty() {
                    default_seeds
                } else {
                    seeds_override.to_vec()
                },
            );
        }
    };

    let val: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("# WARN  {} unparseable ({e}); using defaults",
                      config_path.display());
            return (
                default_d_models,
                default_inits,
                if seeds_override.is_empty() {
                    default_seeds
                } else {
                    seeds_override.to_vec()
                },
            );
        }
    };

    let d_models: Vec<usize> = val["d_models"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|u| u as usize))
                .collect()
        })
        .unwrap_or(default_d_models);
    // 73rd-pass SEV-1 closure: emit a WARN for any init string in the
    // config that doesn't map to a known Init variant, instead of silently
    // dropping it. Silent dropout would leave the freshness gate expecting
    // cells the binary never produced.
    let inits: Vec<Init> = val["inits"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let s = v.as_str()?;
                    let parsed = Init::from_str(s);
                    if parsed.is_none() {
                        eprintln!(
                            "# WARN  grid config init `{}` not recognized; \
                             dropping (known: xavier, he, normal_002 | \
                             normal)", s
                        );
                    }
                    parsed
                })
                .collect()
        })
        .unwrap_or(default_inits);
    let cfg_seeds: Vec<u64> = val["seeds"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64())
                .collect()
        })
        .unwrap_or_default();

    // CLI seeds override config seeds when provided; this lets a user run a
    // subset for quick iteration without editing the config.
    let seeds = if !seeds_override.is_empty() {
        seeds_override.to_vec()
    } else if !cfg_seeds.is_empty() {
        cfg_seeds
    } else {
        default_seeds
    };

    (d_models, inits, seeds)
}

fn grid_mode(seeds_cli: &[u64]) {
    let (d_models_vec, inits_vec, seeds_eff) = load_grid_config(seeds_cli);
    let d_models = d_models_vec.as_slice();
    let inits = inits_vec.as_slice();
    let seeds = seeds_eff.as_slice();
    let n_cells = d_models.len() * inits.len() * seeds.len();
    println!(
        "# format_microbench --grid: {} cells ({} d_model × {} init × {} seeds)",
        n_cells,
        d_models.len(),
        inits.len(),
        seeds.len(),
    );
    let grid_dir = Path::new(".trinity/results/format_microbench_grid");
    let _ = fs::create_dir_all(grid_dir);

    // Per-(init, d_model) aggregator: collect rel_l2 across seeds.
    let mut agg: std::collections::BTreeMap<(String, usize), Vec<f64>> = std::collections::BTreeMap::new();

    for &seed in seeds {
        for &d in d_models {
            for &init in inits {
                let embed = embed_init(init, seed, d);
                let cell = run_one(init, seed, d, &embed);
                let posit_vs_gf16 = cell["headline"]["delta_posit16_vs_gf16"]
                    .as_f64()
                    .unwrap_or(0.0);
                agg.entry((init.slug().to_string(), d))
                    .or_default()
                    .push(posit_vs_gf16);
                let fname = format!("d{}_{}_seed{}.json", d, init.slug(), seed);
                let path = grid_dir.join(&fname);
                let envelope = serde_json::json!({
                    "tool": "format_microbench",
                    "mode": "grid_cell",
                    "init": init.slug(),
                    "d_model": d,
                    "seed": seed,
                    "vocab": VOCAB,
                    "n_values": embed.len(),
                    "cell": cell,
                });
                fs::File::create(&path)
                    .unwrap()
                    .write_all(serde_json::to_string_pretty(&envelope).unwrap().as_bytes())
                    .unwrap();
                print!(".");
                std::io::stdout().flush().ok();
            }
        }
    }
    println!();

    // Summary table.
    println!("\n## Δ(posit16 vs gf16) by (init, d_model) — % rel L2, {}-seed mean ± std", seeds.len());
    let mut header = String::from("              ");
    for d in d_models {
        header.push_str(&format!("  d={:<6}", d));
    }
    println!("{header}");
    let mut summary_rows = serde_json::Map::new();
    for init in inits {
        let mut row = format!("  {:10}", init.slug());
        let mut row_obj = serde_json::Map::new();
        for &d in d_models {
            let key = (init.slug().to_string(), d);
            let v = agg.get(&key).cloned().unwrap_or_default();
            let n = v.len() as f64;
            let mean = if n > 0.0 { v.iter().sum::<f64>() / n } else { 0.0 };
            let var = if n > 1.0 {
                v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
            } else {
                0.0
            };
            let std = var.sqrt();
            row.push_str(&format!("  {:+.1}±{:.2}%", 100.0 * mean, 100.0 * std));
            row_obj.insert(
                format!("d{}", d),
                serde_json::json!({
                    "delta_posit16_vs_gf16_mean": mean,
                    "delta_posit16_vs_gf16_std": std,
                    "n_seeds": v.len(),
                }),
            );
        }
        println!("{row}");
        summary_rows.insert(init.slug().to_string(), serde_json::Value::Object(row_obj));
    }

    let sum_lo = seeds.iter().min().copied().unwrap_or(0);
    let sum_hi = seeds.iter().max().copied().unwrap_or(0);
    let sum_path = grid_dir.join(format!(
        "format_microbench_grid_summary_seeds_{}-{}.json",
        sum_lo, sum_hi
    ));
    let envelope = serde_json::json!({
        "tool": "format_microbench",
        "mode": "grid_summary",
        "seeds": seeds,
        "vocab": VOCAB,
        "d_models": d_models,
        "inits": inits.iter().map(|i| i.slug()).collect::<Vec<_>>(),
        "n_cells": n_cells,
        "delta_posit16_vs_gf16": summary_rows,
        "git_anchor_hint": "f2-methodology branch HEAD at run time",
    });
    fs::File::create(&sum_path)
        .unwrap()
        .write_all(serde_json::to_string_pretty(&envelope).unwrap().as_bytes())
        .unwrap();
    println!("\n# Written summary: {}", sum_path.display());
    println!("# Per-cell JSONs: {}/", grid_dir.display());
}

fn main() {
    let args = parse_args();
    if args.grid {
        // In grid mode: explicit --seeds wins; otherwise fall back to the
        // config's seeds (Loop 151 A — config is the single source of truth).
        let cli_seeds = if args.seeds_explicit {
            args.seeds.clone()
        } else {
            Vec::new()
        };
        grid_mode(&cli_seeds);
    } else {
        let seed = if args.seeds_explicit && !args.seeds.is_empty() {
            args.seeds[0]
        } else {
            42
        };
        single_mode(seed);
    }
}
