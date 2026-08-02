//! format_champion_sweep — champion-scale per-format BPB sweep.
//!
//! Runs the champion HybridModel harness (`train_loop::run_single`) across
//! every real-kernel numeric format from `fake_quant::FormatKind`, over a
//! configurable seed set, and emits per-format median/mean/std BPB as JSONL.
//!
//! This closes the honest gap flagged in issue #183: instead of the toy
//! byte-level `matrix_runner` path (n-gram, near-random BPB≈7.0), it drives
//! the real champion harness with per-format fake-quant applied to ALL weight
//! tensors (embed/proj/lm_head/attn_*) at init AND inside the training loop.
//!
//! HONESTY NOTES (do not remove):
//!   * `train_loop::VOCAB` is a compile-time const = 128 → BPB here is a
//!     byte-level (vocab-128) metric. It is INTERNALLY consistent across
//!     formats at a fixed (hidden, steps, corpus) but is NOT the vocab-32000
//!     champion-lock number (issue #181, BPB=2.1919). Compare formats to each
//!     other within one sweep — not against the #181 lock.
//!   * `--train` and `--val` are REQUIRED and have no defaults. They previously
//!     defaulted to a corpus pair in `data/` where the "val" file was a 160-byte
//!     pangram (one single eval window at stride SEQ+1=129) and the "heldout"
//!     file was byte-identical to the training set. Those defaults manufactured
//!     unqualified numbers such as `bpb=6.4358`; see `data/README.md`.
//!   * `preflight_corpora` hard-errors BEFORE any training if train and val hash
//!     to the same SHA-256, or if val is shorter than `MIN_VAL_BYTES`. It prints
//!     the accepted hashes and byte counts so every reported row can be traced
//!     to exact corpus bytes.
//!   * Formats in `is_unsupported_in_f32()` are no-ops in this f32 sim and are
//!     skipped by default (they would report == f32 and mislead).
//!
//! Usage:
//!   format_champion_sweep \
//!       --hidden 828 --steps 27000 --lr 0.003 --attn-layers 2 \
//!       --eval-every 1000 --seeds 1,2,3,5,8,13,21,34,55,89,144 \
//!       --train <train corpus> --val <disjoint val corpus> \
//!       --out sweep_out.jsonl

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::process;

use trios_trainer::checkpoint::sha256_hex;
use trios_trainer::fake_quant::FormatKind;
use trios_trainer::train_loop::{self, TrainArgs};

/// Minimum accepted validation corpus size, in bytes.
///
/// `evaluate` walks the val stream with stride `SEQ + 1 = 129` and needs a full
/// window, so a corpus of N bytes yields `1 + (N - 129) / 129` chunks. 8192
/// bytes is 63 chunks - still small, but no longer a single-window reading
/// reported as if it were a validation average.
const MIN_VAL_BYTES: u64 = 8192;

/// Canonical real-kernel format list (env-name, has non-trivial f32 kernel).
/// Derived from `FormatKind::from_env` accepted names, minus the
/// `is_unsupported_in_f32` set (they no-op in f32 sim). Grouped for the report.
// Groups (in order): f32 baseline; IEEE narrow floats; OCP FP8/FP6/FP4;
// MX block floats; GoldenFloat phi-family; integer/fixed; NormalFloat;
// posit (tapered); log-number-system; additional-float-precision.
const REAL_KERNEL_FORMATS: &[&str] = &[
    "f32", "fp16", "bf16", "tf32", "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "fp4_e2m1",
    "mxfp4", "mxfp6", "mxfp8", "gf4", "gf8", "gf12", "gf16", "gf20", "gf24", "gf32", "gf64",
    "int4", "int8", "int16", "int32", "uint8", "q15", "nf4", "nf8", "posit8", "posit16", "posit32",
    "lns8", "afp",
];

fn fib_lucas_seeds() -> Vec<u64> {
    // Fibonacci/Lucas canon, ≥11 seeds. FORBIDDEN: 42,43,44,45.
    vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
}

struct Args {
    hidden: usize,
    steps: usize,
    lr: f32,
    attn_layers: u8,
    eval_every: usize,
    seeds: Vec<u64>,
    /// REQUIRED. No default: see `preflight_corpora` and `data/README.md`.
    train_path: Option<String>,
    /// REQUIRED. No default: see `preflight_corpora` and `data/README.md`.
    val_path: Option<String>,
    out: String,
    formats: Vec<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        hidden: 828,
        steps: 27000,
        lr: 0.003,
        attn_layers: 2,
        eval_every: 1000,
        seeds: fib_lucas_seeds(),
        train_path: None,
        val_path: None,
        out: "format_champion_sweep.jsonl".to_string(),
        formats: REAL_KERNEL_FORMATS.iter().map(|s| s.to_string()).collect(),
    };
    let argv: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let k = argv[i].as_str();
        let mut next = || {
            i += 1;
            argv.get(i).cloned().unwrap_or_default()
        };
        match k {
            "--hidden" => a.hidden = next().parse().unwrap(),
            "--steps" => a.steps = next().parse().unwrap(),
            "--lr" => a.lr = next().parse().unwrap(),
            "--attn-layers" => a.attn_layers = next().parse().unwrap(),
            "--eval-every" => a.eval_every = next().parse().unwrap(),
            "--seeds" => {
                a.seeds = next()
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--train" => a.train_path = Some(next()),
            "--val" => a.val_path = Some(next()),
            "--out" => a.out = next(),
            "--formats" => {
                a.formats = next()
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect()
            }
            _ => {}
        }
        i += 1;
    }
    // `--train` with nothing after it yields an empty string; that is a missing
    // required argument, not a path.
    a.train_path = a.train_path.filter(|s| !s.is_empty());
    a.val_path = a.val_path.filter(|s| !s.is_empty());
    a
}

fn usage_line() -> String {
    "usage: format_champion_sweep --train <train corpus> --val <disjoint val corpus> \
     [--hidden N] [--steps N] [--lr F] [--attn-layers N] [--eval-every N] \
     [--seeds a,b,c] [--formats a,b,c] [--out FILE]"
        .to_string()
}

/// Accepted corpus pair: paths, byte counts and SHA-256, printed before any run.
struct CorpusPair {
    train_path: String,
    train_bytes: usize,
    train_sha256: String,
    val_path: String,
    val_bytes: usize,
    val_sha256: String,
}

/// Hard precondition on the corpus pair, evaluated BEFORE any training.
///
/// Refuses on: a missing `--train`/`--val`, an unreadable file, train and val
/// hashing to the same SHA-256 (a "held out" set that is the training set), or
/// a val corpus shorter than `MIN_VAL_BYTES` (a single-window BPB reported as
/// if it were a validation average). Every message names the actual paths and
/// the actual hashes, so an operator cannot misread which files were rejected.
fn preflight_corpora(a: &Args) -> Result<CorpusPair, String> {
    let train_path = a.train_path.clone().ok_or_else(|| {
        format!(
            "missing required argument --train (no default).\n{}",
            usage_line()
        )
    })?;
    let val_path = a.val_path.clone().ok_or_else(|| {
        format!(
            "missing required argument --val (no default).\n{}",
            usage_line()
        )
    })?;

    let train_bytes = std::fs::read(&train_path)
        .map_err(|e| format!("cannot read train corpus '{}': {}", train_path, e))?;
    let val_bytes = std::fs::read(&val_path)
        .map_err(|e| format!("cannot read val corpus '{}': {}", val_path, e))?;

    let train_sha256 = sha256_hex(&train_bytes);
    let val_sha256 = sha256_hex(&val_bytes);

    if train_sha256 == val_sha256 {
        return Err(format!(
            "train and val corpora are byte-identical - the val set IS the train set.\n  \
             train: {} ({} bytes) sha256={}\n  \
             val:   {} ({} bytes) sha256={}\n  \
             Refusing to run: any BPB from this pair measures memorization of the \
             training bytes. See data/README.md.",
            train_path,
            train_bytes.len(),
            train_sha256,
            val_path,
            val_bytes.len(),
            val_sha256
        ));
    }

    if (val_bytes.len() as u64) < MIN_VAL_BYTES {
        return Err(format!(
            "val corpus is too small to produce a validation average.\n  \
             val:   {} ({} bytes) sha256={}\n  \
             minimum: {} bytes. At stride SEQ+1=129 a {}-byte corpus yields {} eval \
             chunk(s); a BPB over that few windows is a single-window reading, not a \
             validation set average. See data/README.md.",
            val_path,
            val_bytes.len(),
            val_sha256,
            MIN_VAL_BYTES,
            val_bytes.len(),
            eval_chunks(val_bytes.len())
        ));
    }

    Ok(CorpusPair {
        train_path,
        train_bytes: train_bytes.len(),
        train_sha256,
        val_path,
        val_bytes: val_bytes.len(),
        val_sha256,
    })
}

/// Number of full eval windows a corpus of `n` bytes yields at stride SEQ+1=129.
fn eval_chunks(n: usize) -> usize {
    const STRIDE: usize = 129;
    if n < STRIDE {
        0
    } else {
        1 + (n - STRIDE) / STRIDE
    }
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if n == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let m = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, var.sqrt())
}

fn main() {
    let a = parse_args();

    // Corpus precondition first: nothing is trained, and no output file is
    // created, until the corpus pair is accepted and its hashes are on the
    // record.
    let corpus = match preflight_corpora(&a) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("[sweep][FATAL] {}", msg);
            process::exit(1);
        }
    };

    eprintln!(
        "[sweep] hidden={} steps={} lr={} attn_layers={} eval_every={} seeds={:?}",
        a.hidden, a.steps, a.lr, a.attn_layers, a.eval_every, a.seeds
    );
    eprintln!(
        "[sweep] train={} bytes={} sha256={}",
        corpus.train_path, corpus.train_bytes, corpus.train_sha256
    );
    eprintln!(
        "[sweep] val={} bytes={} sha256={} eval_chunks={}",
        corpus.val_path,
        corpus.val_bytes,
        corpus.val_sha256,
        eval_chunks(corpus.val_bytes)
    );

    let mut out = File::create(&a.out).expect("create out");

    for fmt_name in &a.formats {
        // Validate + skip f32-noop (unsupported) formats honestly.
        let kind = match FormatKind::from_env(fmt_name) {
            Some(k) => k,
            None => {
                eprintln!("[sweep][skip] unknown format '{}'", fmt_name);
                continue;
            }
        };
        let is_baseline = kind == FormatKind::F32;
        if kind.is_unsupported_in_f32() {
            eprintln!(
                "[sweep][skip] '{}' is no-op in f32 sim (is_unsupported_in_f32) — excluded",
                fmt_name
            );
            continue;
        }

        let mut bpbs: Vec<f64> = Vec::new();
        let mut per_seed: BTreeMap<u64, f64> = BTreeMap::new();

        for &seed in &a.seeds {
            // run_single reads TRIOS_FORMAT_TYPE via resolve_fake_quant_format().
            env::set_var("TRIOS_FORMAT_TYPE", fmt_name);
            let args = TrainArgs {
                seed,
                steps: a.steps,
                hidden: a.hidden,
                lr: a.lr,
                attn_layers: a.attn_layers,
                eval_every: a.eval_every,
                train_path: corpus.train_path.clone(),
                val_path: corpus.val_path.clone(),
            };
            match train_loop::run_single(&args) {
                Ok(outcome) => {
                    eprintln!(
                        "[sweep] fmt={:<10} seed={:<4} bpb={:.6}",
                        fmt_name, seed, outcome.final_bpb
                    );
                    bpbs.push(outcome.final_bpb);
                    per_seed.insert(seed, outcome.final_bpb);
                }
                Err(e) => {
                    eprintln!("[sweep][err] fmt={} seed={} err={}", fmt_name, seed, e);
                }
            }
            env::remove_var("TRIOS_FORMAT_TYPE");
        }

        if bpbs.is_empty() {
            eprintln!("[sweep][skip] fmt={} produced no results", fmt_name);
            continue;
        }
        let mut sorted = bpbs.clone();
        let med = median(&mut sorted);
        let (m, s) = mean_std(&bpbs);
        let per_seed_json: Vec<String> = per_seed
            .iter()
            .map(|(k, v)| format!("\"{}\":{:.6}", k, v))
            .collect();
        let line = format!(
            "{{\"format\":\"{}\",\"is_baseline\":{},\"n_seeds\":{},\"bpb_median\":{:.6},\"bpb_mean\":{:.6},\"bpb_std\":{:.6},\"hidden\":{},\"steps\":{},\"lr\":{},\"attn_layers\":{},\"per_seed\":{{{}}}}}",
            fmt_name, is_baseline, bpbs.len(), med, m, s,
            a.hidden, a.steps, a.lr, a.attn_layers,
            per_seed_json.join(",")
        );
        writeln!(out, "{}", line).expect("write line");
        out.flush().ok();
        eprintln!(
            "[sweep] === fmt={:<10} n={} median={:.6} mean={:.6} std={:.6} ===",
            fmt_name,
            bpbs.len(),
            med,
            m,
            s
        );
    }

    eprintln!("[sweep] done → {}", a.out);
}
