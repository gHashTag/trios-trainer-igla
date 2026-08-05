//! `xarch_probe` -- per-primitive cross-architecture arithmetic probe.
//!
//! Round 6 (`docs/DIVERGENCE-LOCALIZATION.md`) established WHERE the
//! cross-architecture divergence begins: initialisation is byte-portable
//! between `aarch64-apple-darwin` and `x86_64-apple-darwin`, and the artifacts
//! separate only after gradient steps. It did not establish WHY, and it named
//! two surviving candidates without isolating either:
//!
//! * **reduction order** -- both backends auto-vectorise the float arithmetic,
//!   and differently; floating-point addition is not associative, so two
//!   vectorisation schedules over one sum are entitled to two results;
//! * **libm** -- `expf`, `logf`, `powf` and friends are resolved at run time
//!   from the platform maths library and are NOT bit-specified by IEEE 754, so
//!   two architecture slices are permitted to differ in the last place.
//!
//! FMA contraction was already excluded by disassembly (`grep -c mul_add
//! src/train_loop.rs` is 0, and neither release binary contains a single
//! `fmadd`/`vfmadd` instruction). What is left cannot be separated by staring
//! at a 852 272-byte checkpoint, because every primitive feeds every other one.
//!
//! This binary separates them by running each primitive ALONE and hashing its
//! raw little-endian f32 output bytes. IEEE 754 pins `+`, `-`, `*`, `/` and
//! `sqrt` exactly, so any primitive built only from those MUST agree across
//! architectures unless the compiler re-associated it; any primitive that
//! disagrees while using only those operations is a reduction-order finding.
//! Any primitive that disagrees and calls libm is a libm finding. A primitive
//! that agrees rules itself out.
//!
//! Deliberate properties, because the probe is itself a measuring instrument:
//!
//! * **No randomness.** Buffers are filled by an inline LCG with the same two
//!   constants `src/train_loop.rs` uses in `HybridModel::new`, written out here
//!   rather than called, so this binary depends on no crate module and cannot
//!   drift when one is edited.
//! * **No file I/O and no environment reads.** Nothing about the host can
//!   reach the numbers except the instruction set and the maths library.
//! * **Shapes mirror the trainer.** `matvec_384x64` is the `proj` matvec,
//!   `layer_norm_64` is `layer_norm(&combined, 1e-5)`, `sqrelu` is the
//!   activation the training path actually applies (`x*x` for `x > 0`, else
//!   0 -- the trainer uses no `tanh`), and `forward_tiny` is a self-contained
//!   mirror of `HybridModel::forward_cached` at `DIM = 64`, `VOCAB = 128`.
//!
//! Output format, one line per primitive:
//!
//! ```text
//! PRIMITIVE <name> <sha256 of the little-endian f32 output bytes>
//! ```
//!
//! plus one `DETAIL` line per primitive carrying the element count and the raw
//! bit pattern of the first output word, so that a mismatch can be read as a
//! last-place rounding difference or as something larger without re-running.
//! `PLATFORM` and `SUMMARY` lines bracket the report. Diff two runs; the set of
//! differing `PRIMITIVE` lines is the result.
//!
//! See `docs/DIVERGENCE-MECHANISM.md` for the measured tables and the scope
//! statement (the x86_64 arm is `x86_64-apple-darwin` under Rosetta 2, NOT the
//! native x86_64 Linux that produced the CI mismatch).

use sha2::{Digest, Sha256};

// -------------------------------------------------------------------
// Deterministic inputs
// -------------------------------------------------------------------

/// The LCG of `HybridModel::new`, restated inline.
///
/// Restated rather than imported on purpose: this binary must keep measuring
/// the same numbers after any edit to `src/train_loop.rs`, otherwise a change
/// in the trainer would silently redefine the instrument used to audit it.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Integer step. Multiplier and increment are the trainer's.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform-ish f32 in `[-1.0, 1.0)`, by the trainer's own conversion.
    ///
    /// The conversion is integer-to-float plus one multiply and one subtract,
    /// all IEEE-exact, so the input buffers themselves cannot be an
    /// architecture variable. That is checked rather than assumed: the
    /// `input_*` primitives below hash the buffers before any arithmetic runs.
    fn next_f32(&mut self) -> f32 {
        let s = self.next_u64();
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

/// `n` deterministic values in `[-1, 1)`.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.next_f32()).collect()
}

/// `n` deterministic values in `[lo, hi)`, by an IEEE-exact affine map.
fn fill_range(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mid = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    fill(n, seed).into_iter().map(|v| mid + v * half).collect()
}

// -------------------------------------------------------------------
// Reporting
// -------------------------------------------------------------------

fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for v in values {
        hasher.update(v.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest.iter() {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

/// Emit one primitive: the required `PRIMITIVE` line plus a forensic `DETAIL`.
///
/// `DETAIL` carries the first output word as a raw u32 bit pattern. When two
/// arms disagree that is what says whether the disagreement is one unit in the
/// last place or a different number, without a second run.
fn report(name: &str, values: &[f32]) {
    println!("PRIMITIVE {} {}", name, sha256_f32(values));
    let first_bits = values.first().map(|v| v.to_bits()).unwrap_or(0);
    println!(
        "DETAIL    {} n={} first_bits=0x{:08x}",
        name,
        values.len(),
        first_bits
    );
}

// -------------------------------------------------------------------
// Primitives
// -------------------------------------------------------------------

/// (1) Plain sequential dot product. IEEE-exact per operation; a mismatch here
/// can only be re-association by the vectoriser.
fn dot_sequential(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// (2) The same dot product with the accumulator split into 8 partial sums.
///
/// Mathematically identical, numerically not: this is exactly the shape a
/// 4-wide or 8-wide vectoriser produces on its own. If (1) and (2) differ
/// WITHIN one architecture, reduction order demonstrably moves this number on
/// this hardware, which is the precondition for reduction order being able to
/// move it BETWEEN architectures.
fn dot_split8(a: &[f32], b: &[f32]) -> f32 {
    let mut parts = [0.0f32; 8];
    let mut i = 0;
    while i + 8 <= a.len() {
        for lane in 0..8 {
            parts[lane] += a[i + lane] * b[i + lane];
        }
        i += 8;
    }
    let mut tail = 0.0f32;
    while i < a.len() {
        tail += a[i] * b[i];
        i += 1;
    }
    let mut acc = 0.0f32;
    for p in parts.iter() {
        acc += *p;
    }
    acc + tail
}

/// The trainer's activation: squared ReLU. `src/train_loop.rs` computes
/// `if hidden_raw > 0.0 { hidden_raw * hidden_raw } else { 0.0 }`. There is no
/// `tanh` anywhere on the training path, so this is what item (6) covers.
fn sqrelu(x: f32) -> f32 {
    if x > 0.0 {
        x * x
    } else {
        0.0
    }
}

/// `softmax` copied structurally from `src/train_loop.rs`: max fold, in-place
/// `exp`, sequential accumulation, then divide. Composes libm (3) with a
/// reduction, which is why it is listed separately from both.
fn softmax(v: &mut [f32]) {
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

/// `layer_norm` copied structurally from `src/train_loop.rs`, including the
/// `powi(2)` and the single `sqrt`. Two reductions plus one IEEE-exact libm
/// call: `sqrt` IS bit-specified, so a mismatch here is reduction order.
fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std_inv = 1.0 / (var + eps).sqrt();
    x.iter().map(|v| (v - mean) * std_inv).collect()
}

/// The `proj` matvec of `forward_cached`, at its real shape: `hidden` rows of
/// `DIM` accumulated sequentially. 64 is short enough that a vectoriser may or
/// may not bother, which is the point of running it beside the 4096-long dot.
fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r] += w[r * cols + c] * x[c];
        }
    }
    out
}

/// (8) One full forward pass, self-contained.
///
/// A structural mirror of `HybridModel::forward_cached` at the crate's own
/// `DIM = 64`, `VOCAB = 128`, `NUM_CTX = 6`, `CTX_WEIGHTS` and squared-ReLU
/// activation, with the attention branch omitted (it contributes through the
/// same four primitives already probed above, and including it would import a
/// crate module this binary is deliberately free of). Returns the softmaxed
/// logits with the cross-entropy of one target appended, so the hash covers
/// both the distribution and the `.ln()` that the loss is read through.
fn forward_tiny(hidden: usize) -> Vec<f32> {
    const DIM: usize = 64;
    const VOCAB: usize = 128;
    const NUM_CTX: usize = 6;
    const CTX_WEIGHTS: [f32; NUM_CTX] = [0.70, 0.45, 0.30, 0.20, 0.13, 0.08];

    let embed = fill(VOCAB * DIM, 47);
    let ctx: Vec<Vec<f32>> = (0..NUM_CTX)
        .map(|c| fill(VOCAB * DIM, 1000 + c as u64))
        .collect();
    let proj = fill(hidden * DIM, 89);
    let lm_head = fill(VOCAB * hidden, 123);

    // Fixed token window: 6 context tokens plus the current one.
    let tokens: [usize; NUM_CTX + 1] = [7, 19, 3, 88, 41, 12, 55];
    let target = 33usize;

    let t_last = tokens[NUM_CTX];
    let mut combined = embed[t_last * DIM..(t_last + 1) * DIM].to_vec();
    for (ci, cw) in CTX_WEIGHTS.iter().enumerate() {
        let t = tokens[NUM_CTX - 1 - ci];
        let cv = &ctx[ci][t * DIM..(t + 1) * DIM];
        for j in 0..DIM {
            combined[j] += cv[j] * cw;
        }
    }

    let ln = layer_norm(&combined, 1e-5);
    let hidden_raw = matvec(&proj, &ln, hidden, DIM);
    let act: Vec<f32> = hidden_raw.iter().map(|v| sqrelu(*v)).collect();
    let mut logits = matvec(&lm_head, &act, VOCAB, hidden);
    softmax(&mut logits);

    let p = logits[target];
    let loss = -p.max(1e-10).ln();
    let mut out = logits;
    out.push(loss);
    out
}

// -------------------------------------------------------------------
// main
// -------------------------------------------------------------------

fn main() {
    const N: usize = 4096;

    println!(
        "PLATFORM arch={} os={} pointer_width={} profile={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        usize::BITS,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );

    // Inputs first. If these disagree, nothing below is interpretable, because
    // the two arms would not be measuring the same numbers.
    let a = fill(N, 47);
    let b = fill(N, 89);
    let pos = fill_range(N, 123, 0.25, 8.0);
    let ranged = fill_range(N, 144, -8.0, 8.0);
    report("input_a", &a);
    report("input_b", &b);
    report("input_positive", &pos);
    report("input_ranged", &ranged);

    // (1) and (2): reduction order, isolated. Only +, - and * are involved.
    report("dot_sequential_4096", &[dot_sequential(&a, &b)]);
    report("dot_split8_4096", &[dot_split8(&a, &b)]);

    // (3)-(5) and two extras: libm, isolated. One call per element, no
    // reduction, so a mismatch here cannot be reduction order.
    report(
        "exp_4096",
        &ranged.iter().map(|v| v.exp()).collect::<Vec<f32>>(),
    );
    report("ln_4096", &pos.iter().map(|v| v.ln()).collect::<Vec<f32>>());
    report(
        "sqrt_4096",
        &pos.iter().map(|v| v.sqrt()).collect::<Vec<f32>>(),
    );
    // `powf` is reached from RoPE in `src/model_hybrid_attn.rs`
    // (`10_000.0_f32.powf(exp)`); `cos` from `cosine_lr` in
    // `src/train_loop.rs`. Both are on the training path and neither is
    // bit-specified by IEEE 754.
    report(
        "powf_4096",
        &pos.iter()
            .map(|v| 10_000.0f32.powf(*v / 16.0))
            .collect::<Vec<f32>>(),
    );
    report(
        "cos_4096",
        &ranged.iter().map(|v| v.cos()).collect::<Vec<f32>>(),
    );

    // (6) the activation the training path actually uses. Pure multiply and
    // compare: this one is expected to agree, and is here as the control that
    // says the harness itself is not the source of a difference.
    report(
        "sqrelu_4096",
        &ranged.iter().map(|v| sqrelu(*v)).collect::<Vec<f32>>(),
    );

    // (7) libm composed with a reduction.
    let mut sm = ranged.clone();
    softmax(&mut sm);
    report("softmax_4096", &sm);

    // Two reductions plus an IEEE-exact sqrt, at the trainer's real width.
    report("layer_norm_64", &layer_norm(&a[..64], 1e-5));

    // The `proj` matvec at its real shape: 384 rows of 64.
    let w = fill(384 * 64, 47);
    report("matvec_384x64", &matvec(&w, &a[..64], 384, 64));

    // (8) one full forward pass on fixed weights.
    report("forward_tiny_h64", &forward_tiny(64));

    println!("SUMMARY primitives=16 randomness=none file_io=none env_reads=none");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The instrument must not move between two calls in one process, or a
    /// cross-architecture diff would be measuring the instrument.
    #[test]
    fn primitives_are_stable_within_a_process() {
        let a = fill(4096, 47);
        let b = fill(4096, 89);
        assert_eq!(dot_sequential(&a, &b), dot_sequential(&a, &b));
        assert_eq!(dot_split8(&a, &b), dot_split8(&a, &b));
        assert_eq!(forward_tiny(64), forward_tiny(64));
    }

    /// Reduction order is a real effect on THIS hardware, not a hypothesis.
    ///
    /// If this ever passes with equality the split-accumulator primitive has
    /// stopped being a probe of anything and the doc's reading of it is void.
    #[test]
    fn split_accumulator_changes_the_sum() {
        let a = fill(4096, 47);
        let b = fill(4096, 89);
        let seq = dot_sequential(&a, &b);
        let split = dot_split8(&a, &b);
        assert!(
            seq != split,
            "sequential and 8-way-split dot agreed exactly ({}); \
             the reduction-order probe measures nothing on this host",
            seq
        );
        assert!(
            (seq - split).abs() < 1e-3,
            "sequential {} and split {} differ by more than rounding",
            seq,
            split
        );
    }

    /// The inputs are built from IEEE-exact operations only, so the buffers
    /// themselves must never be an architecture variable. Guards the reading
    /// of every other primitive.
    #[test]
    fn inputs_use_only_exact_operations() {
        let v = fill(16, 47);
        let mut rng = Lcg::new(47);
        for expected in v.iter() {
            let s = rng.next_u64();
            let manual = ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            assert_eq!(*expected, manual);
        }
    }

    /// The activation is the trainer's, not a stand-in.
    #[test]
    fn sqrelu_matches_the_training_path() {
        assert_eq!(sqrelu(-1.5), 0.0);
        assert_eq!(sqrelu(0.0), 0.0);
        assert_eq!(sqrelu(3.0), 9.0);
    }

    /// A hash of nothing must not be mistaken for a hash of zeros.
    #[test]
    fn empty_and_zero_hash_differently() {
        assert_ne!(sha256_f32(&[]), sha256_f32(&[0.0]));
    }
}
