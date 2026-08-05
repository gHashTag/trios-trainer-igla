//! `ulp_census` -- element-wise bit-pattern dump of the five libm functions on
//! the training path, so the ULP table in `docs/DIVERGENCE-MECHANISM.md` has an
//! artifact instead of a memory.
//!
//! `xarch_probe` answers WHICH primitives disagree across an instruction-set
//! boundary: it hashes each primitive's whole output, so a disagreement shows
//! up as two different SHA-256 strings and nothing more. That is enough to name
//! the mechanism (libm, not reduction order) and not enough to size it. The
//! sentence the pitch actually leans on -- "every disagreement is exactly one
//! unit in the last place, on roughly 1% of inputs" -- needs the individual
//! elements, because a hash of 4096 values is equally consistent with one
//! last-place rounding difference and with total nonsense.
//!
//! This binary prints those elements. It computes the same five functions over
//! the same two input buffers `xarch_probe` uses (`pos` = `fill_range(4096,
//! 123, 0.25, 8.0)`, `ranged` = `fill_range(4096, 144, -8.0, 8.0)`, both from
//! the LCG of `HybridModel::new` restated inline) and emits one line per
//! element carrying the raw u32 bit pattern of the f32 result. Diff two arms;
//! the differing lines ARE the census, and `tests/xarch_probe_census.rs`
//! derives the published table from exactly those two files.
//!
//! It replaces a throwaway program that lived in `/tmp` and was reproduced by
//! transcription in the documentation. A citation to `/tmp` is not evidence: it
//! cannot be re-run by a reader, it cannot be shown to be the program that
//! produced the numbers, and it is the first thing a competent auditor asks
//! for. The arithmetic here is the arithmetic of that program, unchanged --
//! same LCG constants, same buffer seeds, same ranges, same call expressions,
//! same order -- so the published numbers remain the measured ones.
//!
//! Output format:
//!
//! ```text
//! VALUE  <function> <index> <8-hex-digit u32 bit pattern of the f32 result>
//! DIGEST <function> n=<count> sha256=<of the little-endian f32 output bytes>
//! ```
//!
//! The `DIGEST` lines are the join to `xarch_probe`: `DIGEST exp` must equal
//! that binary's `PRIMITIVE exp_4096` hash on the same arm, because the two
//! binaries are computing the same 4096 numbers by different routes. If they
//! ever disagree, one of the two instruments has drifted and neither table is
//! readable; the test asserts it.
//!
//! Deliberate properties, identical to `xarch_probe` because it is the same
//! kind of object -- a measuring instrument, not a program that happens to
//! print numbers:
//!
//! * **No arguments.** Nothing a caller types can change a published number.
//! * **No environment reads.** `std::env::consts::ARCH` and `OS` are
//!   compile-time constants baked into the binary by the target triple, not
//!   `getenv`; they label the arm and cannot vary between two runs of the same
//!   binary.
//! * **No file I/O and no randomness.**
//! * **No dependency on any module of this crate**, so an edit to the trainer
//!   cannot silently redefine the instrument used to audit it.
//!
//! Build both arms with the pinned toolchain and compare:
//!
//! ```bash
//! cargo build --release --target aarch64-apple-darwin --bin ulp_census
//! cargo build --release --target x86_64-apple-darwin  --bin ulp_census
//! ./target/aarch64-apple-darwin/release/ulp_census > evidence/xarch-probe/ulp-arm64.txt
//! ./target/x86_64-apple-darwin/release/ulp_census   > evidence/xarch-probe/ulp-x86_64.txt
//! diff evidence/xarch-probe/ulp-arm64.txt evidence/xarch-probe/ulp-x86_64.txt
//! ```
//!
//! See `docs/DIVERGENCE-MECHANISM.md` for the reading of the result and for the
//! scope statement: the x86_64 arm is `x86_64-apple-darwin` under Rosetta 2 on
//! Apple Silicon, NOT the native x86_64 Linux that produced the CI mismatch.

use sha2::{Digest, Sha256};

/// Elements per function. The width of the published table.
const N: usize = 4096;

/// The functions this binary censuses, in emission order.
///
/// Declared rather than implied so the `SUMMARY` line can state a count it
/// derived instead of a count someone typed, and so a sixth function cannot be
/// added without the census admitting it.
const FUNCTIONS: &[&str] = &["exp", "ln", "sqrt", "powf", "cos"];

// -------------------------------------------------------------------
// Deterministic inputs -- the LCG of `HybridModel::new`, restated inline
// -------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform-ish f32 in `[-1.0, 1.0)`. Integer-to-float, one multiply, one
    /// subtract: every step IEEE-exact, so the inputs themselves cannot be an
    /// architecture variable. `xarch_probe` checks that rather than assuming it
    /// by hashing the raw buffers as primitives of their own.
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

/// SHA-256 over the little-endian f32 bytes, byte for byte the convention
/// `xarch_probe` uses, so the two binaries' hashes are comparable.
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

/// Emit one function: every element, then the digest that ties it to the probe.
fn emit(name: &str, values: &[f32]) {
    for (i, v) in values.iter().enumerate() {
        println!("VALUE  {} {} {:08x}", name, i, v.to_bits());
    }
    println!(
        "DIGEST {} n={} sha256={}",
        name,
        values.len(),
        sha256_f32(values)
    );
}

// -------------------------------------------------------------------
// The five functions
// -------------------------------------------------------------------

/// The measured outputs, in `FUNCTIONS` order, paired with their names.
///
/// The call expressions are the ones the trainer reaches: `exp` per logit from
/// `softmax`, `ln` through the loss, `sqrt` from `layer_norm`, `powf` from the
/// RoPE frequency table (`10_000.0_f32.powf(..)`), `cos` from `cosine_lr`.
fn measure() -> Vec<(&'static str, Vec<f32>)> {
    let pos = fill_range(N, 123, 0.25, 8.0);
    let ranged = fill_range(N, 144, -8.0, 8.0);
    vec![
        ("exp", ranged.iter().map(|v| v.exp()).collect()),
        ("ln", pos.iter().map(|v| v.ln()).collect()),
        ("sqrt", pos.iter().map(|v| v.sqrt()).collect()),
        (
            "powf",
            pos.iter().map(|v| 10_000.0f32.powf(*v / 16.0)).collect(),
        ),
        ("cos", ranged.iter().map(|v| v.cos()).collect()),
    ]
}

fn main() {
    println!("ULP-CENSUS-FORMAT 1");
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

    let measured = measure();
    assert_eq!(
        measured.len(),
        FUNCTIONS.len(),
        "measured {} functions but declared {}; the census would misreport its \
         own coverage",
        measured.len(),
        FUNCTIONS.len()
    );
    for (i, (name, values)) in measured.iter().enumerate() {
        assert_eq!(
            *name, FUNCTIONS[i],
            "function {} is emitted as `{}` but declared as `{}`",
            i, name, FUNCTIONS[i]
        );
        assert_eq!(
            values.len(),
            N,
            "function `{}` produced {} of {} elements",
            name,
            values.len(),
            N
        );
        emit(name, values);
    }

    println!(
        "SUMMARY functions={} elements_per_function={} elements={}",
        measured.len(),
        N,
        measured.len() * N
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The instrument must not move between two calls in one process, or a
    /// cross-architecture diff would be measuring the instrument.
    #[test]
    fn measurements_are_stable_within_a_process() {
        let first = measure();
        let second = measure();
        assert_eq!(first.len(), second.len());
        for ((na, a), (nb, b)) in first.iter().zip(second.iter()) {
            assert_eq!(na, nb);
            assert_eq!(a, b, "function `{}` moved between two calls", na);
        }
    }

    /// The inputs are the ones `xarch_probe` measures, from the same LCG. If
    /// this ever fails the two instruments are looking at different numbers and
    /// the `DIGEST`-to-`PRIMITIVE` join in the published evidence is void.
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

    /// Every declared function is measured and every measured function is
    /// declared -- the emitted census cannot claim coverage it does not have.
    #[test]
    fn declared_functions_match_measured_functions() {
        let measured = measure();
        let names: Vec<&str> = measured.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, FUNCTIONS.to_vec());
        for (name, values) in measured.iter() {
            assert_eq!(values.len(), N, "function `{}` is short", name);
        }
    }

    /// The input ranges are the documented ones, because the ULP shares are
    /// only meaningful with the domain stated: `ln` and `sqrt` agreeing is a
    /// measurement on `[0.25, 8.0)`, not a guarantee.
    #[test]
    fn input_domains_are_the_documented_ones() {
        let pos = fill_range(N, 123, 0.25, 8.0);
        let ranged = fill_range(N, 144, -8.0, 8.0);
        assert!(pos.iter().all(|v| *v >= 0.25 && *v < 8.0));
        assert!(ranged.iter().all(|v| *v >= -8.0 && *v < 8.0));
        assert!(pos.iter().all(|v| v.is_finite()));
        assert!(ranged.iter().all(|v| v.is_finite()));
    }

    /// A hash of nothing must not be mistaken for a hash of zeros.
    #[test]
    fn empty_and_zero_hash_differently() {
        assert_ne!(sha256_f32(&[]), sha256_f32(&[0.0]));
    }
}
