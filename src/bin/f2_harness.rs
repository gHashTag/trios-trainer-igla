//! F2 breadth-as-moat harness (CPU proxy).
//!
//! Runs the phi-ladder arm (scale-aware ternary -> GF8 -> GF16 -> GF32) against
//! the zoo arm (BitNet ternary -> INT8 -> FP8 E4M3 fwd / E5M2 bwd -> bf16) over
//! several seeds, then prints a Welch two-sample comparison and the lossy
//! cross-format conversion counts.
//!
//! HONESTY (skill goldenfloat-ladder, FL-002). This is an in-sandbox PROXY at
//! toy scale. Its numeric output is NOT a Verdict. The breadth/toolchain-
//! coherence moat stays [Open conjecture]; only phi^2 + phi^-2 = 3 is [Verified].
//! The accuracy verdict may legitimately come back ZooWins or Tie. The real F2
//! head-to-head (issue gHashTag/t27#1021) is BLOCKED on a real BPB pipeline and
//! compute; this harness only validates mechanics and the conversion-count
//! signature, and ensures the phi arm no longer saturates (scale-aware fix).

use std::process::ExitCode;

use trios_trainer::multi_seed::{run_multi_seed, F2Verdict};

/// Canon #93 allowed seed set, mirrored from `src/seed_canon.rs`
/// (forbidden: {42, 43, 44, 45}; allowed: {47, 89, 123, 144}).
///
/// This harness used to run the contiguous range starting at 43, i.e. three
/// forbidden seeds published straight into its own config header.
/// `seed_canon::parse_seed` could not catch it: it reads only the `SEED`
/// environment variable, and this binary never consults it.
const CANON_SEEDS: &[u64] = &[47, 89, 123, 144];

fn main() -> ExitCode {
    // Default config: the Canon #93 seed set, 40 steps, 8-step full-precision
    // warmup, dim 128.
    let seeds: Vec<u64> = CANON_SEEDS.to_vec();
    let steps = 40usize;
    let warmup = 8usize;
    let dim = 128usize;
    let alpha = 0.05_f64;

    let r = run_multi_seed(&seeds, steps, warmup, dim, alpha);

    println!("==================================================================");
    println!(" F2 BREADTH-AS-MOAT HARNESS  (CPU PROXY -- NOT A VERDICT)");
    println!("==================================================================");
    println!(" status: breadth/toolchain-coherence moat = [Open conjecture] (FL-002)");
    println!("         only phi^2 + phi^-2 = 3 is [Verified]; this proxy cannot");
    println!("         promote the moat. Accuracy verdict may be Tie/ZooWins.");
    println!("------------------------------------------------------------------");
    println!(
        " config: seeds={:?} (Canon #93 allowed set) steps={steps} \
         warmup_unquantized={warmup} dim={dim} alpha={alpha}",
        seeds
    );
    println!("------------------------------------------------------------------");
    println!(" ACCURACY AXIS (proxy_bits, lower is better -- NOT real BPB)");
    let phi_mean = r.phi.bits.iter().sum::<f64>() / r.phi.bits.len() as f64;
    let zoo_mean = r.zoo.bits.iter().sum::<f64>() / r.zoo.bits.len() as f64;
    println!("   phi-ladder mean proxy_bits : {phi_mean:.6}");
    println!("   zoo        mean proxy_bits : {zoo_mean:.6}");
    println!(
        "   mean_diff (phi - zoo)      : {:.6}  (negative favours phi)",
        r.mean_diff
    );
    // The Welch statistics exist only when the samples carry resolvable spread.
    // When they do not there is no t, no df and no p to print: printing a
    // placeholder number under these headings is the failure this harness is
    // supposed to detect, not a formatting convenience.
    match &r.welch {
        Ok(w) => {
            println!("   Welch t                    : {:.6}", w.t_stat);
            println!("   Welch-Satterthwaite df     : {:.4}", w.df);
            println!("   two-sided p                : {:.3e}", w.p_two_sided);
        }
        Err(e) => {
            println!("   Welch t                    : undefined");
            println!("   Welch-Satterthwaite df     : undefined");
            println!("   two-sided p                : undefined ({e})");
        }
    }
    println!("------------------------------------------------------------------");
    println!(" BREADTH AXIS (lossy cross-format conversions -- THE actual moat)");
    println!("   phi-ladder lossy conversions : {}", r.phi.total_lossy);
    println!("   phi-ladder coherent widenings: {}", r.phi.total_coherent);
    println!("   zoo        lossy conversions : {}", r.zoo.total_lossy);
    println!("   zoo        coherent widenings: {}", r.zoo.total_coherent);
    println!("------------------------------------------------------------------");

    // A verdict without a p-value is not a weaker verdict, it is not a verdict.
    let verdict_ok = match r.verdict {
        Some(F2Verdict::PhiWins) => {
            println!(
                " ACCURACY VERDICT: PHI WINS (accuracy proxy) -- NOT a Verdict; \
                 moat stays [Open conjecture]"
            );
            true
        }
        Some(F2Verdict::Tie) => {
            println!(" ACCURACY VERDICT: TIE (accuracy proxy) -- moat stays [Open conjecture]");
            true
        }
        Some(F2Verdict::ZooWins) => {
            println!(
                " ACCURACY VERDICT: ZOO WINS (accuracy proxy) -- demote breadth-as-moat \
                 to [Risk] FIRST"
            );
            true
        }
        None => {
            let reason = match &r.welch {
                Err(e) => e.to_string(),
                // Unreachable by construction: `run_multi_seed` derives the
                // verdict from `welch`, so `None` implies `Err`. Printed rather
                // than panicked so the breadth axis above still stands.
                Ok(_) => "verdict absent despite computable Welch statistics".to_string(),
            };
            println!(" VERDICT: none ({reason})");
            false
        }
    };

    // The breadth claim's mechanical signature, reported independently of the
    // accuracy verdict. This is the only thing this proxy can show cleanly.
    if r.phi.total_lossy < r.zoo.total_lossy {
        println!(
            " BREADTH SIGNATURE: phi ladder used {} fewer lossy cross-format",
            r.zoo.total_lossy - r.phi.total_lossy
        );
        println!("   conversions than the zoo at matched stage count. This is");
        println!("   CONSISTENT WITH the [Open conjecture] moat -- it does NOT prove it.");
    } else {
        println!(" BREADTH SIGNATURE: phi ladder did NOT reduce lossy conversions.");
        println!("   This WEAKENS the breadth claim; record in FL-002.");
    }
    println!("==================================================================");

    if verdict_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
