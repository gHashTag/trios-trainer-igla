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
use trios_trainer::seed_canon;

/// The seed set this harness proposes to run: the Canon #93 allowed set
/// (forbidden: {42, 43, 44, 45}; allowed: {47, 89, 123, 144}).
///
/// This harness used to run the contiguous range `(43..51)`, i.e. three
/// forbidden seeds published straight into its own printed config header.
/// `seed_canon::parse_seed` could not catch that, because it reads only the
/// `SEED` environment variable and this binary never consulted it.
///
/// A literal is a PROPOSAL here, not a permission: it is submitted to
/// [`canon_checked_seeds`] before anything is printed or run.
const PROPOSED_SEEDS: &[u64] = &[47, 89, 123, 144];

/// Submit each proposed seed to the canon's own parser and return the set only
/// if every one of them is allowed.
///
/// `seed_canon::parse_seed` takes no argument and reads only `SEED`, so the
/// only way to put a seed in front of the canon's own check - rather than in
/// front of a second copy of its forbidden set, which is how two lists drift
/// apart - is to present each candidate in that variable. This runs in `main`
/// before any thread is spawned, and the caller's `SEED` is restored before
/// returning.
///
/// A caller's `SEED` is deliberately NOT adopted as configuration: one seed
/// cannot make a two-sample comparison, and silently running a one-seed
/// "multi-seed" harness is how an undefined p-value gets manufactured.
fn canon_checked_seeds(proposed: &[u64]) -> Result<Vec<u64>, String> {
    let previous = std::env::var_os("SEED");
    let mut accepted = Vec::with_capacity(proposed.len());
    let mut refusal = None;
    for &candidate in proposed {
        std::env::set_var("SEED", candidate.to_string());
        match seed_canon::parse_seed() {
            Ok(seed) => accepted.push(seed),
            Err(e) => {
                refusal = Some(e);
                break;
            }
        }
    }
    match previous {
        Some(v) => std::env::set_var("SEED", v),
        None => std::env::remove_var("SEED"),
    }
    match refusal {
        Some(e) => Err(e),
        None => Ok(accepted),
    }
}

fn main() -> ExitCode {
    // Default config: the Canon #93 seed set, 40 steps, 8-step full-precision
    // warmup, dim 128. Nothing below prints until the seeds are canon-checked,
    // so a forbidden seed cannot reach the config header even for one line.
    let seeds: Vec<u64> = match canon_checked_seeds(PROPOSED_SEEDS) {
        Ok(seeds) => seeds,
        Err(e) => {
            eprintln!("[canon-93] refusing to run f2_harness: {e}");
            eprintln!(
                "           the proposed seed set is {PROPOSED_SEEDS:?}; fix \
                 PROPOSED_SEEDS in src/bin/f2_harness.rs"
            );
            return ExitCode::FAILURE;
        }
    };
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
        " config: seeds={:?} (each checked by seed_canon::parse_seed) steps={steps} \
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// `canon_checked_seeds` mutates the process-wide `SEED`, so the tests that
    /// call it cannot run concurrently with each other.
    static SEED_ENV_LOCK: Mutex<()> = Mutex::new(());

    /// The set this binary ships must survive the canon's own parser. If a
    /// future edit reintroduces `(43..51)` or any other forbidden seed, this
    /// fails in CI instead of appearing in a published config header.
    #[test]
    fn proposed_seeds_pass_the_canon() {
        let _g = SEED_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let seeds = canon_checked_seeds(PROPOSED_SEEDS)
            .unwrap_or_else(|e| panic!("shipped seed set must be canon-clean: {e}"));
        assert_eq!(seeds, PROPOSED_SEEDS.to_vec());
    }

    /// The check is the canon's, not a local copy of it: a forbidden seed is
    /// refused with the canon's own wording.
    #[test]
    fn a_forbidden_seed_is_refused_by_the_canon_not_by_a_local_copy() {
        let _g = SEED_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for forbidden in [42u64, 43, 44, 45] {
            let err = canon_checked_seeds(&[47, forbidden])
                .expect_err("a forbidden seed must not reach the config header");
            assert!(
                err.contains("forbidden") && err.contains(&forbidden.to_string()),
                "seed {forbidden}: the refusal must name the canon and the seed: {err}"
            );
        }
    }

    /// The caller's environment is left as it was found, whether the set is
    /// accepted or refused.
    #[test]
    fn the_callers_seed_env_is_restored() {
        let _g = SEED_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("SEED", "89");
        let _ = canon_checked_seeds(PROPOSED_SEEDS);
        assert_eq!(std::env::var("SEED").ok().as_deref(), Some("89"));
        let _ = canon_checked_seeds(&[43]);
        assert_eq!(std::env::var("SEED").ok().as_deref(), Some("89"));

        std::env::remove_var("SEED");
        let _ = canon_checked_seeds(PROPOSED_SEEDS);
        assert!(
            std::env::var_os("SEED").is_none(),
            "an unset SEED must stay unset"
        );
    }
}
