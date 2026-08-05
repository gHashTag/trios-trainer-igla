//! A clamped probability is a fabricated measurement, and a ledger writer with
//! no corpus precondition is a leak waiting for a DSN.
//!
//! Two defects, both previously live in files that publish BPB:
//!
//! * **The underflow door.** Round 1 closed the NaN door: `f32::max` returns
//!   the OTHER operand when one side is NaN, so `p.max(1e-10)` turned a
//!   poisoned forward pass into a finite reading. The door beside it stayed
//!   open. An f32 softmax returns a finite, exact `0.0` for a target the model
//!   finds impossible - not NaN, not infinite, so `is_nan()` and `is_finite()`
//!   both accept it - and the same clamp then produced exactly `-ln(1e-10)` =
//!   23.02585 nats = 33.21928 bpb. That is the number this crate documents as
//!   the fake-measurement signature, and `final_val_bpb` is a SEALED field, so
//!   it shipped inside an authenticated declaration.
//! * **The unguarded ledger writers.** `hybrid_train` and `tjepa_train` call
//!   `neon_writer::bpb_sample` and carried no train/val precondition at all. A
//!   run with `--train-path` equal to `--val-path` completed and reported
//!   `bpb=3.9591`; only an unset DSN kept the row off the ledger.
//!   `reject_bpb`'s floor of 2.0 cannot catch this, because a 100% verbatim
//!   overlap lands near 2.48 on this architecture - above the floor and BELOW
//!   `BPB_CHAMPION` = 2.5193, i.e. it reads as a new champion.
//!
//! The behavioural half of the underflow test lives where the eval path is
//! reachable, because `loss_on_seq` is private to each trainer and
//! `src/bin/*.rs` compile as separate crates:
//! `train_loop::measurement_truth_tests::an_underflowed_target_probability_is_an_absence_not_33_bpb`
//! and the same-named test in `src/bin/hybrid_train.rs` build a model whose
//! softmax underflows to an exact `0.0` and assert the reading is `None`.
//! What CANNOT be reached that way is the other nine files, so this census
//! stands in for them: it fails if a clamp is reintroduced in any measurement
//! path in any of the eleven.

use std::path::{Path, PathBuf};

/// Every file that computes a loss or a BPB in this crate. A new trainer that
/// forgets its guard has to be added here, and the census will then say so.
const MEASUREMENT_PATHS: &[&str] = &[
    "src/train_loop.rs",
    "src/bin/hybrid_train.rs",
    "src/bin/tjepa_train.rs",
    "src/bin/arch_explorer.rs",
    "src/bin/ngram_train.rs",
    "src/bin/ngram_train_gf16.rs",
    "src/bin/concat_train.rs",
    "src/bin/lstm_train.rs",
    "src/bin/trinity_pr1722.rs",
    "src/bin/cpu_train.rs",
    "src/bin/train_v2.rs",
];

/// Clamps that feed a logarithm and are NOT measurements, listed line by line
/// so the exemption is auditable rather than a hole in a pattern.
///
/// All three are `hybrid_train`'s NCA entropy regulariser and its gradient.
/// `-sum p*ln p` needs the `0*ln 0 = 0` limit and the floor supplies it: at
/// `p == 0` the term is `0 * ln(1e-10) == 0`, the mathematically correct value.
/// Nothing here is ever reported as a number - it shapes a gradient. The
/// identical expression in `loss_on_seq` was fabricating 33.21928 bpb, which is
/// why the two are told apart by the name of the floor.
///
/// The census also fails if an entry here goes MISSING, so a stale exemption
/// cannot quietly widen.
const NON_MEASUREMENT_CLAMPS: &[(&str, &str)] = &[
    (
        "src/bin/hybrid_train.rs",
        ".map(|&p: &f32| p.max(NCA_ENTROPY_PROB_FLOOR).ln() * p)",
    ),
    (
        "src/bin/hybrid_train.rs",
        "let d_ent = probs[vi] * (probs[vi].max(NCA_ENTROPY_PROB_FLOOR).ln() + entropy);",
    ),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// Line numbers of `#[cfg(test)]` blocks, so the census can tell a regression
/// test that DEMONSTRATES the laundering from a production path that DOES it.
///
/// Brace-counted from the `#[cfg(test)]` attribute. Crude, and sufficient:
/// every test module in these files is a top-level `mod`.
fn test_line_ranges(src: &str) -> Vec<(usize, usize)> {
    let lines: Vec<&str> = src.lines().collect();
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() == "#[cfg(test)]" {
            let mut depth = 0i32;
            let mut seen_open = false;
            let mut j = i;
            while j < lines.len() {
                for c in lines[j].chars() {
                    if c == '{' {
                        depth += 1;
                        seen_open = true;
                    } else if c == '}' {
                        depth -= 1;
                    }
                }
                if seen_open && depth <= 0 {
                    break;
                }
                j += 1;
            }
            ranges.push((i, j.min(lines.len() - 1)));
            i = j + 1;
        } else {
            i += 1;
        }
    }
    ranges
}

fn in_test_block(ranges: &[(usize, usize)], line: usize) -> bool {
    ranges.iter().any(|&(a, b)| line >= a && line <= b)
}

/// The name of the probability a loss line accumulates, e.g. `p` in
/// `total -= p.ln();`. `None` for any other use of `ln`, such as a cosine
/// schedule or the entropy regulariser (whose receiver is a parenthesised
/// expression, not a bare identifier).
fn accumulated_probability(line: &str) -> Option<&str> {
    let t = line.trim();
    let is_accumulation = t.contains("-=") || t.starts_with('-');
    if !is_accumulation || !t.contains(".ln()") {
        return None;
    }
    let idx = t.find(".ln()")?;
    let head = &t[..idx];
    let name: String = head
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        return None;
    }
    // The receiver must be a bare identifier, not the tail of `foo()` or
    // `x.max(FLOOR)`.
    let before = head.len() - name.len();
    if before > 0 && head.as_bytes()[before - 1] == b')' {
        return None;
    }
    let start = head.len() - name.chars().count();
    Some(&head[start..])
}

/// A zero target probability must be an absence, and the 23.02585-nat /
/// 33.21928-bpb reading it used to become must not be producible by any
/// measurement path.
///
/// Two claims, both checkable here:
///
/// 1. No measurement path clamps a probability any more. The clamp is the only
///    way a `0.0` becomes a number, so removing it everywhere IS the refusal.
/// 2. Every loss accumulation is preceded by `!p.is_finite() || p <= 0.0` -
///    i.e. the clamp was replaced by a refusal and not merely deleted, which
///    would have produced `-inf`. ONE spelling, the one the four gradient paths
///    already used: two equivalent spellings of the same guard is how a guard
///    starts to drift.
#[test]
fn no_measurement_path_can_launder_a_zero_probability_into_33_bpb() {
    // Stated as a computation, not a literal: this is the value under audit.
    let laundered_nats = -(1e-10f32).ln();
    let laundered_bpb = laundered_nats / std::f32::consts::LN_2;
    assert!(
        (laundered_nats - 23.02585).abs() < 1e-3,
        "the clamp's reading is 23.02585 nats, got {laundered_nats}"
    );
    assert!(
        (laundered_bpb - 33.21928).abs() < 1e-3,
        "the clamp's reading is 33.21928 bpb, got {laundered_bpb}"
    );

    let root = repo_root();
    let mut clamps: Vec<String> = Vec::new();
    let mut unguarded: Vec<String> = Vec::new();
    let mut exempt_seen: Vec<(&str, &str)> = Vec::new();

    for rel in MEASUREMENT_PATHS {
        let path = root.join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{rel} must be readable: {e}"));
        let lines: Vec<&str> = src.lines().collect();
        let tests = test_line_ranges(&src);

        for (i, line) in lines.iter().enumerate() {
            if in_test_block(&tests, i) {
                continue;
            }

            // 1. No probability clamp feeding a logarithm, under any spelling
            //    of the floor.
            let stripped = line.split("//").next().unwrap_or("");
            if stripped.contains(".max(") && stripped.contains(".ln()") {
                let exempt = NON_MEASUREMENT_CLAMPS
                    .iter()
                    .find(|&&(f, l)| f == *rel && l == line.trim());
                match exempt {
                    Some(&entry) => exempt_seen.push(entry),
                    None => clamps.push(format!("{rel}:{}: {}", i + 1, line.trim())),
                }
            }
            // 2. And no bare `0.0`-producing constant left in a loss line.
            if stripped.contains("23.02585") || stripped.contains("33.21928") {
                clamps.push(format!("{rel}:{}: {}", i + 1, line.trim()));
            }

            // 3. Every loss accumulation is guarded.
            if let Some(var) = accumulated_probability(line) {
                let guard = format!("!{var}.is_finite() || {var} <= 0.0");
                let lo = i.saturating_sub(16);
                let guarded = lines[lo..i].iter().any(|l| l.contains(&guard));
                if !guarded {
                    unguarded.push(format!("{rel}:{}: {}", i + 1, line.trim()));
                }
            }
        }
    }

    assert!(
        clamps.is_empty(),
        "a probability clamp survives in a measurement path; each of these can \
         turn an underflowed 0.0 into 33.21928 bpb:\n  {}",
        clamps.join("\n  ")
    );
    assert!(
        unguarded.is_empty(),
        "a loss accumulation with no `!p.is_finite() || p <= 0.0` refusal \
         within the preceding 16 lines:\n  {}",
        unguarded.join("\n  ")
    );
    for entry in NON_MEASUREMENT_CLAMPS {
        assert!(
            exempt_seen.contains(entry),
            "the exemption for {}:{} is stale - delete it rather than leave a \
             widened hole in this census",
            entry.0,
            entry.1
        );
    }
}

/// Deterministic pseudo-text. Hermetic, and non-periodic enough to satisfy the
/// guard's own 8-gram entropy precondition, so a rejection is evidence about
/// overlap rather than about degeneracy.
fn lcg_tokens(seed: u64, len: usize) -> Vec<usize> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((b'a' + ((s >> 33) % 95) as u8) as usize) % 128
        })
        .collect()
}

/// The shared guard still rejects a fully overlapping split - now that it is
/// `pub` and the four local copies delegate to it.
///
/// This is the case `hybrid_train` and `tjepa_train` had no guard for at all:
/// `--train-path` equal to `--val-path` is a 100% verbatim overlap, and the
/// BPB it produces is BELOW `BPB_CHAMPION`, so it reads as a new record rather
/// than as a leak.
#[test]
fn assert_train_val_disjoint_still_rejects_a_fully_overlapping_pair() {
    use trios_trainer::train_loop::{
        assert_train_val_disjoint, check_train_val_disjoint, eval_chunk_count,
    };

    let train = lcg_tokens(0xA1CE, 20_000);
    let val = train.clone(); // `--val-path` == `--train-path`

    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // the panic IS the expected result
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_train_val_disjoint(&train, &val)
    }));
    std::panic::set_hook(prev);

    let msg = match out {
        Ok(()) => panic!("a val stream identical to train must be rejected"),
        Err(e) => e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "<non-string panic>".to_string()),
    };
    assert!(msg.contains("TRAIN/VAL OVERLAP DETECTED"), "{msg}");
    assert!(msg.contains("100.00% of val windows"), "{msg}");

    // The `Result` form the binaries call carries the same finding, so a
    // caller that exits instead of unwinding is not getting a weaker check.
    let err = check_train_val_disjoint(&train, &val, eval_chunk_count(val.len(), 40))
        .expect_err("the Result form must reject the same split");
    assert!(err.contains("TRAIN/VAL OVERLAP DETECTED"), "{err}");

    // Positive control: a disjoint stream of the same shape is ACCEPTED, so
    // the rejection above is evidence about overlap and not about a guard that
    // refuses everything.
    let disjoint = lcg_tokens(0xB0BA, 20_000);
    assert_train_val_disjoint(&train, &disjoint);
    assert_eq!(
        check_train_val_disjoint(&train, &disjoint, eval_chunk_count(disjoint.len(), 40)),
        Ok(())
    );
}
