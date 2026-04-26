//! `trios-igla` — query helpers for the IGLA RACE ledger.
//!
//! Provides the primitives used by the `trios-igla` binary:
//!
//! - filter (search) over `assertions/seed_results.jsonl`
//! - last-N listing in canonical R7 triplet form
//! - Gate-2 quorum verdict (≥3 seeds with `bpb < T` AND `step ≥ 4000`)
//! - embargo refusal against `assertions/embargo.txt` (R9)
//! - canonical R7 triplet rendering
//!
//! Triplet (R7):
//! ```text
//! BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>
//! ```
//!
//! All commands are read-only — they never mutate the ledger.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------
// Constants — Gate-2 / R8 / R9 anchors
// ---------------------------------------------------------------------

/// Default needle target (Gate-2 victory threshold). Aligns with
/// [`crate::TrainConfig::target_bpb`] default in [`configs/`].
pub const DEFAULT_TARGET_BPB: f64 = 1.85;

/// R8 — only rows with `step >= STEP_MIN_FOR_LEDGER` count toward Gate-2.
/// Mirrors the floor enforced in [`crate::ledger::emit_row`].
pub const STEP_MIN_FOR_LEDGER: u64 = 4_000;

/// Gate-2 stop rule: at least this many distinct seeds must satisfy the
/// `bpb < target` AND `step >= STEP_MIN_FOR_LEDGER` predicate.
pub const GATE2_SEED_QUORUM: usize = 3;

/// Canonical short-SHA prefix length used in the R7 triplet.
pub const SHA_PREFIX_LEN: usize = 7;

/// Default count for `trios-igla list`.
pub const DEFAULT_LIST_LAST_N: usize = 10;

/// Default ledger location (relative to repo root or working dir).
pub const DEFAULT_LEDGER_PATH: &str = "assertions/seed_results.jsonl";

/// Default embargo location (relative to repo root or working dir).
pub const DEFAULT_EMBARGO_PATH: &str = "assertions/embargo.txt";

// ---------------------------------------------------------------------
// Ledger row schema (read-side mirror of `crate::ledger::LedgerRow`)
// ---------------------------------------------------------------------

/// Read-side mirror of [`crate::ledger::LedgerRow`]. Kept independent so
/// that the trainer's emitter (`Serialize`) and the CLI reader
/// (`Deserialize`) can evolve without forcing one struct to do both.
///
/// Schema must stay in sync with the writer; see [`crate::ledger`].
#[derive(Debug, Clone, Deserialize)]
pub struct LedgerRow {
    #[serde(default)]
    pub agent: String,
    pub bpb: f64,
    pub step: u64,
    pub seed: u64,
    pub sha: String,
    #[serde(default)]
    pub jsonl_row: u64,
    #[serde(default)]
    pub gate_status: String,
    #[serde(default)]
    pub ts: String,
}

/// Filter predicate composed from CLI flags. Each `Option` is `None`
/// when the flag is omitted; when `Some(_)` the clause must match.
#[derive(Debug, Default, Clone)]
pub struct SearchFilter {
    pub seed: Option<u64>,
    pub bpb_max: Option<f64>,
    pub step_min: Option<u64>,
    pub sha: Option<String>,
    pub gate_status: Option<String>,
}

// ---------------------------------------------------------------------
// Triplet rendering (R7)
// ---------------------------------------------------------------------

/// Canonical R7 triplet:
/// ```text
/// BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>
/// ```
///
/// `<g>` falls back to `"unknown"` when the row has no `gate_status`
/// (e.g. a legacy row predating the field).
pub fn render_triplet(row: &LedgerRow) -> String {
    let sha7: &str = if row.sha.len() >= SHA_PREFIX_LEN {
        &row.sha[..SHA_PREFIX_LEN]
    } else {
        &row.sha
    };
    let gate = if row.gate_status.is_empty() {
        "unknown"
    } else {
        row.gate_status.as_str()
    };
    format!(
        "BPB={} @ step={} seed={} sha={} jsonl_row={} gate_status={}",
        format_bpb(row.bpb),
        row.step,
        row.seed,
        sha7,
        row.jsonl_row,
        gate,
    )
}

/// Compact float formatter:
/// - `2.5` → `"2.5"`
/// - `2.2393` → `"2.2393"`
/// - `1.0` → `"1"`
fn format_bpb(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 {
        return format!("{}", v as i64);
    }
    let mut s = format!("{:.6}", v);
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    s
}

// ---------------------------------------------------------------------
// Filter evaluation
// ---------------------------------------------------------------------

/// True iff every `Some(_)` clause of the filter matches the row.
pub fn matches(filter: &SearchFilter, row: &LedgerRow) -> bool {
    if let Some(s) = filter.seed {
        if s != row.seed {
            return false;
        }
    }
    if let Some(bm) = filter.bpb_max {
        if !(row.bpb < bm) {
            return false;
        }
    }
    if let Some(sm) = filter.step_min {
        if row.step < sm {
            return false;
        }
    }
    if let Some(ref sha_pref) = filter.sha {
        if !row.sha.starts_with(sha_pref) {
            return false;
        }
    }
    if let Some(ref g) = filter.gate_status {
        if &row.gate_status != g {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------
// Gate-2 quorum
// ---------------------------------------------------------------------

/// Number of distinct seeds with at least one row satisfying
/// `bpb < target_bpb` AND `step >= STEP_MIN_FOR_LEDGER`.
///
/// The verdict is `count >= GATE2_SEED_QUORUM`.
pub fn gate2_seed_count(rows: &[LedgerRow], target_bpb: f64) -> usize {
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for row in rows {
        if row.bpb < target_bpb && row.step >= STEP_MIN_FOR_LEDGER {
            seen.insert(row.seed);
        }
    }
    seen.len()
}

/// `Some(true)` if PASS, `Some(false)` if NOT YET, helper for callers
/// that want a verdict without inspecting the count themselves.
pub fn gate2_verdict(rows: &[LedgerRow], target_bpb: f64) -> bool {
    gate2_seed_count(rows, target_bpb) >= GATE2_SEED_QUORUM
}

// ---------------------------------------------------------------------
// Embargo (R9)
// ---------------------------------------------------------------------

/// True iff `sha` matches an embargo entry.
///
/// Match rules (case-insensitive on hex):
/// - exact: an embargo line equals the candidate
/// - prefix: candidate and entry share the same 7-char prefix when
///   both are at least 7 chars long
///
/// Lines starting with `#` and empty lines are ignored.
pub fn is_embargoed(embargo_lines: &[String], sha: &str) -> bool {
    let needle = sha.trim().to_lowercase();
    if needle.is_empty() {
        return false;
    }
    for line in embargo_lines {
        let entry = line.trim().to_lowercase();
        if entry.is_empty() || entry.starts_with('#') {
            continue;
        }
        if entry == needle {
            return true;
        }
        if needle.len() >= SHA_PREFIX_LEN
            && entry.len() >= SHA_PREFIX_LEN
            && entry[..SHA_PREFIX_LEN] == needle[..SHA_PREFIX_LEN]
        {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------
// JSONL helpers
// ---------------------------------------------------------------------

/// Read a JSONL ledger and return only the parseable rows. Lines that
/// fail to parse (e.g. legacy schema headers) are skipped silently so
/// operational queries keep working as the schema evolves.
pub fn read_ledger(path: &Path) -> Result<Vec<LedgerRow>> {
    let f =
        File::open(path).with_context(|| format!("failed to open ledger {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Ok(row) = serde_json::from_str::<LedgerRow>(trimmed) {
            rows.push(row);
        }
    }
    Ok(rows)
}

/// Read an embargo file into a vector of trimmed, non-empty,
/// non-comment lines.
pub fn read_embargo(path: &Path) -> Result<Vec<String>> {
    let f =
        File::open(path).with_context(|| format!("failed to open embargo {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut lines = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let t = line.trim().to_string();
        if !t.is_empty() && !t.starts_with('#') {
            lines.push(t);
        }
    }
    Ok(lines)
}

// ---------------------------------------------------------------------
// Command actions (consumed by the binary)
// ---------------------------------------------------------------------

/// Write the matched rows as triplet lines to `out`. Returns the number
/// of matches.
pub fn run_search(
    ledger: &Path,
    filter: &SearchFilter,
    out: &mut dyn std::io::Write,
) -> Result<usize> {
    let rows = read_ledger(ledger)?;
    let mut hits = 0usize;
    for row in &rows {
        if matches(filter, row) {
            writeln!(out, "{}", render_triplet(row))?;
            hits += 1;
        }
    }
    Ok(hits)
}

/// Write the last `n` rows as triplet lines. Returns the number emitted.
pub fn run_list(ledger: &Path, n: usize, out: &mut dyn std::io::Write) -> Result<usize> {
    let rows = read_ledger(ledger)?;
    let take = n.min(rows.len());
    let start = rows.len() - take;
    for row in &rows[start..] {
        writeln!(out, "{}", render_triplet(row))?;
    }
    Ok(take)
}

/// Result of `gate` — `(pass, distinct_seed_count, total_rows)`.
pub fn run_gate(ledger: &Path, target_bpb: f64) -> Result<(bool, usize, usize)> {
    let rows = read_ledger(ledger)?;
    let count = gate2_seed_count(&rows, target_bpb);
    Ok((count >= GATE2_SEED_QUORUM, count, rows.len()))
}

/// Result of `check` — `Ok(())` if clean, `Err(_)` with R9 message if embargoed.
pub fn run_check(embargo: &Path, sha: &str) -> Result<()> {
    let lines = read_embargo(embargo)?;
    if is_embargoed(&lines, sha) {
        bail!(
            "R9 embargo refusal: sha={} is on {}",
            sha,
            embargo.display()
        );
    }
    Ok(())
}

/// Print the canonical triplet for a given row index.
pub fn run_triplet(ledger: &Path, row_index: usize, out: &mut dyn std::io::Write) -> Result<()> {
    let rows = read_ledger(ledger)?;
    if row_index >= rows.len() {
        bail!(
            "row_index {} out of bounds (ledger has {} parseable rows)",
            row_index,
            rows.len()
        );
    }
    writeln!(out, "{}", render_triplet(&rows[row_index]))?;
    Ok(())
}

// ---------------------------------------------------------------------
// Default-path helpers
// ---------------------------------------------------------------------

/// Default ledger path as a [`PathBuf`].
pub fn default_ledger_path() -> PathBuf {
    PathBuf::from(DEFAULT_LEDGER_PATH)
}

/// Default embargo path as a [`PathBuf`].
pub fn default_embargo_path() -> PathBuf {
    PathBuf::from(DEFAULT_EMBARGO_PATH)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn row_43_below_target() -> LedgerRow {
        LedgerRow {
            agent: "trios-trainer-igla-gate2".into(),
            bpb: 2.497,
            step: 12_000,
            seed: 43,
            sha: "6a40e17".into(),
            jsonl_row: 1,
            gate_status: "below_target_evidence".into(),
            ts: "2026-04-26T12:34:38Z".into(),
        }
    }

    #[test]
    fn search_hit() {
        let f = SearchFilter {
            seed: Some(43),
            bpb_max: Some(2.50),
            step_min: Some(4_000),
            sha: None,
            gate_status: None,
        };
        assert!(matches(&f, &row_43_below_target()));
    }

    #[test]
    fn search_miss_step_too_low() {
        let f = SearchFilter {
            step_min: Some(4_000),
            ..Default::default()
        };
        let row = LedgerRow {
            agent: "trios-trainer-igla-pretrain".into(),
            bpb: 3.5,
            step: 1_000,
            seed: 43,
            sha: "deadbee".into(),
            jsonl_row: 2,
            gate_status: "below_champion".into(),
            ts: "2026-04-26T00:00:00Z".into(),
        };
        assert!(!matches(&f, &row));
    }

    #[test]
    fn search_miss_seed() {
        let f = SearchFilter {
            seed: Some(44),
            ..Default::default()
        };
        assert!(!matches(&f, &row_43_below_target()));
    }

    fn make(seed: u64, bpb: f64, step: u64) -> LedgerRow {
        LedgerRow {
            agent: "a".into(),
            bpb,
            step,
            seed,
            sha: "aaaaaaa".into(),
            jsonl_row: 0,
            gate_status: "victory_candidate".into(),
            ts: "t".into(),
        }
    }

    #[test]
    fn gate_pass_three_seeds() {
        let rows = vec![
            make(43, 1.80, 27_000),
            make(44, 1.79, 27_000),
            make(45, 1.84, 27_000),
        ];
        assert_eq!(gate2_seed_count(&rows, 1.85), 3);
        assert!(gate2_verdict(&rows, 1.85));
    }

    #[test]
    fn gate_not_yet_two_seeds() {
        let rows = vec![
            make(43, 1.80, 27_000),
            make(44, 1.79, 27_000),
            make(45, 2.00, 27_000),
        ];
        assert_eq!(gate2_seed_count(&rows, 1.85), 2);
        assert!(!gate2_verdict(&rows, 1.85));
    }

    #[test]
    fn gate_ignores_low_step() {
        let rows = vec![
            make(43, 1.50, 100),
            make(44, 1.50, 200),
            make(45, 1.50, 300),
        ];
        assert_eq!(gate2_seed_count(&rows, 1.85), 0);
    }

    #[test]
    fn embargo_refuses_full_match() {
        let embargo: Vec<String> = vec![
            "477e3377", "b3ee6a36", "2f6e4c2", "4a158c01", "6393be94", "5950174", "32d1dd3",
            "a7574c3",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert!(is_embargoed(&embargo, "477e3377"));
    }

    #[test]
    fn embargo_refuses_prefix_match() {
        let embargo: Vec<String> = vec!["477e3377abc".into()];
        assert!(is_embargoed(&embargo, "477e337"));
    }

    #[test]
    fn embargo_accepts_clean() {
        let embargo: Vec<String> = vec!["477e3377".into(), "b3ee6a36".into()];
        assert!(!is_embargoed(&embargo, "2446855"));
    }

    #[test]
    fn embargo_skips_comments_and_blank() {
        let embargo: Vec<String> = vec![
            "# this is the IGLA RACE embargo".into(),
            "".into(),
            "477e3377".into(),
        ];
        assert!(!is_embargoed(&embargo, "2446855"));
        assert!(is_embargoed(&embargo, "477e3377"));
    }

    #[test]
    fn triplet_renders_canonical() {
        let row = LedgerRow {
            agent: "trios-trainer-igla-gate2".into(),
            bpb: 2.2393,
            step: 27_000,
            seed: 43,
            sha: "2446855abcde".into(),
            jsonl_row: 7,
            gate_status: "below_champion".into(),
            ts: "2026-04-26T12:34:38Z".into(),
        };
        assert_eq!(
            render_triplet(&row),
            "BPB=2.2393 @ step=27000 seed=43 sha=2446855 jsonl_row=7 gate_status=below_champion"
        );
    }

    #[test]
    fn triplet_sha_is_seven_chars() {
        let row = LedgerRow {
            agent: "x".into(),
            bpb: 2.0,
            step: 5_000,
            seed: 43,
            sha: "abcdef0123456789".into(),
            jsonl_row: 0,
            gate_status: "below_champion".into(),
            ts: "t".into(),
        };
        let line = render_triplet(&row);
        assert!(line.contains("sha=abcdef0"));
        assert!(!line.contains("sha=abcdef01"));
    }

    #[test]
    fn triplet_falls_back_to_unknown_gate() {
        let row = LedgerRow {
            agent: "x".into(),
            bpb: 2.5,
            step: 5_000,
            seed: 43,
            sha: "1234567".into(),
            jsonl_row: 9,
            gate_status: String::new(),
            ts: "t".into(),
        };
        assert!(render_triplet(&row).ends_with("gate_status=unknown"));
    }

    #[test]
    fn constants_align_with_trainer() {
        assert_eq!(GATE2_SEED_QUORUM, 3);
        assert_eq!(STEP_MIN_FOR_LEDGER, 4_000);
        assert!(DEFAULT_TARGET_BPB < 2.2393); // strictly below champion
    }

    #[test]
    fn phi_anchor_holds() {
        // The TRINITY anchor is the constitutional reason this CLI exists.
        let phi: f64 = (1.0 + 5f64.sqrt()) / 2.0;
        let lhs = phi * phi + 1.0 / (phi * phi);
        assert!((lhs - 3.0).abs() < 1e-10);
    }

    #[test]
    fn format_bpb_handles_specials() {
        assert_eq!(format_bpb(2.5), "2.5");
        assert_eq!(format_bpb(2.2393), "2.2393");
        assert_eq!(format_bpb(1.0), "1");
    }

    #[test]
    fn gate_run_against_temp_ledger() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("seed_results.jsonl");
        let body = "{\"agent\":\"a\",\"bpb\":1.80,\"gate_status\":\"victory_candidate\",\"seed\":43,\"sha\":\"aaaaaaa\",\"step\":27000,\"jsonl_row\":1,\"ts\":\"t\"}\n\
                    {\"agent\":\"a\",\"bpb\":1.79,\"gate_status\":\"victory_candidate\",\"seed\":44,\"sha\":\"bbbbbbb\",\"step\":27000,\"jsonl_row\":2,\"ts\":\"t\"}\n\
                    {\"agent\":\"a\",\"bpb\":1.84,\"gate_status\":\"victory_candidate\",\"seed\":45,\"sha\":\"ccccccc\",\"step\":27000,\"jsonl_row\":3,\"ts\":\"t\"}\n";
        std::fs::write(&p, body).unwrap();
        let (pass, count, total) = run_gate(&p, 1.85).unwrap();
        assert!(pass);
        assert_eq!(count, 3);
        assert_eq!(total, 3);
    }
}
