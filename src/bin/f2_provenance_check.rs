//! F2 provenance verifier — Loop 33 / Loop 32 Option C.
//!
//! Reads a CSV produced by `f2_ablation_sweep` (Loop 32+), parses the W3C-PROV
//! preamble (`# prov:*` lines), and validates each provenance claim against the
//! current repository / code state. Emits PASS/WARN/FAIL per claim.
//!
//! Standards: W3C PROV + Workflow Run RO-Crate (arXiv:2312.07852).
//!
//! Exit codes:
//!   0  all claims PASS
//!   1  ≥ 1 WARN (mismatch but not unsafe — e.g. older git SHA)
//!   2  ≥ 1 FAIL (schema mismatch → BPB numbers may not be reproducible)
//!   3  malformed input (no preamble or missing `prov:trainer_internals_schema`)
//!
//! Per the Loop 31 finding (LOCO_wd drifted 0.07 → 0.58 at identical config hash),
//! this binary closes the residual reproducibility gap.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Default)]
struct Provenance {
    fields: BTreeMap<String, String>,
}

fn parse_provenance(path: &str) -> Provenance {
    let f = File::open(path).expect("open input CSV");
    let r = BufReader::new(f);
    let mut prov = Provenance::default();
    for line in r.lines() {
        let line = line.expect("read");
        if line.is_empty() {
            continue;
        }
        if !line.starts_with('#') {
            break; // provenance ends at the first non-comment line.
        }
        // Format: `# prov:<key> = <value>`. Loop 34 fix 1: split on the FIRST '='
        // only (not " = " token), so values containing '=' (e.g. URL query strings,
        // host names like 'macbook = pro') are preserved verbatim.
        if let Some(rest) = line.strip_prefix("# prov:") {
            if let Some((k, v)) = rest.split_once('=') {
                let key = k.trim().to_string();
                let val = v.trim().to_string();
                prov.fields.insert(key, val);
            }
        }
    }
    prov
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Severity {
    Pass,
    Warn,
    Fail,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn current_git_sha() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Loop 36 fix 5: backward-compat field lookup. Returns canonical name's
/// value if present; otherwise tries each alias in order. Used so pre-Loop-32
/// CSVs (with `git_sha` / `schema_version` instead of W3C-PROV names) still
/// validate.
fn lookup_field<'a>(prov: &'a Provenance, canonical: &str, aliases: &[&str]) -> Option<&'a String> {
    if let Some(v) = prov.fields.get(canonical) {
        return Some(v);
    }
    for alias in aliases {
        if let Some(v) = prov.fields.get(*alias) {
            return Some(v);
        }
    }
    None
}

/// Loop 43 fix 6 → Loop 44 fix 2: probe the CSV's first data column for known
/// stratum prefixes. Derives the prefix list from `race::ablation::Stratum::ALL`
/// so adding a new variant (e.g. `Stratum::ClampZero`) auto-extends detection
/// without touching this binary.
///
/// Returns a sorted/deduped list of stratum identifiers (e.g. `["wd0", "warmup0"]`)
/// when stratified rows are detected. Returns `Some([])` for canonical-only CSVs.
fn detect_strata_in_data(path: &str) -> Option<Vec<String>> {
    use trios_trainer::race::ablation::Stratum;
    let f = File::open(path).ok()?;
    let r = BufReader::new(f);
    let mut found: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Build the prefix list from Stratum::ALL, skipping Canonical (empty prefix
    // would match every mode).
    let known_strata: Vec<&'static str> = Stratum::ALL
        .iter()
        .filter_map(|s| {
            let p = s.prefix();
            if p.is_empty() {
                None
            } else {
                Some(p)
            }
        })
        .collect();
    for line in r.lines().flatten() {
        if line.is_empty() || line.starts_with('#') || line.starts_with("mode,") {
            continue;
        }
        let mode = line.split(',').next().unwrap_or("");
        for &prefix in &known_strata {
            if mode.starts_with(prefix) {
                found.insert(prefix.trim_end_matches('_').to_string());
            }
        }
    }
    Some(found.into_iter().collect())
}

fn check(prov: &Provenance) -> Vec<(Severity, String, String)> {
    let mut out = Vec::new();

    // Required field: trainer_internals_schema.
    let csv_schema = lookup_field(
        prov,
        "trainer_internals_schema",
        &["schema", "schema_version", "trainer_schema"],
    );
    let code_schema = trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA;
    match csv_schema {
        None => out.push((
            Severity::Fail,
            "trainer_internals_schema".into(),
            "missing from CSV preamble (was this CSV produced before Loop 32?)".into(),
        )),
        Some(s) if s == code_schema => out.push((
            Severity::Pass,
            "trainer_internals_schema".into(),
            format!("matches current code: {}", s),
        )),
        Some(s) => out.push((
            Severity::Fail,
            "trainer_internals_schema".into(),
            format!(
                "CSV reports '{}' but current code is '{}' — BPB numbers may not reproduce. Re-run sweep at HEAD or check out the older commit.",
                s, code_schema
            ),
        )),
    }

    // Git SHA — WARN if mismatched (older SHA may be OK; later loops still work).
    let csv_sha = lookup_field(prov, "agent_git_sha", &["git_sha", "sha", "commit"]);
    let head_sha = current_git_sha();
    match (csv_sha, head_sha.as_deref()) {
        (Some(c), Some(h)) if c == h => {
            out.push((
                Severity::Pass,
                "agent_git_sha".into(),
                format!("matches HEAD: {}", c),
            ));
        }
        (Some(c), Some(h)) => {
            out.push((
                Severity::Warn,
                "agent_git_sha".into(),
                format!("CSV=({}), HEAD=({}). Check that the schema also bumped if trainer code changed.", c, h),
            ));
        }
        (Some(c), None) => {
            out.push((
                Severity::Warn,
                "agent_git_sha".into(),
                format!("CSV=({}); current git unavailable (not in repo?).", c),
            ));
        }
        (None, _) => {
            out.push((
                Severity::Warn,
                "agent_git_sha".into(),
                "missing from CSV preamble".into(),
            ));
        }
    }

    // Timestamp — WARN if >365 days old or in the future.
    let now = now_secs();
    let ts_field = lookup_field(prov, "generatedAt", &["timestamp", "ts", "generated_at"]);
    if let Some(ts_s) = ts_field {
        let ts_token = ts_s.split_whitespace().next().unwrap_or("");
        if let Ok(ts) = ts_token.parse::<u64>() {
            let age_days = if ts <= now { (now - ts) / 86400 } else { 0 };
            if ts > now + 86400 {
                out.push((
                    Severity::Warn,
                    "generatedAt".into(),
                    format!("CSV timestamp {} is in the future (clock skew?).", ts),
                ));
            } else if age_days > 365 {
                out.push((
                    Severity::Warn,
                    "generatedAt".into(),
                    format!(
                        "CSV is {} days old. Conventions/schemas may have changed.",
                        age_days
                    ),
                ));
            } else {
                out.push((
                    Severity::Pass,
                    "generatedAt".into(),
                    format!("{} ({} days ago)", ts, age_days),
                ));
            }
        }
    }

    // Loop 34 fix 2: cargo_pkg_version sanity check.
    // The crate version is part of the build's identity; a mismatch with the CSV's
    // value indicates someone bumped `version =` in Cargo.toml between sweep and
    // verification. WARN (not FAIL) because the version often moves while the
    // trainer internals stay identical.
    let csv_pkg = lookup_field(prov, "cargo_pkg_version", &["pkg_version", "crate_version"]);
    let build_pkg = env!("CARGO_PKG_VERSION");
    match csv_pkg {
        Some(s) if s == build_pkg => {
            out.push((Severity::Pass, "cargo_pkg_version".into(), s.clone()));
        }
        Some(s) => {
            out.push((
                Severity::Warn,
                "cargo_pkg_version".into(),
                format!("CSV={}, build={} — crate version moved; verify trainer_internals_schema also bumped if forward kernels changed.", s, build_pkg),
            ));
        }
        None => {
            out.push((
                Severity::Warn,
                "cargo_pkg_version".into(),
                "missing from CSV preamble".into(),
            ));
        }
    }

    // Host — informational PASS regardless of value.
    if let Some(h) = prov.fields.get("host") {
        out.push((Severity::Pass, "host".into(), h.clone()));
    }
    if let Some(by) = prov.fields.get("wasGeneratedBy") {
        out.push((Severity::Pass, "wasGeneratedBy".into(), by.clone()));
    }

    out
}

fn print_help() {
    println!("f2_provenance_check — Loop 33: verify CSV provenance preamble vs current code");
    println!();
    println!("USAGE: f2_provenance_check CSV...");
    println!();
    println!("Reads the W3C-PROV preamble (`# prov:*` lines) and validates each claim:");
    println!("  - trainer_internals_schema matches current code constant");
    println!("  - agent_git_sha matches HEAD (WARN if not)");
    println!("  - generatedAt timestamp is within 365 days, not future-dated");
    println!();
    println!("Exit codes: 0=PASS, 1=≥1 WARN, 2=≥1 FAIL, 3=malformed input.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let inputs: Vec<String> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .collect();
    if inputs.is_empty() {
        eprintln!("# ERROR: no input CSVs given. See --help.");
        std::process::exit(3);
    }
    let mut exit = 0i32;
    for path in &inputs {
        println!("=== {} ===", path);
        let prov = parse_provenance(path);
        if prov.fields.is_empty() {
            println!("FAIL  no provenance preamble (CSV pre-dates Loop 32?)");
            exit = exit.max(3);
            continue;
        }
        // Loop 43 fix 6: scan data section for stratum prefixes and emit an
        // informational banner. Helps users understand "this CSV is a stratified
        // sweep" without having to grep modes by hand.
        if let Some(strata) = detect_strata_in_data(path) {
            if !strata.is_empty() {
                println!(
                    "INFO  {:30}  contains stratified rows: {}",
                    "stratum",
                    strata.join(", ")
                );
            }
        }
        let checks = check(&prov);
        for (sev, key, msg) in &checks {
            let tag = match sev {
                Severity::Pass => "PASS ",
                Severity::Warn => "WARN ",
                Severity::Fail => "FAIL ",
            };
            println!("{} {:30}  {}", tag, key, msg);
            exit = match sev {
                Severity::Fail => exit.max(2),
                Severity::Warn => exit.max(1),
                Severity::Pass => exit,
            };
        }
    }
    std::process::exit(exit);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_csv(path: &std::path::Path, schema: &str, sha: &str, ts: u64) {
        let mut f = File::create(path).unwrap();
        writeln!(
            f,
            "# f2_ablation_sweep provenance (W3C PROV / RO-Crate, arXiv:2312.07852)"
        )
        .unwrap();
        writeln!(f, "# prov:generatedAt = {} (unix seconds UTC)", ts).unwrap();
        writeln!(
            f,
            "# prov:wasGeneratedBy = f2_ablation_sweep --mode loco --steps 200"
        )
        .unwrap();
        writeln!(f, "# prov:agent_git_sha = {}", sha).unwrap();
        writeln!(f, "# prov:host = test_host").unwrap();
        writeln!(f, "# prov:trainer_internals_schema = {}", schema).unwrap();
        writeln!(f, "# prov:cargo_pkg_version = 0.1.0").unwrap();
        writeln!(
            f,
            "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
        )
        .unwrap();
        writeln!(f, "loco,wd,0,,42,0.5,0xdead,0.1").unwrap();
    }

    #[test]
    fn parse_provenance_extracts_fields() {
        let tmp = std::env::temp_dir().join("f2_prov_parse.csv");
        write_csv(&tmp, "trainer_internals_vX", "abc123", 1700000000);
        let prov = parse_provenance(tmp.to_str().unwrap());
        assert_eq!(prov.fields.get("agent_git_sha").unwrap(), "abc123");
        assert_eq!(
            prov.fields.get("trainer_internals_schema").unwrap(),
            "trainer_internals_vX"
        );
        assert_eq!(prov.fields.get("host").unwrap(), "test_host");
    }

    #[test]
    fn check_emits_fail_on_schema_mismatch() {
        let tmp = std::env::temp_dir().join("f2_prov_fail.csv");
        write_csv(&tmp, "trainer_internals_v0_old", "abc123", 1700000000);
        let prov = parse_provenance(tmp.to_str().unwrap());
        let results = check(&prov);
        let schema_check = results
            .iter()
            .find(|(_, k, _)| k == "trainer_internals_schema")
            .unwrap();
        assert_eq!(
            schema_check.0,
            Severity::Fail,
            "expected FAIL for old schema"
        );
    }

    #[test]
    fn check_emits_pass_when_schema_matches() {
        let tmp = std::env::temp_dir().join("f2_prov_pass.csv");
        let cur = trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA;
        write_csv(&tmp, cur, "abc123", 1700000000);
        let prov = parse_provenance(tmp.to_str().unwrap());
        let results = check(&prov);
        let schema_check = results
            .iter()
            .find(|(_, k, _)| k == "trainer_internals_schema")
            .unwrap();
        assert_eq!(
            schema_check.0,
            Severity::Pass,
            "expected PASS for matching schema"
        );
    }

    #[test]
    fn check_accepts_legacy_field_aliases() {
        // Loop 36 fix 5: pre-Loop-32 CSV may use 'git_sha' instead of
        // 'agent_git_sha'. The check() function should find the value via
        // alias lookup rather than missing it.
        let tmp = std::env::temp_dir().join("f2_prov_legacy_alias.csv");
        let cur = trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA;
        let mut f = File::create(&tmp).unwrap();
        writeln!(f, "# prov:trainer_internals_schema = {}", cur).unwrap();
        writeln!(f, "# prov:git_sha = abc123").unwrap(); // legacy alias
        writeln!(f, "# prov:timestamp = 1700000000").unwrap(); // legacy alias
        writeln!(f, "mode,a,b\n").unwrap();
        drop(f);
        let prov = parse_provenance(tmp.to_str().unwrap());
        let results = check(&prov);
        // git SHA via alias should NOT be a "missing" WARN.
        let sha = results
            .iter()
            .find(|(_, k, _)| k == "agent_git_sha")
            .unwrap();
        assert!(
            !sha.2.contains("missing"),
            "alias 'git_sha' should resolve agent_git_sha, got: {:?}",
            sha
        );
        // Timestamp via alias should produce a result (not skipped).
        assert!(results.iter().any(|(_, k, _)| k == "generatedAt"));
    }

    #[test]
    fn parse_provenance_handles_value_with_equals_sign() {
        // Loop 34 fix 1: value containing '=' must survive parsing.
        let tmp = std::env::temp_dir().join("f2_prov_eq_value.csv");
        let mut f = File::create(&tmp).unwrap();
        writeln!(
            f,
            "# prov:trainer_internals_schema = trainer_internals_v1_2026_06_01"
        )
        .unwrap();
        writeln!(f, "# prov:host = host = with = equals").unwrap();
        writeln!(f, "mode,a,b\n").unwrap();
        drop(f);
        let prov = parse_provenance(tmp.to_str().unwrap());
        assert_eq!(prov.fields.get("host").unwrap(), "host = with = equals");
    }

    #[test]
    fn check_pkg_version_pass_when_matching() {
        // Loop 34 fix 2: matching cargo_pkg_version is PASS.
        let tmp = std::env::temp_dir().join("f2_prov_pkg_pass.csv");
        let cur = trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA;
        let mut f = File::create(&tmp).unwrap();
        writeln!(f, "# prov:trainer_internals_schema = {}", cur).unwrap();
        writeln!(
            f,
            "# prov:cargo_pkg_version = {}",
            env!("CARGO_PKG_VERSION")
        )
        .unwrap();
        writeln!(f, "mode,a,b\n").unwrap();
        drop(f);
        let prov = parse_provenance(tmp.to_str().unwrap());
        let results = check(&prov);
        let pkg = results
            .iter()
            .find(|(_, k, _)| k == "cargo_pkg_version")
            .unwrap();
        assert_eq!(pkg.0, Severity::Pass);
    }

    #[test]
    fn check_pkg_version_warns_on_mismatch() {
        // Loop 34 fix 2: mismatched cargo_pkg_version triggers WARN.
        let tmp = std::env::temp_dir().join("f2_prov_pkg_warn.csv");
        let cur = trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA;
        let mut f = File::create(&tmp).unwrap();
        writeln!(f, "# prov:trainer_internals_schema = {}", cur).unwrap();
        writeln!(f, "# prov:cargo_pkg_version = 99.99.99").unwrap();
        writeln!(f, "mode,a,b\n").unwrap();
        drop(f);
        let prov = parse_provenance(tmp.to_str().unwrap());
        let results = check(&prov);
        let pkg = results
            .iter()
            .find(|(_, k, _)| k == "cargo_pkg_version")
            .unwrap();
        assert_eq!(pkg.0, Severity::Warn);
    }

    #[test]
    fn binary_exit_code_2_on_schema_fail() {
        // Loop 34 fix 3: end-to-end subprocess exit check.
        let tmp = std::env::temp_dir().join("f2_prov_exit_fail.csv");
        write_csv(&tmp, "trainer_internals_v0_obsolete", "abc123", 1700000000);
        let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into());
        let bin = format!("{}/debug/f2_provenance_check", target);
        // If binary not present (cargo test --bin doesn't build the release target),
        // try building it on demand.
        if !std::path::Path::new(&bin).exists() {
            let _ = std::process::Command::new("cargo")
                .args(["build", "--bin", "f2_provenance_check"])
                .output();
        }
        // Skip the subprocess test if binary isn't there — environment-specific.
        if !std::path::Path::new(&bin).exists() {
            eprintln!("# skip: {} not found, can't subprocess-test", bin);
            return;
        }
        let status = std::process::Command::new(&bin)
            .arg(tmp.to_str().unwrap())
            .status()
            .expect("spawn binary");
        let code = status.code().unwrap_or(-1);
        assert_eq!(code, 2, "expected exit code 2 (FAIL), got {}", code);
    }

    #[test]
    fn detect_strata_in_data_finds_known_prefixes() {
        // Loop 43 fix 6: scans data column for stratum prefixes.
        let tmp = std::env::temp_dir().join("f2_prov_stratum_detect.csv");
        let mut f = File::create(&tmp).unwrap();
        writeln!(
            f,
            "# prov:trainer_internals_schema = trainer_internals_v1_2026_06_01"
        )
        .unwrap();
        writeln!(
            f,
            "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
        )
        .unwrap();
        writeln!(f, "wd0_pairwise,full_stack,-1,,42,4.0,0xdead,0.1").unwrap();
        writeln!(f, "warmup0_loco,rms,0,,42,5.0,0xdead,0.1").unwrap();
        writeln!(f, "pairwise,full_stack,-1,,42,4.0,0xdead,0.1").unwrap();
        drop(f);
        let strata = detect_strata_in_data(tmp.to_str().unwrap()).unwrap();
        assert!(strata.contains(&"wd0".to_string()));
        assert!(strata.contains(&"warmup0".to_string()));
        assert_eq!(strata.len(), 2);
    }

    #[test]
    fn detect_strata_in_data_empty_for_canonical_only() {
        let tmp = std::env::temp_dir().join("f2_prov_stratum_canonical.csv");
        let mut f = File::create(&tmp).unwrap();
        writeln!(
            f,
            "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
        )
        .unwrap();
        writeln!(f, "pairwise,full_stack,-1,,42,4.0,0xdead,0.1").unwrap();
        writeln!(f, "loco,rms,0,,42,5.0,0xdead,0.1").unwrap();
        drop(f);
        let strata = detect_strata_in_data(tmp.to_str().unwrap()).unwrap();
        assert!(strata.is_empty());
    }

    #[test]
    fn check_handles_provenance_without_data_rows() {
        // Loop 34 fix 4: half-written CSV (preamble + header, no rows) is still
        // a valid provenance preamble — `check()` must not panic and should report
        // the schema/git/ts results as if the file were complete.
        let tmp = std::env::temp_dir().join("f2_prov_no_rows.csv");
        let cur = trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA;
        let mut f = File::create(&tmp).unwrap();
        writeln!(f, "# prov:generatedAt = 1700000000").unwrap();
        writeln!(f, "# prov:agent_git_sha = abc123").unwrap();
        writeln!(f, "# prov:trainer_internals_schema = {}", cur).unwrap();
        writeln!(
            f,
            "# prov:cargo_pkg_version = {}",
            env!("CARGO_PKG_VERSION")
        )
        .unwrap();
        writeln!(
            f,
            "mode,fix_name,fix_index,cumulative_n,seed,bpb,config_hash,wall_s"
        )
        .unwrap();
        // No data rows!
        drop(f);
        let prov = parse_provenance(tmp.to_str().unwrap());
        assert!(!prov.fields.is_empty(), "preamble fields should be parsed");
        let results = check(&prov);
        let schema = results
            .iter()
            .find(|(_, k, _)| k == "trainer_internals_schema")
            .unwrap();
        assert_eq!(schema.0, Severity::Pass);
    }

    #[test]
    fn check_warns_when_timestamp_in_future() {
        let tmp = std::env::temp_dir().join("f2_prov_future.csv");
        let cur = trios_trainer::race::multi_seed::TRAINER_INTERNALS_SCHEMA;
        write_csv(&tmp, cur, "abc123", now_secs() + 10 * 86400);
        let prov = parse_provenance(tmp.to_str().unwrap());
        let results = check(&prov);
        let ts_check = results.iter().find(|(_, k, _)| k == "generatedAt").unwrap();
        assert_eq!(ts_check.0, Severity::Warn);
    }
}
