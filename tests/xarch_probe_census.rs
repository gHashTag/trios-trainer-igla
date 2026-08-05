//! The two cross-architecture instruments must not misreport their own
//! coverage, and the ULP table must be derived from the published dumps.
//!
//! `src/bin/xarch_probe.rs` used to end its report with
//! `SUMMARY primitives=16`, a typed literal that nothing checked. A count an
//! instrument states about itself is a claim like any other: add a `report()`
//! call and the line still says 16, delete one and it still says 16, and the
//! only reader who would ever notice is the auditor who counts the lines by
//! hand. The same defect family as an empty provenance field -- it looks
//! controlled and is not.
//!
//! `src/bin/ulp_census.rs` carries the other half. The sharpest technical claim
//! this repository owns -- the cross-architecture divergence is libm, and every
//! disagreement is exactly one unit in the last place -- was measured by a
//! program that lived in `/tmp`. This file is what makes the published table
//! `evidence/xarch-probe/ULP-CENSUS.txt` a derivation rather than a
//! transcription: it recomputes every cell from
//! `evidence/xarch-probe/ulp-arm64.txt` and `evidence/xarch-probe/ulp-x86_64.txt`
//! and fails if the artifact and the dumps have drifted apart.
//!
//! What each test would catch, stated so a reader can check the tests are worth
//! their runtime:
//!
//! * a `report()` added or removed without the declared list moving with it;
//! * the census and the probe silently measuring different numbers (the
//!   `DIGEST` / `PRIMITIVE` join);
//! * a hand-edited cell in the published table;
//! * either instrument acquiring a file read, an environment read or a clock,
//!   any of which would let something about the host reach a published number.
//!
//! Scope: the evidence-file tests read `evidence/xarch-probe/`, which was
//! produced on one host with both arms of one ISA pair (`aarch64-apple-darwin`
//! natively and `x86_64-apple-darwin` under Rosetta 2). They check that the
//! published table follows from the published dumps. They do NOT re-run the
//! x86_64 arm, and they cannot: the x86_64 dump is an observation, not
//! something a test on any host can regenerate. See
//! `docs/DIVERGENCE-MECHANISM.md` and `evidence/xarch-probe/PROVENANCE.txt`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Functions the census covers, in emission order.
const FUNCTIONS: [&str; 5] = ["exp", "ln", "sqrt", "powf", "cos"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn evidence(name: &str) -> PathBuf {
    repo_root().join("evidence/xarch-probe").join(name)
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e))
}

/// Run the probe and return its stdout, refusing a partial report.
fn run_probe() -> String {
    let exe = env!("CARGO_BIN_EXE_xarch_probe");
    let out = Command::new(exe)
        .output()
        .unwrap_or_else(|e| panic!("cannot execute {}: {}", exe, e));
    assert!(
        out.status.success(),
        "xarch_probe exited with {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("xarch_probe emitted non-UTF-8")
}

/// `SUMMARY key=value` pairs, parsed into a map.
fn summary_fields(report: &str) -> BTreeMap<String, String> {
    let line = report
        .lines()
        .find(|l| l.starts_with("SUMMARY "))
        .expect("the report has no SUMMARY line");
    line.split_whitespace()
        .skip(1)
        .filter_map(|tok| tok.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

// -------------------------------------------------------------------
// The self-census
// -------------------------------------------------------------------

/// The emitted count, the declared count and the lines on stdout are one
/// number.
///
/// This is the test the work item exists for. Adding a `report()` call without
/// extending `PRIMITIVES` makes the binary abort inside `report()` (the name at
/// that position does not match), so the failure arrives as a red test rather
/// than as a report that understates its own coverage.
#[test]
fn summary_count_equals_the_declared_list_and_the_emitted_lines() {
    let report = run_probe();
    let fields = summary_fields(&report);

    let emitted: usize = fields
        .get("primitives")
        .expect("SUMMARY has no `primitives` field")
        .parse()
        .expect("`primitives` is not a number");
    let declared: usize = fields
        .get("declared")
        .expect("SUMMARY has no `declared` field")
        .parse()
        .expect("`declared` is not a number");
    let printed = report
        .lines()
        .filter(|l| l.starts_with("PRIMITIVE "))
        .count();
    let details = report.lines().filter(|l| l.starts_with("DETAIL")).count();

    assert_eq!(
        emitted, declared,
        "the probe emitted {} primitives but declares {}",
        emitted, declared
    );
    assert_eq!(
        printed, declared,
        "{} PRIMITIVE lines on stdout against {} declared",
        printed, declared
    );
    assert_eq!(
        details, printed,
        "{} DETAIL lines against {} PRIMITIVE lines",
        details, printed
    );
    assert!(
        printed > 0,
        "the probe reported nothing; a census of zero primitives is not a pass"
    );
}

/// The count is derived, not typed. A literal in the source would survive every
/// assertion above, because the assertions compare the line against itself.
#[test]
fn the_summary_count_is_not_a_source_literal() {
    let source = read(&repo_root().join("src/bin/xarch_probe.rs"));
    let printed = run_probe()
        .lines()
        .filter(|l| l.starts_with("PRIMITIVE "))
        .count();
    let literal = format!("primitives={}", printed);
    assert!(
        !source.contains(&literal),
        "src/bin/xarch_probe.rs contains the literal `{}`; the SUMMARY count \
         must be derived from the emitted primitives, not typed",
        literal
    );
}

/// The live probe still reproduces the published aarch64 arm, byte for byte.
///
/// Skipped rather than failed when the host is not the arm that produced the
/// evidence: a x86_64 or Linux host disagreeing here is the documented result,
/// not a regression. The PLATFORM line is what decides, so the skip cannot hide
/// a mismatch on the arm the evidence claims.
#[test]
fn the_live_probe_reproduces_the_published_arm64_report() {
    let published = read(&evidence("probe-arm64.txt"));
    let live = run_probe();

    let published_platform = published
        .lines()
        .find(|l| l.starts_with("PLATFORM "))
        .expect("published report has no PLATFORM line");
    let live_platform = live
        .lines()
        .find(|l| l.starts_with("PLATFORM "))
        .expect("live report has no PLATFORM line");
    if published_platform != live_platform {
        eprintln!(
            "skipping: published `{}` against live `{}`",
            published_platform, live_platform
        );
        return;
    }

    assert_eq!(
        published, live,
        "the aarch64 arm no longer reproduces evidence/xarch-probe/probe-arm64.txt; \
         same-machine determinism is a bigger finding than any table on this page"
    );
}

// -------------------------------------------------------------------
// The ULP census
// -------------------------------------------------------------------

/// `(function, index) -> u32 bit pattern`, from a `ulp_census` dump.
fn load_dump(path: &Path) -> BTreeMap<(String, usize), u32> {
    let text = read(path);
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"VALUE") {
            continue;
        }
        assert_eq!(f.len(), 4, "malformed VALUE line: {}", line);
        let index: usize = f[2]
            .parse()
            .unwrap_or_else(|_| panic!("bad index: {}", line));
        let bits =
            u32::from_str_radix(f[3], 16).unwrap_or_else(|_| panic!("bad bit pattern: {}", line));
        assert!(
            out.insert((f[1].to_string(), index), bits).is_none(),
            "duplicate element {} {} in {}",
            f[1],
            index,
            path.display()
        );
    }
    assert!(!out.is_empty(), "{} carries no VALUE lines", path.display());
    out
}

/// One row of the published table, recomputed from the two dumps.
struct Row {
    function: String,
    differing: usize,
    total: usize,
    share: f64,
    max_ulp: u32,
    first_index: Option<usize>,
    first_arm64: u32,
    first_x86_64: u32,
}

impl Row {
    fn render(&self) -> String {
        match self.first_index {
            Some(i) => format!(
                "{} {} {} {:.3}% {} {} 0x{:08x} 0x{:08x}",
                self.function,
                self.differing,
                self.total,
                self.share,
                self.max_ulp,
                i,
                self.first_arm64,
                self.first_x86_64
            ),
            None => format!(
                "{} {} {} {:.3}% {} - - -",
                self.function, self.differing, self.total, self.share, self.max_ulp
            ),
        }
    }
}

/// Recompute the census. The ULP figure is the absolute difference of the two
/// u32 bit patterns, which equals the count of representable floats between the
/// two values only because every value involved is finite and of one sign on
/// both arms -- asserted below rather than assumed.
fn derive_rows() -> Vec<Row> {
    let arm = load_dump(&evidence("ulp-arm64.txt"));
    let x86 = load_dump(&evidence("ulp-x86_64.txt"));
    assert_eq!(
        arm.keys().collect::<Vec<_>>(),
        x86.keys().collect::<Vec<_>>(),
        "the two dumps do not cover the same elements; they are not comparable"
    );

    let mut rows = Vec::new();
    for function in FUNCTIONS.iter() {
        let mut total = 0usize;
        let mut differing = 0usize;
        let mut max_ulp = 0u32;
        let mut first: Option<(usize, u32, u32)> = None;
        for ((f, index), a) in arm.iter() {
            if f != function {
                continue;
            }
            let b = x86[&(f.clone(), *index)];
            total += 1;
            let (a, b) = (*a, b);
            assert!(
                f32::from_bits(a).is_finite() && f32::from_bits(b).is_finite(),
                "{} element {} is not finite on both arms; the ULP reading of a \
                 raw bit-pattern difference would be meaningless",
                f,
                index
            );
            assert!(
                f32::from_bits(a).is_sign_positive() == f32::from_bits(b).is_sign_positive(),
                "{} element {} changed sign across the arms; a bit-pattern \
                 difference is not a ULP count across zero",
                f,
                index
            );
            if a != b {
                differing += 1;
                max_ulp = max_ulp.max(a.abs_diff(b));
                if first.is_none() {
                    first = Some((*index, a, b));
                }
            }
        }
        assert!(total > 0, "no elements for function `{}`", function);
        rows.push(Row {
            function: (*function).to_string(),
            differing,
            total,
            share: differing as f64 * 100.0 / total as f64,
            max_ulp,
            first_index: first.map(|(i, _, _)| i),
            first_arm64: first.map(|(_, a, _)| a).unwrap_or(0),
            first_x86_64: first.map(|(_, _, b)| b).unwrap_or(0),
        });
    }
    rows
}

fn render_table(rows: &[Row]) -> String {
    let mut out = String::new();
    out.push_str("ULP-CENSUS-TABLE 1\n");
    out.push_str("derived_from ulp-arm64.txt ulp-x86_64.txt\n");
    out.push_str(
        "columns function differing of share max_ulp first_index arm64_bits x86_64_bits\n",
    );
    for row in rows {
        out.push_str(&row.render());
        out.push('\n');
    }
    out
}

/// The published table is what the published dumps say, cell for cell.
#[test]
fn published_census_table_is_derived_from_the_published_dumps() {
    let derived = render_table(&derive_rows());
    let published = read(&evidence("ULP-CENSUS.txt"));
    assert_eq!(
        published, derived,
        "evidence/xarch-probe/ULP-CENSUS.txt does not follow from the dumps \
         beside it.\n--- published ---\n{}\n--- derived ---\n{}",
        published, derived
    );
}

/// The result the pitch leans on: every disagreement is exactly one unit in the
/// last place, and it is libm that disagrees.
///
/// This asserts the SHAPE of the finding, not the counts -- the counts live in
/// the derived table above and would legitimately move if the inputs or the
/// platform changed. If a difference ever exceeds 1 ULP, or if `ln` or `sqrt`
/// start to differ, `docs/DIVERGENCE-MECHANISM.md` is wrong on its central
/// paragraph and must be re-read before anything is quoted from it.
#[test]
fn every_disagreement_is_exactly_one_ulp() {
    for row in derive_rows() {
        if row.differing == 0 {
            assert_eq!(
                row.max_ulp, 0,
                "`{}` has no differing elements but a nonzero ULP",
                row.function
            );
            continue;
        }
        assert_eq!(
            row.max_ulp, 1,
            "`{}` differs by up to {} ULP on {} of {} elements; the \
             one-unit-in-the-last-place reading of the mechanism no longer holds",
            row.function, row.max_ulp, row.differing, row.total
        );
    }
}

/// `sqrt` is pinned by IEEE 754 and lowered to a hardware instruction, so it
/// must agree. A disagreement here would mean the method itself is broken, not
/// that a libm slice rounds differently.
#[test]
fn the_ieee_pinned_control_agrees_across_the_arms() {
    let sqrt = derive_rows()
        .into_iter()
        .find(|r| r.function == "sqrt")
        .expect("no sqrt row");
    assert_eq!(
        sqrt.differing, 0,
        "sqrt disagreed on {} of {} elements; IEEE 754 pins it exactly, so \
         either the dumps are not comparable or the harness is the source of \
         the difference",
        sqrt.differing, sqrt.total
    );
}

/// The census and the probe are measuring the same numbers.
///
/// `ulp_census` recomputes `exp`/`ln`/`sqrt`/`powf`/`cos` over the same buffers
/// `xarch_probe` hashes as `exp_4096` and friends. If the two ever disagree,
/// one instrument has drifted and NEITHER table in
/// `docs/DIVERGENCE-MECHANISM.md` is readable -- the per-primitive hashes and
/// the ULP counts would no longer be about the same measurement.
#[test]
fn census_digests_join_to_the_probe_hashes_on_both_arms() {
    for (census, probe) in [
        ("ulp-arm64.txt", "probe-arm64.txt"),
        ("ulp-x86_64.txt", "probe-x86_64.txt"),
    ] {
        let census_text = read(&evidence(census));
        let probe_text = read(&evidence(probe));

        let digests: BTreeMap<&str, &str> = census_text
            .lines()
            .filter(|l| l.starts_with("DIGEST "))
            .map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                let sha = f[3]
                    .strip_prefix("sha256=")
                    .unwrap_or_else(|| panic!("malformed DIGEST line: {}", l));
                (f[1], sha)
            })
            .collect();
        let primitives: BTreeMap<&str, &str> = probe_text
            .lines()
            .filter(|l| l.starts_with("PRIMITIVE "))
            .map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                (f[1], f[2])
            })
            .collect();

        assert_eq!(
            digests.len(),
            FUNCTIONS.len(),
            "{} carries {} DIGEST lines, expected {}",
            census,
            digests.len(),
            FUNCTIONS.len()
        );
        for function in FUNCTIONS.iter() {
            let key = format!("{}_4096", function);
            let census_sha = digests
                .get(function)
                .unwrap_or_else(|| panic!("{} has no DIGEST for `{}`", census, function));
            let probe_sha = primitives
                .get(key.as_str())
                .unwrap_or_else(|| panic!("{} has no PRIMITIVE `{}`", probe, key));
            assert_eq!(
                census_sha, probe_sha,
                "`{}` hashes to {} in {} and to {} in {}; the two instruments \
                 are not measuring the same numbers",
                function, census_sha, census, probe_sha, probe
            );
        }
    }
}

// -------------------------------------------------------------------
// The instruments' own hygiene
// -------------------------------------------------------------------

/// Neither instrument may read a file, read the environment or look at a clock.
///
/// `SUMMARY ... randomness=none file_io=none env_reads=none` is prose the
/// binary asserts about itself and cannot check at run time. This checks it
/// where it is decidable -- in the source -- so the claim on the line is backed
/// by something. `std::env::consts::ARCH` is exempt and deliberately so: it is
/// a compile-time constant baked in by the target triple, not a read of the
/// host.
#[test]
fn the_instruments_read_no_files_no_environment_and_no_clock() {
    for name in ["src/bin/xarch_probe.rs", "src/bin/ulp_census.rs"] {
        let source = read(&repo_root().join(name));
        let code: String = source
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for forbidden in [
            "std::fs",
            "File::open",
            "File::create",
            "env::var",
            "env::args",
            "rand::",
            "SystemTime",
            "Instant::now",
            "thread_rng",
        ] {
            assert!(
                !code.contains(forbidden),
                "{} contains `{}`; the instrument claims \
                 randomness=none file_io=none env_reads=none",
                name,
                forbidden
            );
        }
    }
}

/// Source stays ASCII (L3 PURITY), including the evidence the doc quotes from.
#[test]
fn instruments_and_evidence_are_ascii() {
    for name in [
        "src/bin/xarch_probe.rs",
        "src/bin/ulp_census.rs",
        "evidence/xarch-probe/ULP-CENSUS.txt",
        "evidence/xarch-probe/PROVENANCE.txt",
    ] {
        let text = read(&repo_root().join(name));
        assert!(
            text.is_ascii(),
            "{} is not ASCII; L3 PURITY applies to published evidence too",
            name
        );
    }
}

/// The provenance file must not name the machine that produced the evidence.
///
/// `evidence/xarch-local-isa/probe.json` already leaks a host name into the
/// published record. That is a privacy defect in an artifact meant to travel to
/// a regulator, and it must not be repeated here.
#[test]
fn provenance_carries_no_host_path_or_machine_name() {
    let text = read(&evidence("PROVENANCE.txt"));
    for leak in ["/Users/", "/home/", "MacBook", ".local\n"] {
        assert!(
            !text.contains(leak),
            "evidence/xarch-probe/PROVENANCE.txt contains `{}`; published \
             provenance records the platform, not the person",
            leak.trim_end()
        );
    }
    for required in [
        "rustc",
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "n=1",
    ] {
        assert!(
            text.contains(required),
            "PROVENANCE.txt does not record `{}`",
            required
        );
    }
}
