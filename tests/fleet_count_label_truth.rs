//! fleet_count_label_truth -- a published count must carry its denominator, and
//! a published uncertainty must be a function of the record it cites.
//!
//! Three defects, all of the same family: a figure that looks authoritative
//! because a document states it, where the document is the only thing behind it.
//!
//!   A  THE BUDGET ANCHOR WAS A NUMBER THE TRAINER NO LONGER PRODUCES.
//!      `docs/EVAL-UNCERTAINTY.md` section 3b calls row 0 "the only quantified
//!      row measured ON the artifact the budget is about", and it read `0.0551`
//!      from the schema-`/6` headline record. The current trainer writes
//!      `0.053656626492738724` into `evidence/r9-headline/12000.json` for the
//!      SAME weights -- both records carry `final_val_bpb = 2.6347548961639404`
//!      and `sha256 = 8a86fe69...`. The finite-population correction of section
//!      5b entered at `ba272b9` (`src/train_loop.rs:2445-2452`) and section 5b
//!      said so on 2026-08-03, four days before section 3b acted on it.
//!
//!      MEASURED 2026-08-07: the schema-`/6` record was written to
//!      `checkpoints/r6-headline/12000.json`, and
//!      `git check-ignore -v` reports `.gitignore:13:/checkpoints/` for that
//!      path. `git add` on it is REFUSED, so no commit could ever make this
//!      suite -- which READS that record -- pass in CI, and the panic this file
//!      raised when the read failed told the reader to `git add` it anyway.
//!      A verbatim byte-for-byte copy now lives at `HISTORICAL_RECORD` below,
//!      on a path git accepts; `evidence/r6-headline-historical/PROVENANCE.txt`
//!      records the copy and its digest. Nothing about the record's CONTENT
//!      changed, and this suite still recomputes from it rather than freezing
//!      any literal.
//!
//!      MEASURED 2026-08-07, against a brief that said otherwise: the current
//!      record is schema `trios-checkpoint-record/9`, NOT `/8`. `/9` IS
//!      committed -- `CHECKPOINT_RECORD_SCHEMA` at `src/checkpoint.rs:1921` in
//!      `HEAD` is `"trios-checkpoint-record/9"`, introduced by `e3edf9f`, an
//!      ancestor of `HEAD` -- and IS persisted: a census of the sidecars on
//!      this tree finds 50 `/9` records against 63 `/8`. This file therefore
//!      asserts the record's schema equals the CRATE'S OWN constant rather than
//!      any literal, so the test cannot go stale the way that claim did.
//!
//!      This case RECOMPUTES rather than freezing: it reads `val_bpb_stderr` out
//!      of the record and requires the doc to contain that value to at least six
//!      significant figures. Pinning the literal `0.053657` here would be the
//!      same transcription defect one layer down -- the doc's anchor has to be a
//!      function of the record, so that a record which moves again drags the doc
//!      with it or fails.
//!
//!   B  `8,037` WITHOUT `8,195` MINTS A FOURTH FLEET SIZE. Three figures already
//!      disagreed in this programme's own artifacts (1,851 / 1,878 / 7,927).
//!      `docs/FLEET-COUNT.md` resolved them, but two consumer documents still
//!      labelled 8,037 "sweep log files" with no denominator and no `v6_*.log`
//!      scoping. A reader who runs the obvious `ls .trinity/results/*.log | wc -l`
//!      gets 8,195 -- and now has a fourth number against three that already
//!      disagree, handed to them by the page whose job was to stop that.
//!
//!   C  `7,274` HAD NO DEFINITION ANYWHERE. The census printed it beside two
//!      labelled counts. It is the number of logs that emitted at least one
//!      `val_bpb` line -- runs that reached an eval -- and it is neither the
//!      completion count (5) nor a fleet size.
//!
//! WHY THESE THREE LIVE IN ONE FILE: all three are the same assertion, that a
//! number in a document must name the population or the record it came from.
//!
//! Nothing here trains, spawns a binary, or touches `.trinity/results/`. These
//! are assertions about committed text and committed records, and they must run
//! on a clone -- which is exactly why they cannot assert anything about the
//! 8,037 log files themselves: `.trinity/results/` is gitignored and exists on
//! one workstation. See `docs/FLEET-COUNT.md`.
//!
//! Anchor: phi^2 + phi^-2 = 3.

use std::path::{Path, PathBuf};

/// Repo root, so the suite is independent of the working directory `cargo test`
/// happens to be invoked from.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}\n\
             Every file this suite reads is UNTRACKED at HEAD and, crucially, \
             sits on a path git will ACCEPT: {CURRENT_RECORD}, \
             {HISTORICAL_RECORD}, {FLEET_DOC} and {CENSUS_SCRIPT}. `git add` \
             them together -- a clone otherwise receives the documents that \
             cite them and not the files themselves, and this test is the thing \
             that says so out loud.\n\
             Do NOT repoint any constant in this file at \
             {HISTORICAL_RECORD_ORIGIN}. `git check-ignore -v` reports \
             `.gitignore:13:/checkpoints/` for that path, so `git add` on it is \
             REFUSED and no commit can make this suite pass in CI. \
             {HISTORICAL_RECORD} is a verbatim byte-for-byte copy of it kept on \
             an addable path for exactly that reason -- see \
             evidence/r6-headline-historical/PROVENANCE.txt.",
            path.display()
        )
    })
}

/// The record section 3b row 0 must be anchored to: what the CURRENT trainer
/// writes for the headline weights.
const CURRENT_RECORD: &str = "evidence/r9-headline/12000.json";
/// The record row 0 used to be anchored to, retained in the doc as the
/// historical value beside the current one.
///
/// This is a VERBATIM copy of [`HISTORICAL_RECORD_ORIGIN`], not a re-mint. The
/// copy exists because the original path is gitignored (`.gitignore:13
/// /checkpoints/`) and `git add` on it is refused, so a suite that READS the
/// original could never pass in CI on any commit.
const HISTORICAL_RECORD: &str = "evidence/r6-headline-historical/12000.json";
/// Where the schema-`/6` record was WRITTEN, and the path
/// `docs/EVAL-UNCERTAINTY.md` section 3b still cites. Kept as a separate
/// constant because two different claims were previously conflated into one:
/// which file this suite READS, and which path the document NAMES. Only the
/// first has to be a path git accepts.
const HISTORICAL_RECORD_ORIGIN: &str = "checkpoints/r6-headline/12000.json";

const CENSUS_SCRIPT: &str = "scripts/fleet_census.py";

const UNCERTAINTY_DOC: &str = "docs/EVAL-UNCERTAINTY.md";
const V6_RESULTS_DOC: &str = "IGLA_V6_FINAL_RESULTS.md";
const FLEET_DOC: &str = "docs/FLEET-COUNT.md";

/// Documents that publish 8,037. Every LINE mentioning it must also carry the
/// 8,195 denominator.
const DENOMINATOR_CONSUMERS: [&str; 2] = [UNCERTAINTY_DOC, V6_RESULTS_DOC];

const COUNTED: &str = "8,037";
const DENOMINATOR: &str = "8,195";
const REACHED_AN_EVAL: &str = "7,274";

/// One field out of a checkpoint record, without pulling in a JSON dependency
/// this test does not otherwise need. The records are machine-written with one
/// `"key": value` per occurrence, so a scan for the key and a parse of the
/// numeric run after it is exact for these files -- and it PANICS rather than
/// defaulting if the key is absent, because a missing anchor that silently
/// reads as 0.0 is the defect class this file exists to remove.
fn record_f64(rel: &str, key: &str) -> f64 {
    let text = read(rel);
    let needle = format!("\"{key}\"");
    let at = text.find(&needle).unwrap_or_else(|| {
        panic!(
            "{rel} has no {needle} field; the doc's anchor cites a record that does not carry it"
        )
    });
    let after = &text[at + needle.len()..];
    let after = after
        .trim_start()
        .strip_prefix(':')
        .unwrap_or_else(|| panic!("{rel}: {needle} is not followed by ':'"))
        .trim_start();
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == '+'))
        .unwrap_or(after.len());
    after[..end].parse::<f64>().unwrap_or_else(|e| {
        panic!(
            "{rel}: {needle} value {:?} is not a number: {e}",
            &after[..end]
        )
    })
}

/// `key` looked up only AFTER the first occurrence of `anchor`. The records
/// nest `corpus.train.bytes` before `corpus.val.bytes`, and an unanchored scan
/// for "bytes" silently returns the TRAINING corpus -- a wrong number that
/// parses cleanly, which is the shape of defect this suite exists to refuse.
fn record_f64_after(rel: &str, anchor: &str, key: &str) -> f64 {
    let text = read(rel);
    let at = text
        .find(anchor)
        .unwrap_or_else(|| panic!("{rel} has no {anchor:?} to anchor the lookup of {key:?} to"));
    let scoped = &text[at..];
    let needle = format!("\"{key}\"");
    let found = scoped
        .find(&needle)
        .unwrap_or_else(|| panic!("{rel}: no {needle} after {anchor:?}"));
    let after = &scoped[found + needle.len()..];
    let after = after
        .trim_start()
        .strip_prefix(':')
        .unwrap_or_else(|| panic!("{rel}: {needle} is not followed by ':'"))
        .trim_start();
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == '+'))
        .unwrap_or(after.len());
    after[..end].parse::<f64>().unwrap_or_else(|e| {
        panic!(
            "{rel}: {needle} value {:?} is not a number: {e}",
            &after[..end]
        )
    })
}

fn record_str(rel: &str, key: &str) -> String {
    let text = read(rel);
    let needle = format!("\"{key}\"");
    let at = text
        .find(&needle)
        .unwrap_or_else(|| panic!("{rel} has no {needle} field"));
    let after = &text[at + needle.len()..];
    let after = after
        .trim_start()
        .strip_prefix(':')
        .unwrap_or_else(|| panic!("{rel}: {needle} is not followed by ':'"))
        .trim_start()
        .strip_prefix('"')
        .unwrap_or_else(|| panic!("{rel}: {needle} is not a string"));
    let end = after.find('"').expect("unterminated string");
    after[..end].to_string()
}

/// Section 3b of the uncertainty document, isolated so the anchor assertion
/// cannot be satisfied by the value appearing somewhere else in a 1,200-line
/// file. Ends at section 3c.
fn section_3b(doc: &str) -> &str {
    let start = doc
        .find("## 3b. The budget proper")
        .expect("EVAL-UNCERTAINTY.md has no section 3b heading");
    let rest = &doc[start..];
    let end = rest.find("\n## 3c.").unwrap_or(rest.len());
    &rest[..end]
}

/// (a) THE ANCHOR IS RECOMPUTED, NOT TRANSCRIBED.
///
/// Read `val_bpb_stderr` out of the record the budget must cite, render it to
/// six significant figures, and require section 3b to contain it. Changing the
/// doc back to the historical `0.0551` fails this, and the failure names the
/// record, the value the record states, and what the doc says instead.
///
/// MEASURED 2026-08-07, and NOT what one would assume from reading it: this
/// test does NOT depend on [`HISTORICAL_RECORD`] on the passing path. It names
/// that record only inside `assert!` message arguments, and Rust evaluates
/// those only when the assertion fires. Deleting the historical record leaves
/// this test green; three others in this file go red. That is fine -- the
/// historical value is diagnostic here, not asserted -- but it is written down
/// because "this test reads both records" is the natural misreading of the code
/// below, and an assumed dependency is not a dependency.
#[test]
fn budget_row_zero_anchor_matches_the_current_record() {
    let stderr_now = record_f64(CURRENT_RECORD, "val_bpb_stderr");
    assert!(
        stderr_now.is_finite() && stderr_now > 0.0,
        "{CURRENT_RECORD} states val_bpb_stderr = {stderr_now}, which is not a \
         usable uncertainty. An anchor of 0 or NaN would make every band in \
         section 3b collapse silently."
    );

    let doc = read(UNCERTAINTY_DOC);
    let budget = section_3b(&doc);

    // Six significant figures of a value near 0.05 is six decimal places.
    let rendered = format!("{stderr_now:.6}");

    assert!(
        budget.contains(&rendered),
        "SECTION 3b ANCHOR IS STALE.\n\
         \x20 record   {CURRENT_RECORD}\n\
         \x20 states   val_bpb_stderr = {stderr_now}\n\
         \x20 to 6sf   {rendered}\n\
         and section 3b of {UNCERTAINTY_DOC} does not contain {rendered}.\n\
         Section 3b calls row 0 \"the only quantified row measured ON the \
         artifact the budget is about\". If it cites a value the current \
         trainer will not produce, the budget's only artifact-measured anchor \
         is a transcription of a superseded record.\n\
         The historical record {HISTORICAL_RECORD} states {} -- it may appear \
         beside the current value as history, but it may not BE the anchor.",
        record_f64(HISTORICAL_RECORD, "val_bpb_stderr")
    );

    // The historical value must remain visible, with its source. Deleting a
    // number that was published is the failure mode this repository refuses:
    // a reader who opens the r6 record has to find its figure explained.
    //
    // EITHER path satisfies this, and that is deliberate rather than lax. The
    // record now exists at two paths -- the gitignored location it was written
    // to, and the copy under evidence/ that a clone actually receives -- and
    // both are its source. Hard-coding the origin here would make the CORRECT
    // future edit to section 3b (repointing readers at the clone-visible copy)
    // break this test, which is the wrong incentive. What must not happen is
    // the doc naming NEITHER.
    assert!(
        budget.contains(HISTORICAL_RECORD) || budget.contains(HISTORICAL_RECORD_ORIGIN),
        "section 3b names neither {HISTORICAL_RECORD} nor \
         {HISTORICAL_RECORD_ORIGIN}. The historical value must stay beside the \
         current one with its source, or a reader who opens that record finds a \
         number this document does not admit to."
    );

    // And the reason the two differ must be stated, not left to be inferred.
    for phrase in ["finite-population", "ba272b9"] {
        assert!(
            budget.contains(phrase),
            "section 3b states two different values for val_bpb_stderr on the \
             same weights and never mentions {phrase:?}. Two numbers for one \
             measurand with no stated cause is the disagreement this document \
             exists to remove, not an instance of it."
        );
    }
}

/// The two records must be about the SAME weights, or the revision in section
/// 3b is not a re-estimation at all but a comparison of two different models.
/// This is the load-bearing premise of the whole row-0 rewrite, so it is
/// asserted rather than assumed.
#[test]
fn both_records_describe_byte_identical_weights() {
    let sha_now = record_str(CURRENT_RECORD, "sha256");
    let sha_hist = record_str(HISTORICAL_RECORD, "sha256");
    assert_eq!(
        sha_now, sha_hist,
        "{CURRENT_RECORD} and {HISTORICAL_RECORD} name DIFFERENT checkpoints \
         ({sha_now} vs {sha_hist}). Section 3b presents their differing \
         val_bpb_stderr as one measurement under two estimators; that reading \
         requires identical weights and this says they are not."
    );

    let bpb_now = record_f64(CURRENT_RECORD, "final_val_bpb");
    let bpb_hist = record_f64(HISTORICAL_RECORD, "final_val_bpb");
    assert_eq!(
        bpb_now, bpb_hist,
        "the two records state different final_val_bpb ({bpb_now} vs \
         {bpb_hist}) for the same sha256. Then the stderr difference is not \
         purely an estimator change."
    );

    // The current record must actually be on the CURRENT schema, or "current"
    // is a claim about a file rather than a property of it. Compared against
    // the crate's own constant, never a literal: a hardcoded "/8" here would be
    // the same transcription defect this file exists to remove, and it is
    // exactly the claim that was wrong in the brief this test was written from
    // (the record is /9, and /9 is both committed and persisted).
    let schema = record_str(CURRENT_RECORD, "schema");
    assert_eq!(
        schema,
        trios_trainer::checkpoint::CHECKPOINT_RECORD_SCHEMA,
        "{CURRENT_RECORD} is schema {schema:?} but the crate writes {:?}. \
         Section 3b calls that record the current one -- the value the trainer \
         would produce today. If the schemas differ, either the record predates \
         the current trainer or the constant moved without the evidence being \
         re-minted, and in both cases the budget's anchor is historical again.",
        trios_trainer::checkpoint::CHECKPOINT_RECORD_SCHEMA
    );

    // The historical record must be on an OLDER schema, or the current/history
    // framing in section 3b is backwards.
    let hist_schema = record_str(HISTORICAL_RECORD, "schema");
    assert_ne!(
        hist_schema,
        trios_trainer::checkpoint::CHECKPOINT_RECORD_SCHEMA,
        "{HISTORICAL_RECORD} is on the CURRENT schema {hist_schema:?}. Section \
         3b presents it as the superseded record; if it is not superseded, the \
         two values are not a before/after of one estimator change."
    );
}

/// The revision must be arithmetic, not a new measurement: the current value
/// has to be the historical one times the finite-population factor. If it is
/// not, section 3b's explanation of WHY the two differ is wrong, and the two
/// numbers are evidence of something this document has not identified.
#[test]
fn the_two_stderrs_differ_only_by_the_finite_population_factor() {
    let now = record_f64(CURRENT_RECORD, "val_bpb_stderr");
    let hist = record_f64(HISTORICAL_RECORD, "val_bpb_stderr");
    let n = record_f64(CURRENT_RECORD, "eval_chunks");

    // N is what full coverage of the SAME stream walks. Both the stream length
    // and the planner come from outside this file -- the record's own
    // `corpus.val.bytes` and the crate's own `eval_chunk_count` -- so the test
    // cannot drift from the trainer, and a hardcoded 775 can never paper over a
    // planner change.
    let val_len = record_f64_after(CURRENT_RECORD, "\"val\"", "bytes") as usize;
    assert_eq!(
        val_len, 100_000,
        "the record's val corpus is {val_len} bytes, not the 100,000 section 3b \
         and section 5b both state. N, and therefore the whole correction, is a \
         function of this length."
    );
    let population = trios_trainer::train_loop::eval_chunk_count(val_len, 0) as f64;
    assert!(
        population > n,
        "eval_chunk_count says full coverage is {population} windows against \
         n = {n}; the correction is only meaningful for a strict subset"
    );

    let fpc = (1.0 - n / population).sqrt();
    let predicted = hist * fpc;

    // Both records store f32, so agreement is required at f32 precision.
    let tol = (predicted.abs() * 1e-6).max(1e-9);
    assert!(
        (predicted - now).abs() < tol,
        "THE STERR REVISION IS NOT THE FINITE-POPULATION CORRECTION.\n\
         \x20 historical {hist} ({HISTORICAL_RECORD})\n\
         \x20 n = {n}, N = {population}, sqrt(1 - n/N) = {fpc}\n\
         \x20 predicted  {predicted}\n\
         \x20 current    {now} ({CURRENT_RECORD})\n\
         Section 3b explains the gap between these two records as the FPC of \
         section 5b applied to byte-identical weights. If that arithmetic does \
         not close, the explanation is wrong and the two numbers mean \
         something this document has not identified."
    );
}

/// (b) EVERY LINE PUBLISHING 8,037 MUST CARRY 8,195.
///
/// Line-scoped on purpose: a denominator three paragraphs away is not a
/// denominator, because the sentence gets quoted without it. The failure names
/// the file, the 1-indexed line number and the line.
#[test]
fn every_published_8037_carries_its_denominator() {
    let mut offenders: Vec<String> = Vec::new();
    let mut seen = 0usize;

    for rel in DENOMINATOR_CONSUMERS {
        for (idx, line) in read(rel).lines().enumerate() {
            if !line.contains(COUNTED) {
                continue;
            }
            seen += 1;
            if !line.contains(DENOMINATOR) {
                offenders.push(format!("{rel}:{}: {}", idx + 1, line.trim()));
            }
        }
    }

    assert!(
        seen > 0,
        "no line in {DENOMINATOR_CONSUMERS:?} mentions {COUNTED} at all. Either \
         the figure was renamed -- in which case this guard is now blind and \
         must be updated to the new one -- or it was deleted without this test \
         noticing."
    );

    assert!(
        offenders.is_empty(),
        "BARE {COUNTED} WITH NO DENOMINATOR, on {} line(s):\n{}\n\n\
         A reader who runs `ls .trinity/results/*.log | wc -l` gets {DENOMINATOR}. \
         A bare {COUNTED} hands them a FOURTH figure against the three that \
         already disagree (1,851 / 1,878 / 7,927), from the documents whose \
         purpose is to remove that disagreement. Every line publishing {COUNTED} \
         must carry {DENOMINATOR} and the `v6_*.log` scoping in the same \
         sentence -- reuse the wording in {FLEET_DOC}, do not invent a fourth \
         phrasing.",
        offenders.len(),
        offenders.join("\n")
    );
}

/// The scoping half of the same claim: somewhere in each consumer, the counted
/// family must be named. A denominator alone does not tell a reader WHICH 8,037.
#[test]
fn every_consumer_names_the_counted_family() {
    for rel in DENOMINATOR_CONSUMERS {
        let text = read(rel);
        if !text.contains(COUNTED) {
            continue;
        }
        assert!(
            text.contains("v6_*.log"),
            "{rel} publishes {COUNTED} without ever naming the `v6_*.log` \
             family it counts. The denominator says the number is a subset; \
             only the glob says which subset."
        );
    }
}

/// (c) 7,274 MUST BE EXPLAINED WHERE THE COUNTS ARE RESOLVED.
///
/// The census prints three rows. Two had definitions; this one did not, which
/// is how an unlabelled number becomes a fourth fleet size. The explanation
/// must say what it counts AND both things it is not.
#[test]
fn the_val_bpb_row_is_explained() {
    let doc = read(FLEET_DOC);

    assert!(
        doc.contains(REACHED_AN_EVAL),
        "{FLEET_DOC} never mentions {REACHED_AN_EVAL}. `scripts/fleet_census.py` \
         prints it as one of three rows; a count in the output of a script this \
         document prescribes, with no definition in the document, is exactly \
         the unlabelled figure this page exists to eliminate."
    );

    // What it counts.
    assert!(
        doc.contains("val_bpb") && doc.contains("reached an eval"),
        "{FLEET_DOC} mentions {REACHED_AN_EVAL} without saying what it counts: \
         logs that emitted at least one `val_bpb` line, i.e. runs that reached \
         an eval."
    );

    // And what it is not. Both denials are load-bearing: the count sits
    // between a completion figure and a fleet size and gets mistaken for each.
    let lowered = doc.to_lowercase();
    for denial in ["not a completion count", "not a fleet size"] {
        assert!(
            lowered.contains(denial),
            "{FLEET_DOC} explains {REACHED_AN_EVAL} without stating it is \
             {denial:?}. The completion figure is 5 of 8,037 and the fleet size \
             is neither; a definition that omits the two things a number is \
             going to be mistaken for is not a definition."
        );
    }

    // The completion count must stay attached to its own denominator here too.
    assert!(
        doc.contains("5 of 8,037") || doc.contains("5 DONE out of 8,037"),
        "{FLEET_DOC} does not state the completion count with its denominator. \
         5 of 8,037 is the load-bearing figure on this page."
    );
}

/// Everything this suite and its two consumer documents point a reader at.
const PRESCRIBED: [&str; 4] = [CENSUS_SCRIPT, FLEET_DOC, CURRENT_RECORD, HISTORICAL_RECORD];

/// The census script and this page are prescribed by two TRACKED documents, and
/// both are untracked at HEAD. A clone gets the documents and not the script.
/// This test cannot fix that -- only `git add` can -- but it can refuse to let
/// the documents point at files that are not there on THIS machine, which is
/// the weaker half of the same claim.
#[test]
fn the_prescribed_script_and_page_exist() {
    for rel in PRESCRIBED {
        let path: PathBuf = root().join(rel);
        assert!(
            Path::new(&path).exists(),
            "{UNCERTAINTY_DOC} and {V6_RESULTS_DOC} both point readers at \
             {rel}, and it is not present at {}. Note that this checks the \
             WORKING TREE only: these files are untracked at HEAD, so a clone \
             receives the two documents and neither the page they link to, nor \
             the script they prescribe, nor the records they cite. `git add` \
             them together or the link is a 404 for everyone but this \
             workstation.",
            path.display()
        );
    }
}

/// `git add` advice is only advice if git will take the path.
///
/// THE DEFECT THIS EXISTS FOR: this suite used to READ
/// `checkpoints/r6-headline/12000.json`, and `.gitignore:13` is `/checkpoints/`.
/// `git add` on it is refused, so NO commit could make the three tests above
/// pass in CI -- and the panic they raised on the missing file told the reader
/// to `git add` it. A remedy git will not accept is worse than no remedy: it
/// reads as actionable and consumes the reader's time proving it is not.
///
/// The existence test above is not enough on its own. A file that exists here
/// and cannot be committed passes it forever on this workstation and fails for
/// everybody else, which is the same failure-reports-success shape.
#[test]
fn every_prescribed_path_is_one_git_will_accept() {
    let mut refused = Vec::new();
    for rel in PRESCRIBED {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(root())
            .args(["check-ignore", "-v", "--"])
            .arg(rel)
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "`git check-ignore` could not be run in {}: {e}. This test \
                     asserts that every path this suite reads is one `git add` \
                     will accept, and without git that claim cannot be checked. \
                     It is reported as a failure rather than skipped: a guard \
                     that quietly does nothing is the defect class this file \
                     exists to remove.",
                    root().display()
                )
            });
        // exit 0 == the path IS ignored, and stdout names the rule.
        if out.status.success() {
            refused.push(format!(
                "{rel} -- {}",
                String::from_utf8_lossy(&out.stdout).trim()
            ));
        }
    }

    assert!(
        refused.is_empty(),
        "GITIGNORED PATH IN A SUITE THAT MUST PASS IN CI, on {} path(s):\n  {}\n\n\
         `git add` on each of those is REFUSED, so no commit can put the file in \
         a clone and no CI run can read it. If a record you need lives under an \
         ignored tree, copy it verbatim onto an accepted path and say so in a \
         PROVENANCE file -- {HISTORICAL_RECORD} is that copy of \
         {HISTORICAL_RECORD_ORIGIN}, and \
         evidence/r6-headline-historical/PROVENANCE.txt is that file. Do not \
         instead soften the read into a skip: a skip turns a loud break into a \
         CI no-op.",
        refused.len(),
        refused.join("\n  ")
    );
}
