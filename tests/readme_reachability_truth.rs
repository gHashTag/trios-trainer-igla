//! Resolve every README claim about clone-reachability against the ref a
//! stranger actually gets, not against `HEAD`.
//!
//! WHY THIS FILE EXISTS. `README.md` used to certify, in the first table a
//! reviewer reads, that `evidence/r9-headline/12000.json` "is TRACKED, so a
//! fresh clone receives it". It is not tracked anywhere:
//! `git ls-tree -r HEAD --name-only evidence/r9-headline/` prints nothing, and
//! neither does the same command against the remote branch tip. The directory is
//! working-tree output that is not gitignored either, so `ls` finds it and `git`
//! does not. A sentence that certifies the exact property it lacks is this
//! repository's named defect class, not a broken link.
//!
//! `HEAD` IS NOT THE RIGHT REF, AND NEITHER IS A STALE TRACKING REF. Three
//! different populations answer three different questions, and only one of them
//! answers "what does a clone get":
//!
//! * `git ls-files evidence`               -- the author's index.
//! * `git ls-tree -r HEAD ... evidence`    -- the author's last commit.
//! * the tree of the REMOTE branch tip     -- what `git clone` hands a stranger.
//!
//! MEASURED ON 2026-08-07, and one half of it contradicts the brief this file
//! was written from, so the measurement is recorded here rather than the brief:
//!
//! ```text
//! git ls-files evidence | wc -l                              ->  55
//! git ls-tree -r HEAD --name-only evidence | wc -l           ->  55
//! git ls-tree -r origin/fix/509-qat-v2 --name-only evidence  ->  21
//! git rev-list --count origin/fix/509-qat-v2..HEAD           ->   4
//! git ls-remote origin refs/heads/fix/509-qat-v2             ->  e9eec901d228...
//! git rev-parse HEAD                                         ->  e9eec901d228...
//! ```
//!
//! The last two lines are the same commit. The local branch is NOT ahead of the
//! remote by 4 commits: `refs/remotes/origin/fix/509-qat-v2` is a STALE CACHE,
//! because `remote.origin.fetch` on this checkout is the single refspec
//! `+refs/heads/main:refs/remotes/origin/main`, so no `git fetch` ever updates
//! the tracking ref for this branch. A guard that trusted that ref would have
//! called three genuinely clone-reachable paths unreachable
//! (`evidence/SEALS.txt`, `evidence/xarch-aarch64-reference/`,
//! `docs/CANONICAL-SERIALIZATION.md`). Trusting `HEAD` and trusting the tracking
//! ref are two different ways to answer a reachability question from a local
//! artifact, and both can be wrong in either direction.
//!
//! So the clone ref is RESOLVED, loudly, by this chain (see `resolve_clone_ref`):
//!
//! 1. `git ls-remote --heads <remote> refs/heads/<upstream>` -- authoritative;
//!    the object must also be present locally so its tree can be listed.
//! 2. otherwise `refs/remotes/<remote>/<upstream>`, and ONLY if that ref is
//!    covered by a `remote.<remote>.fetch` refspec, i.e. only if it is a ref git
//!    actually maintains. The caveat is printed.
//! 3. otherwise FAIL, with the remedy. Never skip. A skipped reachability check
//!    is indistinguishable from a passing one, which is the defect being fixed.
//!
//! WHAT IS CHECKED. Three layers, and the first is the complete one:
//!
//! * every relative markdown link target in `README.md` must be in the clone
//!   ref's tree, unless its line carries the literal marker `not in a clone` --
//!   and a line carrying that marker whose links are ALL reachable fails too,
//!   because a stale warning is also a false sentence;
//! * every line carrying a trackedness phrase (`a fresh clone receives`,
//!   `is tracked at`, ...) and naming a path must have that path in the state
//!   the phrase claims, in both polarities;
//! * every hex token in the README that resolves to a commit in this repository
//!   must be an ancestor of the clone ref, which is what catches an instruction
//!   to "run from a clean clone at <sha>" naming a commit no clone can check
//!   out. This is deliberately stronger than a keyword scan around the word
//!   "clone": a keyword list is a loophole, an ancestry test is not.
//!
//! A failure names the path, the README line number, and which of the three
//! states the path is really in.
//!
//! ANTI-VACUITY. `the_scan_population_is_not_empty` fails if the parser stops
//! finding links or path-bearing sentences. A guard over an empty population is
//! a vacuous pass wearing a measurement's clothes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// git plumbing
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct GitOut {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn git(args: &[&str]) -> GitOut {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo_root())
        // Never let a credential or host-key prompt turn a measurement into a hang.
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ASKPASS", "echo")
        .env(
            "GIT_SSH_COMMAND",
            "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
        )
        .output()
        .unwrap_or_else(|e| panic!("failed to run `git {}`: {e}", args.join(" ")));
    GitOut {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

// ---------------------------------------------------------------------------
// clone-ref resolution
// ---------------------------------------------------------------------------

struct CloneRef {
    sha: String,
    remote: String,
    branch: String,
    /// How the sha was obtained. Printed on every run; it is part of the result.
    method: String,
    /// Non-fatal but loud observations about the local ref configuration.
    notes: Vec<String>,
}

/// Does `remote.<remote>.fetch` actually maintain `refs/remotes/<remote>/<branch>`?
fn refspec_covers(remote: &str, branch: &str) -> bool {
    let want = format!("refs/remotes/{remote}/{branch}");
    let cfg = git(&["config", "--get-all", &format!("remote.{remote}.fetch")]);
    for line in cfg.stdout.lines() {
        let spec = line.trim().trim_start_matches('+');
        let Some((_src, dst)) = spec.split_once(':') else {
            continue;
        };
        if let Some(prefix) = dst.strip_suffix('*') {
            if want.starts_with(prefix) {
                return true;
            }
        } else if dst == want {
            return true;
        }
    }
    false
}

fn resolve_clone_ref() -> CloneRef {
    let head_branch = git(&["symbolic-ref", "--quiet", "--short", "HEAD"]);
    assert!(
        head_branch.ok,
        "CANNOT DETERMINE WHAT A CLONE GETS: HEAD is detached, so there is no \
         upstream branch to resolve. Check out the branch this README documents \
         and re-run. This test refuses to skip: an unresolved reachability check \
         is indistinguishable from a passing one."
    );
    let head_branch = head_branch.stdout.trim().to_string();

    let remote = git(&["config", "--get", &format!("branch.{head_branch}.remote")]);
    let merge = git(&["config", "--get", &format!("branch.{head_branch}.merge")]);
    assert!(
        remote.ok && merge.ok,
        "CANNOT DETERMINE WHAT A CLONE GETS: branch `{head_branch}` has no \
         upstream configured (branch.{head_branch}.remote / .merge are unset). \
         Set one with `git branch --set-upstream-to=<remote>/<branch>`. \
         This test refuses to skip."
    );
    let remote = remote.stdout.trim().to_string();
    let upstream_branch = merge
        .stdout
        .trim()
        .strip_prefix("refs/heads/")
        .unwrap_or(merge.stdout.trim())
        .to_string();

    let mut notes = Vec::new();

    let tracking = format!("refs/remotes/{remote}/{upstream_branch}");
    let tracking_sha = git(&["rev-parse", "--quiet", "--verify", &tracking])
        .stdout
        .trim()
        .to_string();

    // 1. Authoritative: ask the remote.
    let ls = git(&[
        "ls-remote",
        "--heads",
        &remote,
        &format!("refs/heads/{upstream_branch}"),
    ]);
    if ls.ok {
        if let Some(sha) = ls
            .stdout
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().next())
        {
            let sha = sha.to_string();
            let have = git(&["cat-file", "-e", &format!("{sha}^{{commit}}")]);
            assert!(
                have.ok,
                "CANNOT DETERMINE WHAT A CLONE GETS: `git ls-remote {remote} \
                 refs/heads/{upstream_branch}` reports tip {sha}, but that commit \
                 is not present locally, so its tree cannot be listed. Run \
                 `git fetch {remote} {upstream_branch}` and re-run. This test \
                 refuses to fall back to a ref it knows is behind the remote."
            );
            if !tracking_sha.is_empty() && tracking_sha != sha {
                notes.push(format!(
                    "STALE LOCAL REF: {tracking} points at {} but the remote tip is {}. \
                     remote.{remote}.fetch does not cover this branch, so no `git fetch` \
                     updates it; `git ls-tree -r {tracking}` answers a question about the \
                     past. Remedy: `git fetch {remote} {upstream_branch}:{tracking}`.",
                    &tracking_sha[..12.min(tracking_sha.len())],
                    &sha[..12.min(sha.len())],
                ));
            }
            return CloneRef {
                sha,
                remote,
                branch: upstream_branch,
                method: "git ls-remote (authoritative)".to_string(),
                notes,
            };
        }
    }

    // 2. Fall back to a tracking ref, but only one git actually maintains.
    let covered = refspec_covers(&remote, &upstream_branch);
    assert!(
        !tracking_sha.is_empty() && covered,
        "CANNOT DETERMINE WHAT A CLONE GETS.\n\
         `git ls-remote {remote} refs/heads/{upstream_branch}` did not answer \
         (exit ok={}, stderr: {})\n\
         and the local ref {tracking} is {}.\n\
         Refusing to certify README reachability from an unverifiable ref. \
         Remedy: restore network access, or run \
         `git fetch {remote} {upstream_branch}:{tracking}` and add a matching \
         refspec to remote.{remote}.fetch.",
        ls.ok,
        ls.stderr.trim(),
        if tracking_sha.is_empty() {
            "absent".to_string()
        } else {
            format!(
                "present at {} but NOT covered by any remote.{remote}.fetch refspec, \
                 i.e. a cache no fetch refreshes",
                &tracking_sha[..12.min(tracking_sha.len())]
            )
        }
    );
    notes.push(format!(
        "The remote could not be queried; using {tracking} as of the last fetch. \
         Reachability below is only as fresh as that fetch."
    ));
    CloneRef {
        sha: tracking_sha,
        remote,
        branch: upstream_branch,
        method: "remote-tracking ref (last fetch)".to_string(),
        notes,
    }
}

fn tree_paths(rev: &str) -> BTreeSet<String> {
    let out = git(&["ls-tree", "-r", rev, "--name-only"]);
    assert!(out.ok, "git ls-tree -r {rev} failed: {}", out.stderr.trim());
    out.stdout
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn present(tree: &BTreeSet<String>, path: &str) -> bool {
    let p = path.trim_end_matches('/');
    if p.is_empty() {
        return false;
    }
    if tree.contains(p) {
        return true;
    }
    let dir = format!("{p}/");
    tree.range(dir.clone()..)
        .next()
        .is_some_and(|f| f.starts_with(&dir))
}

// ---------------------------------------------------------------------------
// the three states
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
enum State {
    InClone,
    CommittedLocallyOnly,
    Untracked {
        ignored: Option<String>,
        on_disk: bool,
    },
}

impl State {
    fn describe(&self) -> String {
        match self {
            State::InClone => "REACHABLE IN A CLONE (present in the clone ref's tree)".to_string(),
            State::CommittedLocallyOnly => {
                "COMMITTED LOCALLY BUT NOT IN THE CLONE REF -- present at HEAD, absent from \
                 what a clone gets; it needs a push"
                    .to_string()
            }
            State::Untracked { ignored, on_disk } => {
                let ign = match ignored {
                    Some(rule) => format!(", and gitignored via {rule}"),
                    None => {
                        ", and not gitignored either -- `ls` finds it, `git` does not".to_string()
                    }
                };
                format!(
                    "UNTRACKED (absent from HEAD and from the clone ref{ign}; {} in this working tree)",
                    if *on_disk { "present" } else { "absent" }
                )
            }
        }
    }
}

struct Trees {
    clone: BTreeSet<String>,
    head: BTreeSet<String>,
}

fn classify(trees: &Trees, path: &str) -> State {
    if present(&trees.clone, path) {
        return State::InClone;
    }
    if present(&trees.head, path) {
        return State::CommittedLocallyOnly;
    }
    let p = path.trim_end_matches('/');
    // `git check-ignore -v` exits 0 for a NEGATION match too: on this tree it
    // printed `.gitignore:78:!evidence/**/*.log` for `evidence/r9-headline/run.log`,
    // a file that is NOT ignored. Reading the exit code of `-v` made the first
    // version of this guard report a wrong state in its own failure message.
    // `-q` is the boolean; `-v` is only good for the rule text.
    let ignored = if git(&["check-ignore", "-q", p]).ok {
        git(&["check-ignore", "-v", p])
            .stdout
            .split_whitespace()
            .next()
            .map(|s| s.to_string())
    } else {
        None
    };
    State::Untracked {
        ignored,
        on_disk: repo_root().join(p).exists(),
    }
}

// ---------------------------------------------------------------------------
// README parsing
// ---------------------------------------------------------------------------

fn readme_lines() -> Vec<String> {
    let p = repo_root().join("README.md");
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
        .lines()
        .map(|l| l.to_string())
        .collect()
}

/// Relative markdown link targets: `(line_number, target)`.
fn relative_links(lines: &[String]) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        let bytes: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == '[' {
                // find the matching ']' then an immediately following '('
                let mut j = i + 1;
                while j < bytes.len() && bytes[j] != ']' {
                    j += 1;
                }
                if j + 1 < bytes.len() && bytes[j] == ']' && bytes[j + 1] == '(' {
                    let mut k = j + 2;
                    while k < bytes.len() && bytes[k] != ')' {
                        k += 1;
                    }
                    if k < bytes.len() {
                        let target: String = bytes[j + 2..k].iter().collect();
                        let target = target.trim().to_string();
                        let skip = target.starts_with("http")
                            || target.starts_with("mailto:")
                            || target.starts_with('#')
                            || target.is_empty();
                        if !skip {
                            // drop any `#anchor`
                            let clean = target.split('#').next().unwrap_or("").trim().to_string();
                            if !clean.is_empty() {
                                out.push((idx + 1, clean));
                            }
                        }
                        i = k + 1;
                        continue;
                    }
                }
            }
            i += 1;
        }
    }
    out
}

/// Top-level entries that can start a repository path, derived from the repo
/// rather than hand-listed, so a new top-level directory does not blind this.
fn top_level_names(trees: &Trees) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = BTreeSet::new();
    for set in [&trees.clone, &trees.head] {
        for p in set {
            if let Some(first) = p.split('/').next() {
                names.insert(first.to_string());
            }
        }
    }
    if let Ok(rd) = std::fs::read_dir(repo_root()) {
        for e in rd.flatten() {
            if let Some(n) = e.file_name().to_str() {
                names.insert(n.to_string());
            }
        }
    }
    names
}

/// Path-like tokens on one line: inside backticks or markdown link targets.
fn path_tokens(line: &str, tops: &BTreeSet<String>) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    let mut in_tick = false;
    let mut buf = String::new();
    for ch in line.chars() {
        if ch == '`' {
            if in_tick {
                candidates.push(std::mem::take(&mut buf));
            } else {
                buf.clear();
            }
            in_tick = !in_tick;
        } else if in_tick {
            buf.push(ch);
        }
    }
    let mut out = Vec::new();
    for cand in candidates {
        for word in cand.split_whitespace() {
            let w = word
                .trim_matches(|c: char| {
                    matches!(c, ',' | ';' | ':' | '(' | ')' | '"' | '\'' | '*' | '.')
                })
                .to_string();
            if !w.contains('/') {
                continue;
            }
            let first = w.split('/').next().unwrap_or("");
            if tops.contains(first) && w.len() > first.len() + 1 {
                out.push(w);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

const NOT_IN_A_CLONE_MARKER: &str = "not in a clone";

const POSITIVE_PHRASES: &[&str] = &[
    "a fresh clone receives",
    "a fresh clone has",
    "a fresh clone gets",
    "reaches a fresh clone",
    "a clone receives",
    "a clone has",
    "a clone gets",
    "a clone can run",
    "it is tracked",
    "is tracked at",
    "is tracked in",
    "the tracked record",
    "are tracked at",
];

const NEGATIVE_PHRASES: &[&str] = &[
    NOT_IN_A_CLONE_MARKER,
    "not tracked",
    "untracked",
    "not yet tracked",
    "in no clone",
    "no clone",
    "a clone does not",
    "does not reach",
    "gitignored",
    "is zero",
    "got nothing",
    "prints nothing",
];

fn contains_any(hay: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| hay.contains(n))
}

// ---------------------------------------------------------------------------
// findings
// ---------------------------------------------------------------------------

struct Finding {
    line: usize,
    subject: String,
    claim: String,
    state: String,
}

fn report(header: &str, findings: &[Finding], cref: &CloneRef) -> String {
    let mut s = String::new();
    s.push_str(header);
    s.push_str(&format!(
        "\n\nclone ref: {}/{} @ {} (resolved by {})\n",
        cref.remote, cref.branch, cref.sha, cref.method
    ));
    for n in &cref.notes {
        s.push_str(&format!("note: {n}\n"));
    }
    s.push_str(&format!("\n{} finding(s):\n", findings.len()));
    for f in findings {
        s.push_str(&format!(
            "\n  README.md:{}\n    subject : {}\n    claims  : {}\n    actually: {}\n",
            f.line, f.subject, f.claim, f.state
        ));
    }
    s.push_str(
        "\nSettle any one of these yourself with:\n  \
         git ls-tree -r <remote>/<branch> --name-only <path>\n\
         `ls` cannot answer this question and neither can `git ls-tree HEAD`.\n",
    );
    s
}

fn setup() -> (CloneRef, Trees, Vec<String>) {
    let cref = resolve_clone_ref();
    println!(
        "clone ref: {}/{} @ {} (resolved by {})",
        cref.remote, cref.branch, cref.sha, cref.method
    );
    for n in &cref.notes {
        println!("note: {n}");
    }
    let trees = Trees {
        clone: tree_paths(&cref.sha),
        head: tree_paths("HEAD"),
    };
    println!(
        "clone ref carries {} files ({} under evidence/); HEAD carries {} ({} under evidence/)",
        trees.clone.len(),
        trees
            .clone
            .iter()
            .filter(|p| p.starts_with("evidence/"))
            .count(),
        trees.head.len(),
        trees
            .head
            .iter()
            .filter(|p| p.starts_with("evidence/"))
            .count(),
    );
    let lines = readme_lines();
    (cref, trees, lines)
}

// ---------------------------------------------------------------------------
// the guards
// ---------------------------------------------------------------------------

#[test]
fn readme_relative_links_resolve_in_a_fresh_clone() {
    let (cref, trees, lines) = setup();
    let links = relative_links(&lines);

    let mut findings = Vec::new();
    // group by line so a marker can be judged against every link on that line
    let mut by_line: std::collections::BTreeMap<usize, Vec<String>> = Default::default();
    for (ln, target) in &links {
        by_line.entry(*ln).or_default().push(target.clone());
    }

    for (ln, targets) in by_line {
        let marked = lines[ln - 1].to_lowercase().contains(NOT_IN_A_CLONE_MARKER);
        let mut all_reachable = true;
        for t in &targets {
            let state = classify(&trees, t);
            if state == State::InClone {
                continue;
            }
            all_reachable = false;
            if marked {
                continue; // the line says so, in the words the guard reads
            }
            findings.push(Finding {
                line: ln,
                subject: t.clone(),
                claim: "linked as if a reader on GitHub could follow it".to_string(),
                state: state.describe(),
            });
        }
        if marked && all_reachable && !targets.is_empty() {
            findings.push(Finding {
                line: ln,
                subject: targets.join(", "),
                claim: format!("carries the marker `{NOT_IN_A_CLONE_MARKER}`"),
                state: "REACHABLE IN A CLONE -- the warning is stale and must be removed"
                    .to_string(),
            });
        }
    }

    assert!(
        findings.is_empty(),
        "{}",
        report(
            "README.md links a reader to paths a fresh clone does not have (or warns about paths it does have).",
            &findings,
            &cref
        )
    );
}

#[test]
fn readme_trackedness_sentences_are_true_against_the_clone_ref() {
    let (cref, trees, lines) = setup();
    let tops = top_level_names(&trees);
    let mut findings = Vec::new();
    let mut scanned = 0usize;

    for (idx, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        let negative = contains_any(&lower, NEGATIVE_PHRASES);
        let positive = contains_any(&lower, POSITIVE_PHRASES);
        if !positive && !negative {
            continue;
        }
        let paths = path_tokens(line, &tops);
        if paths.is_empty() {
            continue;
        }
        scanned += 1;
        // A line carrying ANY negative phrase is never read as a positive
        // assertion ("the TRACKED figure for checkpoints/ is zero" is a denial).
        //
        // The negative direction is asserted only for the canonical marker, and
        // only as "at least one path on this line is unreachable". A line-wide
        // "every path here is unreachable" rule fires on the reachable paths a
        // warning sentence legitimately names -- `src/checkpoint.rs` in the row
        // that warns about `evidence/r9-headline/` -- which would be a guard
        // failing on true prose.
        if lower.contains(NOT_IN_A_CLONE_MARKER) {
            let any_unreachable = paths.iter().any(|p| classify(&trees, p) != State::InClone);
            if !any_unreachable {
                findings.push(Finding {
                    line: idx + 1,
                    subject: paths.join(", "),
                    claim: format!("carries the marker `{NOT_IN_A_CLONE_MARKER}`"),
                    state: "every path named on this line is REACHABLE IN A CLONE -- the \
                            warning is stale and must be removed"
                        .to_string(),
                });
            }
            continue;
        }
        if negative {
            continue;
        }
        for p in paths {
            let state = classify(&trees, &p);
            if state != State::InClone {
                findings.push(Finding {
                    line: idx + 1,
                    subject: p,
                    claim: "asserts the path IS tracked / reaches a clone".to_string(),
                    state: state.describe(),
                });
            }
        }
    }

    println!("trackedness sentences scanned: {scanned}");
    assert!(
        findings.is_empty(),
        "{}",
        report(
            "README.md states a trackedness the clone ref contradicts.",
            &findings,
            &cref
        )
    );
}

#[test]
fn readme_never_points_a_reader_at_a_commit_absent_from_the_clone() {
    let (cref, _trees, lines) = setup();
    let mut findings = Vec::new();
    let mut checked = 0usize;

    for (idx, line) in lines.iter().enumerate() {
        for token in hex_tokens(line) {
            let rev = git(&[
                "rev-parse",
                "--quiet",
                "--verify",
                &format!("{token}^{{commit}}"),
            ]);
            if !rev.ok {
                continue; // not a commit in this repository: a blob digest, a hash in prose
            }
            let sha = rev.stdout.trim().to_string();
            checked += 1;
            let anc = git(&["merge-base", "--is-ancestor", &sha, &cref.sha]);
            if !anc.ok {
                findings.push(Finding {
                    line: idx + 1,
                    subject: format!("{token} ({})", &sha[..12.min(sha.len())]),
                    claim: "named as a commit a reader can reach".to_string(),
                    state: format!(
                        "NOT AN ANCESTOR OF THE CLONE REF {} -- no clone of {}/{} can check it out",
                        &cref.sha[..12.min(cref.sha.len())],
                        cref.remote,
                        cref.branch
                    ),
                });
            }
        }
    }

    println!("commit-ish tokens checked: {checked}");
    assert!(
        findings.is_empty(),
        "{}",
        report(
            "README.md names a commit that is not in the history a clone receives.",
            &findings,
            &cref
        )
    );
}

/// Hex tokens of 7..=40 characters standing alone (a 64-character digest has no
/// word boundary inside it and is therefore not a candidate).
fn hex_tokens(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let is_hex = |c: char| c.is_ascii_hexdigit() && !c.is_ascii_uppercase();
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if is_hex(chars[i]) && (i == 0 || !is_word(chars[i - 1])) {
            let start = i;
            while i < chars.len() && is_hex(chars[i]) {
                i += 1;
            }
            let end = i;
            let after_ok = end >= chars.len() || !is_word(chars[end]);
            let len = end - start;
            if after_ok && (7..=40).contains(&len) {
                let tok: String = chars[start..end].iter().collect();
                if tok.chars().any(|c| c.is_ascii_digit()) && tok.chars().any(|c| c.is_alphabetic())
                {
                    out.push(tok);
                }
            }
        } else {
            i += 1;
        }
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn the_scan_population_is_not_empty() {
    let (_cref, trees, lines) = setup();
    let links = relative_links(&lines);
    assert!(
        links.len() >= 20,
        "the link parser found only {} relative links in README.md; it has gone blind and \
         every other assertion in this file is vacuous",
        links.len()
    );
    let tops = top_level_names(&trees);
    let sentences = lines
        .iter()
        .filter(|l| {
            let lower = l.to_lowercase();
            (contains_any(&lower, POSITIVE_PHRASES) || contains_any(&lower, NEGATIVE_PHRASES))
                && !path_tokens(l, &tops).is_empty()
        })
        .count();
    assert!(
        sentences >= 3,
        "only {sentences} path-bearing trackedness sentences were found in README.md; the \
         phrase list has stopped matching the prose and the sentence guard is vacuous"
    );
    let evidence_in_clone = trees
        .clone
        .iter()
        .filter(|p| p.starts_with("evidence/"))
        .count();
    assert!(
        evidence_in_clone > 0,
        "the clone ref carries no evidence/ files at all; the classification has no population"
    );
    println!(
        "population: {} relative links, {} trackedness sentences, {} evidence/ files in the clone ref",
        links.len(),
        sentences,
        evidence_in_clone
    );
    let _ = Path::new("");
}
