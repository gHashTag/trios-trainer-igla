//! Scarab Contract Tests
//!
//! These tests verify critical contracts between scarab and the rest of the system.
//! Failures here indicate regressions that would break production.

use std::process::Command;

#[test]
/// CONTRACT: scarab MUST use rustls TLS, not NoTls
/// Bug A regression check — NoTls cannot connect to Neon
fn scarab_uses_rustls_tls() {
    let output = Command::new("grep")
        .args([
            "tokio_postgres_rustls::MakeRustlsConnect",
            "src/bin/scarab.rs",
        ])
        .output()
        .expect("grep should run");

    let stdout = String::from_utf8_lossy(&output.stdout);

    if !stdout.contains("MakeRustlsConnect") {
        panic!(
            "CRITICAL: scarab.rs does NOT use MakeRustlsConnect! \
             Bug A regression — NoTls cannot connect to Neon.\n\
             Expected: 'use tokio_postgres_rustls::MakeRustlsConnect;' in scarab.rs"
        );
    }

    // Also verify NoTls is NOT used
    let no_tls_output = Command::new("grep")
        .args(["tokio_postgres::NoTls", "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if String::from_utf8_lossy(&no_tls_output.stdout).contains("NoTls") {
        panic!(
            "CRITICAL: scarab.rs uses NoTls! \
             Bug A regression — NoTls cannot connect to Neon.\n\
             Found: 'use tokio_postgres::NoTls;' in scarab.rs"
        );
    }
}

#[test]
/// CONTRACT: TrainerSpec MUST have train_path and val_path fields
/// Bug B regression check — without these, custom corpus paths are ignored
fn trainer_spec_has_corpus_paths() {
    let output = Command::new("grep")
        .args(["train_path: Option<String>", "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("train_path") {
        panic!(
            "CRITICAL: TrainerSpec missing train_path field! \
             Bug B regression — custom corpus paths will be ignored."
        );
    }

    let output = Command::new("grep")
        .args(["val_path: Option<String>", "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("val_path") {
        panic!(
            "CRITICAL: TrainerSpec missing val_path field! \
             Bug B regression — custom corpus paths will be ignored."
        );
    }
}

#[test]
/// CONTRACT: scarab MUST pass --train-data, --val-data, --ctx to trios-train
/// Bug C regression check — without these, trainer receives hardcoded defaults
fn scarab_passes_corpus_args() {
    let output = Command::new("grep")
        .args(["--train-data", "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("--train-data") {
        panic!(
            "CRITICAL: scarab does NOT pass --train-data to trios-train! \
             Bug C regression — trainer cannot receive custom corpus."
        );
    }

    let output = Command::new("grep")
        .args(["--val-data", "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("--val-data") {
        panic!(
            "CRITICAL: scarab does NOT pass --val-data to trios-train! \
             Bug C regression — trainer cannot receive custom corpus."
        );
    }

    let output = Command::new("grep")
        .args(["--ctx", "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("--ctx") {
        panic!(
            "CRITICAL: scarab does NOT pass --ctx to trios-train! \
             Bug C regression — trainer cannot receive custom context size."
        );
    }
}

#[test]
/// CONTRACT: scarab MUST query bpb_samples and write final_bpb/final_step
/// Bug E regression check — without this, final metrics are lost
fn scarab_writes_final_metrics() {
    let output = Command::new("grep")
        .args(["final_bpb.*=.*\\$", "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("final_bpb") {
        panic!(
            "CRITICAL: scarab does NOT write final_bpb to strategy_queue! \
             Bug E regression — final metrics will be NULL in DB."
        );
    }

    let output = Command::new("grep")
        .args(["final_step.*=.*\\$", "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("final_step") {
        panic!(
            "CRITICAL: scarab does NOT write final_step to strategy_queue! \
             Bug E regression — final metrics will be NULL in DB."
        );
    }

    // Verify UPDATE statement includes final_bpb and final_step
    let output = Command::new("grep")
        .args([
            "UPDATE strategy_queue.*final_bpb.*final_step",
            "src/bin/scarab.rs",
        ])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("final_bpb") {
        panic!(
            "CRITICAL: UPDATE strategy_queue does NOT include final_bpb/final_step! \
             Bug E regression — metrics not persisted."
        );
    }
}

#[test]
/// CONTRACT: scarab MUST query bpb_samples table after trainer completes
fn scarab_queries_bpb_samples() {
    let output = Command::new("grep")
        .args([
            "SELECT.*FROM bpb_samples.*ORDER BY ts DESC LIMIT 1",
            "src/bin/scarab.rs",
        ])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("bpb_samples") {
        panic!(
            "CRITICAL: scarab does NOT query bpb_samples table! \
             Bug E regression — cannot read final metrics."
        );
    }
}

#[test]
/// CONTRACT: scarab MUST export NEON_DATABASE_URL to trainer subprocess
/// Bug A part 2 — trainer needs env var for bpb_sample writes
fn scarab_exports_neon_dsn() {
    let output = Command::new("grep")
        .args([r#".env\("NEON_DATABASE_URL""#, "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("NEON_DATABASE_URL") {
        panic!(
            "CRITICAL: scarab does NOT export NEON_DATABASE_URL to trainer! \
             Bug A part 2 — trainer cannot write bpb_samples."
        );
    }
}

#[test]
/// CONTRACT: scarab MUST export TRIOS_CANON_NAME to trainer subprocess
/// Required for bpb_sample writes to use correct canon_name
fn scarab_exports_canon_name() {
    let output = Command::new("grep")
        .args([r#".env\("TRIOS_CANON_NAME""#, "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("TRIOS_CANON_NAME") {
        panic!(
            "CRITICAL: scarab does NOT export TRIOS_CANON_NAME to trainer! \
             bpb_sample writes will use incorrect canon_name."
        );
    }
}

#[test]
/// CONTRACT: scarab MUST use FOR UPDATE SKIP LOCKED for claim
/// Prevents race conditions when multiple scarabs claim same strategy
fn scarab_uses_skip_locked() {
    let output = Command::new("grep")
        .args(["FOR UPDATE SKIP LOCKED", "src/bin/scarab.rs"])
        .output()
        .expect("grep should run");

    if !String::from_utf8_lossy(&output.stdout).contains("SKIP LOCKED") {
        panic!(
            "CRITICAL: scarab does NOT use SKIP LOCKED for claim! \
             Race condition risk — multiple scarabs could claim same strategy."
        );
    }
}

#[test]
/// CONTRACT: scarab MUST NOT filter claim by account
/// Fungible pool requirement — any scarab takes any pending task
fn scarab_claim_is_account_free() {
    // Verify claim query does NOT have "AND account = $1"
    let output = Command::new("grep")
        .args([
            "UPDATE strategy_queue.*WHERE.*FOR UPDATE",
            "src/bin/scarab.rs",
        ])
        .output()
        .expect("grep should run");

    let query = String::from_utf8_lossy(&output.stdout);

    if query.contains("account") {
        panic!(
            "CRITICAL: scarab claim query filters by account! \
             Fungible pool broken — scarabs cannot take arbitrary tasks."
        );
    }
}

#[test]
/// CONTRACT: Migration 0003 MUST exist for final_bpb/final_step columns
fn migration_0003_exists() {
    let output = Command::new("cat")
        .arg("migrations/0003_strategy_queue_final_metrics.sql")
        .output()
        .expect("cat should run");

    let content = String::from_utf8_lossy(&output.stdout);

    if !content.contains("ADD COLUMN IF NOT EXISTS final_bpb") {
        panic!(
            "CRITICAL: Migration 0003 missing or malformed! \
             final_bpb column not added to strategy_queue."
        );
    }

    if !content.contains("ADD COLUMN IF NOT EXISTS final_step") {
        panic!(
            "CRITICAL: Migration 0003 missing or malformed! \
             final_step column not added to strategy_queue."
        );
    }
}
