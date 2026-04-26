//! L-T* falsification witness: EMBARGO BYPASS is impossible.
//!
//! The hypothesis (Gate G1) is **falsified iff** any embargoed SHA is accepted
//! by `ledger::emit_row`. This test asserts the rule for every entry in
//! `assertions/embargo.txt` and for the 8 canonical SHAs from issue #2.
//!
//! Mission: TRAINER-IGLA-SOT (`phi^2 + phi^-2 = 3`).
//! Refs: gHashTag/trios-trainer-igla#2 (ONE SHOT) — R9 standing rule.

use std::io::Write;
use trios_trainer::config::{
    DataConfig, LedgerConfig, ModelConfig, ObjectiveConfig, OptimizerConfig, TrainConfig,
};
use trios_trainer::ledger::{emit_row_with_sha, is_embargoed};

/// 8 canonical SHAs published by the embargo standing rule (issue #2).
const CANONICAL_EMBARGO: &[&str] = &[
    "477e3377", "b3ee6a36", "2f6e4c2", "4a158c01", "6393be94", "5950174", "32d1dd3", "a7574c3",
];

fn cfg_for(jsonl: &str, embargo: &str) -> TrainConfig {
    TrainConfig {
        name: "embargo-test".to_string(),
        steps: 5000,
        seed: 43,
        target_bpb: 1.85,
        champion_bpb: Some(2.2393),
        model: ModelConfig {
            d_model: 64,
            n_layers: 1,
            n_heads: 1,
            vocab_size: 128,
            seq_len: 64,
            hybrid_attn: false,
        },
        optimizer: OptimizerConfig {
            kind: "adamw".to_string(),
            lr: 0.004,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            schedule: "phi".to_string(),
            warmup_steps: 0,
        },
        data: DataConfig {
            corpus: "tinyshakespeare".to_string(),
            batch_size: 1,
            batch_tokens: 64,
        },
        objective: ObjectiveConfig {
            w_ce: 1.0,
            w_jepa: 0.0,
            w_nca: 0.0,
        },
        ledger: LedgerConfig {
            jsonl_path: jsonl.to_string(),
            push: false,
            embargo_path: embargo.to_string(),
        },
    }
}

#[test]
fn embargo_file_blocks_every_canonical_sha() {
    // Walk the canonical embargo file shipped in the repo.
    let path = "assertions/embargo.txt";
    for sha in CANONICAL_EMBARGO {
        let blocked = is_embargoed(path, sha).unwrap_or_else(|e| {
            panic!("is_embargoed({sha}) failed: {e}");
        });
        assert!(
            blocked,
            "R9 violation: SHA {sha} should be embargoed by {path}"
        );
    }
}

#[test]
fn emit_row_bails_on_embargoed_sha() {
    // Synthesise a row with an embargoed SHA and require `bail!`.
    let dir = tempfile::tempdir().expect("tempdir");
    let jsonl = dir.path().join("seed_results.jsonl");
    let embargo = dir.path().join("embargo.txt");
    {
        let mut f = std::fs::File::create(&embargo).unwrap();
        for sha in CANONICAL_EMBARGO {
            writeln!(f, "{sha}").unwrap();
        }
    }
    let cfg = cfg_for(jsonl.to_str().unwrap(), embargo.to_str().unwrap());
    let result = emit_row_with_sha(&cfg, 1.50, 27000, "477e3377");
    assert!(
        result.is_err(),
        "EMBARGO BYPASS: emit_row accepted embargoed SHA"
    );
    let err = result.err().unwrap().to_string();
    assert!(err.contains("embargo"), "error must mention embargo: {err}");
    // And the jsonl file must be empty (no row appended)
    let appended = std::fs::read_to_string(&jsonl).unwrap_or_default();
    assert!(
        appended.is_empty(),
        "EMBARGO BYPASS: row got appended despite bail: {appended:?}"
    );
}

#[test]
fn emit_row_accepts_clean_sha_after_step_floor() {
    // Positive control: a non-embargoed SHA at step ≥ 4000 must succeed.
    let dir = tempfile::tempdir().expect("tempdir");
    let jsonl = dir.path().join("seed_results.jsonl");
    let embargo = dir.path().join("embargo.txt");
    std::fs::write(&embargo, "# none\n").unwrap();
    let cfg = cfg_for(jsonl.to_str().unwrap(), embargo.to_str().unwrap());
    let row = emit_row_with_sha(&cfg, 1.50, 4001, "deadbee").expect("clean emit must succeed");
    assert_eq!(row.sha, "deadbee");
    assert_eq!(row.step, 4001);
    assert_eq!(row.gate_status, "victory_candidate");
}

#[test]
fn emit_row_bails_below_step_floor() {
    let dir = tempfile::tempdir().expect("tempdir");
    let jsonl = dir.path().join("seed_results.jsonl");
    let embargo = dir.path().join("embargo.txt");
    std::fs::write(&embargo, "# none\n").unwrap();
    let cfg = cfg_for(jsonl.to_str().unwrap(), embargo.to_str().unwrap());
    let result = emit_row_with_sha(&cfg, 1.50, 3999, "deadbee");
    assert!(
        result.is_err(),
        "R8 violation: step 3999 row should bail, got {:?}",
        result
    );
}
