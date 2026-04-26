//! Falsification witness: embargo enforcement
//!
//! Per the standing rule (R9 + Risk Register row 4), `ledger::emit_row`
//! must `bail!` if the SHA is in `assertions/embargo.txt`. This test
//! synthesises a row and asserts every embargoed SHA is refused.

use std::io::Write;
use tempfile::tempdir;
use trios_trainer::config::{
    DataConfig, LedgerConfig, ModelConfig, ObjectiveConfig, OptimizerConfig, TrainConfig,
};
use trios_trainer::ledger::{emit_row_with_sha, is_embargoed};

const EMBARGOED_SHAS: &[&str] = &[
    "477e3377", "b3ee6a36", "2f6e4c2", "4a158c01", "6393be94", "5950174", "32d1dd3", "a7574c3",
];

fn synth_cfg(jsonl: &str, embargo: &str) -> TrainConfig {
    TrainConfig {
        name: "embargo-test".into(),
        steps: 4000,
        seed: 43,
        target_bpb: 1.85,
        champion_bpb: Some(2.2393),
        model: ModelConfig {
            d_model: 64,
            n_layers: 1,
            n_heads: 1,
            vocab_size: 256,
            seq_len: 32,
            hybrid_attn: false,
        },
        optimizer: OptimizerConfig {
            kind: "adamw".into(),
            lr: 0.004,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.0,
            schedule: "phi".into(),
            warmup_steps: 0,
        },
        data: DataConfig {
            corpus: "tinyshakespeare".into(),
            batch_size: 4,
            batch_tokens: 64,
        },
        objective: ObjectiveConfig {
            w_ce: 1.0,
            w_jepa: 0.0,
            w_nca: 0.0,
        },
        ledger: LedgerConfig {
            jsonl_path: jsonl.into(),
            push: false,
            embargo_path: embargo.into(),
        },
    }
}

fn write_embargo(path: &std::path::Path) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "# Embargo test list").unwrap();
    for sha in EMBARGOED_SHAS {
        writeln!(f, "{sha}").unwrap();
    }
}

#[test]
fn every_embargoed_sha_is_refused() {
    let dir = tempdir().unwrap();
    let embargo = dir.path().join("embargo.txt");
    let jsonl = dir.path().join("seed_results.jsonl");
    write_embargo(&embargo);

    let cfg = synth_cfg(jsonl.to_str().unwrap(), embargo.to_str().unwrap());
    for sha in EMBARGOED_SHAS {
        let res = emit_row_with_sha(&cfg, 1.50, 4000, sha);
        assert!(
            res.is_err(),
            "embargoed SHA {sha} was NOT refused — embargo bypass detected"
        );
        let msg = format!("{}", res.unwrap_err());
        assert!(
            msg.to_lowercase().contains("embargo"),
            "error for {sha} did not mention embargo: {msg}"
        );
    }
}

#[test]
fn non_embargoed_sha_passes_embargo_check() {
    let dir = tempdir().unwrap();
    let embargo = dir.path().join("embargo.txt");
    write_embargo(&embargo);
    assert!(!is_embargoed(&embargo, "deadbeef").unwrap());
    assert!(is_embargoed(&embargo, "477e3377").unwrap());
}

#[test]
fn embargo_check_below_4000_steps_bails() {
    let dir = tempdir().unwrap();
    let embargo = dir.path().join("embargo.txt");
    let jsonl = dir.path().join("seed_results.jsonl");
    write_embargo(&embargo);
    let cfg = synth_cfg(jsonl.to_str().unwrap(), embargo.to_str().unwrap());
    // Below 4000 should bail per R8 even with non-embargoed SHA.
    let res = emit_row_with_sha(&cfg, 1.50, 100, "deadbeef");
    assert!(res.is_err());
    assert!(format!("{}", res.unwrap_err()).contains("R8"));
}
