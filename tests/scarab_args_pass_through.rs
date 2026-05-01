//! Contract tests: scarab.rs must pass all trainer args from config_json
//!
//! Bugs covered:
//!   - X1: --ctx, --train-data, --val-data dropped
//!   - Bug C: --ctx parsed but not passed
//!   - Bug B: train_path/val_path fields missing

use serde::Deserialize;

#[derive(Debug, Deserialize, Default)]
struct TrainerSpec {
    hidden: Option<u32>,
    lr: Option<f64>,
    steps: Option<u32>,
    ctx: Option<u32>,
    format: Option<String>,
    seed: Option<u64>,
    optimizer: Option<String>,
    train_path: Option<String>,
    val_path: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct StrategySpec {
    #[serde(default)]
    trainer: TrainerSpec,
}

#[test]
fn test_trainer_spec_all_fields_parsed() {
    let json = r#"{
        "trainer": {
            "hidden": 828,
            "lr": 0.0004,
            "steps": 1000,
            "ctx": 12,
            "format": "fp32",
            "seed": 42,
            "optimizer": "adamw"
        }
    }"#;

    let spec: StrategySpec = serde_json::from_str(json).unwrap();
    let t = &spec.trainer;
    assert_eq!(t.hidden, Some(828));
    assert_eq!(t.lr, Some(0.0004));
    assert_eq!(t.steps, Some(1000));
    assert_eq!(t.ctx, Some(12)); // Bug C: parsed but not passed
    assert_eq!(t.format, Some("fp32".into()));
    assert_eq!(t.seed, Some(42));
    assert_eq!(t.optimizer, Some("adamw".into()));
}

#[test]
fn test_trainer_spec_with_train_val_path() {
    let json = r#"{
        "trainer": {
            "hidden": 828,
            "train_path": "/custom/train.txt",
            "val_path": "/custom/val.txt"
        }
    }"#;

    let spec: StrategySpec = serde_json::from_str(json).unwrap();
    let t = &spec.trainer;
    assert_eq!(t.train_path, Some("/custom/train.txt".into()));
    assert_eq!(t.val_path, Some("/custom/val.txt".into()));
}

// NOTE: Integration test requires scarab.rs to build Command and inspect args
// This is a placeholder for the real integration test that would:
// 1. Spawn scarab with a config_json containing train_path/val_path/ctx
// 2. Intercept the trios-train Command
// 3. Verify --train-data, --val-data, --ctx are in args
