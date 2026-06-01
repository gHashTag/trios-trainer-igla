//! F2 adapter — encode/decode F2 protocol params for scarab integration.
//!
//! Loop 16 QQ. Offline-only; provides struct + canon-name helpers so F2 experiments
//! can be expressed as scarab_strategy + f2_strategy rows (see migrations/0008).
//!
//! No database connection required at this layer; pure data types.

use crate::race::format_ladder::LadderKind;
use crate::race::multi_seed::{MultiSeedConfig, TaskKind};

/// F2 protocol params persisted in `public.f2_strategy` (one row per scarab).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct F2StrategyRow {
    pub service_id: String,
    pub arm: F2Arm,
    pub precision_bits: f64,
    pub quantizer: F2Quantizer,
    pub task_kind: F2TaskKind,
    pub task_params: serde_json::Value,
    pub use_ffn: bool,
    pub d_hidden: usize,
    pub iso_neff_target_n: Option<u64>,
    pub config_fingerprint: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum F2Arm {
    Phi,
    Zoo,
    Fp32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum F2Quantizer {
    ParetoQSeq,
    ParetoQLsq,
    Int4Rtn,
    Bf16E4m3,
    Fp32Baseline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum F2TaskKind {
    Counter,
    SparseParity,
    BytesFile,
}

impl F2StrategyRow {
    /// Derive the appropriate quantizer for an (arm, precision_bits) pair.
    pub fn pick_quantizer(arm: F2Arm, p_w: f64) -> F2Quantizer {
        match arm {
            F2Arm::Fp32 => F2Quantizer::Fp32Baseline,
            F2Arm::Phi => {
                if p_w <= 2.5 {
                    F2Quantizer::ParetoQSeq
                } else {
                    F2Quantizer::ParetoQLsq
                }
            }
            F2Arm::Zoo => {
                if p_w <= 4.5 {
                    F2Quantizer::Int4Rtn
                } else {
                    F2Quantizer::Bf16E4m3
                }
            }
        }
    }

    /// Convert this row into a MultiSeedConfig for local sandbox runs.
    /// Uses default seeds [42..46] — production scarab supplies its own seed via scarab_strategy.
    pub fn to_multiseed_config(&self) -> MultiSeedConfig {
        let task_kind = match self.task_kind {
            F2TaskKind::Counter => TaskKind::Counter,
            F2TaskKind::SparseParity => {
                let n_bits = self
                    .task_params
                    .get("n_bits")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(40) as usize;
                let k = self
                    .task_params
                    .get("k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3) as usize;
                let n_tasks = self
                    .task_params
                    .get("n_tasks")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(64) as usize;
                TaskKind::SparseParity { n_bits, k, n_tasks }
            }
            F2TaskKind::BytesFile => TaskKind::Counter, // corpus drives instead of task
        };
        let ladder_kind = match self.arm {
            F2Arm::Phi | F2Arm::Fp32 => LadderKind::PhiLadder,
            F2Arm::Zoo => LadderKind::FormatZoo,
        };
        MultiSeedConfig {
            seeds: vec![42, 43, 44, 45, 46],
            train_ratio: 0.8,
            vocab_size: 64,
            d_model: 128,
            steps: 200,
            lr: 0.004,
            ladder_kind,
            warmup_steps_unquantized: 20,
            spike_injection_steps: Vec::new(),
            iso_neff_n_target: self.iso_neff_target_n,
            corpus: crate::race::multi_seed::CorpusKind::Synthetic,
            paretoq_precision: if matches!(self.arm, F2Arm::Fp32) {
                None
            } else {
                Some(self.precision_bits)
            },
            disable_quantization: matches!(self.arm, F2Arm::Fp32),
            task_kind,
            use_ffn: self.use_ffn,
            d_hidden: self.d_hidden,
            label_smoothing: 0.0,
            weight_decay: 0.1,
            apply_rmsnorm: true,
            grad_clip_l2: Some(1.0),
            latent_clamp_max: Some(1.0),
            dropout_p: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_quantizer_dispatches_by_arm_and_precision() {
        assert_eq!(
            F2StrategyRow::pick_quantizer(F2Arm::Fp32, 32.0),
            F2Quantizer::Fp32Baseline
        );
        assert_eq!(
            F2StrategyRow::pick_quantizer(F2Arm::Phi, 1.58),
            F2Quantizer::ParetoQSeq
        );
        assert_eq!(
            F2StrategyRow::pick_quantizer(F2Arm::Phi, 4.0),
            F2Quantizer::ParetoQLsq
        );
        assert_eq!(
            F2StrategyRow::pick_quantizer(F2Arm::Zoo, 4.0),
            F2Quantizer::Int4Rtn
        );
        assert_eq!(
            F2StrategyRow::pick_quantizer(F2Arm::Zoo, 8.0),
            F2Quantizer::Bf16E4m3
        );
    }

    #[test]
    fn to_multiseed_config_fp32_disables_quantization() {
        let row = F2StrategyRow {
            service_id: "scarab-test".to_string(),
            arm: F2Arm::Fp32,
            precision_bits: 32.0,
            quantizer: F2Quantizer::Fp32Baseline,
            task_kind: F2TaskKind::SparseParity,
            task_params: serde_json::json!({"n_bits": 40, "k": 3, "n_tasks": 64}),
            use_ffn: true,
            d_hidden: 64,
            iso_neff_target_n: None,
            config_fingerprint: 0,
        };
        let cfg = row.to_multiseed_config();
        assert!(cfg.disable_quantization);
        assert!(cfg.paretoq_precision.is_none());
        assert!(cfg.use_ffn);
    }

    #[test]
    fn to_multiseed_config_phi_reads_task_params() {
        let row = F2StrategyRow {
            service_id: "scarab-phi-158".to_string(),
            arm: F2Arm::Phi,
            precision_bits: 1.58,
            quantizer: F2Quantizer::ParetoQSeq,
            task_kind: F2TaskKind::SparseParity,
            task_params: serde_json::json!({"n_bits": 50, "k": 5, "n_tasks": 128}),
            use_ffn: true,
            d_hidden: 96,
            iso_neff_target_n: Some(41152),
            config_fingerprint: 0xdeadbeef,
        };
        let cfg = row.to_multiseed_config();
        assert!(matches!(
            cfg.task_kind,
            TaskKind::SparseParity {
                n_bits: 50,
                k: 5,
                n_tasks: 128
            }
        ));
        assert_eq!(cfg.d_hidden, 96);
        assert_eq!(cfg.iso_neff_n_target, Some(41152));
        assert_eq!(cfg.paretoq_precision, Some(1.58));
    }

    #[test]
    fn round_trip_serde_json() {
        let row = F2StrategyRow {
            service_id: "scarab-x".to_string(),
            arm: F2Arm::Phi,
            precision_bits: 2.0,
            quantizer: F2Quantizer::ParetoQSeq,
            task_kind: F2TaskKind::SparseParity,
            task_params: serde_json::json!({}),
            use_ffn: true,
            d_hidden: 64,
            iso_neff_target_n: None,
            config_fingerprint: 0,
        };
        let json = serde_json::to_string(&row).unwrap();
        let back: F2StrategyRow = serde_json::from_str(&json).unwrap();
        assert_eq!(row, back);
    }
}
