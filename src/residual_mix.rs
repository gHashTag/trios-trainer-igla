//! Residual Mix Ratio Sweep
//!
//! P07: Residual Mix ratio sweep [0.4, 0.5, 0.618, 0.75]
//! Expected ΔBPB: −0.01

const RATIOS: [f32; 4] = [0.4, 0.5, 0.618, 0.75];

pub struct ResidualMixConfig {
    pub ratio: f32,
}

impl Default for ResidualMixConfig {
    fn default() -> Self {
        Self { ratio: 0.618 }  // 1/φ
    }
}

impl ResidualMixConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sweep() -> Vec<ResidualMixConfig> {
        RATIOS.iter().map(|&ratio| ResidualMixConfig { ratio }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_ratio() {
        let config = ResidualMixConfig::new();
        assert!((config.ratio - 0.618).abs() < 1e-6);
    }

    #[test]
    fn test_sweep_count() {
        let configs = ResidualMixConfig::sweep();
        assert_eq!(configs.len(), 4);
    }
}
