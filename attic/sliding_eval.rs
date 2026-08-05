//! Sliding Eval with Stride=64
//!
//! P11: Sliding eval stride=64
//! Expected ΔBPB: −0.03
//!
//! Reduces evaluation compute by skipping positions while maintaining accuracy.

pub struct SlidingEvalConfig {
    pub stride: usize,
}

impl Default for SlidingEvalConfig {
    fn default() -> Self {
        Self { stride: 64 }
    }
}

impl SlidingEvalConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stride_64() -> Self {
        Self { stride: 64 }
    }

    /// Get eval positions with stride
    pub fn eval_positions(&self, seq_len: usize) -> Vec<usize> {
        (0..seq_len).step_by(self.stride).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_stride() {
        let config = SlidingEvalConfig::default();
        assert_eq!(config.stride, 64);
    }

    #[test]
    fn test_eval_positions() {
        let config = SlidingEvalConfig::stride_64();
        let positions = config.eval_positions(200);
        assert_eq!(positions, vec![0, 64, 128, 192]);
    }

    #[test]
    fn test_eval_positions_short() {
        let config = SlidingEvalConfig::stride_64();
        let positions = config.eval_positions(100);
        assert_eq!(positions, vec![0, 64]);
    }
}
