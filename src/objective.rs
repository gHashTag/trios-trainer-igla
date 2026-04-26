//! Objective stub with build() function (L-T1)
//!
//! This file provides the `build()` function for objective construction.
//! During L-T1, we provide a stub to allow compilation.

/// Build objective from configuration
///
/// Placeholder for L-T1 - returns default cross-entropy loss.
pub fn build(_cfg: &str) -> Box<dyn Fn(&[f32], &[usize]) -> f64> {
    // TODO: Parse config and construct appropriate objective
    Box::new(|_logits, _targets| cross_entropy_loss(_logits, _targets))
}

/// Combined objective function placeholder
///
/// For L-T1, this is a simple cross-entropy loss.
pub fn combined_loss(logits: &[f32], targets: &[usize]) -> f64 {
    cross_entropy_loss(logits, targets)
}

/// Simple cross-entropy loss
///
/// Placeholder for L-T1 - calculates cross-entropy loss.
pub fn cross_entropy_loss(logits: &[f32], targets: &[usize]) -> f64 {
    if logits.is_empty() || targets.is_empty() {
        return 0.0;
    }
    let vocab_size = logits.len() / targets.len();
    let mut total_loss = 0.0f32;

    for (batch, &target) in targets.iter().enumerate() {
        let offset = batch * vocab_size;

        // Find max for numerical stability
        let max_logit = logits[offset..offset + vocab_size]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute log-softmax for target
        let mut sum_exp = 0.0f32;
        for v in 0..vocab_size {
            sum_exp += (logits[offset + v] - max_logit).exp();
        }

        let log_prob = logits[offset + target] - max_logit - sum_exp.ln();
        total_loss -= log_prob;
    }

    total_loss / targets.len() as f64
}
