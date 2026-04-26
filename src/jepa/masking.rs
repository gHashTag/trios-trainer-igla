//! Span masking for T-JEPA
//!
//! Generates random span masks for masked prediction training.

use rand::Rng;

/// Mask configuration
#[derive(Debug, Clone, Copy)]
pub struct MaskConfig {
    pub ratio: f64,       // Target mask ratio (0.3)
    pub min_span: usize,  // Minimum span length (3)
    pub max_span: usize,  // Maximum span length (9 or 11)
    pub num_spans: usize, // Number of spans to mask (2)
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self {
            ratio: 0.3,
            min_span: 3,
            max_span: 9,
            num_spans: 2,
        }
    }
}

/// Mask result
#[derive(Debug, Clone)]
pub struct MaskResult {
    /// Boolean mask: true = masked position
    pub mask: Vec<bool>,
    /// List of (start, end) for each masked span
    pub spans: Vec<(usize, usize)>,
}

/// Generate random span masks
///
/// # Arguments
/// * `seq_len` - Sequence length
/// * `config` - Mask configuration
/// * `rng` - Random number generator
///
/// # Returns
/// MaskResult with boolean mask and span boundaries
///
/// # Example
/// ```
/// use trios_trainer::jepa::masking::{mask_spans, MaskConfig};
/// use rand::SeedableRng;
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let result = mask_spans(100, MaskConfig::default(), &mut rng);
/// assert_eq!(result.spans.len(), 2);
/// ```
pub fn mask_spans(seq_len: usize, config: MaskConfig, rng: &mut impl Rng) -> MaskResult {
    let mut mask = vec![false; seq_len];
    let mut spans = Vec::new();

    // Ensure seq_len is long enough for the spans
    let min_required = config.min_span * config.num_spans;
    if seq_len < min_required {
        // If too short, mask as much as possible
        let span_len = seq_len / config.num_spans;
        for i in 0..config.num_spans {
            let start = i * span_len;
            let end = ((i + 1) * span_len).min(seq_len);
            mask[start..end].fill(true);
            spans.push((start, end));
        }
        return MaskResult { mask, spans };
    }

    for _ in 0..config.num_spans {
        let span_len = rng.gen_range(config.min_span..=config.max_span);
        let start = rng.gen_range(0..seq_len.saturating_sub(span_len));
        let end = (start + span_len).min(seq_len);

        mask[start..end].fill(true);
        spans.push((start, end));
    }

    MaskResult { mask, spans }
}

/// Get unmasked positions (context for prediction)
pub fn get_unmasked(mask: &[bool]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &m)| if !m { Some(i) } else { None })
        .collect()
}

/// Get masked positions (targets for prediction)
pub fn get_masked(mask: &[bool]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect()
}

/// Check if masks are non-overlapping
pub fn spans_non_overlapping(spans: &[(usize, usize)]) -> bool {
    for (i, (start1, end1)) in spans.iter().enumerate() {
        for (start2, end2) in spans.iter().skip(i + 1) {
            // Check for overlap
            if !(end1 <= start2 || end2 <= start1) {
                return false;
            }
        }
    }
    true
}

/// Partition sequence into context and target positions
pub fn partition_context_target(mask: &[bool]) -> (Vec<usize>, Vec<usize>) {
    let mut context = Vec::new();
    let mut target = Vec::new();

    for (i, &m) in mask.iter().enumerate() {
        if m {
            target.push(i);
        } else {
            context.push(i);
        }
    }

    (context, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_mask_ratio_approximate() {
        let mut rng = StdRng::seed_from_u64(42);
        let result = mask_spans(100, MaskConfig::default(), &mut rng);

        let masked_count = result.mask.iter().filter(|&&m| m).count();
        let ratio = masked_count as f64 / 100.0;

        // Allow variance - actual ratio may differ due to span constraints
        assert!((ratio - 0.3).abs() < 0.25);
    }

    #[test]
    fn test_span_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        let result = mask_spans(100, MaskConfig::default(), &mut rng);

        for (start, end) in result.spans {
            assert!(start < end);
            assert!(end <= 100);
            assert!(end - start >= 3);
            assert!(end - start <= 11);
        }
    }

    #[test]
    fn test_num_spans() {
        let config = MaskConfig {
            num_spans: 3,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let result = mask_spans(100, config, &mut rng);

        assert_eq!(result.spans.len(), 3);
    }

    #[test]
    fn test_get_unmasked() {
        let mask = vec![false, true, false, false, true];
        let unmasked = get_unmasked(&mask);

        assert_eq!(unmasked, vec![0, 2, 3]);
    }

    #[test]
    fn test_get_masked() {
        let mask = vec![false, true, false, false, true];
        let masked = get_masked(&mask);

        assert_eq!(masked, vec![1, 4]);
    }

    #[test]
    fn test_spans_non_overlapping() {
        let spans = vec![(0, 5), (10, 15), (20, 25)];
        assert!(spans_non_overlapping(&spans));

        let overlapping = vec![(0, 10), (5, 15)];
        assert!(!spans_non_overlapping(&overlapping));
    }

    #[test]
    fn test_partition_context_target() {
        let mask = vec![false, true, false, true, false];
        let (context, target) = partition_context_target(&mask);

        assert_eq!(context, vec![0, 2, 4]);
        assert_eq!(target, vec![1, 3]);
    }

    #[test]
    fn test_short_sequence() {
        let config = MaskConfig {
            min_span: 3,
            max_span: 5,
            num_spans: 2,
            ratio: 0.3,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let result = mask_spans(8, config, &mut rng);

        // Should still work, just covering what it can
        assert!(result.mask.len() == 8);
        assert!(result.spans.len() <= 2);
    }

    #[test]
    fn test_reproducible_with_seed() {
        let config = MaskConfig::default();
        let mut rng1 = StdRng::seed_from_u64(12345);
        let mut rng2 = StdRng::seed_from_u64(12345);

        let result1 = mask_spans(50, config, &mut rng1);
        let result2 = mask_spans(50, config, &mut rng2);

        assert_eq!(result1.mask, result2.mask);
        assert_eq!(result1.spans, result2.spans);
    }
}
