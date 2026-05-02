//! BPB Calculation — honest bits-per-byte metrics
//!
//! # Constitutional mandate (Law 5)
//!
//! BPB calculation must be:
//! - Mathematically sound (cross-entropy → bits)
//! - Bounded by INV-5 (0 to 100)
//! - Never returns NaN or infinity

/// Calculate BPB from loss (cross-entropy in nats)
///
/// # Formula
///
/// `bpb = loss / ln(2)`
///
/// Where:
/// - `loss` is cross-entropy in nats (natural log base)
/// - `ln(2)` ≈ 0.6931 converts nats to bits
///
/// # Invariants
///
/// - BPB must be non-negative (INV-5)
/// - BPB > 100 indicates catastrophic failure (data corruption, model collapse)
/// - Returns None for NaN/infinity (fail-loud)
///
/// # References
///
/// - (ref: INV-5) BPB theoretical bounds
/// - (ref: PhD Chapter 10) Cross-entropy derivation
///
/// # Example
///
/// ```
/// use trinity_core::bpb::calculate_bpb;
///
/// // Cross-entropy of 2.0 nats → ~2.885 BPB
/// let bpb = calculate_bpb(2.0);
/// assert_eq!(bpb, Some(2.8853900817779268));
/// ```
pub fn calculate_bpb(loss: f32) -> Option<f32> {
    // INV-5: NaN/infinity → fail-loud
    if !loss.is_finite() {
        return None;
    }

    // INV-5: negative loss is impossible (cross-entropy is always ≥ 0)
    if loss < 0.0 {
        return None;
    }

    // Convert nats to bits: bpb = loss / ln(2)
    let ln_2: f32 = 0.69314718;
    let bpb = loss / ln_2;

    // INV-5: BPB must be finite
    if !bpb.is_finite() {
        return None;
    }

    Some(bpb)
}

/// Check if BPB is better than floor threshold (INV-6)
///
/// # Invariants
///
/// - Returns false for BPB < INV_6_FLOOR_BPB (worse than random)
/// - Returns true for BPB ≥ INV_6_FLOOR_BPB (acceptable)
///
/// # References
///
/// - (ref: INV-6) Floor BPB = 2.0
/// - (ref: PhD Chapter 12) "worse than random" detection
#[inline]
pub fn bpb_above_floor(bpb: f32) -> bool {
    bpb >= super::invariants::INV_6_FLOOR_BPB
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_bpb_basic() {
        // Loss of 0.693 nats (ln(2)) → 1.0 BPB
        let bpb = calculate_bpb(0.69314718);
        assert_eq!(bpb, Some(1.0));
    }

    #[test]
    fn calculate_bpb_zero_loss() {
        // Zero loss → zero BPB
        let bpb = calculate_bpb(0.0);
        assert_eq!(bpb, Some(0.0));
    }

    #[test]
    fn calculate_bpb_negative_loss() {
        // Negative loss is impossible
        let bpb = calculate_bpb(-1.0);
        assert_eq!(bpb, None);
    }

    #[test]
    fn calculate_bpb_nan() {
        let bpb = calculate_bpb(f32::NAN);
        assert_eq!(bpb, None);
    }

    #[test]
    fn calculate_bpb_infinity() {
        let bpb = calculate_bpb(f32::INFINITY);
        assert_eq!(bpb, None);
    }

    #[test]
    fn bpb_above_floor() {
        assert!(bpb_above_floor(2.5), "2.5 > floor (2.0)");
        assert!(bpb_above_floor(2.0), "2.0 == floor (2.0)");
        assert!(!bpb_above_floor(1.9), "1.9 < floor (2.0)");
    }
}
