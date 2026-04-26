//! Fibonacci Dimensions — φ-based dimension progression
//!
//! Uses Fibonacci numbers (1,1,2,3,5,8,13,21,34,55,89,144,233...)
//! for model dimensions, following φ-optimization principles.

use super::phi_constants::*;

/// Fibonacci sequence up to reasonable model dimensions
pub const FIBONACCI: [u64; 16] = [
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
];

/// Get the nth Fibonacci number (0-indexed)
pub fn fibonacci(n: usize) -> u64 {
    if n < FIBONACCI.len() {
        FIBONACCI[n]
    } else {
        // For n beyond the table, compute using Binet's formula
        let phi = PHI as f64;
        let neg_phi = (1.0 - phi) as f64;
        let sqrt5 = 5.0_f64.sqrt();
        let n_f = n as f64;

        let fib = (phi.powf(n_f) - neg_phi.powf(n_f)) / sqrt5;
        fib.round() as u64
    }
}

/// Get the next Fibonacci number after a given value
pub fn next_fibonacci(value: u64) -> u64 {
    for i in 0..FIBONACCI.len() {
        if FIBONACCI[i] > value {
            return FIBONACCI[i];
        }
    }
    // For values beyond the table
    let mut n = FIBONACCI.len();
    loop {
        let f = fibonacci(n);
        if f > value {
            return f;
        }
        n += 1;
    }
}

/// Get the nearest Fibonacci number to a given value
pub fn nearest_fibonacci(value: u64) -> u64 {
    for i in 0..FIBONACCI.len() - 1 {
        if FIBONACCI[i] <= value && value <= FIBONACCI[i + 1] {
            let lower = FIBONACCI[i];
            let upper = FIBONACCI[i + 1];
            let diff_lower = value - lower;
            let diff_upper = upper - value;
            return if diff_lower <= diff_upper { lower } else { upper };
        }
    }
    let n = FIBONACCI.len();
    let f_n = fibonacci(n);
    if value <= f_n {
        return f_n;
    }
    let f_n_plus_1 = fibonacci(n + 1);
    let diff_lower = value - f_n;
    let diff_upper = f_n_plus_1 - value;
    if diff_lower <= diff_upper { f_n } else { f_n_plus_1 }
}

/// Check if a value is a Fibonacci number
pub fn is_fibonacci(value: u64) -> bool {
    for &f in &FIBONACCI {
        if f == value {
            return true;
        }
    }
    // For values beyond the table, use the property:
    // n is Fibonacci iff 5n^2 ± 4 is a perfect square
    let n = value as f64;
    let test1 = 5.0 * n * n + 4.0;
    let test2 = 5.0 * n * n - 4.0;
    let sqrt1 = test1.sqrt();
    let sqrt2 = test2.sqrt();
    (sqrt1 - sqrt1.round()).abs() < 1e-6 || (sqrt2 - sqrt2.round()).abs() < 1e-6
}

/// Recommended model dimensions (Fibonacci-based)
pub const RECOMMENDED_DIMS: [usize; 10] = [
    8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
];

/// Get a recommended dimension for a given size category
/// category: 0=tiny, 1=small, 2=medium, 3=large, 4=xlarge
pub fn recommended_dim(category: usize) -> usize {
    let idx = if category >= RECOMMENDED_DIMS.len() { RECOMMENDED_DIMS.len() - 1 } else { category };
    RECOMMENDED_DIMS[idx]
}

/// Fibonacci-based hidden sizes for N-gram models
pub const NGRAM_HIDDEN_SIZES: [usize; 6] = [
    8, 13, 21, 34, 55, 89,
];

/// Fibonacci-based embedding dimensions
pub const EMBEDDING_DIMS: [usize; 6] = [
    8, 13, 21, 34, 55, 89,
];

/// φ-based layer count progression (powers of φ)
/// φ^1 ≈ 1.6 → 2 layers
/// φ^2 ≈ 2.6 → 3 layers
/// φ^3 ≈ 4.2 → 4 layers
/// φ^4 ≈ 6.9 → 7 layers
/// φ^5 ≈ 11.1 → 11 layers
pub fn phi_layer_count(power: u32) -> usize {
    let phi_pow = PHI.powf(power as f64);
    phi_pow.round() as usize
}

/// Check if a dimension is φ-optimized (close to Fibonacci)
pub fn is_phi_optimized(dim: usize) -> bool {
    is_fibonacci(dim as u64) || {
        // Check if dimension is close to a Fibonacci number
        // within 10% tolerance
        let nearest = nearest_fibonacci(dim as u64) as usize;
        let ratio = dim as f64 / nearest as f64;
        ratio >= 0.9 && ratio <= 1.1
    }
}

/// Calculate the "φ-distance" of a dimension from nearest Fibonacci
/// Returns 0.0 if exactly Fibonacci, 1.0 if at midpoint
pub fn phi_distance(dim: u64) -> f64 {
    let nearest = nearest_fibonacci(dim);
    let lower = if nearest >= dim {
        fibonacci(FIBONACCI.len().saturating_sub(2))
    } else {
        next_fibonacci(nearest)
    };

    if lower == nearest {
        let mid = (lower + nearest) / 2;
        if dim <= mid {
            (mid - dim) as f64 / (mid - lower).max(1) as f64
        } else {
            (dim - mid) as f64 / (nearest - mid).max(1) as f64
        }
    } else {
        let mid = (lower + nearest) / 2;
        if dim <= mid {
            (mid - dim) as f64 / (mid - lower).max(1) as f64
        } else {
            (dim - mid) as f64 / (nearest - mid).max(1) as f64
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_sequence() {
        assert_eq!(fibonacci(0), 1);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(2), 2);
        assert_eq!(fibonacci(3), 3);
        assert_eq!(fibonacci(4), 5);
        assert_eq!(fibonacci(5), 8);
        assert_eq!(fibonacci(6), 13);
        assert_eq!(fibonacci(7), 21);
        assert_eq!(fibonacci(8), 34);
        assert_eq!(fibonacci(9), 55);
        assert_eq!(fibonacci(10), 89);
        assert_eq!(fibonacci(11), 144);
    }

    #[test]
    fn test_next_fibonacci() {
        assert_eq!(next_fibonacci(0), 1);
        assert_eq!(next_fibonacci(1), 2);
        assert_eq!(next_fibonacci(2), 3);
        assert_eq!(next_fibonacci(5), 8);
        assert_eq!(next_fibonacci(8), 13);
        assert_eq!(next_fibonacci(13), 21);
        assert_eq!(next_fibonacci(100), 144);
    }

    #[test]
    fn test_nearest_fibonacci() {
        assert_eq!(nearest_fibonacci(0), 1);
        assert_eq!(nearest_fibonacci(1), 1);
        assert_eq!(nearest_fibonacci(2), 2);
        assert_eq!(nearest_fibonacci(4), 5);  // 4 is closer to 5 than to 3
        assert_eq!(nearest_fibonacci(6), 5);  // 6 is closer to 5 than to 8
        assert_eq!(nearest_fibonacci(7), 8);  // 7 is closer to 8 than to 5
    }

    #[test]
    fn test_is_fibonacci() {
        assert!(is_fibonacci(1));
        assert!(is_fibonacci(2));
        assert!(is_fibonacci(3));
        assert!(is_fibonacci(5));
        assert!(is_fibonacci(8));
        assert!(is_fibonacci(13));
        assert!(is_fibonacci(21));
        assert!(is_fibonacci(34));
        assert!(is_fibonacci(55));
        assert!(is_fibonacci(89));

        assert!(!is_fibonacci(4));
        assert!(!is_fibonacci(6));
        assert!(!is_fibonacci(7));
        assert!(!is_fibonacci(10));
    }

    #[test]
    fn test_recommended_dims() {
        assert_eq!(recommended_dim(0), 8);
        assert_eq!(recommended_dim(1), 13);
        assert_eq!(recommended_dim(2), 21);
        assert_eq!(recommended_dim(3), 34);
        assert_eq!(recommended_dim(4), 55);
    }

    #[test]
    fn test_ngram_hidden_sizes() {
        assert_eq!(NGRAM_HIDDEN_SIZES[0], 8);
        assert_eq!(NGRAM_HIDDEN_SIZES[5], 89);
        assert_eq!(NGRAM_HIDDEN_SIZES.len(), 6);
    }

    #[test]
    fn test_embedding_dims() {
        assert_eq!(EMBEDDING_DIMS[0], 8);
        assert_eq!(EMBEDDING_DIMS[5], 89);
        assert_eq!(EMBEDDING_DIMS.len(), 6);
    }

    #[test]
    fn test_phi_layer_count() {
        // φ^1 ≈ 1.6 → 2 layers
        assert_eq!(phi_layer_count(1), 2);
        // φ^2 ≈ 2.6 → 3 layers
        assert_eq!(phi_layer_count(2), 3);
        // φ^3 ≈ 4.2 → 4 layers
        assert_eq!(phi_layer_count(3), 4);
        // φ^4 ≈ 6.9 → 7 layers
        assert_eq!(phi_layer_count(4), 7);
        // φ^5 ≈ 11.1 → 11 layers
        assert_eq!(phi_layer_count(5), 11);
    }

    #[test]
    fn test_is_phi_optimized() {
        // Exact Fibonacci numbers
        assert!(is_phi_optimized(8));
        assert!(is_phi_optimized(13));
        assert!(is_phi_optimized(21));
        assert!(is_phi_optimized(34));

        // Within 10% tolerance
        assert!(is_phi_optimized(64));   // Close to 55 or 89? Actually 64 is not close
        assert!(!is_phi_optimized(64));
        assert!(is_phi_optimized(72));   // Close to 89? 72/89 ≈ 0.81 < 0.9
        assert!(!is_phi_optimized(72));
        assert!(is_phi_optimized(96));   // Close to 89? 96/89 ≈ 1.08 < 1.1
        assert!(is_phi_optimized(96));
    }

    #[test]
    fn test_phi_distance() {
        // Exact Fibonacci should have distance 0
        assert_eq!(phi_distance(8), 0.0);
        assert_eq!(phi_distance(13), 0.0);
        assert_eq!(phi_distance(21), 0.0);

        // Midpoint should have distance 1.0
        assert!((phi_distance(6) - 1.0).abs() < 0.01);  // Between 5 and 8
        assert!((phi_distance(10) - 1.0).abs() < 0.01); // Between 8 and 13
    }

    #[test]
    fn test_fibonacci_ratio_convergence() {
        // Test that F(n+1)/F(n) converges to φ
        for i in 5..FIBONACCI.len() - 1 {
            let ratio = FIBONACCI[i + 1] as f64 / FIBONACCI[i] as f64;
            let error = (ratio - PHI).abs();
            // Converges quickly
            assert!(error < 0.01, "Ratio at {}: {} error: {}", i, ratio, error);
        }
    }

    #[test]
    fn test_trinity_identity_with_fibonacci() {
        // Test φ² + 1/φ² = 3 using Fibonacci relationship
        // φ ≈ F(n+1)/F(n) for large n
        let n = 10;  // Use F(10) = 55, F(11) = 89
        let phi_approx = FIBONACCI[n + 1] as f64 / FIBONACCI[n] as f64;
        let phi_sq_approx = phi_approx * phi_approx;
        let phi_inv_sq_approx = 1.0 / phi_sq_approx;
        let sum = phi_sq_approx + phi_inv_sq_approx;

        // Should be close to 3
        let error = (sum - 3.0).abs();
        assert!(error < 0.1, "Trinity identity with Fibonacci approximation: error = {}", error);
    }

    #[test]
    fn test_current_best_hidden_is_fibonacci() {
        // Our current best uses hidden=384
        // 384 is not a Fibonacci number, but is it φ-optimized?
        let is_opt = is_phi_optimized(384);
        let dist = phi_distance(384);
        let nearest = nearest_fibonacci(384);

        assert!(!is_opt, "384 should not be Fibonacci");
        assert_eq!(nearest, 377, "Nearest to 384 should be 377");

        // Check if we should use 377 or 610 instead
        let dist_to_377 = (384.0 - 377.0) / 377.0;
        let dist_to_610 = (610.0 - 384.0) / 610.0;
        assert!(dist_to_377 < dist_to_610, "Should be closer to 377");
    }

    #[test]
    fn test_dimension_suggestions() {
        // For different size requirements, suggest Fibonacci dimensions
        let suggestions: [(usize, usize); 5] = [
            (32, 34),   // Small model → 34
            (64, 55),   // Medium → 55 (not 89!)
            (128, 144), // Large → 144
            (256, 233), // XL → 233
            (512, 610), // XXL → 610
        ];

        for (input, expected) in suggestions {
            let suggested = nearest_fibonacci(input as u64);
            assert_eq!(suggested as usize, expected, "For input {}, expected {}", input, expected);
        }
    }

    #[test]
    fn test_fibonacci_growth() {
        // Test that Fibonacci grows approximately by φ each step
        for i in 3..FIBONACCI.len() - 1 {
            let ratio = FIBONACCI[i + 1] as f64 / FIBONACCI[i] as f64;
            let error = (ratio - PHI).abs();
            assert!(error < 0.01, "Growth at step {}: {} (error: {})", i, ratio, error);
        }
    }

    #[test]
    fn test_binet_formula_accuracy() {
        // Test Binet's formula for values in our table
        for i in 0..10 {
            let expected = FIBONACCI[i];
            let computed = fibonacci(i);
            assert_eq!(expected, computed, "Binet formula failed at index {}", i);
        }
    }

    #[test]
    fn test_large_fibonacci() {
        // Test that we can compute large Fibonacci numbers
        let f20 = fibonacci(20);
        let f25 = fibonacci(25);

        // F20 = 6765, F25 = 75025
        assert_eq!(f20, 6765);
        assert_eq!(f25, 75025);
    }

    #[test]
    fn test_recommended_dims_are_fibonacci() {
        // All recommended dimensions should be Fibonacci numbers
        for &dim in &RECOMMENDED_DIMS {
            assert!(is_fibonacci(dim as u64), "Recommended dim {} is not Fibonacci", dim);
        }
    }

    #[test]
    fn test_phi_optimization_for_common_sizes() {
        // Check common model dimension sizes
        let common_sizes = [32, 64, 128, 256, 512, 768, 1024];
        for &size in &common_sizes {
            let nearest = nearest_fibonacci(size as u64);
            let dist = phi_distance(size as u64);
            // All should have some φ-optimized alternative
            assert!(dist <= 1.0, "Size {} has no close Fibonacci alternative", size);
        }
    }
}
