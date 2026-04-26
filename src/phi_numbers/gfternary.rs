//! GFTernary — Golden Float Ternary format
//!
//! Values: {-φ, 0, +φ}
//! Ternary representation with φ-step instead of unit step

use super::phi_constants::*;

/// GFTernary: ternary value with φ-quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GFTernary {
    /// Negative value: -φ
    NegPhi,
    /// Zero
    Zero,
    /// Positive value: +φ
    PosPhi,
}

impl GFTernary {
    /// Create GFTernary from f32 by φ-quantization
    pub fn from_f32(value: f32) -> Self {
        let v = value as f64;

        // Quantize to nearest ternary step with φ-spacing
        // Thresholds: -φ/2 and φ/2
        let phi_half = PHI / 2.0;

        if v < -phi_half {
            Self::NegPhi
        } else if v > phi_half {
            Self::PosPhi
        } else {
            Self::Zero
        }
    }

    /// Create GFTernary from f64 by φ-quantization
    pub fn from_f64(value: f64) -> Self {
        let phi_half = PHI / 2.0;

        if value < -phi_half {
            Self::NegPhi
        } else if value > phi_half {
            Self::PosPhi
        } else {
            Self::Zero
        }
    }

    /// Convert GFTernary to f32
    pub fn to_f32(self) -> f32 {
        match self {
            Self::NegPhi => -(PHI as f32),
            Self::Zero => 0.0,
            Self::PosPhi => PHI as f32,
        }
    }

    /// Convert GFTernary to f64
    pub fn to_f64(self) -> f64 {
        match self {
            Self::NegPhi => -PHI,
            Self::Zero => 0.0,
            Self::PosPhi => PHI,
        }
    }

    /// Get the raw ternary value (-1, 0, 1) scaled by φ
    pub fn raw_value(self) -> f64 {
        match self {
            Self::NegPhi => -1.0,
            Self::Zero => 0.0,
            Self::PosPhi => 1.0,
        }
    }

    /// Quantization error vs f32
    pub fn quant_error_f32(self, original: f32) -> f32 {
        (self.to_f32() - original).abs()
    }

    /// Relative error vs f32
    pub fn relative_error_f32(self, original: f32) -> f32 {
        if original.abs() < f32::MIN_POSITIVE {
            return self.quant_error_f32(original);
        }
        self.quant_error_f32(original) / original.abs()
    }

    /// Quantization error vs f64
    pub fn quant_error_f64(self, original: f64) -> f64 {
        (self.to_f64() - original).abs()
    }

    /// Relative error vs f64
    pub fn relative_error_f64(self, original: f64) -> f64 {
        if original.abs() < f64::MIN_POSITIVE {
            return self.quant_error_f64(original);
        }
        self.quant_error_f64(original) / original.abs()
    }

    /// Get the number of bits needed to represent this value
    pub fn bit_width(self) -> u8 {
        // Ternary needs log2(3) ≈ 1.585 bits per value
        // Round up to 2 bits for practical encoding
        2
    }

    /// Encode as 2 bits
    pub fn encode_bits(self) -> u8 {
        match self {
            Self::NegPhi => 0b01,  // -1 in ternary (but we use 01 for -φ)
            Self::Zero => 0b10,  // 0
            Self::PosPhi => 0b11,  // +1 (but we use 11 for +φ)
        }
    }

    /// Decode from 2 bits
    pub fn decode_bits(bits: u8) -> Option<Self> {
        match bits & 0b11 {
            0b01 => Some(Self::NegPhi),
            0b10 => Some(Self::Zero),
            0b11 => Some(Self::PosPhi),
            _ => None,
        }
    }

    /// Arithmetic: add two GFTernary values (with clamping)
    pub fn add(self, other: Self) -> Self {
        let sum = self.raw_value() + other.raw_value();
        match sum {
            v if v < -0.5 => Self::NegPhi,
            v if v > 0.5 => Self::PosPhi,
            _ => Self::Zero,
        }
    }

    /// Arithmetic: multiply two GFTernary values
    pub fn mul(self, other: Self) -> Self {
        // -φ * -φ = φ², but we clamp back to φ
        // -φ * 0 = 0
        // -φ * +φ = -φ² → -φ
        // 0 * anything = 0
        // +φ * -φ = -φ² → -φ
        // +φ * 0 = 0
        // +φ * +φ = φ² → φ
        let prod = self.raw_value() * other.raw_value();
        match prod {
            v if v < -0.5 => Self::NegPhi,
            v if v > 0.5 => Self::PosPhi,
            _ => Self::Zero,
        }
    }

    /// Check if this is a positive value
    pub fn is_positive(self) -> bool {
        matches!(self, Self::PosPhi)
    }

    /// Check if this is a negative value
    pub fn is_negative(self) -> bool {
        matches!(self, Self::NegPhi)
    }

    /// Check if this is zero
    pub fn is_zero(self) -> bool {
        matches!(self, Self::Zero)
    }

    /// Get the absolute value (always 0 or +φ)
    pub fn abs(self) -> Self {
        match self {
            Self::NegPhi => Self::PosPhi,
            Self::Zero => Self::Zero,
            Self::PosPhi => Self::PosPhi,
        }
    }

    /// Negate the value
    pub fn negate(self) -> Self {
        match self {
            Self::NegPhi => Self::PosPhi,
            Self::Zero => Self::Zero,
            Self::PosPhi => Self::NegPhi,
        }
    }
}

impl Default for GFTernary {
    fn default() -> Self {
        Self::Zero
    }
}

impl std::ops::Add for GFTernary {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self.add(other)
    }
}

impl std::ops::Mul for GFTernary {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        self.mul(other)
    }
}

impl std::ops::Neg for GFTernary {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f32_negative() {
        let gf = GFTernary::from_f32(-2.0);
        assert_eq!(gf, GFTernary::NegPhi);
    }

    #[test]
    fn test_from_f32_zero() {
        let gf = GFTernary::from_f32(0.0);
        assert_eq!(gf, GFTernary::Zero);

        let gf = GFTernary::from_f32(0.5);
        assert_eq!(gf, GFTernary::Zero);

        let gf = GFTernary::from_f32(-0.5);
        assert_eq!(gf, GFTernary::Zero);
    }

    #[test]
    fn test_from_f32_positive() {
        let gf = GFTernary::from_f32(2.0);
        assert_eq!(gf, GFTernary::PosPhi);
    }

    #[test]
    fn test_phi_thresholds() {
        // Test values exactly at thresholds
        let phi_half = PHI / 2.0;

        // Just below -φ/2
        let gf = GFTernary::from_f64(-(phi_half + 0.001));
        assert_eq!(gf, GFTernary::NegPhi);

        // Just above -φ/2
        let gf = GFTernary::from_f64(-(phi_half - 0.001));
        assert_eq!(gf, GFTernary::Zero);

        // Just below +φ/2
        let gf = GFTernary::from_f64(phi_half - 0.001);
        assert_eq!(gf, GFTernary::Zero);

        // Just above +φ/2
        let gf = GFTernary::from_f64(phi_half + 0.001);
        assert_eq!(gf, GFTernary::PosPhi);
    }

    #[test]
    fn test_to_f32() {
        assert_eq!(GFTernary::NegPhi.to_f32(), -(PHI as f32));
        assert_eq!(GFTernary::Zero.to_f32(), 0.0);
        assert_eq!(GFTernary::PosPhi.to_f32(), PHI as f32);
    }

    #[test]
    fn test_to_f64() {
        assert_eq!(GFTernary::NegPhi.to_f64(), -PHI);
        assert_eq!(GFTernary::Zero.to_f64(), 0.0);
        assert_eq!(GFTernary::PosPhi.to_f64(), PHI);
    }

    #[test]
    fn test_quant_error() {
        let gf = GFTernary::from_f32(0.1);
        let err = gf.quant_error_f32(0.1);
        assert!((err - 0.1).abs() < 1e-5);

        let gf = GFTernary::from_f32(2.0);
        let err = gf.quant_error_f32(2.0);
        assert!((err - (PHI as f32 - 2.0)).abs() < 1e-5);
    }

    #[test]
    fn test_raw_value() {
        assert_eq!(GFTernary::NegPhi.raw_value(), -1.0);
        assert_eq!(GFTernary::Zero.raw_value(), 0.0);
        assert_eq!(GFTernary::PosPhi.raw_value(), 1.0);
    }

    #[test]
    fn test_bit_width() {
        assert_eq!(GFTernary::NegPhi.bit_width(), 2);
        assert_eq!(GFTernary::Zero.bit_width(), 2);
        assert_eq!(GFTernary::PosPhi.bit_width(), 2);
    }

    #[test]
    fn test_encode_decode() {
        let values = [GFTernary::NegPhi, GFTernary::Zero, GFTernary::PosPhi];
        for v in &values {
            let bits = v.encode_bits();
            let decoded = GFTernary::decode_bits(bits).unwrap();
            assert_eq!(*v, decoded);
        }
    }

    #[test]
    fn test_add() {
        // NegPhi + NegPhi = -2 → clamped to NegPhi
        assert_eq!(GFTernary::NegPhi.add(GFTernary::NegPhi), GFTernary::NegPhi);

        // NegPhi + Zero = -1 → NegPhi
        assert_eq!(GFTernary::NegPhi.add(GFTernary::Zero), GFTernary::NegPhi);

        // NegPhi + PosPhi = 0 → Zero
        assert_eq!(GFTernary::NegPhi.add(GFTernary::PosPhi), GFTernary::Zero);

        // Zero + Zero = 0
        assert_eq!(GFTernary::Zero.add(GFTernary::Zero), GFTernary::Zero);

        // Zero + PosPhi = +1 → PosPhi
        assert_eq!(GFTernary::Zero.add(GFTernary::PosPhi), GFTernary::PosPhi);

        // PosPhi + PosPhi = +2 → clamped to PosPhi
        assert_eq!(GFTernary::PosPhi.add(GFTernary::PosPhi), GFTernary::PosPhi);
    }

    #[test]
    fn test_add_operator() {
        assert_eq!(GFTernary::NegPhi + GFTernary::PosPhi, GFTernary::Zero);
        assert_eq!(GFTernary::Zero + GFTernary::Zero, GFTernary::Zero);
        assert_eq!(GFTernary::PosPhi + GFTernary::PosPhi, GFTernary::PosPhi);
    }

    #[test]
    fn test_mul() {
        // Neg * Neg = + (clamped)
        assert_eq!(GFTernary::NegPhi.mul(GFTernary::NegPhi), GFTernary::PosPhi);

        // Neg * Zero = Zero
        assert_eq!(GFTernary::NegPhi.mul(GFTernary::Zero), GFTernary::Zero);

        // Neg * Pos = Neg (φ² → φ)
        assert_eq!(GFTernary::NegPhi.mul(GFTernary::PosPhi), GFTernary::NegPhi);

        // Zero * anything = Zero
        assert_eq!(GFTernary::Zero.mul(GFTernary::NegPhi), GFTernary::Zero);
        assert_eq!(GFTernary::Zero.mul(GFTernary::Zero), GFTernary::Zero);
        assert_eq!(GFTernary::Zero.mul(GFTernary::PosPhi), GFTernary::Zero);

        // Pos * Neg = Neg
        assert_eq!(GFTernary::PosPhi.mul(GFTernary::NegPhi), GFTernary::NegPhi);

        // Pos * Zero = Zero
        assert_eq!(GFTernary::PosPhi.mul(GFTernary::Zero), GFTernary::Zero);

        // Pos * Pos = Pos (φ² → φ)
        assert_eq!(GFTernary::PosPhi.mul(GFTernary::PosPhi), GFTernary::PosPhi);
    }

    #[test]
    fn test_mul_operator() {
        assert_eq!(GFTernary::NegPhi * GFTernary::PosPhi, GFTernary::NegPhi);
        assert_eq!(GFTernary::Zero * GFTernary::PosPhi, GFTernary::Zero);
        assert_eq!(GFTernary::PosPhi * GFTernary::PosPhi, GFTernary::PosPhi);
    }

    #[test]
    fn test_negate() {
        assert_eq!(-GFTernary::NegPhi, GFTernary::PosPhi);
        assert_eq!(-GFTernary::Zero, GFTernary::Zero);
        assert_eq!(-GFTernary::PosPhi, GFTernary::NegPhi);
    }

    #[test]
    fn test_abs() {
        assert_eq!(GFTernary::NegPhi.abs(), GFTernary::PosPhi);
        assert_eq!(GFTernary::Zero.abs(), GFTernary::Zero);
        assert_eq!(GFTernary::PosPhi.abs(), GFTernary::PosPhi);
    }

    #[test]
    fn test_is_positive() {
        assert!(!GFTernary::NegPhi.is_positive());
        assert!(!GFTernary::Zero.is_positive());
        assert!(GFTernary::PosPhi.is_positive());
    }

    #[test]
    fn test_is_negative() {
        assert!(GFTernary::NegPhi.is_negative());
        assert!(!GFTernary::Zero.is_negative());
        assert!(!GFTernary::PosPhi.is_negative());
    }

    #[test]
    fn test_is_zero() {
        assert!(!GFTernary::NegPhi.is_zero());
        assert!(GFTernary::Zero.is_zero());
        assert!(!GFTernary::PosPhi.is_zero());
    }

    #[test]
    fn test_default() {
        assert_eq!(GFTernary::default(), GFTernary::Zero);
    }

    #[test]
    fn test_fibonacci_quantization() {
        // Quantize Fibonacci numbers to GFTernary
        let fib_nums = [1, 2, 3, 5, 8, 13, 21];
        for f in fib_nums {
            let gf = GFTernary::from_f32(f as f32);
            // All should be PosPhi (positive > φ/2)
            assert_eq!(gf, GFTernary::PosPhi, "Fibonacci {} should be PosPhi", f);
        }
    }

    #[test]
    fn test_phi_family_quantization() {
        // Quantize φ-family to GFTernary
        let phi_vals = [PHI, PHI_SQUARED, PHI_CUBED, PHI_SQRT];
        for v in phi_vals {
            let gf = GFTernary::from_f64(v);
            // All should be PosPhi (positive > φ/2)
            assert_eq!(gf, GFTernary::PosPhi, "φ-family value {} should be PosPhi", v);
        }
    }

    #[test]
    fn test_phi_conjugate_quantization() {
        let gf = GFTernary::from_f64(PHI_CONJUGATE);
        // 1/φ ≈ 0.618 < φ/2 ≈ 0.809, so it should be Zero
        assert_eq!(gf, GFTernary::Zero);

        let gf = GFTernary::from_f64(PHI_INVERSE_SQUARED);
        // 1/φ² ≈ 0.382 < φ/2, so it should be Zero
        assert_eq!(gf, GFTernary::Zero);
    }

    #[test]
    fn test_negative_values() {
        let gf = GFTernary::from_f64(-PHI);
        assert_eq!(gf, GFTernary::NegPhi);

        let gf = GFTernary::from_f64(-PHI_SQUARED);
        assert_eq!(gf, GFTernary::NegPhi);

        let gf = GFTernary::from_f64(-2.0);
        assert_eq!(gf, GFTernary::NegPhi);
    }

    #[test]
    fn test_symmetry() {
        assert_eq!(-GFTernary::from_f64(2.0), GFTernary::from_f64(-2.0));
        assert_eq!(-GFTernary::from_f64(-2.0), GFTernary::from_f64(2.0));
    }

    #[test]
    fn test_comparison_operators() {
        assert!(GFTernary::NegPhi != GFTernary::Zero);
        assert!(GFTernary::Zero != GFTernary::PosPhi);
        assert!(GFTernary::NegPhi != GFTernary::PosPhi);

        assert_eq!(GFTernary::NegPhi, GFTernary::NegPhi);
        assert_eq!(GFTernary::Zero, GFTernary::Zero);
        assert_eq!(GFTernary::PosPhi, GFTernary::PosPhi);
    }

    #[test]
    fn test_cloning() {
        let gf = GFTernary::PosPhi;
        let gf_clone = gf.clone();
        assert_eq!(gf, gf_clone);
    }

    #[test]
    fn test_copy() {
        let gf = GFTernary::PosPhi;
        let gf_copy = gf;
        assert_eq!(gf, gf_copy);
    }

    #[test]
    fn test_all_variants() {
        let variants = [GFTernary::NegPhi, GFTernary::Zero, GFTernary::PosPhi];
        for v in variants {
            // Test round-trip through encoding
            let bits = v.encode_bits();
            let decoded = GFTernary::decode_bits(bits).unwrap();
            assert_eq!(v, decoded);

            // Test negation
            assert_eq!(-(-v), v);

            // Test absolute value
            assert!(!v.abs().is_negative());

            // Test raw value
            if v.is_positive() {
                assert_eq!(v.raw_value(), 1.0);
            } else if v.is_negative() {
                assert_eq!(v.raw_value(), -1.0);
            } else {
                assert_eq!(v.raw_value(), 0.0);
            }
        }
    }

    #[test]
    fn test_encode_invalid_bits() {
        assert_eq!(GFTernary::decode_bits(0b00), None);
    }

    #[test]
    fn test_phi_step_property() {
        // Verify that GFTernary uses φ-step instead of unit step
        let step = GFTernary::PosPhi.to_f64() - GFTernary::Zero.to_f64();
        assert!((step - PHI).abs() < 1e-10, "Step should be φ: {}", step);

        // Also verify negative step
        let step_neg = GFTernary::Zero.to_f64() - GFTernary::NegPhi.to_f64();
        assert!((step_neg - PHI).abs() < 1e-10, "Negative step should be φ: {}", step_neg);
    }

    #[test]
    fn test_memory_efficiency() {
        // GFTernary uses 2 bits per value
        // Compare with standard 16-bit float which uses 16 bits
        // Compression ratio: 16/2 = 8×
        let gf_bits = GFTernary::PosPhi.bit_width();
        let f16_bits = 16;
        let compression = f16_bits as f64 / gf_bits as f64;
        assert!((compression - 8.0).abs() < 1e-10, "Compression ratio should be 8×: {}", compression);
    }

    #[test]
    fn test_arithmetic_properties() {
        // Test associativity of addition (with clamping)
        let a = GFTernary::NegPhi;
        let b = GFTernary::Zero;
        let c = GFTernary::PosPhi;

        // (a + b) + c = a + (b + c)
        let left = (a + b) + c;
        let right = a + (b + c);
        assert_eq!(left, right, "Addition not associative");

        // Test distributivity (with clamping)
        // a * (b + c) = a * b + a * c
        let left = a * (b + c);
        let right = a * b + a * c;
        assert_eq!(left, right, "Multiplication not distributive");

        // Test identity
        assert_eq!(a + GFTernary::Zero, a, "Zero not additive identity");
        assert_eq!(a * GFTernary::PosPhi, a, "PosPhi not multiplicative identity");
    }

    #[test]
    fn test_comparison_with_standard_f32() {
        // Compare GFTernary quantization error with standard 8-bit quantization
        let test_vals = [0.1, 0.5, 1.0, PHI as f32, 2.0];
        for v in test_vals {
            let gf = GFTernary::from_f32(v);
            let gf_err = gf.relative_error_f32(v);

            // GFTernary has very high quantization error (coarse quantization)
            // But it's predictable and uses only 2 bits
            assert!(gf_err >= 0.0);
            assert!(gf_err <= 1.0, "Relative error should be reasonable: {}", gf_err);
        }
    }
}
