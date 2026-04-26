//! GoldenFloat16 (GF16) - φ-optimized 16-bit floating point format
//!
//! Format: 1 sign | 6 exponent | 9 mantissa
//! Bias: 15 (to allow representing 1.0, 2.0, etc.)
//! No subnormals
//! Range: 4.66×10⁻⁵ to 6.55×10⁴
//!
//! Based on: https://github.com/gHashTag/zig-golden-float

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct GF16(pub u16);

impl GF16 {
    const SIGN_MASK: u16 = 0x8000;
    const EXP_MASK: u16 = 0x7E00;
    const MANTISSA_MASK: u16 = 0x01FF;
    const EXP_BIAS: i32 = 15;

    pub const ZERO: GF16 = GF16(0x0000);
    pub const NEG_ZERO: GF16 = GF16(0x8000);
    pub const INF: GF16 = GF16(0x7E00);
    pub const NEG_INF: GF16 = GF16(0xFE00);
    pub const NAN: GF16 = GF16(0x7FFF);

    #[must_use]
    pub fn from_f32(val: f32) -> Self {
        if val == 0.0 {
            return if val.is_sign_negative() { Self::NEG_ZERO } else { Self::ZERO };
        }
        if val.is_nan() {
            return Self::NAN;
        }
        if !val.is_finite() {
            return if val.is_sign_negative() { Self::NEG_INF } else { Self::INF };
        }

        let bits = val.to_bits();
        let f32_sign = (bits >> 31) & 0x1;
        let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let f32_mant = bits & 0x007F_FFFF;

        if f32_exp <= -15 {
            return GF16((f32_sign as u16) << 15);
        }

        let mut exp = f32_exp + Self::EXP_BIAS;

        if exp >= 63 {
            exp = 62;
        } else if exp <= 0 {
            return GF16((f32_sign as u16) << 15);
        }

        let mantissa = (f32_mant >> (23 - 9)) as u16;
        let sign = (f32_sign as u16) << 15;
        GF16(sign | ((exp as u16) << 9) | mantissa)
    }

    #[must_use]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = (bits & Self::SIGN_MASK) >> 15;
        let exp  = (bits & Self::EXP_MASK) >> 9;
        let mantissa = bits & Self::MANTISSA_MASK;

        if exp == 0 {
            return if sign != 0 { -0.0_f32 } else { 0.0_f32 };
        }

        if exp == 63 {
            return if mantissa == 0 {
                if sign != 0 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            };
        }

        let f32_exp = exp as i32 - Self::EXP_BIAS + 127;
        if !(0..=255).contains(&f32_exp) {
            return if sign != 0 { f32::NEG_INFINITY } else { f32::INFINITY };
        }

        let f32_mant = (mantissa as u32) << (23 - 9);
        f32::from_bits((sign as u32) << 31 | (f32_exp as u32) << 23 | f32_mant)
    }

    #[must_use]
    pub fn phi_distance() -> f64 {
        let ratio = 6.0_f64 / 9.0_f64;
        let inv_phi = 1.0_f64 / 1.618_033_988_749_895_f64;
        (ratio - inv_phi).abs()
    }

    #[must_use]
    pub fn exponent(self) -> i32 {
        ((self.0 & Self::EXP_MASK) >> 9) as i32
    }

    #[must_use]
    pub fn mantissa(self) -> u16 {
        self.0 & Self::MANTISSA_MASK
    }

    #[must_use]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 & Self::EXP_MASK) >> 9;
        exp == 63 && (self.0 & Self::MANTISSA_MASK) != 0
    }

    #[must_use]
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 & Self::EXP_MASK) >> 9;
        exp == 63 && (self.0 & Self::MANTISSA_MASK) == 0
    }

    #[must_use]
    pub fn is_finite(self) -> bool {
        (self.0 & Self::EXP_MASK) >> 9 != 63
    }

    #[must_use]
    pub fn is_sign_negative(self) -> bool {
        (self.0 & Self::SIGN_MASK) != 0
    }

    #[must_use]
    pub fn to_bits(self) -> u16 {
        self.0
    }

    #[must_use]
    pub fn from_bits(bits: u16) -> Self {
        GF16(bits)
    }
}

impl fmt::Debug for GF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bits = self.0;
        let exp = (bits & Self::EXP_MASK) >> 9;
        let mantissa = bits & Self::MANTISSA_MASK;

        if exp == 63 {
            if mantissa == 0 {
                if bits & Self::SIGN_MASK != 0 {
                    write!(f, "GF16(-Inf)")
                } else {
                    write!(f, "GF16(Inf)")
                }
            } else {
                write!(f, "GF16(NaN)")
            }
        } else if exp == 0 {
            write!(f, "GF16(0)")
        } else {
            write!(f, "GF16({})", self.to_f32())
        }
    }
}

impl fmt::Display for GF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = self.to_f32();
        if val.is_nan() {
            write!(f, "NaN")
        } else if val.is_infinite() {
            if val.is_sign_negative() {
                write!(f, "-Inf")
            } else {
                write!(f, "Inf")
            }
        } else {
            write!(f, "{val}")
        }
    }
}

impl From<f32> for GF16 {
    fn from(val: f32) -> Self {
        Self::from_f32(val)
    }
}

impl From<GF16> for f32 {
    fn from(val: GF16) -> Self {
        val.to_f32()
    }
}

impl From<f64> for GF16 {
    fn from(val: f64) -> Self {
        Self::from_f32(val as f32)
    }
}

#[derive(Clone, Default)]
pub struct GF16Vec {
    data: Vec<GF16>,
}

impl GF16Vec {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        GF16Vec { data: Vec::with_capacity(capacity) }
    }

    pub fn push(&mut self, val: f32) {
        self.data.push(GF16::from_f32(val));
    }

    pub fn push_gf16(&mut self, val: GF16) {
        self.data.push(val);
    }

    #[must_use]
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.data.iter().map(|x| x.to_f32()).collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = GF16> + '_ {
        self.data.iter().copied()
    }
}

impl FromIterator<f32> for GF16Vec {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        let mut vec = GF16Vec::new(0);
        for val in iter {
            vec.push(val);
        }
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let z = GF16::from_f32(0.0);
        assert_eq!(z, GF16::ZERO);
        assert_eq!(z.to_f32(), 0.0);
    }

    #[test]
    fn test_negative_zero() {
        let z = GF16::from_f32(-0.0);
        assert_eq!(z, GF16::NEG_ZERO);
        assert!(z.is_sign_negative());
        assert_eq!(z.to_f32().to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn test_one() {
        let one = GF16::from_f32(1.0);
        let back = one.to_f32();
        assert!((back - 1.0).abs() < 0.01, "Expected ~1.0, got {back}");
    }

    #[test]
    fn test_negative() {
        let neg = GF16::from_f32(-1.5);
        assert!(neg.is_sign_negative());
        let f = neg.to_f32();
        assert!(f < -1.0 && f > -2.0, "Expected ~-1.5, got {f}");
    }

    #[test]
    fn test_roundtrip() {
        let values = [0.1_f32, 0.5, 1.0, 2.0, 10.0, 100.0, 0.001, 0.0001];
        for &v in &values {
            let gf = GF16::from_f32(v);
            let back = gf.to_f32();
            let rel_error = (back - v).abs() / v.abs().max(1e-10);
            assert!(rel_error < 0.02, "Roundtrip error={rel_error}");
        }
    }

    #[test]
    fn test_phi_distance() {
        let pd = GF16::phi_distance();
        assert!((pd - 0.0486).abs() < 0.001, "φ-distance = {pd}, expected ~0.049");
    }

    #[test]
    fn test_special_values() {
        assert!(GF16::NAN.is_nan());
        assert!(!GF16::NAN.is_finite());
        assert!(GF16::INF.is_infinite());
        assert!(!GF16::INF.is_finite());
        assert!(GF16::NEG_INF.is_infinite());
        assert!(GF16::NEG_INF.is_sign_negative());
    }

    #[test]
    fn test_clamping() {
        let tiny = GF16::from_f32(1e-10);
        assert_eq!(tiny, GF16::ZERO);
        let huge = GF16::from_f32(1e10);
        assert!(huge.is_finite());
    }

    #[test]
    fn test_gf16_vec() {
        let mut vec = GF16Vec::new(10);
        vec.push(1.0);
        vec.push(2.0);
        vec.push(3.0);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.to_f32_vec().len(), 3);
    }

    #[test]
    fn test_neg_zero_to_f32() {
        let nz = GF16::NEG_ZERO;
        assert_eq!(nz.to_f32().to_bits(), (-0.0_f32).to_bits());
    }
}

#[derive(Debug, Clone)]
pub struct QuantizationMetrics {
    pub max_error_pct: f64,
    pub avg_error_pct: f64,
    pub mse: f64,
    pub mae: f64,
    pub phi_error: f64,
}

impl QuantizationMetrics {
    #[must_use]
    pub fn compute(original: &[f32], quantized: &[f32]) -> Self {
        assert_eq!(original.len(), quantized.len());
        let n = original.len();
        let mut max_error = 0.0_f64;
        let mut sum_error_pct = 0.0_f64;
        let mut sum_abs_error = 0.0_f64;
        let mut sum_sq_error = 0.0_f64;

        for (&orig, &quant) in original.iter().zip(quantized.iter()) {
            let abs_orig = (orig.abs() as f64).max(1e-10);
            let abs_error = ((quant - orig).abs()) as f64;
            let error_pct = abs_error / abs_orig * 100.0;

            if error_pct > max_error { max_error = error_pct; }
            sum_error_pct += error_pct;
            sum_abs_error += abs_error;
            sum_sq_error  += abs_error * abs_error;
        }

        let n_f = n as f64;
        QuantizationMetrics {
            max_error_pct: max_error,
            avg_error_pct: sum_error_pct / n_f,
            mse: sum_sq_error / n_f,
            mae: sum_abs_error / n_f,
            phi_error: GF16::phi_distance(),
        }
    }
}

#[must_use]
pub fn benchmark_quantization(n: usize) -> QuantizationMetrics {
    use rand::Rng;
    use rand::thread_rng;
    let mut rng = thread_rng();

    let original: Vec<f32> = (0..n)
        .map(|_| rng.gen::<f32>() * 0.2 - 0.1)
        .collect();

    let quantized: Vec<f32> = original
        .iter()
        .map(|&x| GF16::from_f32(x).to_f32())
        .collect();

    QuantizationMetrics::compute(&original, &quantized)
}
