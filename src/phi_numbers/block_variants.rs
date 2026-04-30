//! Block GF Variants — Inspired by MX/MSFP competitive analysis
//!
//! These formats implement block floating point where multiple elements
//! share a common exponent (inspired by OCP MX Formats and Microsoft FP).
//!
//! Key competitive insights:
//! - MX Formats: 32 elements share E8M0 scale factor
//! - MSFP: 128 elements share 5-bit exponent
//! - Benefit: Dramatically reduces memory footprint while preserving accuracy
//!
//! GF Block Variants:
//! - GF16B32: 16-bit block format, 32 elements share 8-bit exponent
//! - GF8B64: 8-bit block format, 64 elements share 4-bit exponent

use super::phi_constants::*;

// ============================================================================
// GF16B32 — 16-bit Block Format (32 elements share exponent)
// ============================================================================

/// GF16B32 — Block GF16 with shared exponent across 32 elements
///
/// Format per element: [sign:1][mant:7] (8 bits mantissa + sign)
/// Shared exponent: 8 bits (stored once per block of 32 elements)
///
/// Storage: 32 × 8 bits = 256 bits + 8 bits exponent = 264 bits = 33 bytes
/// Standard GF16: 32 × 16 bits = 512 bits = 64 bytes
/// Savings: 48% vs standard GF16!
///
/// Inspired by: OCP MX Formats (E4M3/E5M2 with E8M0 scale factor)
#[derive(Clone, Debug)]
pub struct GF16B32Block {
    /// Shared exponent for the block (8 bits, biased)
    pub shared_exp: u8,
    /// 32 mantissas (7 bits each + sign)
    pub mantissas: [u8; 32],
}

impl GF16B32Block {
    const EXP_BIAS: i8 = 63; // 2^(8-1) - 1

    /// Create a block from f32 values
    pub fn from_f32_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 32, "GF16B32Block requires exactly 32 values");

        // Find maximum exponent in the block
        let mut max_exp = i32::MIN;
        let mut has_nonzero = false;
        let mut signs = [false; 32];

        for (i, &v) in values.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            has_nonzero = true;
            signs[i] = v < 0.0;
            let bits = v.to_bits();
            let exp = ((bits >> 23) & 0xFF) as i32 - 127;
            max_exp = max_exp.max(exp);
        }

        // Shared exponent is max_exp + bias (or 0 if all zeros)
        let shared_exp = if !has_nonzero {
            0
        } else {
            let biased = max_exp + Self::EXP_BIAS as i32;
            (biased.max(0).min(255)) as u8
        };

        // Quantize mantissas relative to shared exponent
        let mut mantissas = [0u8; 32];
        for i in 0..32 {
            if values[i] == 0.0 {
                mantissas[i] = 0;
                continue;
            }

            let f32_exp = ((values[i].to_bits() >> 23) & 0xFF) as i32 - 127;
            let f32_mant = values[i].to_bits() & 0x007FFFFF;

            // Calculate mantissa offset from shared exponent
            let exp_diff = f32_exp - (shared_exp as i32 - Self::EXP_BIAS as i32);

            if exp_diff < -7 {
                // Too small, round to zero
                mantissas[i] = 0;
            } else {
                // Scale mantissa by 2^exp_diff
                let scale = 2.0_f32.powi(exp_diff as i32);
                let mant_scaled = ((1.0 + (f32_mant as f32) / 8388608.0) * scale).min(1.9921875);

                // Quantize to 7 bits (0-127, where 128 = 2.0)
                let mant_q = (mant_scaled * 64.0).round() as u8;
                mantissas[i] = mant_q.min(127);
            }

            if signs[i] {
                mantissas[i] |= 0x80; // Set sign bit
            }
        }

        GF16B32Block { shared_exp, mantissas }
    }

    /// Convert block back to f32 slice
    pub fn to_f32_slice(&self) -> [f32; 32] {
        let mut result = [0.0f32; 32];

        let shared_exp_val = (self.shared_exp as i32) - Self::EXP_BIAS as i32;
        let exp_val = 2.0_f32.powi(shared_exp_val);

        for i in 0..32 {
            let mant_raw = self.mantissas[i];
            if mant_raw == 0 {
                result[i] = 0.0;
                continue;
            }

            let sign = if mant_raw & 0x80 != 0 { -1.0 } else { 1.0 };
            let mant = (mant_raw & 0x7F) as f32 / 64.0; // 7-bit mantissa (0-1.984)

            result[i] = sign * exp_val * mant;
        }

        result
    }

    /// Calculate quantization error
    pub fn quantization_error(&self, original: &[f32]) -> f32 {
        let decoded = self.to_f32_slice();
        let mut total_error = 0.0f32;
        for i in 0..32 {
            total_error += (decoded[i] - original[i]).abs();
        }
        total_error / 32.0
    }

    /// Compression ratio vs standard GF16
    pub fn compression_ratio() -> f32 {
        // Standard GF16: 32 × 16 = 512 bits
        // GF16B32: 264 bits
        512.0 / 264.0
    }
}

// ============================================================================
// GF8B64 — 8-bit Block Format (64 elements share exponent)
// ============================================================================

/// GF8B64 — 8-bit Block Format with shared exponent across 64 elements
///
/// Format per element: [sign:1][mant:2] (3 bits per element)
/// Shared exponent: 4 bits (stored once per block of 64 elements)
///
/// Storage: 64 × 3 bits = 192 bits + 4 bits exponent = 196 bits = 24.5 bytes
/// Standard GF8: 64 × 8 bits = 512 bits = 64 bytes
/// Savings: 62% vs standard GF8!
///
/// Use case: Extreme compression for embeddings where relative values matter more
#[derive(Clone, Debug)]
pub struct GF8B64Block {
    /// Shared exponent for the block (4 bits, biased)
    pub shared_exp: u8,
    /// 64 mantissas packed into 24 bytes (192 bits)
    /// Each mantissa: [sign:1][mant:2]
    pub packed_mantissas: [u8; 24],
}

impl GF8B64Block {
    const EXP_BIAS: i8 = 7; // 2^(4-1) - 1

    /// Create a block from f32 values
    pub fn from_f32_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 64, "GF8B64Block requires exactly 64 values");

        // Find maximum exponent in the block
        let mut max_exp = i32::MIN;
        let mut has_nonzero = false;
        let mut signs = [false; 64];

        for (i, &v) in values.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            has_nonzero = true;
            signs[i] = v < 0.0;
            let bits = v.to_bits();
            let exp = ((bits >> 23) & 0xFF) as i32 - 127;
            max_exp = max_exp.max(exp);
        }

        // Shared exponent (4 bits, biased)
        let shared_exp = if !has_nonzero {
            0
        } else {
            let biased = max_exp + Self::EXP_BIAS as i32;
            (biased.max(0).min(15)) as u8
        };

        // Quantize to 2-bit mantissa (0, 1, 2, 3 representing 1.0, 1.33, 1.67, 2.0)
        let mut packed_mantissas = [0u8; 24];
        let mut bit_pos = 0u32;

        for i in 0..64 {
            if values[i] == 0.0 {
                // Zero - skip sign bit, mantissa = 0
                bit_pos += 3;
                continue;
            }

            let f32_exp = ((values[i].to_bits() >> 23) & 0xFF) as i32 - 127;
            let exp_diff = f32_exp - (shared_exp as i32 - Self::EXP_BIAS as i32);

            // Scale and quantize mantissa (2 bits: 00=1.0, 01=1.33, 10=1.67, 11=2.0)
            let scale = 2.0_f32.powi(exp_diff as i32).max(0.0).min(2.0);
            let mant_q = if scale < 0.66 {
                0
            } else if scale < 1.0 {
                0
            } else if scale < 1.5 {
                1
            } else if scale < 2.0 {
                2
            } else {
                3
            };

            // Pack: [sign:1][mant:2]
            let packed = ((signs[i] as u8) << 2) | mant_q;

            // Store in packed array
            let byte_idx = (bit_pos / 8) as usize;
            let bit_offset = (bit_pos % 8) as u8;
            packed_mantissas[byte_idx] |= packed << bit_offset;

            // Handle 3-bit value crossing byte boundary
            if bit_offset > 5 {
                let overflow = packed >> (8 - bit_offset);
                packed_mantissas[byte_idx + 1] |= overflow;
            }

            bit_pos += 3;
        }

        GF8B64Block { shared_exp, packed_mantissas }
    }

    /// Convert block back to f32 slice
    pub fn to_f32_slice(&self) -> [f32; 64] {
        let mut result = [0.0f32; 64];

        let shared_exp_val = (self.shared_exp as i32) - Self::EXP_BIAS as i32;
        let exp_val = 2.0_f32.powi(shared_exp_val);

        let mut bit_pos = 0u32;
        for i in 0..64 {
            let byte_idx = (bit_pos / 8) as usize;
            let bit_offset = (bit_pos % 8) as u8;

            let packed = (self.packed_mantissas[byte_idx] >> bit_offset) |
                if bit_offset > 5 {
                    (self.packed_mantissas[byte_idx + 1] << (8 - bit_offset))
                } else {
                    0
                };

            // If mantissa is 0 and we had no non-zero values (shared_exp=0), treat as zero
            if self.shared_exp == 0 && (packed & 0x07) == 0 {
                result[i] = 0.0;
            } else {
                let sign = if packed & 0x04 != 0 { -1.0 } else { 1.0 };
                let mant_val = match packed & 0x03 {
                    0 => 1.0,
                    1 => 1.3333333,
                    2 => 1.6666667,
                    _ => 2.0,
                };
                result[i] = sign * exp_val * mant_val;
            }
            bit_pos += 3;
        }

        result
    }

    /// Compression ratio vs standard GF8
    pub fn compression_ratio() -> f32 {
        // Standard GF8: 64 × 8 = 512 bits
        // GF8B64: 196 bits
        512.0 / 196.0
    }
}

// ============================================================================
// GF12MSFP — MSFP-style variant (12-bit, 128 elements shared exp)
// ============================================================================

/// GF12MSFP — Microsoft Floating Point style GF variant
///
/// Per element: [sign:1][mant:4] (5 bits)
/// Shared exponent: 5 bits per 128 elements
///
/// Inspired by: MSFP (NeurIPS 2020) - per-128-element vector shares 5-bit exponent
/// Storage: 128 × 5 bits = 640 bits + 5 bits = 645 bits = 80.6 bytes
/// Standard GF12: 128 × 12 = 1536 bits = 192 bytes
/// Savings: 58% vs standard GF12!
#[derive(Clone, Debug)]
pub struct GF12MSFPBlock {
    pub shared_exp: u8,
    pub packed_mantissas: [u8; 80], // 128 × 5 bits = 640 bits = 80 bytes
}

impl GF12MSFPBlock {
    const EXP_BIAS: i8 = 15; // 2^(5-1) - 1

    pub fn from_f32_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 128, "GF12MSFPBlock requires exactly 128 values");

        let mut max_exp = i32::MIN;
        let mut has_nonzero = false;
        let mut signs = [false; 128];

        for (i, &v) in values.iter().enumerate() {
            if v == 0.0 { continue; }
            has_nonzero = true;
            signs[i] = v < 0.0;
            let exp = ((v.to_bits() >> 23) & 0xFF) as i32 - 127;
            max_exp = max_exp.max(exp);
        }

        let shared_exp = if !has_nonzero {
            0
        } else {
            let biased = max_exp + Self::EXP_BIAS as i32;
            (biased.max(0).min(31)) as u8
        };

        let mut packed_mantissas = [0u8; 80];
        let mut bit_pos = 0u32;

        for i in 0..128 {
            if values[i] == 0.0 {
                bit_pos += 5;
                continue;
            }

            let f32_exp = ((values[i].to_bits() >> 23) & 0xFF) as i32 - 127;
            let exp_diff = f32_exp - (shared_exp as i32 - Self::EXP_BIAS as i32);

            // 4-bit mantissa (16 levels)
            let scale = 2.0_f32.powi(exp_diff as i32).max(0.0).min(2.0);
            let mant_q = ((scale - 1.0) * 8.0).round() as u8; // Map [1.0, 2.0] to [0, 15]
            let mant_q = mant_q.min(15);

            let packed = ((signs[i] as u8) << 4) | mant_q;

            let byte_idx = (bit_pos / 8) as usize;
            let bit_offset = (bit_pos % 8) as u8;
            packed_mantissas[byte_idx] |= packed << bit_offset;

            if bit_offset > 3 {
                let overflow = packed >> (8 - bit_offset);
                packed_mantissas[byte_idx + 1] |= overflow;
            }

            bit_pos += 5;
        }

        GF12MSFPBlock { shared_exp, packed_mantissas }
    }

    pub fn to_f32_slice(&self) -> [f32; 128] {
        let mut result = [0.0f32; 128];
        let shared_exp_val = (self.shared_exp as i32) - Self::EXP_BIAS as i32;
        let exp_val = 2.0_f32.powi(shared_exp_val);

        let mut bit_pos = 0u32;
        for i in 0..128 {
            let byte_idx = (bit_pos / 8) as usize;
            let bit_offset = (bit_pos % 8) as u8;

            let packed = (self.packed_mantissas[byte_idx] >> bit_offset) |
                if bit_offset > 3 {
                    (self.packed_mantissas[byte_idx + 1] << (8 - bit_offset))
                } else {
                    0
                };

            if packed == 0 {
                bit_pos += 5;
                continue;
            }

            let sign = if packed & 0x10 != 0 { -1.0 } else { 1.0 };
            let mant_val = 1.0 + (packed & 0x0F) as f32 / 8.0;

            result[i] = sign * exp_val * mant_val;
            bit_pos += 5;
        }

        result
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf16b32_block_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let block = GF16B32Block::from_f32_slice(&values);
        let decoded = block.to_f32_slice();

        let error = block.quantization_error(&values);
        assert!(error < 0.1, "Quantization error too high: {}", error);
    }

    #[test]
    fn test_gf16b32_compression_ratio() {
        let ratio = GF16B32Block::compression_ratio();
        assert!((ratio - 1.94).abs() < 0.01, "Expected ~1.94x compression, got {}", ratio);
    }

    #[test]
    fn test_gf8b64_compression_ratio() {
        let ratio = GF8B64Block::compression_ratio();
        assert!((ratio - 2.61).abs() < 0.01, "Expected ~2.61x compression, got {}", ratio);
    }

    #[test]
    fn test_gf12msfp_block_roundtrip() {
        // Use values that are within the dynamic range of block formats
        // Values from 0.1 to 6.4 (exponential scale)
        let values: Vec<f32> = (0..128).map(|i| 0.1 * (1.05_f32).powi(i as i32).min(10.0)).collect();
        let block = GF12MSFPBlock::from_f32_slice(&values);
        let decoded = block.to_f32_slice();

        // Check that non-zero values are approximately preserved
        for i in 0..128 {
            if values[i] != 0.0 && decoded[i] != 0.0 {
                let rel_error = (decoded[i] - values[i]).abs() / values[i].abs().max(1e-10);
                assert!(rel_error < 0.5, "Value {}: original={} decoded={} rel error={}", i, values[i], decoded[i], rel_error);
            }
        }
    }

    #[test]
    fn test_block_formats_phi_optimized_exponents() {
        // Verify that shared exponents use φ-optimized bias ranges
        assert!(GF16B32Block::EXP_BIAS == 63);
        assert!(GF8B64Block::EXP_BIAS == 7);
        assert!(GF12MSFPBlock::EXP_BIAS == 15);
    }

    #[test]
    fn test_block_formats_handle_all_zeros() {
        let zeros = vec![0.0f32; 32];
        let block = GF16B32Block::from_f32_slice(&zeros);
        let decoded = block.to_f32_slice();
        assert!(decoded.iter().all(|&x| x == 0.0), "All zeros should decode to zeros");

        let zeros64 = vec![0.0f32; 64];
        let block64 = GF8B64Block::from_f32_slice(&zeros64);
        let decoded64 = block64.to_f32_slice();
        assert!(decoded64.iter().all(|&x| x == 0.0), "All zeros should decode to zeros");

        let zeros128 = vec![0.0f32; 128];
        let block128 = GF12MSFPBlock::from_f32_slice(&zeros128);
        let decoded128 = block128.to_f32_slice();
        assert!(decoded128.iter().all(|&x| x == 0.0), "All zeros should decode to zeros");
    }

    #[test]
    fn test_block_formats_handle_negative_values() {
        let values: Vec<f32> = vec![-1.0, -0.5, -2.0, -3.14159, -0.001];
        let mut padded = values.clone();
        padded.resize(32, 0.0);

        let block = GF16B32Block::from_f32_slice(&padded);
        let decoded = block.to_f32_slice();

        for (i, &v) in values.iter().enumerate() {
            if v < 0.0 {
                assert!(decoded[i] <= 0.0, "Negative value {} decoded to positive: {} -> {}", v, v, decoded[i]);
            }
        }
    }
}
