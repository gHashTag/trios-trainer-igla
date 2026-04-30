//! Golden Float Family — φ-based number systems
//!
//! Implements φ-optimized floating point formats and related golden ratio constants.
//!
//! ## Format Family
//!
//! | Format | Total Bits | Exp | Mant | Block Size | Use Case |
//! |--------|-----------|-----|------|------------|----------|
//! | GF4 | 4 | 1 | 2 | 1 | Minimal storage |
//! | GF8 | 8 | 3 | 4 | 1 | Ultra-compression |
//! | GF8B64 | 8 | 4 (shared) | 2 | 64 | Embeddings (62% savings) |
//! | GF12 | 12 | 4 | 7 | 1 | General quantization |
//! | GF12MSFP | 12 | 5 (shared) | 4 | 128 | MSFP-style (58% savings) |
//! | GF16 | 16 | 6 | 9 | 1 | **Normative** (φ-optimized) |
//! | GF16B32 | 16 | 8 (shared) | 7 | 32 | MX-style (48% savings) |
//! | GF20 | 20 | 7 | 12 | 1 | High precision |
//! | GF24 | 24 | 9 | 14 | 1 | Near-FP32 |
//! | GF32 | 32 | 13 | 18 | 1 | FP32 replacement |
//! | GF64 | 64 | 21 | 42 | 1 | FP64 replacement |
//!
//! ## Block Formats
//!
//! Block formats (GF8B64, GF12MSFP, GF16B32) are inspired by OCP MX Formats
//! and Microsoft Floating Point (MSFP), where multiple elements share a
//! common exponent. This dramatically reduces memory footprint while
//! preserving accuracy for correlated values (e.g., embeddings, activations).

pub mod block_variants;
pub mod experimental;
pub mod fibonacci_dims;
pub mod gf12;
pub mod gf20;
pub mod gf24;
pub mod gf32;
pub mod gf4;
pub mod gf64;
pub mod gf8;
pub mod gfternary;
pub mod phi_constants;
pub mod precision_format;

pub use block_variants::{GF16B32Block, GF8B64Block, GF12MSFPBlock};
pub use experimental::{
    GF12Alt, GF12Alt2, GF20Alt, GF24Alt, GF24Alt2,
};
pub use fibonacci_dims::*;
pub use gf12::GF12;
pub use gf20::GF20;
pub use gf24::GF24;
pub use gf32::GF32;
pub use gf4::GF4;
pub use gf64::GF64;
pub use gf8::GF8;
pub use gfternary::GFTernary;
pub use phi_constants::*;
pub use precision_format::{BF16, FP16, FP32, GF4a, GF6a, PrecisionFormat, Quantize};

/// Main Trinity identity: φ² + 1/φ² = 3
pub const TRINITY_IDENTITY: f64 = 3.0;

/// Verify Trinity identity holds
pub fn verify_trinity_identity() -> bool {
    (PHI * PHI + (1.0 / (PHI * PHI)) - TRINITY_IDENTITY).abs() < 1e-10
}
