//! Golden Float Family — φ-based number systems
//!
//! Implements φ-optimized floating point formats and related golden ratio constants.
//! Full family: GFTernary · GF4 · GF8 · GF12 · GF16 (Coq-proven) · GF20 · GF24 ·
//! GF32 · GF64 · GF128 · GF256. PhD Glava 06 / 09 / 23.

pub mod fibonacci_dims;
pub mod gf12;
pub mod gf128;
pub mod gf20;
pub mod gf24;
pub mod gf256;
pub mod gf32;
pub mod gf4;
pub mod gf64;
pub mod gf8;
pub mod gfternary;
pub mod phi_constants;

pub use fibonacci_dims::*;
pub use gf12::GF12;
pub use gf128::GF128;
pub use gf20::GF20;
pub use gf24::GF24;
pub use gf256::GF256;
pub use gf32::GF32;
pub use gf4::GF4;
pub use gf64::GF64;
pub use gf8::GF8;
pub use gfternary::GFTernary;
pub use phi_constants::*;

/// Main Trinity identity: φ² + 1/φ² = 3
pub const TRINITY_IDENTITY: f64 = 3.0;

/// Verify Trinity identity holds
pub fn verify_trinity_identity() -> bool {
    (PHI * PHI + (1.0 / (PHI * PHI)) - TRINITY_IDENTITY).abs() < 1e-10
}
