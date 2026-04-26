//! Golden Float Family — φ-based number systems
//!
//! Implements φ-optimized floating point formats and related golden ratio constants.

pub mod fibonacci_dims;
pub mod gf32;
pub mod gf64;
pub mod gf8;
pub mod gfternary;
pub mod phi_constants;

pub use fibonacci_dims::*;
pub use gf32::GF32;
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
