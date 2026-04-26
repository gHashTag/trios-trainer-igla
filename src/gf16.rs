//! GoldenFloat16 façade.
//!
//! When built with `--features trios-integration`, all symbols are re-exported
//! from the canonical `trios-golden-float` crate. Otherwise this file provides
//! a minimal compile-only stub so the trainer skeleton builds standalone.

#[cfg(feature = "trios-integration")]
pub use trios_golden_float::*;

#[cfg(not(feature = "trios-integration"))]
mod stub {
    /// Placeholder for the GoldenFloat16 type.
    /// Real implementation lives in `trios-golden-float` (gHashTag/trios).
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Gf16(pub u16);
}

#[cfg(not(feature = "trios-integration"))]
pub use stub::*;
