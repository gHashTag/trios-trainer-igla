//! Canon #93 seed enforcement for Wave 29.
//!
//! Reads `SEED` from the environment and validates against the forbidden set.
//!
//! Forbidden: {42, 43, 44, 45}.
//! Allowed canon set: {47, 89, 123, 144}.
//!
//! Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877

/// Parse SEED from environment and enforce Canon #93.
/// Forbidden: {42, 43, 44, 45}. Allowed canon set: {47, 89, 123, 144}.
/// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
pub fn parse_seed() -> Result<u64, String> {
    let raw = std::env::var("SEED")
        .map_err(|_| "SEED env var unset (Canon #93 requires explicit seed)".to_string())?;
    let seed: u64 = raw.parse().map_err(|e| format!("SEED parse error: {e}"))?;
    const FORBIDDEN: &[u64] = &[42, 43, 44, 45];
    if FORBIDDEN.contains(&seed) {
        return Err(format!(
            "seed {seed} forbidden under Canon #93 (allowed: 47, 89, 123, 144)"
        ));
    }
    Ok(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_seed_47() {
        std::env::set_var("SEED", "47");
        assert_eq!(parse_seed(), Ok(47));
    }

    #[test]
    fn allowed_seed_89() {
        std::env::set_var("SEED", "89");
        assert_eq!(parse_seed(), Ok(89));
    }

    #[test]
    fn allowed_seed_123() {
        std::env::set_var("SEED", "123");
        assert_eq!(parse_seed(), Ok(123));
    }

    #[test]
    fn allowed_seed_144() {
        std::env::set_var("SEED", "144");
        assert_eq!(parse_seed(), Ok(144));
    }

    #[test]
    fn forbidden_seed_42() {
        std::env::set_var("SEED", "42");
        let err = parse_seed().unwrap_err();
        assert!(
            err.contains("forbidden"),
            "expected 'forbidden' in error: {err}"
        );
    }

    #[test]
    fn forbidden_seed_43() {
        std::env::set_var("SEED", "43");
        let err = parse_seed().unwrap_err();
        assert!(
            err.contains("forbidden"),
            "expected 'forbidden' in error: {err}"
        );
    }

    #[test]
    fn forbidden_seed_44() {
        std::env::set_var("SEED", "44");
        let err = parse_seed().unwrap_err();
        assert!(err.contains("forbidden"));
    }

    #[test]
    fn forbidden_seed_45() {
        std::env::set_var("SEED", "45");
        let err = parse_seed().unwrap_err();
        assert!(err.contains("forbidden"));
    }
}
