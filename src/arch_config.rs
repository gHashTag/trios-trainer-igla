//! Wave 31 PR-B — env-gated architectural knobs.
//! Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
//!
//! Three knobs read from environment at trainer init:
//!
//! | Env var          | Default | Wave-32 value | Forbidden |
//! |------------------|---------|---------------|-----------|
//! | `HIDDEN_DIM`     | 384     | 1024          | <256      |
//! | `NUM_ATTN_LAYERS`| 1       | 4             | <1        |
//! | `GF16_ENABLED`   | false   | true          | —         |
//!
//! Defaults preserve the Wave-30 baseline (h=384, 1L, no GF16).
//! Wave 32 redeploy sets all three to attack Gate-2 (BPB<1.85).

/// Parse `HIDDEN_DIM` from environment.
///
/// Default: 384 (Wave-30 arch floor).
/// Forbidden: h < 256 (R6).
pub fn parse_hidden_dim() -> Result<usize, String> {
    let raw = std::env::var("HIDDEN_DIM").unwrap_or_else(|_| "384".to_string());
    let h: usize = raw.parse().map_err(|e| format!("HIDDEN_DIM parse: {e}"))?;
    if h < 256 {
        return Err(format!("HIDDEN_DIM={h} < 256 forbidden (R6)"));
    }
    Ok(h)
}

/// Parse `NUM_ATTN_LAYERS` from environment.
///
/// Default: 1 (Wave-30 arch floor).
/// Forbidden: n < 1.
pub fn parse_num_attn_layers() -> Result<usize, String> {
    let raw = std::env::var("NUM_ATTN_LAYERS").unwrap_or_else(|_| "1".to_string());
    let n: usize = raw
        .parse()
        .map_err(|e| format!("NUM_ATTN_LAYERS parse: {e}"))?;
    if n < 1 {
        return Err(format!("NUM_ATTN_LAYERS={n} < 1 forbidden"));
    }
    Ok(n)
}

/// Parse `GF16_ENABLED` from environment.
///
/// Default: false (Wave-30 baseline, no GF16 quantisation).
/// Accepts: true/1/yes/on → true; false/0/no/off → false.
pub fn parse_gf16_enabled() -> Result<bool, String> {
    let raw = std::env::var("GF16_ENABLED").unwrap_or_else(|_| "false".to_string());
    match raw.to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(format!("GF16_ENABLED parse: invalid value {raw}")),
    }
}

/// Parsed arch knobs — returned as a struct for convenience.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchConfig {
    pub hidden_dim: usize,
    pub num_attn_layers: usize,
    pub gf16_enabled: bool,
}

/// Parse all three knobs at once. Fails fast on the first violation.
pub fn parse_arch_config() -> Result<ArchConfig, String> {
    Ok(ArchConfig {
        hidden_dim: parse_hidden_dim()?,
        num_attn_layers: parse_num_attn_layers()?,
        gf16_enabled: parse_gf16_enabled()?,
    })
}
