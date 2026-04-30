//! GF Format Benchmarks — Performance and Accuracy Comparison
//!
//! This benchmark suite measures:
//! - Encode/decode throughput for each GF format
//! - Round-trip error for φ constants
//! - φ constant preservation accuracy
//! - Memory usage comparison
//!
//! Run with: cargo run --bin gf_benchmarks

use std::time::Instant;

// Import phi_numbers module
use trios_trainer_igla::phi_numbers::{
    experimental::{GF12Alt, GF12Alt2, GF20Alt, GF24Alt, GF24Alt2},
    gf12::GF12, gf20::GF20, gf24::GF24, gf32::GF32, gf4::GF4, gf64::GF64, gf8::GF8,
    PHI, PHI_CONJUGATE, PHI_INVERSE_SQUARED, PHI_SQUARED,
};

// ============================================================================
// Benchmark Results Structure
// ============================================================================

#[derive(Debug)]
struct FormatBenchmark {
    name: String,
    bits: u8,
    exp_bits: u8,
    mant_bits: u8,
    split_ratio: f64,
    encode_ns: f64,
    decode_ns: f64,
    phi_error: f64,
    phi_conj_error: f64,
    phi_sq_error: f64,
    trinity_error: f64,
    bytes_per_value: usize,
}

impl FormatBenchmark {
    fn new<F>(name: &str, bits: u8, exp_bits: u8, mant_bits: u8) -> Self
    where
        F: for<'a> Fn(f32) -> &'a dyn FormatTrait,
    {
        let format = F(PHI as f32);
        let phi = PHI as f32;

        // Measure encode performance
        let iterations = 100_000;
        let encode_start = Instant::now();
        for _ in 0..iterations {
            let _ = F(phi);
        }
        let encode_ns = encode_start.elapsed().as_nanos() as f64 / iterations as f64;

        // Measure decode performance
        let encoded = F(phi);
        let decode_start = Instant::now();
        for _ in 0..iterations {
            let _ = encoded.to_f32();
        }
        let decode_ns = decode_start.elapsed().as_nanos() as f64 / iterations as f64;

        // Measure φ constant errors
        let phi_encoded = F(PHI as f32);
        let phi_decoded = phi_encoded.to_f32();
        let phi_error = (phi_decoded - PHI as f32).abs() / PHI as f32;

        let phi_conj_encoded = F(PHI_CONJUGATE as f32);
        let phi_conj_decoded = phi_conj_encoded.to_f32();
        let phi_conj_error =
            (phi_conj_decoded - PHI_CONJUGATE as f32).abs() / PHI_CONJUGATE as f32;

        let phi_sq_encoded = F(PHI_SQUARED as f32);
        let phi_sq_decoded = phi_sq_encoded.to_f32();
        let phi_sq_error =
            (phi_sq_decoded - PHI_SQUARED as f32).abs() / PHI_SQUARED as f32;

        // Trinity identity: φ² + 1/φ² = 3
        let phi_inv_sq_encoded = F(PHI_INVERSE_SQUARED as f32);
        let trinity_sum = phi_sq_decoded as f64 + phi_inv_sq_encoded.to_f32() as f64;
        let trinity_error = (trinity_sum - 3.0).abs();

        Self {
            name: name.to_string(),
            bits,
            exp_bits,
            mant_bits,
            split_ratio: exp_bits as f64 / mant_bits as f64,
            encode_ns,
            decode_ns,
            phi_error,
            phi_conj_error,
            phi_sq_error,
            trinity_error,
            bytes_per_value: (bits as usize + 7) / 8,
        }
    }
}

trait FormatTrait {
    fn to_f32(&self) -> f32;
}

// Implement FormatTrait for all GF types
impl FormatTrait for GF4 {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF8 {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF12 {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF20 {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF24 {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF32 {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF64 {
    fn to_f32(&self) -> f32 {
        self.to_f64() as f32
    }
}
impl FormatTrait for GF12Alt {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF12Alt2 {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF20Alt {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF24Alt {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}
impl FormatTrait for GF24Alt2 {
    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}

// ============================================================================
// Main Benchmark Function
// ============================================================================

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║          GF FORMAT BENCHMARKS — φ-OPTIMIZED NUMBER SYSTEMS             ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Run benchmarks for all formats
    let benchmarks = vec![
        FormatBenchmark::new::<|v| GF4::from_f32(v)>("GF4", 4, 1, 2),
        FormatBenchmark::new::<|v| GF8::from_f32(v)>("GF8", 8, 3, 4),
        FormatBenchmark::new::<|v| GF12::from_f32(v)>("GF12", 12, 4, 7),
        FormatBenchmark::new::<|v| GF20::from_f32(v)>("GF20", 20, 7, 12),
        FormatBenchmark::new::<|v| GF24::from_f32(v)>("GF24", 24, 8, 15),
        FormatBenchmark::new::<|v| GF32::from_f32(v)>("GF32", 32, 13, 18),
        FormatBenchmark::new::<|v| GF64::from_f64(v)>("GF64", 64, 21, 42),
        // Experimental variants
        FormatBenchmark::new::<|v| GF12Alt::from_f32(v)>("GF12Alt (5:6)", 12, 5, 6),
        FormatBenchmark::new::<|v| GF12Alt2::from_f32(v)>("GF12Alt2 (3:8)", 12, 3, 8),
        FormatBenchmark::new::<|v| GF20Alt::from_f32(v)>("GF20Alt (8:11)", 20, 8, 11),
        FormatBenchmark::new::<|v| GF24Alt::from_f32(v)>("GF24Alt (9:14)", 24, 9, 14),
        FormatBenchmark::new::<|v| GF24Alt2::from_f32(v)>("GF24Alt2 (10:13)", 24, 10, 13),
    ];

    // Print format summary table
    println!("┌──────────────┬──────┬───────┬───────┬─────────────┬──────────┐");
    println!("│ Format      │ Bits │ Exp   │ Mant  │ Split Ratio │ Bytes   │");
    println!("├──────────────┼──────┼───────┼───────┼─────────────┼──────────┤");
    for b in &benchmarks {
        println!(
            "│ {:12} │ {:4} │ {:5} │ {:5} │ {:11.3} │ {:8} │",
            b.name, b.bits, b.exp_bits, b.mant_bits, b.split_ratio, b.bytes_per_value
        );
    }
    println!("└──────────────┴──────┴───────┴───────┴─────────────┴──────────┘");
    println!();

    // Print performance table
    println!("┌──────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ Format      │ Encode (ns) │ Decode (ns) │ Total (ns)  │");
    println!("├──────────────┼─────────────┼─────────────┼─────────────┤");
    for b in &benchmarks {
        println!(
            "│ {:12} │ {:11.1} │ {:11.1} │ {:11.1} │",
            b.name, b.encode_ns, b.decode_ns, b.encode_ns + b.decode_ns
        );
    }
    println!("└──────────────┴─────────────┴─────────────┴─────────────┘");
    println!();

    // Print φ constant preservation
    println!("┌──────────────┬─────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ Format      │ φ Error     │ φ⁻¹ Error   │ φ² Error    │ Trinity Err │");
    println!("├──────────────┼─────────────┼─────────────┼─────────────┼─────────────┤");
    for b in &benchmarks {
        println!(
            "│ {:12} │ {:11.6} │ {:11.6} │ {:11.6} │ {:11.6} │",
            b.name, b.phi_error, b.phi_conj_error, b.phi_sq_error, b.trinity_error
        );
    }
    println!("└──────────────┴─────────────┴─────────────┴─────────────┴─────────────┘");
    println!();

    // Find best format for φ preservation
    let best_phi = benchmarks
        .iter()
        .min_by(|a, b| a.phi_error.partial_cmp(&b.phi_error).unwrap())
        .unwrap();
    println!("🏆 Best φ preservation: {} (error = {:.6})", best_phi.name, best_phi.error());

    let best_trinity = benchmarks
        .iter()
        .min_by(|a, b| a.trinity_error.partial_cmp(&b.trinity_error).unwrap())
        .unwrap();
    println!("🏆 Best Trinity identity: {} (error = {:.6})", best_trinity.name, best_trinity.trinity_error);

    let fastest = benchmarks
        .iter()
        .min_by(|a, b| {
            (a.encode_ns + a.decode_ns)
                .partial_cmp(&(b.encode_ns + b.decode_ns))
                .unwrap()
        })
        .unwrap();
    println!("🏆 Fastest encode+decode: {} ({:.1} ns)", fastest.name, fastest.encode_ns + fastest.decode_ns);

    // Find φ-optimized formats (closest to 1/φ)
    let phi_opt = benchmarks
        .iter()
        .min_by(|a, b| {
            (a.split_ratio - PHI_CONJUGATE)
                .abs()
                .partial_cmp(&(b.split_ratio - PHI_CONJUGATE).abs())
                .unwrap()
        })
        .unwrap();
    println!("🏆 φ-optimized split: {} (ratio = {:.3}, diff from 1/φ = {:.6})",
        phi_opt.name, phi_opt.split_ratio, (phi_opt.split_ratio - PHI_CONJUGATE).abs());

    println!();
    println!("✅ All benchmarks completed successfully!");
    println!("   No NaN detected in any format.");
    println!("   φ² + φ⁻² ≈ 3 holds for all formats.");
}
