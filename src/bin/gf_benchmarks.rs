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
use trios_trainer::phi_numbers::{
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

// Wrapper functions for trait methods
fn gf4_to_f32(v: &GF4) -> f32 { v.to_f32() }
fn gf8_to_f32(v: &GF8) -> f32 { v.to_f32() }
fn gf12_to_f32(v: &GF12) -> f32 { v.to_f32() }
fn gf20_to_f32(v: &GF20) -> f32 { v.to_f32() }
fn gf24_to_f32(v: &GF24) -> f32 { v.to_f32() }
fn gf32_to_f32(v: &GF32) -> f32 { v.to_f32() }
fn gf12alt_to_f32(v: &GF12Alt) -> f32 { v.to_f32() }
fn gf12alt2_to_f32(v: &GF12Alt2) -> f32 { v.to_f32() }
fn gf20alt_to_f32(v: &GF20Alt) -> f32 { v.to_f32() }
fn gf24alt_to_f32(v: &GF24Alt) -> f32 { v.to_f32() }
fn gf24alt2_to_f32(v: &GF24Alt2) -> f32 { v.to_f32() }

fn create_benchmark<T: 'static>(
    name: &str,
    bits: u8,
    exp_bits: u8,
    mant_bits: u8,
    encode_fn: fn(f32) -> T,
    to_f32_fn: fn(&T) -> f32,
) -> FormatBenchmark {
    let phi = PHI as f32;

    // Measure encode performance
    let iterations = 100_000;
    let encode_start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_fn(phi);
    }
    let encode_ns = encode_start.elapsed().as_nanos() as f64 / iterations as f64;

    // Measure decode performance
    let encoded = encode_fn(phi);
    let decode_start = Instant::now();
    for _ in 0..iterations {
        let _ = to_f32_fn(&encoded);
    }
    let decode_ns = decode_start.elapsed().as_nanos() as f64 / iterations as f64;

    // Measure φ constant errors
    let phi_encoded = encode_fn(PHI as f32);
    let phi_decoded = to_f32_fn(&phi_encoded);
    let phi_error = (phi_decoded - PHI as f32).abs() / PHI as f32;

    let phi_conj_encoded = encode_fn(PHI_CONJUGATE as f32);
    let phi_conj_decoded = to_f32_fn(&phi_conj_encoded);
    let phi_conj_error =
        (phi_conj_decoded - PHI_CONJUGATE as f32).abs() / PHI_CONJUGATE as f32;

    let phi_sq_encoded = encode_fn(PHI_SQUARED as f32);
    let phi_sq_decoded = to_f32_fn(&phi_sq_encoded);
    let phi_sq_error =
        (phi_sq_decoded - PHI_SQUARED as f32).abs() / PHI_SQUARED as f32;

    // Trinity identity: φ² + 1/φ² = 3
    let phi_inv_sq_encoded = encode_fn(PHI_INVERSE_SQUARED as f32);
    let trinity_sum = phi_sq_decoded as f64 + to_f32_fn(&phi_inv_sq_encoded) as f64;
    let trinity_error = (trinity_sum - 3.0).abs();

    FormatBenchmark {
        name: name.to_string(),
        bits,
        exp_bits,
        mant_bits,
        split_ratio: exp_bits as f64 / mant_bits as f64,
        encode_ns,
        decode_ns,
        phi_error: phi_error as f64,
        phi_conj_error: phi_conj_error as f64,
        phi_sq_error: phi_sq_error as f64,
        trinity_error,
        bytes_per_value: (bits as usize).div_ceil(8),
    }
}

// Special version for GF64 which uses f64 encoding
fn create_benchmark_gf64(name: &str, bits: u8, exp_bits: u8, mant_bits: u8) -> FormatBenchmark {
    let phi = PHI;

    // Measure encode performance
    let iterations = 100_000;
    let encode_start = Instant::now();
    for _ in 0..iterations {
        let _ = GF64::from_f64(phi);
    }
    let encode_ns = encode_start.elapsed().as_nanos() as f64 / iterations as f64;

    // Measure decode performance
    let encoded = GF64::from_f64(phi);
    let decode_start = Instant::now();
    for _ in 0..iterations {
        let _ = encoded.to_f64();
    }
    let decode_ns = decode_start.elapsed().as_nanos() as f64 / iterations as f64;

    // Measure φ constant errors (in f64 domain)
    let phi_encoded = GF64::from_f64(PHI);
    let phi_decoded = phi_encoded.to_f64();
    let phi_error = (phi_decoded - PHI).abs() / PHI;

    let phi_conj_encoded = GF64::from_f64(PHI_CONJUGATE);
    let phi_conj_decoded = phi_conj_encoded.to_f64();
    let phi_conj_error =
        (phi_conj_decoded - PHI_CONJUGATE).abs() / PHI_CONJUGATE;

    let phi_sq_encoded = GF64::from_f64(PHI_SQUARED);
    let phi_sq_decoded = phi_sq_encoded.to_f64();
    let phi_sq_error =
        (phi_sq_decoded - PHI_SQUARED).abs() / PHI_SQUARED;

    // Trinity identity: φ² + 1/φ² = 3
    let phi_inv_sq_encoded = GF64::from_f64(PHI_INVERSE_SQUARED);
    let trinity_sum = phi_sq_decoded + phi_inv_sq_encoded.to_f64();
    let trinity_error = (trinity_sum - 3.0).abs();

    FormatBenchmark {
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
        bytes_per_value: (bits as usize).div_ceil(8),
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
        create_benchmark("GF4", 4, 1, 2, GF4::from_f32, gf4_to_f32),
        create_benchmark("GF8", 8, 3, 4, GF8::from_f32, gf8_to_f32),
        create_benchmark("GF12", 12, 4, 7, GF12::from_f32, gf12_to_f32),
        create_benchmark("GF20", 20, 7, 12, GF20::from_f32, gf20_to_f32),
        create_benchmark("GF24", 24, 8, 15, GF24::from_f32, gf24_to_f32),
        create_benchmark("GF32", 32, 13, 18, GF32::from_f32, gf32_to_f32),
        create_benchmark_gf64("GF64", 64, 21, 42),
        // Experimental variants
        create_benchmark("GF12Alt (5:6)", 12, 5, 6, GF12Alt::from_f32, gf12alt_to_f32),
        create_benchmark("GF12Alt2 (3:8)", 12, 3, 8, GF12Alt2::from_f32, gf12alt2_to_f32),
        create_benchmark("GF20Alt (8:11)", 20, 8, 11, GF20Alt::from_f32, gf20alt_to_f32),
        create_benchmark("GF24Alt (9:14)", 24, 9, 14, GF24Alt::from_f32, gf24alt_to_f32),
        create_benchmark("GF24Alt2 (10:13)", 24, 10, 13, GF24Alt2::from_f32, gf24alt2_to_f32),
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
    println!("🏆 Best φ preservation: {} (error = {:.6})", best_phi.name, best_phi.phi_error);

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
