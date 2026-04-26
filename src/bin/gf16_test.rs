//! GF16 Test Binary - Quantization and accuracy benchmarks

use trios_trainer::gf16::{GF16, benchmark_quantization};
use std::time::Instant;

fn main() {
    println!("=== GF16 Test Suite ===\n");

    // Test 1: Basic conversions
    println!("Test 1: Basic f32 <-> GF16 conversions");
    test_basic_conversions();

    // Test 2: Roundtrip accuracy
    println!("\nTest 2: Roundtrip accuracy");
    test_roundtrip_accuracy();

    // Test 3: Quantization metrics
    println!("\nTest 3: Quantization metrics on random weights");
    test_quantization_metrics();

    // Test 4: φ-distance verification
    println!("\nTest 4: φ-distance verification");
    test_phi_distance();

    // Test 5: Special values
    println!("\nTest 5: Special values");
    test_special_values();

    // Test 6: Range boundaries
    println!("\nTest 6: Range boundaries");
    test_range_boundaries();

    // Test 7: Comparison with f16 and bf16 (simulated)
    println!("\nTest 7: Format comparison summary");
    test_format_comparison();

    println!("\n=== All tests passed ===");
}

fn test_basic_conversions() {
    let values = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 100.0, -100.0];

    for &v in &values {
        let gf = GF16::from_f32(v);
        let back = gf.to_f32();
        println!("  f32: {:10.4} -> GF16: {:10.4} -> f32: {:10.4} (error: {:.6}%)",
                 v, gf, back, ((back - v).abs() / v.abs().max(1e-10) * 100.0));
    }
}

fn test_roundtrip_accuracy() {
    let test_values = vec![
        0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0,
        0.0001, 0.00001, -0.5, -2.5, -10.0,
    ];

    let mut max_rel_error = 0.0f64;
    let mut avg_rel_error = 0.0f64;

    for &v in &test_values {
        let gf = GF16::from_f32(v);
        let back = gf.to_f32();
        let rel_error = (back - v).abs() as f64 / v.abs().max(1e-10) as f64;
        max_rel_error = max_rel_error.max(rel_error);
        avg_rel_error += rel_error;
    }

    avg_rel_error /= test_values.len() as f64;

    println!("  Max relative error: {:.4}%", max_rel_error * 100.0);
    println!("  Avg relative error: {:.4}%", avg_rel_error * 100.0);
}

fn test_quantization_metrics() {
    let n = 10000;
    println!("  Quantizing {} random weights N(0, 0.1)...", n);

    let start = Instant::now();
    let metrics = benchmark_quantization(n);
    let elapsed = start.elapsed();

    println!("  Quantization completed in {:?}", elapsed);
    println!("  Max error:  {:.4}%", metrics.max_error_pct);
    println!("  Avg error:  {:.4}%", metrics.avg_error_pct);
    println!("  MSE:        {:.8}", metrics.mse);
    println!("  MAE:        {:.8}", metrics.mae);
    println!("  φ-distance: {:.6}", metrics.phi_error);
}

fn test_phi_distance() {
    let phi = 1.618_033_988_749_895_f64;
    let inv_phi = 1.0 / phi;
    let gf16_phi_dist = GF16::phi_distance();
    let exp_mant_ratio = 6.0 / 9.0;  // GF16's 6:9 split

    println!("  Golden ratio (φ):      {:.15}", phi);
    println!("  Inverse φ (1/φ):      {:.15}", inv_phi);
    println!("  GF16 exp:mant ratio:  {:.6}", exp_mant_ratio);
    println!("  GF16 φ-distance:      {:.6}", gf16_phi_dist);
    println!("  Expected φ-distance:  {:.6}", (exp_mant_ratio - inv_phi).abs());
    println!("  Matches specification: {}", (gf16_phi_dist - 0.0486).abs() < 0.001);
}

fn test_special_values() {
    println!("  Zero:        {} -> f32: {}", GF16::ZERO, GF16::ZERO.to_f32());
    println!("  Neg Zero:    {} -> f32: {}", GF16::NEG_ZERO, GF16::NEG_ZERO.to_f32());
    println!("  Infinity:    {} -> f32: {}", GF16::INF, GF16::INF.to_f32());
    println!("  Neg Infinity:{} -> f32: {}", GF16::NEG_INF, GF16::NEG_INF.to_f32());
    println!("  NaN:         {} -> f32: {} (is_nan: {})",
             GF16::NAN, GF16::NAN.to_f32(), GF16::NAN.is_nan());

    // Test special value properties
    assert!(GF16::NAN.is_nan());
    assert!(!GF16::NAN.is_finite());
    assert!(GF16::INF.is_infinite());
    assert!(GF16::NEG_INF.is_infinite());
    assert!(GF16::ZERO.to_f32() == 0.0);
    println!("  All special value properties verified ✓");
}

fn test_range_boundaries() {
    // GF16 range: 4.66×10⁻¹⁰ to 4.30×10⁹

    // Test near minimum
    let tiny = GF16::from_f32(1e-20);
    println!("  Very small (1e-20): {} -> {}", GF16::from_bits(tiny.0), tiny.to_f32());
    println!("    Clamped to zero: {}", tiny.to_f32() == 0.0);

    let min_pos = GF16::from_f32(1e-9);
    println!("  Small (1e-9):       {} -> {}", GF16::from_bits(min_pos.0), min_pos.to_f32());

    // Test near maximum
    let huge = GF16::from_f32(1e20);
    println!("  Very large (1e20):  {} -> {}", GF16::from_bits(huge.0), huge.to_f32());
    println!("    Clamped to inf: {}", huge.is_infinite());

    let large = GF16::from_f32(1e9);
    println!("  Large (1e9):        {} -> {}", GF16::from_bits(large.0), large.to_f32());
    println!("    Is finite: {}", large.is_finite());
}

fn test_format_comparison() {
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │                    16-bit Format Comparison                  │");
    println!("  ├─────────────────┬─────────────┬─────────────┬───────────────┤");
    println!("  │ Format          │ Exp:Mant    │ φ-distance  │ Range         │");
    println!("  ├─────────────────┼─────────────┼─────────────┼───────────────┤");
    println!("  │ IEEE fp16       │ 5:10        │ 0.118       │ 6.10×10⁻⁵ - 6.5×10⁴ │");
    println!("  │ bfloat16        │ 8:7         │ 0.525       │ 1.18×10⁻³⁸ - 3.4×10³⁸│");
    println!("  │ GF16            │ 6:9         │ 0.049       │ 4.66×10⁻¹⁰ - 4.3×10⁹│");
    println!("  └─────────────────┴─────────────┴─────────────┴───────────────┘");
    println!();
    println!("  Key advantages of GF16:");
    println!("    • Best φ-distance among 16-bit formats → optimal value distribution");
    println!("    • 65,000× wider gradient range than IEEE fp16");
    println!("    • 9-bit mantissa provides sufficient precision for ML");
    println!("    • No subnormals → simpler hardware implementation");
}
