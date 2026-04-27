//! Trinity 3k FineWeb Training for Parameter Golf #110
//!
//! Training Trinity 3k on actual FineWeb dataset (binary format)

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use trios_trainer::trinity_3k::{AdamWConfig, Trinity3kConfig, Trinity3kModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Trinity 3k FineWeb Training for Parameter Golf #110");
    println!("🎯 Target: <1.15 BPB on FineWeb validation");
    println!("📊 Architecture: 3^k dimensions with byte-level processing");

    // Trinity 3k configuration
    let config = Trinity3kConfig::default();

    println!("📋 Configuration:");
    println!("  • Vocab size: {} (3^6)", config.vocab_size);
    println!("  • Hidden dim: {} (3^5)", config.hidden_dim);
    println!(
        "  • Heads: {} (3^3) x dim: {} (3^2)",
        config.n_heads, config.head_dim
    );
    println!("  • Layers: {}", config.n_layers);
    println!("  • Total params: {}", config.total_params());

    // Create model
    let mut model = Trinity3kModel::new(config)?;
    println!("✅ Model created successfully");

    // Load FineWeb data
    println!("📊 Loading FineWeb dataset...");
    let train_data = load_fineweb_data("./data/datasets/fineweb10B_sp4096/fineweb_train_000.bin")?;
    let val_data = load_fineweb_data("./data/datasets/fineweb10B_sp4096/fineweb_val_000.bin")?;

    println!("📊 FineWeb data loaded:");
    println!("  • Training: {} tokens", train_data.len());
    println!("  • Validation: {} tokens", val_data.len());

    // Training loop
    println!("🔄 Starting training loop...");
    train_trinity_3k_fineweb(&mut model, &train_data, &val_data)?;

    println!("🎉 Training completed!");
    Ok(())
}

/// Load FineWeb data from binary file
fn load_fineweb_data(path: &str) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let path = Path::new(path);
    let mut file = File::open(path)?;

    // Read header (256 x 4-byte integers)
    let mut header_bytes = [0u8; 1024];
    file.read_exact(&mut header_bytes)?;

    // Parse header
    let magic = u32::from_le_bytes(header_bytes[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(header_bytes[4..8].try_into().unwrap());
    let num_tokens = u32::from_le_bytes(header_bytes[8..12].try_into().unwrap()) as usize;

    if magic != 20240520 {
        return Err(format!("Invalid magic number: {}", magic).into());
    }
    if version != 1 {
        return Err(format!("Unsupported version: {}", version).into());
    }

    // Read token data (uint16)
    let mut token_bytes = vec![0u8; num_tokens * 2];
    file.read_exact(&mut token_bytes)?;

    // Convert to u16 tokens
    let tokens = token_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Ok(tokens)
}

/// Training function using FineWeb data
fn train_trinity_3k_fineweb(
    model: &mut Trinity3kModel,
    train_data: &[u16],
    val_data: &[u16],
) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 32;
    let seq_len = 16; // short seq for numerical gradient speed
    let n_epochs = 2;
    let vocab_size = 729;

    let adamw_cfg = AdamWConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };

    println!("📊 Training parameters:");
    println!("  • Batch size: {}", batch_size);
    println!("  • Sequence length: {}", seq_len);
    println!("  • AdamW LR: {}", adamw_cfg.lr);
    println!("  • Epochs: {}", n_epochs);

    let mut total_steps = 0;
    let start_time = Instant::now();

    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_bpb = 0.0;
        let mut epoch_steps = 0;

        println!("🔄 Epoch {}/{}", epoch + 1, n_epochs);

        // Process training data in batches
        for batch_start in (0..train_data.len() - seq_len).step_by(batch_size * seq_len) {
            let batch_end = (batch_start + batch_size * seq_len).min(train_data.len() - seq_len);

            for i in (batch_start..batch_end).step_by(seq_len) {
                if i + seq_len + 1 > train_data.len() {
                    break;
                }

                // Get sequence and convert to Trinity 3k token space (0-728)
                let tokens: Vec<usize> = train_data[i..i + seq_len + 1]
                    .iter()
                    .map(|&t| (t % vocab_size as u16) as usize)
                    .collect();

                let (loss, bpb) = model.loss_bpb(&tokens);
                epoch_loss += loss;
                epoch_bpb += bpb;
                epoch_steps += 1;

                model.train_step(&tokens, &adamw_cfg);

                total_steps += 1;

                if total_steps % 10 == 0 {
                    println!(
                        "  Step {}: Loss = {:.4}, BPB = {:.4}",
                        total_steps, loss, bpb
                    );
                }
            }
        }

        if epoch_steps > 0 {
            let avg_loss = epoch_loss / epoch_steps as f32;
            let avg_bpb = epoch_bpb / epoch_steps as f32;
            println!(
                "  Epoch {}/{}: Train Loss = {:.4}, Train BPB = {:.4}",
                epoch + 1,
                n_epochs,
                avg_loss,
                avg_bpb
            );

            // Validation
            if val_data.len() > seq_len {
                let val_tokens: Vec<usize> = val_data[..seq_len + 1]
                    .iter()
                    .map(|&t| (t % vocab_size as u16) as usize)
                    .collect();

                let (val_loss, val_bpb) = model.loss_bpb(&val_tokens);
                println!(
                    "  Epoch {}/{}: Val Loss = {:.4}, Val BPB = {:.4}",
                    epoch + 1,
                    n_epochs,
                    val_loss,
                    val_bpb
                );
            }
        }
    }

    let duration = start_time.elapsed();
    println!(
        "✅ Training completed in {:.2} seconds",
        duration.as_secs_f32()
    );
    println!("📊 Total steps: {}", total_steps);

    // Final evaluation on full validation set
    println!("🎯 Final evaluation on validation set...");
    let val_seq_len = (val_data.len() - 1).min(1000); // Use up to 1000 tokens
    let val_tokens: Vec<usize> = val_data[..val_seq_len]
        .iter()
        .map(|&t| (t % vocab_size as u16) as usize)
        .collect();

    let (final_loss, final_bpb) = model.loss_bpb(&val_tokens);
    println!(
        "🎯 Final validation: Loss = {:.4}, BPB = {:.4}",
        final_loss, final_bpb
    );

    if final_bpb < 1.15 {
        println!("🎉 SUCCESS: BPB {:.4} < 1.15 target!", final_bpb);
    } else {
        println!(
            "📊 Current BPB: {:.4} (target: <1.15) - need optimization",
            final_bpb
        );
    }

    Ok(())
}
