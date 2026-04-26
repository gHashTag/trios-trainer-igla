//! Trinity 3k Simple Training for Parameter Golf #110
//!
//! Basic training loop for Trinity 3k byte-level model

use std::time::Instant;
use trios_trainer::trinity_3k_model::{Trinity3kModel, Trinity3kConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Trinity 3k Simple Training for Parameter Golf #110");
    println!("🎯 Target: <1.15 BPB on FineWeb");
    println!("📊 Architecture: 3^k dimensions with byte-level processing");

    // Trinity 3k configuration
    let config = Trinity3kConfig::default();
    
    println!("📋 Configuration:");
    println!("  • Vocab size: {} (3^6)", config.vocab_size);
    println!("  • Hidden dim: {} (3^5)", config.hidden_dim);
    println!("  • Heads: {} (3^3) x dim: {} (3^2)", config.n_heads, config.head_dim);
    println!("  • Layers: {}", config.n_layers);
    println!("  • Total params: {}", config.total_params());

    // Create model
    let vocab_size = config.vocab_size;

    let mut model = Trinity3kModel::new(config)?;
    println!("✅ Model created successfully");

    // Create synthetic byte data for testing
    let synthetic_data = create_synthetic_byte_data(10000, vocab_size);
    println!("📊 Synthetic data loaded: {} bytes", synthetic_data.len());

    // Training loop
    println!("🔄 Starting training loop...");
    train_trinity_3k(&mut model, &synthetic_data)?;

    println!("🎉 Training completed!");
    Ok(())
}

fn create_synthetic_byte_data(size: usize, vocab_size: usize) -> Vec<usize> {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen_range(0..vocab_size)).collect()
}

fn train_trinity_3k(
    model: &mut Trinity3kModel,
    data: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 32;
    let seq_len = 64;
    let learning_rate = 0.01;
    let n_epochs = 5;

    println!("📊 Training parameters:");
    println!("  • Batch size: {}", batch_size);
    println!("  • Sequence length: {}", seq_len);
    println!("  • Learning rate: {}", learning_rate);
    println!("  • Epochs: {}", n_epochs);

    let mut total_steps = 0;
    let start_time = Instant::now();

    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_bpb = 0.0;
        let mut epoch_steps = 0;

        println!("🔄 Epoch {}/{}", epoch + 1, n_epochs);

        // Process data in batches
        for batch_start in (0..data.len() - seq_len).step_by(batch_size * seq_len) {
            let batch_end = (batch_start + batch_size * seq_len).min(data.len() - seq_len);
            
            for i in (batch_start..batch_end).step_by(seq_len) {
                let tokens = &data[i..i + seq_len + 1]; // +1 for targets
                
                let (loss, bpb) = model.loss_bpb(tokens);
                epoch_loss += loss;
                epoch_bpb += bpb;
                epoch_steps += 1;

                // Simple SGD step (placeholder - real implementation needs gradients)
                model.sgd_step(tokens, learning_rate);

                total_steps += 1;

                if total_steps % 10 == 0 {
                    println!("  Step {}: Loss = {:.4}, BPB = {:.4}", total_steps, loss, bpb);
                }
            }
        }

        if epoch_steps > 0 {
            let avg_loss = epoch_loss / epoch_steps as f32;
            let avg_bpb = epoch_bpb / epoch_steps as f32;
            println!("  Epoch {}/{}: Avg Loss = {:.4}, Avg BPB = {:.4}", 
                    epoch + 1, n_epochs, avg_loss, avg_bpb);
        }
    }

    let duration = start_time.elapsed();
    println!("✅ Training completed in {:.2} seconds", duration.as_secs_f32());
    println!("📊 Total steps: {}", total_steps);

    // Final evaluation
    let eval_data = &data[data.len() - 1000..]; // Last 1000 tokens for evaluation
    let (final_loss, final_bpb) = model.loss_bpb(eval_data);
    println!("🎯 Final evaluation: Loss = {:.4}, BPB = {:.4}", final_loss, final_bpb);

    if final_bpb < 1.15 {
        println!("🎉 SUCCESS: BPB {:.4} < 1.15 target!", final_bpb);
    } else {
        println!("📊 Current BPB: {:.4} (target: <1.15)", final_bpb);
    }

    Ok(())
}
