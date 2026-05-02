//! Download tiny_shakespeare dataset - RUST ONLY
//!
//! Usage: cargo run --bin download_data

use std::fs;
use std::io::Write;
use std::path::Path;

const URL: &str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
const TRAIN_PATH: &str = "data/tiny_shakespeare.txt";
const VAL_PATH: &str = "data/tiny_shakespeare_val.txt";

fn download(url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Use ureq which is already in dependencies
    let response = ureq::get(url).call()?;
    let mut data = Vec::new();
    response.into_reader().read_to_end(&mut data)?;
    Ok(data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[download_data] Fetching tiny_shakespeare from karpathy/char-rnn...");

    let full = download(URL)?;

    let size = full.len();
    let val_size = 100_000;
    let head_size = size.saturating_sub(val_size);

    if head_size == 0 {
        eprintln!("[download_data] ERROR: Data too small ({size} bytes), need at least {val_size}");
        return Err("Data too small".into());
    }

    // Create data directory
    fs::create_dir_all("data")?;

    // Write train (first N - 100000 bytes)
    let train_path = Path::new(TRAIN_PATH);
    let mut train_file = fs::File::create(train_path)?;
    train_file.write_all(&full[..head_size])?;
    println!("[download_data] train: {} bytes -> {}", head_size, TRAIN_PATH);

    // Write val (last 100000 bytes)
    let val_path = Path::new(VAL_PATH);
    let mut val_file = fs::File::create(val_path)?;
    val_file.write_all(&full[head_size..])?;
    println!("[download_data] val: {} bytes -> {}", val_size, VAL_PATH);

    println!("[download_data] DONE - byte-disjoint split");
    Ok(())
}
