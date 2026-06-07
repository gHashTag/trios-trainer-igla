//! IGLA-Coder P2: byte-level code corpus binarizer.
//!
//! Reads a code corpus (a directory tree, or a single .jsonl with a "text" field)
//! and writes train/val .bin token files in the repo's existing format:
//!   header: 256 x u32 LE  (magic=20240520, version=1, num_tokens, rest 0)
//!   body:   num_tokens x u16 LE
//!
//! Vocabulary is byte-level: token IDs 0..=255 are raw bytes. Control / FIM
//! sentinels live ABOVE the byte range:
//!   256 BOS, 257 EOS, 258 PAD, 259 FIM_PRE, 260 FIM_MID, 261 FIM_SUF, 262 LANG
//! => VOCAB_SIZE = 263. A document is encoded as [BOS] bytes... [EOS].
//!
//! CPU-only, no ML, no network. ASCII-only output. Anchor: phi^2 + phi^-2 = 3.
//!
//! Usage:
//!   code_binarize <input_path> <out_dir> [val_fraction]
//! where <input_path> is a directory of source files OR a .jsonl file whose
//! records each contain a "text" string field.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const MAGIC: u32 = 20240520;
const VERSION: u32 = 1;

pub const BOS: u16 = 256;
pub const EOS: u16 = 257;
pub const VOCAB_SIZE: u16 = 263;

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_files(&p, out);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
}

/// Minimal extraction of the "text" field from one JSON object line.
/// Avoids a serde dependency in this tool; handles standard JSON escapes.
fn extract_text_field(line: &str) -> Option<String> {
    let key = "\"text\"";
    let ki = line.find(key)?;
    let rest = &line[ki + key.len()..];
    let colon = rest.find(':')?;
    let after = &rest[colon + 1..];
    let q = after.find('"')?;
    let mut chars = after[q + 1..].chars();
    let mut s = String::new();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('n') => s.push('\n'),
                Some('t') => s.push('\t'),
                Some('r') => s.push('\r'),
                Some('"') => s.push('"'),
                Some('\\') => s.push('\\'),
                Some('/') => s.push('/'),
                Some('u') => {
                    let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                    if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(cp) {
                            s.push(ch);
                        }
                    }
                }
                Some(other) => s.push(other),
                None => break,
            },
            '"' => return Some(s),
            other => s.push(other),
        }
    }
    Some(s)
}

fn encode_doc(text: &str, tokens: &mut Vec<u16>) {
    tokens.push(BOS);
    for b in text.as_bytes() {
        tokens.push(*b as u16);
    }
    tokens.push(EOS);
}

fn write_bin(path: &Path, tokens: &[u16]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let mut header = [0u8; 1024];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&(tokens.len() as u32).to_le_bytes());
    f.write_all(&header)?;
    let mut body = Vec::with_capacity(tokens.len() * 2);
    for t in tokens {
        body.extend_from_slice(&t.to_le_bytes());
    }
    f.write_all(&body)?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: code_binarize <input_path> <out_dir> [val_fraction]");
        std::process::exit(2);
    }
    let input = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    let val_frac: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.1);
    fs::create_dir_all(&out_dir)?;

    let mut docs: Vec<String> = Vec::new();
    if input.is_file() && input.extension().map(|e| e == "jsonl").unwrap_or(false) {
        let reader = BufReader::new(File::open(&input)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Some(t) = extract_text_field(&line) {
                docs.push(t);
            }
        }
    } else if input.is_dir() {
        let mut files = Vec::new();
        collect_files(&input, &mut files);
        files.sort();
        for f in files {
            if let Ok(t) = fs::read_to_string(&f) {
                docs.push(t);
            }
        }
    } else {
        eprintln!("input must be a directory or a .jsonl file");
        std::process::exit(2);
    }

    if docs.is_empty() {
        eprintln!("no documents found");
        std::process::exit(1);
    }

    // Deterministic split: every Nth doc to val.
    let stride = if val_frac > 0.0 {
        (1.0 / val_frac).round() as usize
    } else {
        0
    };
    let mut train_tokens: Vec<u16> = Vec::new();
    let mut val_tokens: Vec<u16> = Vec::new();
    for (i, d) in docs.iter().enumerate() {
        if stride > 0 && i % stride == 0 {
            encode_doc(d, &mut val_tokens);
        } else {
            encode_doc(d, &mut train_tokens);
        }
    }
    if val_tokens.is_empty() {
        // ensure a non-empty val by moving the last doc
        encode_doc(docs.last().unwrap(), &mut val_tokens);
    }

    let train_path = out_dir.join("code_train.bin");
    let val_path = out_dir.join("code_val.bin");
    write_bin(&train_path, &train_tokens)?;
    write_bin(&val_path, &val_tokens)?;

    println!("docs={} vocab_size={}", docs.len(), VOCAB_SIZE);
    println!(
        "train_tokens={} -> {}",
        train_tokens.len(),
        train_path.display()
    );
    println!("val_tokens={} -> {}", val_tokens.len(), val_path.display());
    Ok(())
}
