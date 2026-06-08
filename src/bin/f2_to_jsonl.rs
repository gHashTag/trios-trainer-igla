//! F2 long-form CSV → JSON Lines adapter — Loop 44.
//!
//! Converts any F2 long-form CSV (dual_mediation, mediation_sensitivity,
//! lambda-sweep, ablation_aggregate, iloco_score) into JSON Lines
//! (`pd.read_json(lines=True)` / Jupyter / Marimo native).
//!
//! Per DoWhy / PyWhy (py-why/dowhy, arXiv:2011.04216) the convention is
//! one flat object per record with no nesting — keeps notebook pivots /
//! filters cheap. CMAverse longitudinal-PSE 2025 uses the same record-per-PSE
//! flat layout.
//!
//! NaN handling: per JSON spec NaN/Infinity are not valid. We emit `null`
//! for non-finite numeric values (superjson 2025-09-07 guide convention).
//! Per-column header is auto-detected; numeric columns are parsed as f64,
//! everything else stays as a string.

use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::Path;

/// Loop 44: format an f64 for JSON, mapping NaN/Inf to `null`.
fn fmt_json_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "null".to_string()
    }
}

/// JSON-escape a string per RFC 8259. Handles `"`, `\`, and control characters.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellKind {
    /// Numeric value — emit as JSON number (or null for NaN/Inf).
    Number,
    /// String — JSON-escape and quote.
    String,
}

/// Heuristic: a column whose every value parses as f64 (after trimming) is numeric.
fn infer_column_kinds(header: &[&str], rows: &[Vec<String>]) -> Vec<CellKind> {
    header
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let all_numeric = rows.iter().all(|r| {
                r.get(i)
                    .map(|c| {
                        let t = c.trim();
                        t.is_empty() || t.parse::<f64>().is_ok()
                    })
                    .unwrap_or(true)
            });
            if all_numeric {
                CellKind::Number
            } else {
                CellKind::String
            }
        })
        .collect()
}

/// Loop 45 fix 1: two-pass streaming conversion. First pass samples up to
/// `INFER_SAMPLE_SIZE` rows to detect column kinds (Polars convention of
/// `infer_schema_length=100`, raised to 1024 here for robustness). Second
/// pass re-opens the input and streams every row through the writer.
/// Memory bound: O(INFER_SAMPLE_SIZE × max_row_size); typically <1 MB.
const INFER_SAMPLE_SIZE: usize = 1024;

/// Helper: read header + sample up to N data rows, skipping `#`-preamble.
/// Loop 46 fix 1: takes an owned File so the caller can `seek(0)` it for
/// pass 2 without reopening — closes the TOCTOU window between passes.
/// Returns `(header, sample_rows)`.
fn read_header_and_sample<R: BufRead>(
    r: &mut R,
    sample_size: usize,
) -> std::io::Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut header: Option<Vec<String>> = None;
    let mut sample: Vec<Vec<String>> = Vec::with_capacity(sample_size);
    let mut line = String::new();
    loop {
        line.clear();
        let n = r.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<String> = trimmed.split(',').map(|s| s.trim().to_string()).collect();
        if header.is_none() {
            header = Some(parts);
            continue;
        }
        if sample.len() >= sample_size {
            break;
        }
        sample.push(parts);
    }
    Ok((header.unwrap_or_default(), sample))
}

/// Write a single JSON record from a row + column kinds.
fn write_record<W: Write>(
    w: &mut W,
    header: &[String],
    row: &[String],
    kinds: &[CellKind],
) -> std::io::Result<()> {
    write!(w, "{{")?;
    let mut first = true;
    for (i, key) in header.iter().enumerate() {
        if !first {
            write!(w, ",")?;
        }
        first = false;
        write!(w, "{}:", json_escape(key))?;
        let empty = String::new();
        let cell = row.get(i).unwrap_or(&empty);
        match kinds[i] {
            CellKind::Number => {
                if cell.trim().is_empty() {
                    write!(w, "null")?;
                } else {
                    match cell.trim().parse::<f64>() {
                        Ok(v) => write!(w, "{}", fmt_json_f64(v))?,
                        Err(_) => write!(w, "{}", json_escape(cell))?,
                    }
                }
            }
            CellKind::String => write!(w, "{}", json_escape(cell))?,
        }
    }
    writeln!(w, "}}")?;
    Ok(())
}

/// Loop 44 → Loop 45 fix 1 → Loop 46 fix 1: convert long-form CSV to JSON
/// Lines via single-open + seek streaming.
///
/// Pass 1 samples first `INFER_SAMPLE_SIZE` rows for column-kind inference;
/// pass 2 `seek(0)` on the same File and streams every row. Single open
/// closes the TOCTOU window between two opens (audit Loop 46 #1).
///
/// Skips `#`-preamble (W3C-PROV / Loop 32+) on both passes.
fn convert_csv_to_jsonl<W: Write>(input: &Path, w: &mut W) -> std::io::Result<usize> {
    let f = File::open(input)?;
    let mut r = BufReader::new(f);
    // Pass 1: sample for inference.
    let (header, sample) = read_header_and_sample(&mut r, INFER_SAMPLE_SIZE)?;
    if header.is_empty() {
        return Ok(0);
    }
    let header_refs: Vec<&str> = header.iter().map(|s| s.as_str()).collect();
    let kinds = infer_column_kinds(&header_refs, &sample);
    // Loop 46 fix 1: seek the same handle back to the start instead of
    // reopening — eliminates the race window where the file could change.
    r.seek(SeekFrom::Start(0))?;
    let mut header_seen = false;
    let mut count = 0usize;
    let mut line = String::new();
    loop {
        line.clear();
        let n = r.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !header_seen {
            header_seen = true;
            continue;
        }
        let parts: Vec<String> = trimmed.split(',').map(|s| s.trim().to_string()).collect();
        write_record(w, &header, &parts, &kinds)?;
        count += 1;
    }
    Ok(count)
}

fn print_help() {
    println!("f2_to_jsonl — Loop 44: long-form CSV → JSON Lines for Jupyter/Marimo");
    println!();
    println!("USAGE: f2_to_jsonl [FLAGS] INPUT_CSV");
    println!();
    println!("Auto-detects numeric columns; emits NaN/Inf as JSON null.");
    println!("Skips W3C-PROV `#`-preamble. One flat object per data row.");
    println!();
    println!("FLAGS:");
    println!("  --help, -h          Print this help and exit");
    println!("  --out PATH          Write JSONL to file (default stdout)");
    println!();
    println!("Refs: DoWhy / PyWhy (arXiv:2011.04216) — flat record-per-estimate convention;");
    println!("      CMAverse longitudinal PSE 2025 — record-per-path-effect.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let mut input: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if a == "--out" {
            out_path = args.get(i + 1).cloned();
            i += 2;
        } else if a.starts_with("--") {
            eprintln!("# ERROR: unknown flag {}", a);
            std::process::exit(2);
        } else {
            input = Some(a.clone());
            i += 1;
        }
    }
    let Some(input) = input else {
        eprintln!("# ERROR: no input CSV. See --help.");
        std::process::exit(2);
    };
    let n = if let Some(p) = out_path {
        let mut f = File::create(&p).expect("create out file");
        let n = convert_csv_to_jsonl(Path::new(&input), &mut f).expect("convert");
        eprintln!("# Wrote {} JSONL records to {}", n, p);
        n
    } else {
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        let n = convert_csv_to_jsonl(Path::new(&input), &mut h).expect("convert");
        eprintln!("# Wrote {} JSONL records to stdout", n);
        n
    };
    if n == 0 {
        eprintln!("# WARN: 0 records emitted (CSV had no data rows?)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn fmt_json_f64_handles_finite_and_nonfinite() {
        assert_eq!(fmt_json_f64(1.5), "1.5");
        assert_eq!(fmt_json_f64(f64::NAN), "null");
        assert_eq!(fmt_json_f64(f64::INFINITY), "null");
        assert_eq!(fmt_json_f64(f64::NEG_INFINITY), "null");
    }

    #[test]
    fn json_escape_handles_quotes_and_specials() {
        assert_eq!(json_escape("rms"), "\"rms\"");
        assert_eq!(json_escape("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_escape("line\nbreak"), "\"line\\nbreak\"");
    }

    #[test]
    fn convert_csv_to_jsonl_roundtrips_simple_table() {
        // Loop 44: synth a tiny dual-mediation-shaped CSV and verify JSONL output.
        let tmp = std::env::temp_dir().join("f2_jsonl_simple.csv");
        let mut f = File::create(&tmp).unwrap();
        writeln!(f, "# preamble line").unwrap();
        writeln!(f, "fix_x,pse_name,estimate,ci95_lo").unwrap();
        writeln!(f, "rms,NDE,-4.12,-4.68").unwrap();
        writeln!(f, "rms,NIE_M1,4.99,4.42").unwrap();
        drop(f);
        let mut buf = Vec::new();
        let n = convert_csv_to_jsonl(&tmp, &mut buf).unwrap();
        assert_eq!(n, 2);
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2);
        // First line should have "fix_x":"rms" and numeric estimate.
        assert!(lines[0].contains("\"fix_x\":\"rms\""));
        assert!(lines[0].contains("\"estimate\":-4.12"));
        // String column quoted, numeric not.
        assert!(lines[0].contains("\"pse_name\":\"NDE\""));
    }

    #[test]
    fn convert_csv_to_jsonl_handles_no_trailing_newline() {
        // Loop 46 fix 3: CSV ending without a final newline.
        let tmp = std::env::temp_dir().join("f2_jsonl_no_newline.csv");
        let mut f = File::create(&tmp).unwrap();
        // Note: no `writeln!` on the last data row — manual write without \n.
        write!(f, "fix_x,value\nrms,5.0\nwd,0.07").unwrap();
        drop(f);
        let mut buf = Vec::new();
        let n = convert_csv_to_jsonl(&tmp, &mut buf).unwrap();
        assert_eq!(n, 2, "missing newline truncated stream: got {}", n);
    }

    #[test]
    fn convert_csv_to_jsonl_handles_blank_lines() {
        // Loop 46 fix 3: blank lines embedded + trailing must be skipped without
        // counting as records or aborting iteration.
        let tmp = std::env::temp_dir().join("f2_jsonl_blank_lines.csv");
        let mut f = File::create(&tmp).unwrap();
        writeln!(f, "fix_x,value").unwrap();
        writeln!(f).unwrap(); // blank
        writeln!(f, "rms,5.0").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "wd,0.07").unwrap();
        writeln!(f).unwrap();
        writeln!(f).unwrap();
        drop(f);
        let mut buf = Vec::new();
        let n = convert_csv_to_jsonl(&tmp, &mut buf).unwrap();
        assert_eq!(n, 2, "blank-line handling off by N: got {}", n);
    }

    #[test]
    fn convert_csv_to_jsonl_handles_row_count_exceeding_sample() {
        // Loop 45 fix 1: streaming pass must emit ALL rows even when row count
        // exceeds INFER_SAMPLE_SIZE. Synth a CSV with > INFER_SAMPLE_SIZE rows.
        let tmp = std::env::temp_dir().join("f2_jsonl_streaming.csv");
        let mut f = File::create(&tmp).unwrap();
        writeln!(f, "name,n").unwrap();
        let total = INFER_SAMPLE_SIZE + 50;
        for i in 0..total {
            writeln!(f, "fix{},{:.6}", i, i as f64 * 0.5).unwrap();
        }
        drop(f);
        let mut buf = Vec::new();
        let n = convert_csv_to_jsonl(&tmp, &mut buf).unwrap();
        assert_eq!(n, total, "streaming missed rows beyond sample window");
        // Spot-check last row is in output.
        let s = String::from_utf8(buf).unwrap();
        let last_marker = format!("\"fix{}\"", total - 1);
        assert!(
            s.contains(&last_marker),
            "last row missing from streamed output"
        );
    }

    #[test]
    fn convert_csv_to_jsonl_skips_w3c_prov_preamble() {
        // Loop 45 fix 4: lock the contract that `#`-preamble lines never appear
        // as JSON records. Synthesizes a full Loop-32 preamble + 2 data rows.
        let tmp = std::env::temp_dir().join("f2_jsonl_preamble.csv");
        let mut f = File::create(&tmp).unwrap();
        writeln!(f, "# f2_ablation_sweep provenance (W3C PROV)").unwrap();
        writeln!(f, "# prov:generatedAt = 1700000000").unwrap();
        writeln!(f, "# prov:agent_git_sha = abc123").unwrap();
        writeln!(
            f,
            "# prov:trainer_internals_schema = trainer_internals_v1_2026_06_01"
        )
        .unwrap();
        writeln!(f, "fix_x,value").unwrap();
        writeln!(f, "rms,5.0").unwrap();
        writeln!(f, "wd,0.07").unwrap();
        drop(f);
        let mut buf = Vec::new();
        let n = convert_csv_to_jsonl(&tmp, &mut buf).unwrap();
        // Exactly 2 records (the preamble + header + 2 data lines → 2 records).
        assert_eq!(n, 2, "expected 2 JSONL records; preamble must not leak");
        let s = String::from_utf8(buf).unwrap();
        // No JSONL line should contain # or 'prov:' (preamble keywords).
        for line in s.lines() {
            assert!(!line.contains("\"#"), "preamble leaked: {}", line);
            assert!(!line.contains("prov:"), "preamble field leaked: {}", line);
        }
    }

    #[test]
    fn convert_csv_to_jsonl_emits_null_for_nan_and_empty() {
        let tmp = std::env::temp_dir().join("f2_jsonl_nan.csv");
        let mut f = File::create(&tmp).unwrap();
        writeln!(f, "fix_x,value").unwrap();
        writeln!(f, "a,NaN").unwrap();
        writeln!(f, "b,").unwrap();
        drop(f);
        let mut buf = Vec::new();
        convert_csv_to_jsonl(&tmp, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        // NaN → null; empty → null (numeric column inferred).
        let lines: Vec<&str> = s.lines().collect();
        assert!(
            lines[0].contains("\"value\":null"),
            "NaN should map to null: {}",
            lines[0]
        );
        assert!(
            lines[1].contains("\"value\":null"),
            "empty should map to null: {}",
            lines[1]
        );
    }

    #[test]
    fn infer_column_kinds_distinguishes_string_from_number() {
        let header = vec!["name", "n"];
        let rows = vec![
            vec!["alpha".to_string(), "1".to_string()],
            vec!["beta".to_string(), "2.5".to_string()],
        ];
        let kinds = infer_column_kinds(&header, &rows);
        assert_eq!(kinds[0], CellKind::String);
        assert_eq!(kinds[1], CellKind::Number);
    }
}
