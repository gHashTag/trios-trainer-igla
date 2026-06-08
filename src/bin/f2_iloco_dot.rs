//! F2 iLOCO interaction-network DOT emitter — Loop 29 Option A.
//!
//! Reads a `f2_iloco_score` CSV (rank,fix_a,fix_b,..,iloco,kind,p_value,q_value_bh)
//! and emits Graphviz DOT with:
//!   - one node per fix
//!   - undirected edges colored by `kind`:
//!     compensatory -> red    (positive iLOCO; sum-of-individual overstates joint cost)
//!     redundant    -> blue   (negative iLOCO; joint cost exceeds sum)
//!     independent  -> grey   (|iLOCO| < eps; omitted by default)
//!   - edge penwidth ∝ |iLOCO|
//!   - edge label = "iLOCO=±X.XX  q=Y.YYe-Z"
//!   - significant edges (q < α) styled solid; non-significant dashed
//!
//! Sign convention follows arXiv:2502.06661 (iLOCO Eq.(3)) and matches our
//! `f2_iloco_score` output. Color choice (synergy=red, redundancy=blue) follows
//! PID convention (Ontivero-Ortega et al. Phys. Rev. E 111 L033301, 2025).
//!
//! Render with: `dot -Tpng f2_interactions.dot -o f2_interactions.png`

use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone)]
struct PairRow {
    fix_a: String,
    fix_b: String,
    iloco: f64,
    kind: String,
    q_value: f64,
}

fn parse_iloco_csv(path: &str) -> Vec<PairRow> {
    let f = File::open(path).expect("open iloco CSV");
    let r = BufReader::new(f);
    let mut rows = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line.expect("read");
        if i == 0 || line.starts_with('#') || line.is_empty() {
            continue;
        }
        // Format: rank,fix_a,fix_b,delta_a,delta_b,delta_ab,iloco,kind,p_value,q_value_bh
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 10 {
            continue;
        }
        rows.push(PairRow {
            fix_a: parts[1].to_string(),
            fix_b: parts[2].to_string(),
            iloco: parts[6].parse().unwrap_or(0.0),
            kind: parts[7].to_string(),
            q_value: parts[9].parse().unwrap_or(1.0),
        });
    }
    rows
}

/// Loop 31 fix 5: emit Mermaid-flavored graph for GitHub-native rendering.
/// No external Graphviz install needed; pastes directly into Markdown/issues.
fn emit_mermaid<W: Write>(
    w: &mut W,
    rows: &[PairRow],
    alpha: f64,
    show_independent: bool,
    min_iloco: f64,
) -> std::io::Result<()> {
    writeln!(w, "graph TD")?;
    // Mermaid uses --- for undirected; --> directed; we use --- with class styles.
    let mut emitted: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for r in rows {
        emitted.insert(r.fix_a.clone());
        emitted.insert(r.fix_b.clone());
    }
    for n in &emitted {
        writeln!(w, "    {}((<b>{}</b>))", n, n)?;
    }
    let mut edge_idx = 0usize;
    for r in rows {
        if r.iloco.abs() < min_iloco {
            continue;
        }
        if !show_independent && r.kind == "independent" {
            continue;
        }
        let sig_tag = if r.q_value < alpha { "*" } else { "" };
        writeln!(
            w,
            "    {} ---|\"iLOCO={:+.2}{} q={:.1e}\"| {}",
            r.fix_a, r.iloco, sig_tag, r.q_value, r.fix_b
        )?;
        let class = match r.kind.as_str() {
            "compensatory" => "comp",
            "redundant" => "red",
            _ => "ind",
        };
        writeln!(
            w,
            "    linkStyle {} stroke:{}",
            edge_idx,
            match class {
                "comp" => "#C0392B,stroke-width:3px",
                "red" => "#2E86AB,stroke-width:3px",
                _ => "#95A5A6,stroke-width:1px",
            }
        )?;
        edge_idx += 1;
    }
    writeln!(w)?;
    writeln!(
        w,
        "    %% Legend: comp(red)=+iLOCO, red(blue)=-iLOCO; * = q<{:.2}",
        alpha
    )?;
    writeln!(
        w,
        "    %% Render: pastes directly into GitHub-flavored Markdown."
    )?;
    Ok(())
}

fn emit_dot<W: Write>(
    w: &mut W,
    rows: &[PairRow],
    alpha: f64,
    show_independent: bool,
    min_iloco: f64,
) -> std::io::Result<()> {
    // Discover unique fix names.
    let mut nodes: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for r in rows {
        nodes.insert(r.fix_a.clone());
        nodes.insert(r.fix_b.clone());
    }
    let max_abs = rows
        .iter()
        .map(|r| r.iloco.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-9);

    writeln!(w, "graph f2_interactions {{")?;
    writeln!(
        w,
        "  graph [layout=neato, overlap=false, splines=true, bgcolor=\"#FAFAFA\"];"
    )?;
    writeln!(
        w,
        "  node  [shape=ellipse, style=\"filled,rounded\", fillcolor=\"#FFFFFF\", \
         fontname=\"Helvetica\", fontsize=12, penwidth=1.5];"
    )?;
    writeln!(w, "  edge  [fontname=\"Helvetica\", fontsize=9];")?;
    writeln!(w)?;
    for n in &nodes {
        writeln!(w, "  \"{}\";", n)?;
    }
    writeln!(w)?;
    for r in rows {
        if r.iloco.abs() < min_iloco {
            continue;
        }
        if !show_independent && r.kind == "independent" {
            continue;
        }
        let color = match r.kind.as_str() {
            "compensatory" => "\"#C0392B\"", // red
            "redundant" => "\"#2E86AB\"",    // blue
            _ => "\"#95A5A6\"",              // grey
        };
        let width = 1.0 + 5.0 * (r.iloco.abs() / max_abs);
        let style = if r.q_value < alpha { "solid" } else { "dashed" };
        let sig_tag = if r.q_value < alpha { "*" } else { "" };
        writeln!(
            w,
            "  \"{}\" -- \"{}\" [color={}, penwidth={:.2}, style={}, label=\"iLOCO={:+.3}{}\\nq={:.2e}\"];",
            r.fix_a, r.fix_b, color, width, style, r.iloco, sig_tag, r.q_value
        )?;
    }
    writeln!(w)?;
    writeln!(
        w,
        "  // Legend: red=compensatory (+iLOCO), blue=redundant (-iLOCO),"
    )?;
    writeln!(
        w,
        "  //         solid=significant (q<{:.2}), dashed=ns.  arXiv:2502.06661.",
        alpha
    )?;
    writeln!(w, "}}")?;
    Ok(())
}

fn print_help() {
    println!("f2_iloco_dot — Loop 29 Option A: render iLOCO interaction network as Graphviz DOT");
    println!();
    println!("USAGE: f2_iloco_dot [FLAGS] ILOCO_CSV");
    println!();
    println!("FLAGS:");
    println!("  --help, -h          Print this help and exit");
    println!("  --out PATH          Write DOT to file (default stdout)");
    println!("  --alpha F           BH q-value threshold for solid edges (default 0.10)");
    println!("  --min-iloco F       Suppress edges with |iLOCO| < F (default 0.0)");
    println!("  --show-independent  Include edges where kind=independent (default off)");
    println!("  --format FMT        Output format: 'dot' (Graphviz, default) or 'mermaid'");
    println!("                      (GitHub-native, no external tool needed)");
    println!();
    println!("Render DOT with:  dot -Tpng <out.dot> -o <out.png>");
    println!("Mermaid pastes directly into Markdown.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let mut input: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut alpha: f64 = 0.10;
    let mut min_iloco: f64 = 0.0;
    let mut show_independent = false;
    let mut format = "dot".to_string();
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if a == "--out" {
            if i + 1 >= args.len() {
                eprintln!("# ERROR: --out requires a path");
                std::process::exit(2);
            }
            out_path = Some(args[i + 1].clone());
            i += 2;
        } else if a == "--alpha" {
            if i + 1 >= args.len() {
                eprintln!("# ERROR: --alpha requires a value");
                std::process::exit(2);
            }
            alpha = args[i + 1].parse().expect("parse --alpha");
            i += 2;
        } else if a == "--min-iloco" {
            if i + 1 >= args.len() {
                eprintln!("# ERROR: --min-iloco requires a value");
                std::process::exit(2);
            }
            min_iloco = args[i + 1].parse().expect("parse --min-iloco");
            i += 2;
        } else if a == "--show-independent" {
            show_independent = true;
            i += 1;
        } else if a == "--format" {
            if i + 1 >= args.len() {
                eprintln!("# ERROR: --format requires 'dot' or 'mermaid'");
                std::process::exit(2);
            }
            format = args[i + 1].clone();
            if format != "dot" && format != "mermaid" {
                eprintln!(
                    "# ERROR: unknown --format '{}' (use 'dot' or 'mermaid')",
                    format
                );
                std::process::exit(2);
            }
            i += 2;
        } else if a.starts_with("--") {
            eprintln!("# ERROR: unknown flag {}", a);
            std::process::exit(2);
        } else {
            input = Some(a.clone());
            i += 1;
        }
    }
    let input = match input {
        Some(p) => p,
        None => {
            eprintln!("# ERROR: no iLOCO CSV given. See --help.");
            std::process::exit(2);
        }
    };
    let rows = parse_iloco_csv(&input);
    eprintln!("# Parsed {} pair rows from {}", rows.len(), input);
    if let Some(path) = out_path.as_deref() {
        let mut f = File::create(path).expect("create output file");
        if format == "mermaid" {
            emit_mermaid(&mut f, &rows, alpha, show_independent, min_iloco).expect("write");
            eprintln!(
                "# Wrote Mermaid graph to {} — paste into GitHub Markdown",
                path
            );
        } else {
            emit_dot(&mut f, &rows, alpha, show_independent, min_iloco).expect("write");
            eprintln!(
                "# Wrote DOT to {} — render with: dot -Tpng {} -o out.png",
                path, path
            );
        }
    } else {
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        if format == "mermaid" {
            emit_mermaid(&mut h, &rows, alpha, show_independent, min_iloco).expect("write stdout");
        } else {
            emit_dot(&mut h, &rows, alpha, show_independent, min_iloco).expect("write stdout");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_rows() -> Vec<PairRow> {
        vec![
            PairRow {
                fix_a: "wd".into(),
                fix_b: "rms".into(),
                iloco: -0.82,
                kind: "redundant".into(),
                q_value: 0.08,
            },
            PairRow {
                fix_a: "warmup".into(),
                fix_b: "wd".into(),
                iloco: 0.33,
                kind: "compensatory".into(),
                q_value: 0.25,
            },
            PairRow {
                fix_a: "clamp".into(),
                fix_b: "smooth".into(),
                iloco: 0.0,
                kind: "independent".into(),
                q_value: 1.0,
            },
        ]
    }

    #[test]
    fn emit_dot_contains_graph_header_and_nodes() {
        let rows = synth_rows();
        let mut buf = Vec::new();
        emit_dot(&mut buf, &rows, 0.10, false, 0.0).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("graph f2_interactions"));
        assert!(s.contains("\"wd\""));
        assert!(s.contains("\"rms\""));
        assert!(s.contains("\"warmup\""));
        // independent edges dropped by default
        assert!(!s.contains("\"clamp\" -- \"smooth\""));
    }

    #[test]
    fn emit_dot_marks_significant_solid_and_others_dashed() {
        let rows = synth_rows();
        let mut buf = Vec::new();
        emit_dot(&mut buf, &rows, 0.10, false, 0.0).unwrap();
        let s = String::from_utf8(buf).unwrap();
        // wd--rms has q=0.08 < 0.10 → solid
        assert!(s.contains("style=solid"));
        // warmup--wd has q=0.25 → dashed
        assert!(s.contains("style=dashed"));
    }

    #[test]
    fn emit_dot_uses_red_for_compensatory_blue_for_redundant() {
        let rows = synth_rows();
        let mut buf = Vec::new();
        emit_dot(&mut buf, &rows, 0.10, false, 0.0).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("#C0392B")); // red, compensatory
        assert!(s.contains("#2E86AB")); // blue, redundant
    }

    #[test]
    fn emit_mermaid_contains_graph_header_and_nodes() {
        let rows = synth_rows();
        let mut buf = Vec::new();
        emit_mermaid(&mut buf, &rows, 0.10, false, 0.0).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("graph TD"), "expected mermaid TD prefix");
        assert!(s.contains("wd(("));
        assert!(s.contains("warmup(("));
        // Edge labels include iLOCO + q.
        assert!(s.contains("iLOCO="));
        assert!(s.contains("q="));
        // Color uses linkStyle directive.
        assert!(s.contains("linkStyle"));
    }

    #[test]
    fn min_iloco_suppresses_below_threshold() {
        let rows = synth_rows();
        let mut buf = Vec::new();
        emit_dot(&mut buf, &rows, 0.10, true, 0.5).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("\"wd\" -- \"rms\"")); // 0.82 > 0.5
        assert!(!s.contains("\"warmup\" -- \"wd\"")); // 0.33 < 0.5
    }
}
