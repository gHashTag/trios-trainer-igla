//! `canon_digest` -- raw vs canonical checkpoint digest, and the signed-zero
//! census that motivates the difference.
//!
//! The reproducibility criterion this repository offers is `sha256` over the
//! raw checkpoint bytes. That digest distinguishes `-0.0` from `+0.0`, which
//! IEEE 754 and every forward pass in the trainer do not, so it can report
//! MISMATCH between two numerically identical artifacts. This binary measures
//! how much of a given mismatch is that artefact of the encoding and how much
//! is real, and prints both digests so the clause in
//! `docs/CANONICAL-SERIALIZATION.md` can be checked rather than believed.
//!
//! ```text
//!   canon_digest <artifact.bin> [--json]
//!   canon_digest <a.bin> <b.bin> [--json]
//! ```
//!
//! One path prints the raw digest, the canonical digest and the negative-zero
//! count. Two paths print the pair census -- params, bitwise-differing,
//! numerically-differing, signed-zero-only, zero-in-both -- and one line
//! stating whether canonicalisation changes the verdict. `--json` prints the
//! same record as JSON on stdout and nothing else, so an evidence file is
//! generated rather than typed.
//!
//! Exit codes: 0 on a completed measurement (a MISMATCH is a result, not a
//! failure), 2 on usage error, 1 on a refused or unreadable artifact.

use trios_trainer::canonical_digest::{canonical_digest, compare};

fn usage() -> String {
    "usage: canon_digest <artifact.bin> [--json]\n       \
     canon_digest <a.bin> <b.bin> [--json]\n\n\
     Prints the raw sha256 and the canonical sha256 (negative zero normalised \
     to positive zero, NaN/Inf refused) of a TRIOSCKP checkpoint. With two \
     artifacts, prints the signed-zero census across the pair."
        .to_string()
}

fn main() {
    let mut json = false;
    let mut paths: Vec<String> = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{}", usage());
                return;
            }
            other if other.starts_with('-') => {
                eprintln!("UNKNOWN FLAG: {other}\n\n{}", usage());
                std::process::exit(2);
            }
            other => paths.push(other.to_string()),
        }
    }
    if paths.is_empty() || paths.len() > 2 {
        eprintln!(
            "EXPECTED ONE OR TWO ARTIFACT PATHS, GOT {}\n\n{}",
            paths.len(),
            usage()
        );
        std::process::exit(2);
    }

    let read = |p: &str| -> Vec<u8> {
        match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("failed to read {p}: {e}");
                std::process::exit(1);
            }
        }
    };

    if paths.len() == 1 {
        let bytes = read(&paths[0]);
        let d = match canonical_digest(&bytes) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{e:#}");
                std::process::exit(1);
            }
        };
        if json {
            match serde_json::to_string_pretty(&d) {
                Ok(s) => println!("{s}"),
                Err(e) => {
                    eprintln!("failed to serialise record: {e}");
                    std::process::exit(1);
                }
            }
            return;
        }
        println!("ARTIFACT  {}", paths[0]);
        println!("BYTES     {}", bytes.len());
        println!("PARAMS    {}", d.census.params);
        println!("RAW       sha256 {}", d.raw_sha256);
        println!("CANONICAL sha256 {}", d.canonical_sha256);
        println!("NEGZERO   {}", d.census.negative_zero_count);
        if d.raw_sha256 == d.canonical_sha256 {
            println!(
                "NOTE      no negative zero in the payload, so the canonical \
                 digest equals the raw one"
            );
        } else {
            println!(
                "NOTE      {} parameter(s) held -0.0; the raw digest and the \
                 canonical digest are therefore different strings for the same \
                 numbers",
                d.census.negative_zero_count
            );
        }
        return;
    }

    let a = read(&paths[0]);
    let b = read(&paths[1]);
    let c = match compare(&a, &b) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
    };
    if json {
        match serde_json::to_string_pretty(&c) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("failed to serialise record: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    println!("A  {}", paths[0]);
    println!("B  {}", paths[1]);
    println!("PARAMS                 {}", c.params);
    println!("BITWISE-DIFFERING      {}", c.bitwise_differing);
    println!("NUMERICALLY-DIFFERING  {}", c.numerically_differing);
    println!("SIGNED-ZERO-ONLY       {}", c.signed_zero_only);
    println!("ZERO-IN-BOTH           {}", c.zero_in_both);
    println!("RAW-A        {}", c.raw_sha256_a);
    println!("RAW-B        {}", c.raw_sha256_b);
    println!("CANONICAL-A  {}", c.canonical_sha256_a);
    println!("CANONICAL-B  {}", c.canonical_sha256_b);
    println!(
        "RAW-VERDICT        {}",
        if c.raw_digests_equal {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!(
        "CANONICAL-VERDICT  {}",
        if c.canonical_digests_equal {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    if c.canonicalization_changes_verdict {
        println!(
            "VERDICT-CHANGED    YES -- the raw digests disagreed only about the \
             sign of zero; canonicalisation makes these two artifacts one"
        );
    } else if c.raw_digests_equal {
        println!("VERDICT-CHANGED    NO -- the raw digests already agreed");
    } else {
        println!(
            "VERDICT-CHANGED    NO -- canonicalisation removed {} encoding-only \
             difference(s) and left {} genuine numeric one(s); the mismatch \
             survives the clause",
            c.signed_zero_only, c.numerically_differing
        );
    }
}
