//! `trios-igla` — needle-search CLI over the IGLA RACE ledger.
//!
//! Read-only queries against `assertions/seed_results.jsonl` and
//! `assertions/embargo.txt`. Never mutates the ledger.
//!
//! ```bash
//! # Filter ledger
//! trios-igla search --seed 43 --bpb-max 1.85 --step-min 4000
//!
//! # Last 5 rows in canonical R7 triplet form
//! trios-igla list --last 5
//!
//! # Gate-2 quorum check (PASS iff ≥3 seeds satisfy bpb<target AND step>=4000)
//! trios-igla gate --target 1.85
//!
//! # Embargo refusal (R9)
//! trios-igla check 2446855
//!
//! # Print canonical R7 triplet for row index 0
//! trios-igla triplet 0
//! ```
//!
//! Triplet (R7):
//! ```text
//! BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

use trios_trainer::igla;

#[derive(Parser, Debug)]
#[command(
    name = "trios-igla",
    about = "Needle-search CLI for the IGLA RACE ledger (gHashTag/trios-trainer-igla#18)",
    version
)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand, Debug)]
enum Action {
    /// Filter ledger rows; emits one R7 triplet line per match.
    Search {
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long = "bpb-max")]
        bpb_max: Option<f64>,
        #[arg(long = "step-min")]
        step_min: Option<u64>,
        #[arg(long)]
        sha: Option<String>,
        #[arg(long = "gate-status")]
        gate_status: Option<String>,
        #[arg(long, default_value = igla::DEFAULT_LEDGER_PATH)]
        ledger: PathBuf,
    },
    /// Print the last N rows in canonical R7 triplet form.
    List {
        #[arg(long, default_value_t = igla::DEFAULT_LIST_LAST_N)]
        last: usize,
        #[arg(long, default_value = igla::DEFAULT_LEDGER_PATH)]
        ledger: PathBuf,
    },
    /// Gate-2 quorum check: PASS iff ≥3 seeds satisfy `bpb < target` AND `step >= 4000`.
    Gate {
        #[arg(long, default_value_t = igla::DEFAULT_TARGET_BPB)]
        target: f64,
        #[arg(long, default_value = igla::DEFAULT_LEDGER_PATH)]
        ledger: PathBuf,
    },
    /// Embargo refusal (R9). Non-zero exit if SHA is on the embargo list.
    Check {
        sha: String,
        #[arg(long, default_value = igla::DEFAULT_EMBARGO_PATH)]
        embargo: PathBuf,
    },
    /// Print the canonical R7 triplet for a row index (0-based).
    Triplet {
        row_index: usize,
        #[arg(long, default_value = igla::DEFAULT_LEDGER_PATH)]
        ledger: PathBuf,
    },
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    let mut stdout = std::io::stdout().lock();
    match cli.action {
        Action::Search {
            seed,
            bpb_max,
            step_min,
            sha,
            gate_status,
            ledger,
        } => {
            let filter = igla::SearchFilter {
                seed,
                bpb_max,
                step_min,
                sha,
                gate_status,
            };
            let hits = igla::run_search(&ledger, &filter, &mut stdout)?;
            eprintln!("trios-igla search: {hits} match(es)");
            if hits == 0 {
                return Ok(ExitCode::from(2));
            }
        }
        Action::List { last, ledger } => {
            let n = igla::run_list(&ledger, last, &mut stdout)?;
            eprintln!("trios-igla list: emitted {n} row(s)");
        }
        Action::Gate { target, ledger } => {
            let (pass, count, total) = igla::run_gate(&ledger, target)?;
            println!(
                "{} target={target} quorum={count}/{quorum} rows={total} ledger={path}",
                if pass { "PASS" } else { "NOT YET" },
                quorum = igla::GATE2_SEED_QUORUM,
                path = ledger.display(),
            );
            if !pass {
                return Ok(ExitCode::from(2));
            }
        }
        Action::Check { sha, embargo } => match igla::run_check(&embargo, &sha) {
            Ok(()) => {
                println!("OK sha={sha} embargo={}", embargo.display());
            }
            Err(e) => {
                println!("REFUSED sha={sha} reason={e}");
                return Ok(ExitCode::from(1));
            }
        },
        Action::Triplet { row_index, ledger } => {
            igla::run_triplet(&ledger, row_index, &mut stdout)?;
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("trios-igla: {e:#}");
            ExitCode::from(1)
        }
    }
}
