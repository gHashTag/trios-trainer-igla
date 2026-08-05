//! Leaderboard printers for `tri race status` / `tri race best`.
//!
//! Both functions may only report what a query actually returned. `show_best`
//! used to answer "No completed trials yet" from the `Err` arm of `query_one`,
//! which conflated "the ledger has no completed trial" with "nothing was
//! contacted" -- and against the stub backend it was always the latter. The
//! error is now propagated, and the no-rows sentence is printed only from an
//! empty row set that a read produced.

use anyhow::Result;

use crate::race::neon::NeonDb;

pub async fn show_status(db: &NeonDb) -> Result<()> {
    let rows = db
        .client()
        .query(
            "SELECT trial_id, machine_id, status, \
                    COALESCE(best_bpb::text, '-'), \
                    COALESCE(config->>'arch', '-') as arch, \
                    created_at \
             FROM igla_race_trials \
             ORDER BY COALESCE(best_bpb, 9999) ASC \
             LIMIT 20",
            &[],
        )
        .await?;

    eprintln!();
    eprintln!("IGLA RACE LEADERBOARD");
    eprintln!(
        "{:<6} {:<20} {:<10} {:<8} {:<10}",
        "Rank", "Trial", "Status", "BPB", "Arch"
    );
    eprintln!("{}", "-".repeat(60));

    for (i, row) in rows.iter().enumerate() {
        let trial_id: String = row.get(0);
        let _machine_id: String = row.get(1);
        let status: String = row.get(2);
        let bpb: String = row.get(3);
        let arch: String = row.get(4);

        let tid_trunc = if trial_id.len() > 18 {
            format!("{}..", &trial_id[..16])
        } else {
            trial_id
        };

        eprintln!(
            "#{:<5} {:<20} {:<10} {:<8} {:<10}",
            i + 1,
            tid_trunc,
            status,
            bpb,
            arch
        );
    }
    eprintln!();
    Ok(())
}

pub async fn show_best(db: &NeonDb) -> Result<()> {
    // `query`, not `query_one`: an empty row set is a fact the ledger stated,
    // while `query_one`'s `Err` cannot distinguish "no such row" from "no
    // connection". Only the first justifies printing a count of zero.
    let rows = db
        .client()
        .query(
            "SELECT trial_id, machine_id, config::text, \
                    COALESCE(best_bpb::text, '-'), status \
             FROM igla_race_trials \
             WHERE best_bpb IS NOT NULL \
             ORDER BY best_bpb ASC LIMIT 1",
            &[],
        )
        .await?;

    let Some(r) = rows.first() else {
        eprintln!("No completed trials yet");
        return Ok(());
    };

    let trial_id: String = r.get(0);
    let machine_id: String = r.get(1);
    let config_str: String = r.get(2);
    let bpb: String = r.get(3);
    let status: String = r.get(4);

    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap_or_default();
    eprintln!("BEST TRIAL");
    eprintln!("  Trial:   {}", trial_id);
    eprintln!("  Machine: {}", machine_id);
    eprintln!("  Status:  {}", status);
    eprintln!("  BPB:     {}", bpb);
    eprintln!(
        "  Config:  {}",
        serde_json::to_string_pretty(&config).unwrap_or_default()
    );
    Ok(())
}
