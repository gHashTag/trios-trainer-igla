// migration/tests/idempotent.rs
//
// Idempotency test: applying the migration twice must not produce an error.
//
// This test requires a live Postgres database reachable via DATABASE_URL.
// It is skipped (passes trivially) if DATABASE_URL is not set, so CI runs
// without a database do not fail.
//
// To run locally:
//   DATABASE_URL=postgres://... cargo test -p migration -- --nocapture

use migration::{Migrator, MigratorTrait};

#[tokio::test]
async fn migration_is_idempotent() {
    let db_url = match std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("TRIOS_DATABASE_URL"))
    {
        Ok(u) => u,
        Err(_) => {
            eprintln!("[idempotent] DATABASE_URL not set — skipping live test");
            return;
        }
    };

    // Strip channel_binding to avoid SCRAM-SHA-256-PLUS failures with rustls.
    let db_url = strip_channel_binding(&db_url);

    let db = sea_orm::Database::connect(&db_url)
        .await
        .expect("connect to Postgres");

    // First apply.
    Migrator::up(&db, None)
        .await
        .expect("first migration::up must succeed");

    // Second apply — must be idempotent (all CREATE IF NOT EXISTS).
    Migrator::up(&db, None)
        .await
        .expect("second migration::up must also succeed (idempotency)");

    db.close().await.expect("close connection");
}

/// Wave 24 — verify that after running Migrator::up() twice, the step column
/// in public.bpb_samples is reported as bigint by information_schema.
///
/// Skipped when DATABASE_URL is not set (no live DB in CI).
#[tokio::test]
#[ignore = "requires live DATABASE_URL"]
async fn step_column_is_bigint_after_migration() {
    let db_url = match std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("TRIOS_DATABASE_URL"))
    {
        Ok(u) => u,
        Err(_) => {
            eprintln!("[step_column_is_bigint] DATABASE_URL not set — skipping");
            return;
        }
    };

    let db_url = strip_channel_binding(&db_url);

    let db = sea_orm::Database::connect(&db_url)
        .await
        .expect("connect to Postgres");

    // First apply.
    Migrator::up(&db, None)
        .await
        .expect("first migration::up must succeed");

    // Second apply — must be idempotent.
    Migrator::up(&db, None)
        .await
        .expect("second migration::up must also succeed (idempotency)");

    // Verify step column type via information_schema.
    use sea_orm::ConnectionTrait;
    let result = db
        .query_one(sea_orm::Statement::from_string(
            sea_orm::DatabaseBackend::Postgres,
            "SELECT data_type FROM information_schema.columns \
             WHERE table_schema='public' AND table_name='bpb_samples' AND column_name='step'"
                .to_owned(),
        ))
        .await
        .expect("information_schema query must succeed")
        .expect("step column must exist in public.bpb_samples");

    let data_type: String = result
        .try_get_by_index(0)
        .expect("data_type column must be present");

    assert_eq!(
        data_type, "bigint",
        "public.bpb_samples.step must be bigint after Wave 24 migration"
    );

    eprintln!("[step_column_is_bigint] public.bpb_samples.step data_type = {data_type} ✓");

    db.close().await.expect("close connection");
}

/// Mirror of neon_writer::strip_channel_binding — duplicated here to keep
/// the migration crate self-contained without a dep on trios_trainer.
fn strip_channel_binding(dsn: &str) -> String {
    let Some(qpos) = dsn.find('?') else {
        return dsn.to_string();
    };
    let (head, query) = dsn.split_at(qpos + 1);
    let kept: Vec<&str> = query
        .split('&')
        .filter(|kv| !kv.trim_start().starts_with("channel_binding="))
        .collect();
    let rebuilt = kept.join("&");
    if rebuilt.is_empty() {
        head.trim_end_matches('?').to_string()
    } else {
        format!("{head}{rebuilt}")
    }
}
