// tests/ledger_seaorm.rs
//
// Smoke test for the SeaORM ledger writer.
//
// Requires a live Postgres database reachable via DATABASE_URL.
// Skipped (passes trivially) if DATABASE_URL is not set, so CI runs
// without a database do not fail.
//
// To run locally:
//   DATABASE_URL=postgres://... cargo test -p trios-trainer ledger_seaorm -- --nocapture
//
// Acceptance gate G2: proves bpb_sample() writes and reads back correctly via SeaORM.

use migration::MigratorTrait;
use trios_trainer::neon_writer::strip_channel_binding;

/// Smoke test: insert one bpb_sample row (seed=47, step=200, bpb=2.19) and read it back.
///
/// Uses sea_orm directly to verify the row is persisted, proving the SeaORM
/// connection and ActiveModel insert work end-to-end.
#[tokio::test]
async fn ledger_seaorm_smoke() {
    let db_url = match std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("TRIOS_DATABASE_URL"))
    {
        Ok(u) => u,
        Err(_) => {
            eprintln!("[ledger_seaorm_smoke] DATABASE_URL not set — skipping live test");
            return;
        }
    };

    let db_url = strip_channel_binding(&db_url);

    use sea_orm::{
        sea_query::OnConflict, ActiveModelTrait, ActiveValue::Set, ColumnTrait, Database,
        EntityTrait, QueryFilter, QuerySelect,
    };
    use trios_trainer::entities::bpb_samples;

    let db = Database::connect(&db_url)
        .await
        .expect("connect to Postgres for smoke test");

    // Ensure schema (idempotent).
    migration::Migrator::up(&db, None)
        .await
        .expect("migration must succeed before smoke test");

    let canon = "ledger_seaorm_smoke_test";
    let seed: i64 = 47; // BIGINT
    let step: i64 = 200; // BIGINT (Wave 24: step lifted from INT to BIGINT)
    let bpb: f64 = 2.19;

    // Insert (idempotent: DO NOTHING on conflict).
    let model = bpb_samples::ActiveModel {
        canon_name: Set(canon.to_string()),
        seed: Set(seed),
        step: Set(step),
        bpb: Set(bpb),
        ts: Set(chrono::Utc::now().into()),
        ..Default::default()
    };

    let on_conflict = OnConflict::columns([
        bpb_samples::Column::CanonName,
        bpb_samples::Column::Seed,
        bpb_samples::Column::Step,
    ])
    .do_nothing()
    .to_owned();

    let insert_result = bpb_samples::Entity::insert(model)
        .on_conflict(on_conflict)
        .exec(&db)
        .await;

    match insert_result {
        Ok(_) => eprintln!("[ledger_seaorm_smoke] insert ok"),
        Err(sea_orm::DbErr::RecordNotInserted) => {
            eprintln!("[ledger_seaorm_smoke] row already exists (OK — idempotent)");
        }
        Err(e) => panic!("bpb_sample insert failed: {e}"),
    }

    // Read back.
    let row = bpb_samples::Entity::find()
        .filter(bpb_samples::Column::CanonName.eq(canon))
        .filter(bpb_samples::Column::Seed.eq(seed))
        .filter(bpb_samples::Column::Step.eq(step))
        .one(&db)
        .await
        .expect("SELECT must succeed")
        .expect("row must exist after insert");

    assert_eq!(row.canon_name, canon, "canon_name must match");
    assert_eq!(row.seed, seed, "seed must match (i64/BIGINT)");
    assert_eq!(row.step, step, "step must match");
    assert!(
        (row.bpb - bpb).abs() < 0.001,
        "bpb must be approximately correct"
    );

    eprintln!(
        "[ledger_seaorm_smoke] verified row: id={} seed={} bpb={}",
        row.id, row.seed, row.bpb
    );

    db.close().await.expect("close connection");
}

/// Verifies the MigratorTrait is available and migration is idempotent.
#[tokio::test]
async fn ledger_seaorm_migration_idempotent() {
    let db_url = match std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("TRIOS_DATABASE_URL"))
    {
        Ok(u) => u,
        Err(_) => {
            eprintln!("[ledger_seaorm_migration_idempotent] DATABASE_URL not set — skipping");
            return;
        }
    };

    let db_url = strip_channel_binding(&db_url);
    let db = sea_orm::Database::connect(&db_url)
        .await
        .expect("connect for idempotency test");

    migration::Migrator::up(&db, None)
        .await
        .expect("first up must succeed");

    migration::Migrator::up(&db, None)
        .await
        .expect("second up must succeed (idempotency)");

    db.close().await.expect("close");
}
