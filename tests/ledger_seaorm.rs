// tests/ledger_seaorm.rs
//
// Smoke test for the SeaORM ledger writer.
//
// ## Why this test is gated twice
//
// It used to run DDL migrations plus an insert against whatever `DATABASE_URL`
// (or `NEON_DATABASE_URL`, or `TRIOS_DATABASE_URL`) happened to be exported,
// and to `return` quietly when none was set - so a plain `cargo test` on a
// machine with the production DSN in its environment would have migrated the
// live ledger, and the skip counted as a pass either way.
//
// A survey run has already inserted `canon_name='ledger_seaorm_smoke_test'`
// into the local trios database by accident. That row is the evidence that the
// old gate was not a gate.
//
// Two conditions now hold before anything touches a database:
//
//   1. `TRIOS_ALLOW_LIVE_LEDGER_TESTS=1` must be set explicitly. Having a DSN
//      in the environment is not consent.
//   2. The DSN host must be `localhost` or `127.0.0.1`. A remote host aborts
//      the test with a panic rather than being silently skipped, because a
//      remote host under an explicit opt-in is a misconfiguration worth seeing.
//
// Both live tests are `#[ignore]`d, so without `--ignored` they are reported as
// ignored rather than counted as passing.
//
// To run locally, against a local Postgres only:
//   TRIOS_ALLOW_LIVE_LEDGER_TESTS=1 \
//   DATABASE_URL=postgres://user@localhost/trios \
//   cargo test -p trios-trainer --test ledger_seaorm -- --ignored --nocapture
//
// Acceptance gate G2: proves bpb_sample() writes and reads back correctly via SeaORM.

use migration::MigratorTrait;
use trios_trainer::neon_writer::strip_channel_binding;

/// Hosts this test is permitted to migrate and write to.
const ALLOWED_HOSTS: [&str; 2] = ["localhost", "127.0.0.1"];

/// Resolve a DSN only if the operator opted in AND the target is local.
///
/// Returns `None` when the test should not run at all. Panics when the operator
/// opted in but pointed the test at something that is not local - a silent skip
/// there would hide exactly the misconfiguration that produced the stray row
/// described above.
fn live_dsn_or_skip(test_name: &str) -> Option<String> {
    if std::env::var("TRIOS_ALLOW_LIVE_LEDGER_TESTS").as_deref() != Ok("1") {
        eprintln!(
            "[{test_name}] TRIOS_ALLOW_LIVE_LEDGER_TESTS=1 not set - not touching any database"
        );
        return None;
    }

    let raw = match std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("TRIOS_DATABASE_URL"))
    {
        Ok(u) => u,
        Err(_) => {
            eprintln!("[{test_name}] opt-in set but no DSN exported - nothing to connect to");
            return None;
        }
    };

    let host = dsn_host(&raw);
    assert!(
        ALLOWED_HOSTS.contains(&host.as_str()),
        "[{test_name}] refusing to run: DSN host {host:?} is not in the allowlist \
         {ALLOWED_HOSTS:?}. This test runs DDL migrations and an INSERT; it must \
         never be pointed at a shared or production ledger."
    );

    Some(strip_channel_binding(&raw))
}

/// Extract the host from a `scheme://[user[:pass]@]host[:port]/db` DSN.
///
/// Deliberately hand-rolled: pulling in a URL parser for a host allowlist would
/// add a dependency to a guard whose whole point is to be obvious.
fn dsn_host(dsn: &str) -> String {
    let after_scheme = dsn.split_once("://").map(|(_, r)| r).unwrap_or(dsn);
    let authority = after_scheme
        .split(['/', '?'])
        .next()
        .unwrap_or(after_scheme);
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(authority);
    // Strip an optional :port. IPv6 literals are not supported and will simply
    // fail the allowlist, which is the safe direction.
    host_port
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or(host_port)
        .to_string()
}

/// Smoke test: insert one bpb_sample row (seed=47, step=200, bpb=2.19) and read it back.
///
/// Uses sea_orm directly to verify the row is persisted, proving the SeaORM
/// connection and ActiveModel insert work end-to-end.
#[tokio::test]
#[ignore = "runs DDL migrations and an INSERT; needs TRIOS_ALLOW_LIVE_LEDGER_TESTS=1 and a localhost DSN"]
async fn ledger_seaorm_smoke() {
    let Some(db_url) = live_dsn_or_skip("ledger_seaorm_smoke") else {
        return;
    };

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
            eprintln!("[ledger_seaorm_smoke] row already exists (OK - idempotent)");
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
#[ignore = "runs DDL migrations twice; needs TRIOS_ALLOW_LIVE_LEDGER_TESTS=1 and a localhost DSN"]
async fn ledger_seaorm_migration_idempotent() {
    let Some(db_url) = live_dsn_or_skip("ledger_seaorm_migration_idempotent") else {
        return;
    };

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

/// The host allowlist is itself testable, and this part needs no database.
#[test]
fn dsn_host_extraction_and_allowlist() {
    assert_eq!(
        dsn_host("postgres://user:pw@localhost:5432/trios"),
        "localhost"
    );
    assert_eq!(dsn_host("postgres://user@127.0.0.1/trios"), "127.0.0.1");
    assert_eq!(dsn_host("postgres://localhost/trios"), "localhost");
    assert_eq!(
        dsn_host("postgres://u:p@ep-cool-name-123.us-east-2.aws.neon.tech/neondb?sslmode=require"),
        "ep-cool-name-123.us-east-2.aws.neon.tech"
    );

    assert!(ALLOWED_HOSTS.contains(&dsn_host("postgres://localhost/x").as_str()));
    assert!(
        !ALLOWED_HOSTS.contains(&dsn_host("postgres://u:p@db.production.example.com/x").as_str())
    );
}
