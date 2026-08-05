//! The race ledger backend must refuse, not answer.
//!
//! `NeonDb::connect` used to sleep 50ms, log "Connected to Neon (STUB)" and
//! return `Ok`. Every writer then returned `Ok(())` having written nothing and
//! `query` returned an empty row set, which `race::status` printed as
//! "No completed trials yet" -- a positive factual claim about a shared ledger,
//! produced without contacting anything. Pointed at the unroutable address
//! 256.256.256.256 it printed exactly the same thing as it would against a
//! live, empty table.
//!
//! These tests pin the refusal. A well-formed DSN must fail the same way an
//! unroutable or empty one does: syntactic validity is not reachability.

use trios_trainer::race::neon::{NeonDb, STUB_REFUSAL};

/// Every connection string, including one that parses cleanly, is refused.
#[tokio::test]
async fn race_stub_connect_refuses_every_connection_string() {
    let cases = [
        // Well-formed and plausible: a real Neon DSN shape.
        "postgresql://user:pw@ep-cool-name-123456.us-east-2.aws.neon.tech/igla?sslmode=require",
        // Well-formed but unroutable: the address the fabrication was caught with.
        "postgresql://u@256.256.256.256:5432/x",
        // Locally reachable in principle.
        "postgresql://postgres@127.0.0.1:5432/postgres",
        // Degenerate inputs.
        "",
        "not a dsn at all",
    ];

    for dsn in cases {
        let result = NeonDb::connect(dsn).await;
        assert!(
            result.is_err(),
            "connect({dsn:?}) returned Ok; a handle to a database nobody contacted \
             is how 'No completed trials yet' got printed about an empty socket"
        );
        let msg = format!("{:#}", result.err().unwrap());
        assert_eq!(
            msg, STUB_REFUSAL,
            "connect({dsn:?}) refused with an unexpected message"
        );
    }
}

/// The refusal names itself, so the operator learns why and not merely that.
#[test]
fn race_stub_refusal_text_names_the_defect() {
    assert!(
        STUB_REFUSAL.contains("stub"),
        "refusal must say it is a stub: {STUB_REFUSAL}"
    );
    assert!(
        STUB_REFUSAL.contains("no database is contacted"),
        "refusal must say nothing was contacted: {STUB_REFUSAL}"
    );
    assert!(
        STUB_REFUSAL.contains("measurement"),
        "refusal must say the non-answer is not a measurement: {STUB_REFUSAL}"
    );
}

/// The refusal is the only outcome, so no caller can be handed a `NeonDb`
/// and go on to print a leaderboard. `race::status::show_status` and
/// `show_best` both take `&NeonDb` by value-of-reference, and this test is
/// the proof that neither is reachable.
#[tokio::test]
async fn race_stub_no_caller_can_obtain_a_handle() {
    // Repeated calls must not become Ok on a warm path, a retry, or a cache.
    for _ in 0..3 {
        assert!(NeonDb::connect("postgresql://u@localhost/db")
            .await
            .is_err());
    }
}
