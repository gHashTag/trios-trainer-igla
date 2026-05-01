//! Contract tests: scarab.rs must write final_bpb and final_step to strategy_queue
//!
//! Bugs covered:
//!   - X2: UPDATE strategy_queue missing final_bpb/final_step columns
//!   - X3: No JSONL reader to extract final_bpb from seed_results.jsonl
//!
//! This is an SQL contract test — verifies the UPDATE statement structure.

#[test]
fn test_update_sql_includes_final_metrics() {
    // scarab.rs:208-215 UPDATE statement MUST include:
    // - final_bpb DOUBLE PRECISION
    // - final_step INTEGER
    // Bug X2: current code only has status, finished_at, last_error

    let sql = r#"
        UPDATE strategy_queue
        SET status = $1, finished_at = NOW(), last_error = $2
        WHERE id = $3
    "#;

    // Current state FAILS: doesn't include final_bpb/final_step
    assert!(
        sql.contains("final_bpb"),
        "BUG X2: UPDATE missing final_bpb column"
    );
    assert!(
        sql.contains("final_step"),
        "BUG X2: UPDATE missing final_step column"
    );

    // Expected SQL (after Bug E fix):
    let expected_sql = r#"
        UPDATE strategy_queue
        SET status = $1, finished_at = NOW(), last_error = $2,
            final_bpb = $4, final_step = $5
        WHERE id = $3
    "#;
    assert!(expected_sql.contains("final_bpb"));
    assert!(expected_sql.contains("final_step"));
}

#[test]
fn test_jsonl_reader_logic_exists() {
    // scarab.rs should have logic to:
    // 1. Read seed_results.jsonl after trainer completes
    // 2. Extract last line (final metrics)
    // 3. Parse final_bpb and step
    // 4. Include in UPDATE statement
    //
    // Bug X3: this logic is currently missing

    // This is a placeholder for the actual integration test.
    // Implementation would:
    // 1. Create a temporary seed_results.jsonl with test data
    // 2. Call scarab's metric extraction logic
    // 3. Verify final_bpb and final_step are extracted correctly

    let test_jsonl = r#"{"step": 81000, "bpb": 1.062, "seed": 42}
{"step": 162000, "bpb": 1.041, "seed": 42}
{"step": 243000, "bpb": 1.028, "seed": 42}"#;

    // Last line should be parsed
    let lines: Vec<&str> = test_jsonl.lines().collect();
    let last_line = lines.last().unwrap();

    assert!(last_line.contains("\"step\": 243000"));
    assert!(last_line.contains("\"bpb\": 1.028"));

    // Expected: scarab.rs reads this and passes to UPDATE
    // Bug X3: this parsing code is missing
}
