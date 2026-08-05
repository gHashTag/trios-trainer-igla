// src/neon_writer.rs  [ledger]
//
// Ledger writer for trios-trainer-igla.
//
// Provides the same public API as the original tokio-postgres version, but
// uses SeaORM internally.  The connection is cached in a Mutex and re-tried
// after a failure.  All public functions keep their synchronous signatures;
// they block on a private Tokio runtime (same pattern as before).
//
// ENV fallback chain (do NOT introduce NEON_* as primary):
//   DATABASE_URL (canonical, set by Railway since #113)
//   -> NEON_DATABASE_URL  (legacy alias)
//   -> TRIOS_NEON_DSN     (legacy alias)
//   -> TRIOS_DATABASE_URL (legacy alias)
//
// WRITE OPT-IN: a DSN alone is NOT permission to write.
//   TRIOS_LEDGER_WRITE=1 must ALSO be set before this process touches the
//   shared ledger. Any of the four aliases above is routinely present in a
//   developer or CI environment for reasons that have nothing to do with this
//   trainer, and connecting on their presence alone is how a scouting run put
//   12 rows on the shared ledger before anyone noticed. Without the opt-in this
//   module behaves exactly as if no DSN were configured -- `db()` returns
//   `None`, `db_status()` is `DbStatus::NoDsn`, `dsn_configured()` is false and
//   `ledger_exit_code()` is 0 -- and says so once, on stderr, naming the
//   variable. The only guard that existed before was `env -i` inside a CI
//   workflow file, which protects CI and nothing else. What it does NOT do is
//   record the wrong cause: the sidecar gets `skipped-not-opted-in`, never
//   `skipped-no-dsn` (see `LedgerWrite::SkippedNotOptedIn`, 2026-08-03).
//
// Constitutional notes:
//   R5 - never panic the trainer on DB errors; log a warn and continue.
//   R7 - emits forward step/seed/bpb verbatim.
//   R9 - writer never touches ledger::emit_row; embargo gate stays in trios.
//
// Anchor: phi^2 + phi^-2 = 3 - DOI 10.5281/zenodo.19227877

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use sea_orm::{
    sea_query::OnConflict, ActiveModelTrait, ActiveValue::Set, ColumnTrait, ConnectionTrait,
    Database, DatabaseConnection, EntityTrait, QueryFilter,
};
use tokio::runtime::Runtime;

use crate::entities::{bpb_samples, igla_agents_heartbeat, igla_race_trials};

// -- Tokio runtime -------------------------------------------------------------

fn rt() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime")
    })
}

// -- DSN helpers ---------------------------------------------------------------

/// Resolve the database DSN from the environment fallback chain.
///
/// A DSN found here is a piece of CONFIGURATION, not permission to write. Use
/// [`active_dsn`] everywhere the answer decides whether shared state is
/// touched.
fn resolve_dsn() -> Option<String> {
    std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("NEON_DATABASE_URL"))
        .or_else(|_| std::env::var("TRIOS_NEON_DSN"))
        .or_else(|_| std::env::var("TRIOS_DATABASE_URL"))
        .ok()
}

/// The env var an operator must set to `1` before this process writes to the
/// shared ledger. Named in every message about the refusal so nobody has to
/// read this file to find it.
pub const LEDGER_WRITE_OPT_IN: &str = "TRIOS_LEDGER_WRITE";

/// Has the operator explicitly asked for ledger writes this run?
///
/// Deliberately strict: exactly `1`, trimmed. "true"/"yes"/"on" are not
/// accepted because a half-recognised spelling is how an opt-in silently
/// becomes a default again.
fn ledger_write_opted_in() -> bool {
    std::env::var(LEDGER_WRITE_OPT_IN)
        .ok()
        .is_some_and(|v| v.trim() == "1")
}

/// Say once, out loud, that a configured DSN is being ignored.
///
/// Printed from inside this module rather than from a binary so every caller
/// gets it, including the ones that never look at [`db_status`].
fn announce_missing_opt_in() {
    static ANNOUNCED: std::sync::Once = std::sync::Once::new();
    ANNOUNCED.call_once(|| {
        eprintln!(
            "[ledger] a DSN is configured but {LEDGER_WRITE_OPT_IN} is not set to 1: \
             this process will NOT write to the shared ledger and reports itself as \
             having no DSN. Set {LEDGER_WRITE_OPT_IN}=1 to opt in."
        );
    });
}

/// The DSN this process is ALLOWED to use: configured AND opted in.
///
/// Every decision that can reach shared state goes through here. When a DSN is
/// configured without the opt-in the answer is `None` -- the same shape as no
/// DSN at all, which is what `dsn_configured`, `db_status` and
/// `ledger_exit_code` then report. That equivalence is deliberate: a run that
/// is not going to be recorded must look, to every caller and to the exit
/// status, exactly like a run that was never asked to record itself. The one
/// thing that must NOT be silent is the configuration being overridden, so
/// that is announced once by name -- and, since 2026-08-03, also written into
/// the record as [`LedgerWrite::SkippedNotOptedIn`] rather than as
/// `SkippedNoDsn`. Behaviour is equivalent; the stated CAUSE is not, and the
/// sidecar is an evidence document.
fn active_dsn() -> Option<String> {
    let dsn = resolve_dsn()?;
    if ledger_write_opted_in() {
        return Some(dsn);
    }
    announce_missing_opt_in();
    None
}

/// Remove `channel_binding=require` (or `=prefer`) from a Neon-style DSN.
///
/// `tokio-postgres-rustls` / `sqlx-postgres` with rustls do not expose the TLS
/// exporter needed for `tls-server-end-point` channel binding, so
/// SCRAM-SHA-256-PLUS auth fails.  Neon Postgres accepts plain SCRAM-SHA-256
/// over TLS; stripping this query-string param is the minimal fix (#84, #113).
pub fn strip_channel_binding(dsn: &str) -> String {
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

// -- SeaORM connection cache ----------------------------------------------------

/// Why `db()` returned what it did.
///
/// EPIC-446: `db()` used to return `None` both when no DSN was configured and
/// when `Database::connect` failed, so an unreachable database printed
/// "DSN unset - skipping". Callers that record their own outcome (the
/// checkpoint sidecar) would inherit that falsehood, so the two cases are now
/// distinguished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbStatus {
    /// A connection is live.
    Ready,
    /// No DSN in the environment at all.
    NoDsn,
    /// A DSN was configured but `Database::connect` failed.
    ConnectFailed,
}

impl DbStatus {
    /// Short reason string for log lines.
    pub fn as_str(self) -> &'static str {
        match self {
            DbStatus::Ready => "ready",
            DbStatus::NoDsn => "DSN unset",
            DbStatus::ConnectFailed => "DSN set but connect failed",
        }
    }
}

/// Minimum wait between reconnect attempts, so a permanently unreachable host
/// costs one connect timeout per interval and not one per training step.
const RECONNECT_BACKOFF: Duration = Duration::from_secs(5);

struct ConnState {
    conn: Option<DatabaseConnection>,
    status: DbStatus,
    last_attempt: Option<Instant>,
}

/// Cached connection plus the bookkeeping needed to retry it.
///
/// The previous version cached the *result* of the first connect in a
/// `OnceLock<Option<DatabaseConnection>>`. One transient failure - a Neon
/// cold start, a DNS blip during boot - therefore disabled every ledger write
/// for the remaining lifetime of the process, and the trainer went on printing
/// BPB lines as if they had been recorded. A `Mutex` lets the connection be
/// re-established, and `DROPPED_WRITES` records what was lost meanwhile.
fn conn_state() -> &'static Mutex<ConnState> {
    static S: OnceLock<Mutex<ConnState>> = OnceLock::new();
    S.get_or_init(|| {
        Mutex::new(ConnState {
            conn: None,
            status: DbStatus::NoDsn,
            last_attempt: None,
        })
    })
}

/// Rows that reached a live connection and were accepted by Postgres.
static LANDED_WRITES: AtomicU64 = AtomicU64::new(0);
/// Rows that were meant to be written and were not, for any reason.
static DROPPED_WRITES: AtomicU64 = AtomicU64::new(0);
/// Rows this writer REFUSED to publish: the value or its provenance was
/// unpublishable, so nothing was lost and something wrong was stopped.
///
/// Kept apart from `DROPPED_WRITES` because the two mean opposite things to an
/// operator -- a drop is a transport failure worth retrying, a refusal is this
/// crate declining to publish. Kept counted at all because `LedgerWrite::
/// Rejected` used to increment neither counter, so a process in which 100% of
/// rows were refused reported `attempted=0 landed=0 dropped=0` and exit 0.
static REJECTED_WRITES: AtomicU64 = AtomicU64::new(0);

fn note_landed() {
    LANDED_WRITES.fetch_add(1, Ordering::Relaxed);
}

fn note_dropped() {
    DROPPED_WRITES.fetch_add(1, Ordering::Relaxed);
}

fn note_rejected() {
    REJECTED_WRITES.fetch_add(1, Ordering::Relaxed);
}

/// Rows accepted by the database so far in this process.
pub fn landed_writes() -> u64 {
    LANDED_WRITES.load(Ordering::Relaxed)
}

/// Writes attempted and lost so far in this process.
pub fn dropped_writes() -> u64 {
    DROPPED_WRITES.load(Ordering::Relaxed)
}

/// Writes this crate refused as unpublishable so far in this process.
pub fn rejected_writes() -> u64 {
    REJECTED_WRITES.load(Ordering::Relaxed)
}

/// Will this run write to a ledger at all?
///
/// Exposed so a binary can say "no DSN configured" instead of printing a
/// tally that reads as an assertion about writes. `trios-train` printed
/// `LEDGER: attempted=3 landed=0 dropped=3 rejected=0` on a run where no DSN
/// was ever set - three writes reported DROPPED when none had been attempted.
/// The counters are truthful about the calls (each write function does note a
/// drop when it finds no connection); the LINE was not, because "dropped"
/// means "lost in transport" to every reader of it.
///
/// Reads [`active_dsn`], so a DSN present without `TRIOS_LEDGER_WRITE=1`
/// answers false: the question a caller is really asking is "is this run
/// recorded", and an un-opted-in run is not. The override is announced once on
/// stderr by [`announce_missing_opt_in`], so it is refused loudly and reported
/// consistently.
pub fn dsn_configured() -> bool {
    active_dsn().is_some()
}

/// The two row kinds that ARE the evidence of a run.
///
/// A heartbeat says the process is alive. A `bpb_sample` says what was
/// measured and a `checkpoint_record` says which artifact produced it. Only
/// the latter two can substantiate a claim, so only the latter two decide the
/// exit code in [`ledger_exit_code`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvidenceKind {
    BpbSample,
    CheckpointRecord,
}

static BPB_ATTEMPTED: AtomicU64 = AtomicU64::new(0);
static BPB_LANDED: AtomicU64 = AtomicU64::new(0);
static CKPT_ATTEMPTED: AtomicU64 = AtomicU64::new(0);
static CKPT_LANDED: AtomicU64 = AtomicU64::new(0);

/// Record one evidence-row outcome, per kind, and hand the outcome back.
///
/// Every call counts as ATTEMPTED - a refusal and a lost write are both
/// attempts that produced no row - and only `Written` counts as landed.
fn note_evidence(kind: EvidenceKind, outcome: LedgerWrite) -> LedgerWrite {
    let (attempted, landed) = match kind {
        EvidenceKind::BpbSample => (&BPB_ATTEMPTED, &BPB_LANDED),
        EvidenceKind::CheckpointRecord => (&CKPT_ATTEMPTED, &CKPT_LANDED),
    };
    attempted.fetch_add(1, Ordering::Relaxed);
    if outcome == LedgerWrite::Written {
        landed.fetch_add(1, Ordering::Relaxed);
    }
    outcome
}

/// `bpb_sample` rows attempted / landed so far in this process.
pub fn bpb_sample_writes() -> (u64, u64) {
    (
        BPB_ATTEMPTED.load(Ordering::Relaxed),
        BPB_LANDED.load(Ordering::Relaxed),
    )
}

/// `checkpoint_record` rows attempted / landed so far in this process.
pub fn checkpoint_record_writes() -> (u64, u64) {
    (
        CKPT_ATTEMPTED.load(Ordering::Relaxed),
        CKPT_LANDED.load(Ordering::Relaxed),
    )
}

/// Process exit code the trainer must honour.
///
/// A configured DSN is a statement that the run is supposed to be recorded. If
/// writes were attempted under one and nothing landed, the run produced logs
/// and no ledger - exiting 0 would let a supervisor record it as a success.
/// With no DSN configured at all there is nothing to be silent about, so the
/// code stays 0.
///
/// A REFUSAL is checked before a drop and is a stronger reason to fail, not a
/// weaker one: a dropped write means the transport lost a value this crate was
/// willing to publish, while a refusal means this crate looked at the value and
/// judged it unpublishable. A run whose every row was refused has not merely
/// failed to record itself -- it produced nothing fit to record.
///
/// The aggregate `landed` counter is NOT sufficient on its own. It counts every
/// kind of row alike, so ONE landed heartbeat licensed a 0 exit while every
/// `bpb_sample` and every `checkpoint_record` -- the two kinds that are the
/// evidence -- dropped. "The process was alive" is not "the run recorded
/// itself". Each evidence kind is therefore judged separately: attempted and
/// never landed is fatal, however many heartbeats got through.
pub fn ledger_exit_code() -> i32 {
    if active_dsn().is_none() {
        return 0;
    }
    for (label, (attempted, landed)) in [
        ("bpb_sample", bpb_sample_writes()),
        ("checkpoint_record", checkpoint_record_writes()),
    ] {
        if attempted > 0 && landed == 0 {
            eprintln!(
                "[ledger] FATAL: a DSN was configured, {attempted} {label} row(s) were \
                 attempted and 0 landed. Exiting non-zero: a landed heartbeat is not \
                 evidence, this run recorded no {label}."
            );
            return 1;
        }
    }
    let landed = landed_writes();
    let dropped = dropped_writes();
    let rejected = rejected_writes();
    if rejected > 0 && landed == 0 {
        eprintln!(
            "[ledger] FATAL: a DSN was configured, {rejected} row(s) were refused as \
             unpublishable and 0 landed. Exiting non-zero: this run published nothing."
        );
        return 1;
    }
    if dropped > 0 && landed == 0 {
        eprintln!(
            "[ledger] FATAL: a DSN was configured, {dropped} write(s) were attempted \
             and 0 landed. Exiting non-zero: this run has no ledger."
        );
        return 1;
    }
    if dropped > 0 {
        eprintln!("[ledger] WARNING: {landed} write(s) landed, {dropped} were dropped.");
    }
    if rejected > 0 {
        eprintln!("[ledger] WARNING: {landed} write(s) landed, {rejected} were refused.");
    }
    0
}

fn db() -> Option<DatabaseConnection> {
    let mut st = conn_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if let Some(conn) = st.conn.as_ref() {
        return Some(conn.clone());
    }

    // `active_dsn`, not `resolve_dsn`: an ambient DSN is configuration, and
    // connecting on its presence alone is how an unrelated scouting run landed
    // rows on the shared ledger.
    let Some(raw_dsn) = active_dsn() else {
        st.status = DbStatus::NoDsn;
        return None;
    };

    if let Some(t) = st.last_attempt {
        if t.elapsed() < RECONNECT_BACKOFF {
            return None;
        }
    }
    st.last_attempt = Some(Instant::now());

    let dsn = strip_channel_binding(&raw_dsn);
    if dsn != raw_dsn {
        eprintln!("[ledger] stripped channel_binding from DSN (rustls limitation)");
    }
    eprintln!("[ledger] connecting via SeaORM ...");
    match rt().block_on(async { Database::connect(&dsn).await }) {
        Ok(conn) => {
            eprintln!("[ledger] connected OK");
            st.status = DbStatus::Ready;
            st.conn = Some(conn.clone());
            Some(conn)
        }
        Err(e) => {
            eprintln!("[ledger] connect failed: {e} (will retry in {RECONNECT_BACKOFF:?})");
            st.status = DbStatus::ConnectFailed;
            None
        }
    }
}

/// Status of the cached connection. Forces the connection attempt if it has
/// not happened yet, so the answer is never stale.
pub fn db_status() -> DbStatus {
    let _ = db();
    conn_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .status
}

/// The outcome to RECORD when a write path found no connection.
///
/// `DbStatus` answers "what happened to the connection"; this answers "why is
/// there no row", and the two are not the same question. `DbStatus::NoDsn`
/// covers both a genuinely empty environment and an un-opted-in one, because
/// [`active_dsn`] collapses them on purpose -- see [`LedgerWrite::
/// SkippedNotOptedIn`] for why that collapse must not reach the sidecar. The
/// distinction is re-derived here from the environment rather than carried on
/// `DbStatus`: the opt-in is decided BEFORE any connect, so no connection
/// state describes it, and inventing a `DbStatus` for it would put a
/// non-connection fact in a connection enum.
fn skipped_write_outcome(status: DbStatus) -> LedgerWrite {
    match status {
        DbStatus::NoDsn => {
            if resolve_dsn().is_some() && !ledger_write_opted_in() {
                LedgerWrite::SkippedNotOptedIn
            } else {
                LedgerWrite::SkippedNoDsn
            }
        }
        _ => LedgerWrite::Failed,
    }
}

// -- Public API ----------------------------------------------------------------

/// Insert a fresh row into `igla_race_trials` (idempotent on `trial_id`).
///
/// Old raw-SQL call:
///   INSERT INTO igla_race_trials ... ON CONFLICT (trial_id) DO UPDATE ...
/// New SeaORM call:
///   igla_race_trials::Entity::insert(active_model).on_conflict(...).exec(&db)
pub fn trial_start(trial_id: &str, config_json: &str, agent_id: &str, branch: &str) {
    let Some(conn) = db() else {
        eprintln!(
            "[ledger] no connection ({}) - skipping trial_start",
            db_status().as_str()
        );
        note_dropped();
        return;
    };

    let trial_uuid = match trial_id.parse::<uuid::Uuid>() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[ledger] trial_start: invalid UUID {trial_id}: {e}");
            note_dropped();
            return;
        }
    };
    // A row whose `config` is an empty object describes a run that never
    // happened. Substituting `{}` for unparseable JSON and then printing
    // "trial_start ok" published a trial with no recipe attached, so the
    // malformed config is now a refusal instead of a default.
    let config_val: serde_json::Value = match serde_json::from_str(config_json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "[ledger] trial_start: invalid JSON for {trial_id}: {e} - refusing to \
                 record a trial with a fabricated empty config"
            );
            note_dropped();
            return;
        }
    };

    let model = igla_race_trials::ActiveModel {
        trial_id: Set(trial_uuid),
        config: Set(config_val),
        status: Set("running".to_string()),
        agent_id: Set(Some(agent_id.to_string())),
        branch: Set(Some(branch.to_string())),
        ..Default::default()
    };

    let on_conflict = OnConflict::column(igla_race_trials::Column::TrialId)
        .update_columns([
            igla_race_trials::Column::Status,
            igla_race_trials::Column::AgentId,
            igla_race_trials::Column::Branch,
        ])
        .to_owned();

    let res = rt().block_on(
        igla_race_trials::Entity::insert(model)
            .on_conflict(on_conflict)
            .exec(&conn),
    );
    match res {
        Ok(_) => {
            eprintln!("[ledger] trial_start ok: {trial_id}");
            note_landed();
        }
        Err(e) => {
            eprintln!("[ledger] trial_start failed: {e}");
            note_dropped();
        }
    }
}

/// What an `UPDATE ... WHERE trial_id = $1` that matched no row means.
///
/// `rows_affected == 0` is the exact shape of a trial whose `trial_start` was
/// dropped: the statement is accepted, nothing is updated, and the caller used
/// to print "ok: ... rows=0". It is a failure, not a success.
fn update_outcome(rows_affected: u64) -> LedgerWrite {
    if rows_affected == 0 {
        LedgerWrite::Failed
    } else {
        LedgerWrite::Written
    }
}

/// Filter a caller-supplied BPB through [`reject_bpb`].
///
/// `None` means the caller took no measurement and there is nothing to record.
/// `Some(v)` that `reject_bpb` refuses is a sentinel (`f32::MAX`, a value at or
/// under the degenerate-corpus floor), not a reading; it is dropped with a
/// printed reason rather than bound to a column.
fn guarded_bpb(what: &str, id: &str, bpb: Option<f32>) -> Option<f32> {
    let v = bpb?;
    match reject_bpb(v) {
        Some(reason) => {
            eprintln!("[ledger] REJECT {what} bpb for {id}: {reason}");
            None
        }
        None => Some(v),
    }
}

/// Render an optional BPB for a log line without inventing a number.
fn fmt_bpb(bpb: Option<f32>) -> String {
    match bpb {
        Some(v) => format!("{v:.4}"),
        None => "unmeasured".to_string(),
    }
}

/// Upsert agent heartbeat and update the latest BPB on the trial row.
///
/// Old raw-SQL calls:
///   INSERT INTO igla_agents_heartbeat ... ON CONFLICT (agent_id) DO UPDATE ...
///   UPDATE igla_race_trials SET bpb_latest=$1, steps_done=$2 WHERE trial_id=$3
/// New SeaORM calls:
///   igla_agents_heartbeat::Entity::insert(...).on_conflict(...).exec(&db)
///   igla_race_trials::Entity::update_many().col_expr(...).filter(...).exec(&db)
pub fn heartbeat(trial_id: &str, agent_id: &str, bpb: Option<f32>, step: usize) {
    // Guard first: a sentinel must not reach `bpb_latest` even when the write
    // itself would have succeeded. `None` leaves the column untouched rather
    // than overwriting a real reading with a placeholder.
    let bpb_ok = guarded_bpb("heartbeat", trial_id, bpb);

    let Some(conn) = db() else {
        eprintln!(
            "[ledger] no connection ({}) - skipping heartbeat",
            db_status().as_str()
        );
        note_dropped();
        return;
    };

    // Upsert heartbeat row.
    let hb_model = igla_agents_heartbeat::ActiveModel {
        agent_id: Set(agent_id.to_string()),
        machine_id: Set("railway".to_string()),
        branch: Set("main".to_string()),
        task: Set(Some(trial_id.to_string())),
        status: Set("active".to_string()),
        last_heartbeat: Set(chrono::Utc::now().into()),
    };
    let on_conflict_hb = OnConflict::column(igla_agents_heartbeat::Column::AgentId)
        .update_columns([
            igla_agents_heartbeat::Column::Status,
            igla_agents_heartbeat::Column::LastHeartbeat,
            igla_agents_heartbeat::Column::Task,
        ])
        .to_owned();
    let res = rt().block_on(
        igla_agents_heartbeat::Entity::insert(hb_model)
            .on_conflict(on_conflict_hb)
            .exec(&conn),
    );
    match res {
        Ok(_) => note_landed(),
        Err(e) => {
            eprintln!("[ledger] heartbeat (upsert) failed: {e}");
            note_dropped();
        }
    }

    // Update trial row with latest bpb / step.
    // steps_done is BIGINT - bind as i64.
    let trial_uuid = match trial_id.parse::<uuid::Uuid>() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[ledger] heartbeat: invalid UUID {trial_id}: {e}");
            note_dropped();
            return;
        }
    };

    use sea_orm::sea_query::Expr;
    let mut upd = igla_race_trials::Entity::update_many().col_expr(
        igla_race_trials::Column::StepsDone,
        Expr::value(step as i64),
    );
    if let Some(v) = bpb_ok {
        upd = upd.col_expr(igla_race_trials::Column::BpbLatest, Expr::value(v as f64));
    }
    let res = rt().block_on(
        upd.filter(igla_race_trials::Column::TrialId.eq(trial_uuid))
            .exec(&conn),
    );
    match res {
        Ok(r) => match update_outcome(r.rows_affected) {
            LedgerWrite::Written => {
                eprintln!(
                    "[ledger] heartbeat ok: trial={trial_id} step={step} bpb_latest={} rows={}",
                    fmt_bpb(bpb_ok),
                    r.rows_affected
                );
                note_landed();
            }
            _ => {
                eprintln!(
                    "[ledger] heartbeat updated 0 rows: trial={trial_id} - no such trial \
                     row, nothing recorded"
                );
                note_dropped();
            }
        },
        Err(e) => {
            eprintln!("[ledger] heartbeat (update) failed: {e}");
            note_dropped();
        }
    }
}

/// Mark a trial complete with the final BPB.
///
/// Old raw-SQL call:
///   UPDATE igla_race_trials SET bpb_final=$1, status='complete' WHERE trial_id=$2
/// New SeaORM call:
///   igla_race_trials::Entity::update_many().col_expr(...).filter(...).exec(&db)
pub fn trial_complete(trial_id: &str, bpb: Option<f32>) {
    let bpb_ok = guarded_bpb("trial_complete", trial_id, bpb);

    let Some(conn) = db() else {
        eprintln!(
            "[ledger] no connection ({}) - skipping trial_complete",
            db_status().as_str()
        );
        note_dropped();
        return;
    };

    let trial_uuid = match trial_id.parse::<uuid::Uuid>() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[ledger] trial_complete: invalid UUID {trial_id}: {e}");
            note_dropped();
            return;
        }
    };

    // The trial is complete either way; `bpb_final` is only set when a value
    // survived the guard, so a run that measured nothing leaves the column
    // NULL instead of stamping it with a sentinel.
    use sea_orm::sea_query::Expr;
    let mut upd = igla_race_trials::Entity::update_many().col_expr(
        igla_race_trials::Column::Status,
        Expr::value("complete".to_string()),
    );
    if let Some(v) = bpb_ok {
        upd = upd.col_expr(igla_race_trials::Column::BpbFinal, Expr::value(v as f64));
    }
    let res = rt().block_on(
        upd.filter(igla_race_trials::Column::TrialId.eq(trial_uuid))
            .exec(&conn),
    );
    match res {
        Ok(r) => match update_outcome(r.rows_affected) {
            LedgerWrite::Written => {
                eprintln!(
                    "[ledger] trial_complete ok: trial={trial_id} bpb_final={} rows={}",
                    fmt_bpb(bpb_ok),
                    r.rows_affected
                );
                note_landed();
            }
            _ => {
                eprintln!(
                    "[ledger] trial_complete updated 0 rows: trial={trial_id} - no such \
                     trial row, nothing recorded"
                );
                note_dropped();
            }
        },
        Err(e) => {
            eprintln!("[ledger] trial_complete failed: {e}");
            note_dropped();
        }
    }
}

/// Parse `IGLA-{LANE}-{format}-h{H}-LR{L}-rng{SEED}-{algo}` canon_name into
/// `(format, algo, hidden)` triple. Returns `None` if the canon_name does not
/// match the canonical IGLA schema (legacy `scarab-*` names, smoke names, etc).
///
/// Examples:
///   `IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng1597-adamw` ->
///     `("gf16", "adamw", 128)`
///   `IGLA-SCARAB-ADAMW-binary16-h384-LR0001-rng123-adamw` ->
///     `("binary16", "adamw", 384)`
///
/// Rule: format = field BEFORE `-h{N}-`; hidden = N; algo = LAST `-` field.
///
/// Anchor: phi^2+phi^-2=3 - DOI 10.5281/zenodo.19227877
pub fn parse_canon_name(canon: &str) -> Option<(String, String, i32)> {
    if !canon.starts_with("IGLA-") {
        return None;
    }
    let parts: Vec<&str> = canon.split('-').collect();
    if parts.len() < 5 {
        return None;
    }

    // Find the `h{N}` token (hidden); the token immediately before it is the format.
    let mut hidden: Option<i32> = None;
    let mut h_idx: Option<usize> = None;
    for (i, tok) in parts.iter().enumerate() {
        if let Some(rest) = tok.strip_prefix('h') {
            if let Ok(n) = rest.parse::<i32>() {
                hidden = Some(n);
                h_idx = Some(i);
                break;
            }
        }
    }
    let h_idx = h_idx?;
    let hidden = hidden?;
    if h_idx == 0 {
        return None;
    }
    let format = parts[h_idx - 1].to_string();
    if format.is_empty() {
        return None;
    }

    // algo = everything AFTER the last `rng{SEED}` token, joined with '-'.
    // This preserves multi-token algos like `muon-cwd`.
    let mut rng_idx: Option<usize> = None;
    for (i, tok) in parts.iter().enumerate() {
        if let Some(rest) = tok.strip_prefix("rng") {
            if rest.parse::<i64>().is_ok() {
                rng_idx = Some(i);
            }
        }
    }
    let rng_idx = rng_idx?;
    if rng_idx + 1 >= parts.len() {
        return None;
    }
    let algo = parts[rng_idx + 1..].join("-");
    if algo.is_empty() {
        return None;
    }

    Some((format, algo, hidden))
}

/// Optimizers that a real trainer in this crate can actually execute.
///
/// Mirror of the matrix_runner whitelist. Rejects fake/silent-fallback algos
/// (soap/lamb/prodigy/lion/...) at the WRITE path so they cannot reach
/// `ssot.bpb_samples` regardless of which trainer binary emitted them.
/// Refs: trios#777, trios#779, migration 0006_quarantine_fake_canons.
/// R5 evidence: gf16-lamb vs gf16-prodigy produced bit-identical BPB at every
/// step (verified 2026-05-14T14:36Z, B-22).
pub const ALGO_WHITELIST: &[&str] = &["adamw", "muon", "muon-cwd"];

/// Reserved canon prefix for operator-supplied and synthetic numbers.
///
/// `bpb_smoke` and `smoke_train` publish a BPB that was typed into an env var,
/// not measured. Under an unconstrained `CANON_NAME` those rows were
/// indistinguishable from a trained result in the leaderboard's own table.
/// Any canon carrying this prefix is refused entry to `ssot.bpb_samples`.
pub const SMOKE_CANON_PREFIX: &str = "SMOKE-";

/// Force a name into the reserved smoke namespace, idempotently.
///
/// Applied by the smoke binaries to whatever the operator supplies, so there
/// is no spelling of `CANON_NAME` that lets a hand-typed BPB into ssot.
pub fn smoke_canon_name(raw: &str) -> String {
    let trimmed = raw.trim();
    let core = trimmed.trim_start_matches(SMOKE_CANON_PREFIX);
    if core.is_empty() {
        return format!("{SMOKE_CANON_PREFIX}UNNAMED");
    }
    format!("{SMOKE_CANON_PREFIX}{core}")
}

/// Upper bound on a BPB that could be a real measurement.
///
/// `evaluate` used to return `f32::MAX` as its "could not measure" sentinel,
/// and every downstream guard tested `is_finite()` -- which `f32::MAX` passes.
/// A row reading 340282346638528859811704183484516925440.0000 was published
/// and then divided by, producing a panic. Over a 128-symbol vocabulary the
/// uniform-model ceiling is log2(128) = 7 bits per byte; 64 leaves two orders
/// of magnitude of headroom for a genuinely broken-but-real model and still
/// rejects every sentinel.
pub const BPB_SENTINEL_CEILING: f32 = 64.0;

/// Reject a BPB that cannot be a measurement of held-out text.
///
/// Mirrors `train_loop::guard_bpb`, but lives on the WRITE side so no caller
/// can bypass it. Returns the reason string when the value must not be
/// published.
fn reject_bpb(bpb: f32) -> Option<String> {
    if !bpb.is_finite() {
        return Some(format!("bpb={bpb} is not finite"));
    }
    if bpb >= BPB_SENTINEL_CEILING {
        return Some(format!(
            "bpb={bpb} >= BPB_SENTINEL_CEILING {BPB_SENTINEL_CEILING}: this is an \
             unmeasured sentinel, not a reading"
        ));
    }
    // Compared in f32, the space the reading actually lives in. Widening to
    // f64 first makes the floor itself pass: `0.1f64 as f32 as f64` is
    // 0.10000000149011612, which is strictly greater than 0.1.
    //
    // The bound is the PUBLICATION floor, not the JEPA-proxy detector floor:
    // everything reaching this function is on its way into `ssot.bpb_samples`.
    // `JEPA_PROXY_BPB_FLOOR` (0.1) is 26x below the band the measured
    // calibration implies and admitted the retracted 1.5492 unchallenged. See
    // `invariants::PUBLISHED_BPB_FLOOR` for the derivation.
    let floor = crate::invariants::PUBLISHED_BPB_FLOOR;
    if bpb <= floor {
        return Some(format!(
            "bpb={bpb} <= PUBLISHED_BPB_FLOOR {floor}: below anything this \
             architecture can honestly reach, so the eval corpus is degenerate or \
             duplicated, or the number did not come from this crate"
        ));
    }
    None
}

/// The optimizer that actually executed, bound once at process start.
static EXECUTED_ALGO: OnceLock<String> = OnceLock::new();

/// Bind the optimizer that will actually execute to this process.
///
/// The write-side whitelist used to gate `algo` parsed out of `canon_name`, so
/// a genuine `adamw` run under a `-soap` canon was silently dropped mid-run and
/// a `--optimizer=adamw` run under a `-muon` canon was recorded as muon. The
/// executed optimizer is the fact; the canon suffix is a label. A trainer calls
/// this before step 1 and hard-errors on `Err`, instead of discovering the
/// disagreement as a silent `return` at the first checkpoint.
pub fn bind_executed_optimizer(canon_name: &str, algo: &str) -> Result<(), String> {
    if !ALGO_WHITELIST.contains(&algo) {
        return Err(format!(
            "executed optimizer '{algo}' is not in {ALGO_WHITELIST:?}; refusing to \
             start a run whose results could not be published"
        ));
    }
    if let Some((_format, suffix_algo, _hidden)) = parse_canon_name(canon_name) {
        if suffix_algo != algo {
            return Err(format!(
                "canon_name '{canon_name}' claims optimizer '{suffix_algo}' but the run \
                 executes '{algo}'. Fix one of them before starting: a run recorded under \
                 the wrong optimizer is worse than a run not recorded at all."
            ));
        }
    }
    match EXECUTED_ALGO.set(algo.to_string()) {
        Ok(()) => Ok(()),
        Err(_) => {
            let bound = EXECUTED_ALGO.get().map(String::as_str).unwrap_or("");
            if bound == algo {
                Ok(())
            } else {
                Err(format!(
                    "executed optimizer already bound to '{bound}', cannot rebind to '{algo}'"
                ))
            }
        }
    }
}

/// The optimizer bound by `bind_executed_optimizer`, if any.
pub fn executed_optimizer() -> Option<&'static str> {
    EXECUTED_ALGO.get().map(String::as_str)
}

/// Where the `algo` attached to a row came from.
///
/// The two used to collapse into one `String`, and that is the whole bug. The
/// anti-mislabelling gate in `bpb_sample_with_algo_inner` asks "does the canon
/// suffix agree with the optimizer that executed?" -- but `bpb_sample` filled
/// the second half of that comparison from the canon suffix itself when nothing
/// had been declared, so the test compared the suffix to itself and could not
/// fail. Provenance therefore travels WITH the value: only a
/// [`bind_executed_optimizer`] declaration is admissible as the right-hand side
/// of that comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AlgoSource {
    /// Bound by [`bind_executed_optimizer`], or passed by a caller that states
    /// it ran this optimizer. The only thing that can CONFIRM a canon suffix.
    Declared(String),
    /// Read out of `canon_name`. A label somebody typed, never evidence.
    CanonSuffix(String),
    /// Nothing was declared and the canon carries no suffix.
    Undeclared,
}

impl AlgoSource {
    /// The optimizer this crate is willing to certify as EXECUTED, if any.
    fn declared(&self) -> Option<&str> {
        match self {
            AlgoSource::Declared(a) => Some(a.as_str()),
            AlgoSource::CanonSuffix(_) | AlgoSource::Undeclared => None,
        }
    }

    /// The string that goes in the row's `algo` column, whatever its origin.
    ///
    /// A LABEL, and only that. `Undeclared` labels as the empty string, which
    /// the legacy path already treated as "the whitelist could not be applied".
    fn label(&self) -> &str {
        match self {
            AlgoSource::Declared(a) | AlgoSource::CanonSuffix(a) => a.as_str(),
            AlgoSource::Undeclared => "",
        }
    }

    /// A caller-supplied optimizer, normalising the empty string to
    /// `Undeclared`: "" is the absence of a declaration, not a declaration of
    /// nothing.
    fn from_declared(algo: &str) -> Self {
        if algo.trim().is_empty() {
            AlgoSource::Undeclared
        } else {
            AlgoSource::Declared(algo.to_string())
        }
    }
}

/// What the canon suffix and the executed optimizer say about each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CanonVerdict {
    /// A declaration was made and it agrees with the suffix. The only state in
    /// which this crate certifies that the canon names what ran.
    Confirmed,
    /// A declaration was made and it contradicts the suffix. Fatal for the row.
    Contradicted,
    /// Nothing was declared, so the suffix stands unconfirmed. NOT `Confirmed`:
    /// a label cannot be its own evidence.
    Undeclared,
}

/// Compare a canon suffix against the optimizer this crate can certify.
///
/// Split out of `bpb_sample_with_algo_inner` so the comparison is testable
/// without a database, because the bug it replaces was invisible in every test
/// that needed one: `bpb_sample` used to fill the right-hand side from the
/// canon suffix itself, so `suffix_algo != algo` compared the suffix to itself
/// and could not fire.
fn canon_verdict(suffix_algo: &str, algo: &AlgoSource) -> CanonVerdict {
    match algo.declared() {
        Some(declared) if declared == suffix_algo => CanonVerdict::Confirmed,
        Some(_) => CanonVerdict::Contradicted,
        None => CanonVerdict::Undeclared,
    }
}

/// Resolve the optimizer for a legacy `bpb_sample` call, keeping its origin.
///
/// The fallback to the canon suffix is retained -- legacy trainers whose canon
/// does not parse still reach `public.bpb_samples` unchanged -- but it is
/// returned as [`AlgoSource::CanonSuffix`], which no gate accepts as a
/// declaration.
fn resolve_algo_source(canon_name: &str) -> AlgoSource {
    if let Some(bound) = executed_optimizer() {
        return AlgoSource::from_declared(bound);
    }
    match parse_canon_name(canon_name) {
        Some((_format, suffix_algo, _hidden)) => AlgoSource::CanonSuffix(suffix_algo),
        None => AlgoSource::Undeclared,
    }
}

/// Insert a single row into `ssot.bpb_samples` with checkpoint telemetry.
///
/// Compatibility entry point. The optimizer is the process-bound executed one
/// when a run declared it via `bind_executed_optimizer`; otherwise the suffix
/// parsed out of `canon_name` is carried as a LABEL only (the pre-existing
/// fallback, kept so legacy callers are not silently dropped) and cannot
/// satisfy the canon-agreement gate. New code should call
/// [`bpb_sample_with_algo`] and pass the optimizer it actually ran.
///
/// Deliberately NOT `#[must_use]`: existing statement-position call sites stay
/// valid, and the truthful outcome is printed by this function itself, so a
/// caller that ignores the return value still cannot contradict it.
pub fn bpb_sample(
    canon_name: &str,
    seed: i32,
    step: i32,
    bpb: f32,
    ema_bpb: Option<f32>,
) -> LedgerWrite {
    note_evidence(
        EvidenceKind::BpbSample,
        bpb_sample_with_algo_inner(
            canon_name,
            seed,
            step,
            bpb,
            ema_bpb,
            &resolve_algo_source(canon_name),
        ),
    )
}

/// The legacy/smoke upsert, verbatim, as a `const`.
///
/// R4-3: every upsert in this module is a `const` so its INSERT column list and
/// its `DO UPDATE SET` list can be audited WITHOUT a database -- see
/// `do_update_set_covers_every_non_key_column`. The test parses the string that
/// actually ships, not a copy of it, so a column added to the INSERT list and
/// forgotten in the update list fails at `cargo test` instead of on a live
/// table months later.
///
/// Last-write-wins. The previous `bpb = LEAST(EXCLUDED.bpb, bpb_samples.bpb)`
/// made this table structurally incapable of recording a regression: a re-run
/// that got worse silently kept the older, better number while the log printed
/// the value SUBMITTED. With last-write-wins the stored value and the submitted
/// value are the same, so the log line is a statement about the row and not
/// about the argument. The same structural limit named on
/// [`CHECKPOINT_UPSERT_SQL`] applies here: replacing a row destroys the earlier
/// measurement, and only `ts` records that anything happened.
const BPB_SAMPLE_LEGACY_UPSERT_SQL: &str = "INSERT INTO public.bpb_samples \
     (canon_name, seed, step, bpb, ema_bpb, ts) \
     VALUES ($1, $2, $3, $4, $5, $6) \
     ON CONFLICT (canon_name, seed, step) DO UPDATE SET \
       bpb = EXCLUDED.bpb, \
       ema_bpb = EXCLUDED.ema_bpb, \
       ts = EXCLUDED.ts";

/// Insert a single row into `ssot.bpb_samples`, whitelisting the optimizer that
/// actually executed rather than a string parsed out of `canon_name`.
///
/// Schema (verified against phd-postgres-ssot 2026-05-14, matrix_runner aligned):
///   id BIGSERIAL, canon_name TEXT, format TEXT, algo TEXT, hidden INT,
///   seed BIGINT, step INT, bpb DOUBLE PRECISION, sha TEXT (nullable),
///   run_id TEXT (nullable), ts TIMESTAMPTZ
///
/// Derived columns:
///   format/hidden -- parsed from canon_name
///   algo          -- the executed optimizer argument, NOT the canon suffix
///   sha           -- GIT_SHA env (build-time), or empty
///   run_id        -- RAILWAY_DEPLOYMENT_ID env (runtime), or empty
///
/// `ema_bpb` is written to `public.bpb_samples.ema_bpb` on the legacy path.
/// `ssot.bpb_samples` has no such column, so on the canonical path the value is
/// printed in the "ok" line and explicitly labelled as not stored - it is never
/// silently discarded.
///
/// If canon_name does not match the IGLA schema (legacy scarab-*, smoke names),
/// we DO NOT fabricate -- fall back to writing into public.bpb_samples to keep
/// non-canonical callers green during the migration. Names carrying
/// [`SMOKE_CANON_PREFIX`] are pinned to that fallback unconditionally.
///
/// 2026-05-14: switched main path from public.bpb_samples -> ssot.bpb_samples
/// to unblock PASS-N monitors and leaderboard auditors that query ssot only.
/// Anchor: phi^2+phi^-2=3 - DOI 10.5281/zenodo.19227877
pub fn bpb_sample_with_algo(
    canon_name: &str,
    seed: i32,
    step: i32,
    bpb: f32,
    ema_bpb: Option<f32>,
    algo: &str,
) -> LedgerWrite {
    // One attempt, counted once, whatever the inner path decides. The tally
    // lives in this wrapper rather than in each of the seven inner branches so
    // a branch added later cannot forget it.
    note_evidence(
        EvidenceKind::BpbSample,
        bpb_sample_with_algo_inner(
            canon_name,
            seed,
            step,
            bpb,
            ema_bpb,
            // The caller states which optimizer it ran, so this IS a
            // declaration -- unlike the canon-suffix fallback in `bpb_sample`.
            &AlgoSource::from_declared(algo),
        ),
    )
}

fn bpb_sample_with_algo_inner(
    canon_name: &str,
    seed: i32,
    step: i32,
    bpb: f32,
    ema_bpb: Option<f32>,
    algo: &AlgoSource,
) -> LedgerWrite {
    // -- VALUE GUARD ----------------------------------------------------------
    // Before anything else, and regardless of whether a DSN exists: a value
    // that cannot be a measurement must never be published, printed as
    // published, or counted as a dropped write worth retrying.
    if let Some(reason) = reject_bpb(bpb) {
        eprintln!(
            "[ledger] REJECT bpb_sample: {canon_name} seed={seed} step={step}: {reason}"
        );
        note_rejected();
        return LedgerWrite::Rejected;
    }

    let is_smoke = canon_name.starts_with(SMOKE_CANON_PREFIX);
    let parsed = if is_smoke {
        None
    } else {
        parse_canon_name(canon_name)
    };

    // -- R5 GUARD -- WRITE-SIDE ALGO_WHITELIST ---------------------------------
    // Gates the EXECUTED optimizer, not the canon suffix.
    //
    // R4-3: this used to be conjoined with a "the canon name parsed" test,
    // which made it inert for exactly the names that need it most. A canon that
    // does NOT parse (the default `trios-train-rng{seed}`, legacy `scarab-*`)
    // skipped the check entirely, so `algo="soap"` under a non-conforming name
    // was written unchallenged. A name that carries no recipe is a reason to
    // check the executed optimizer harder, not a reason to skip it.
    //
    // An undeclared row is handled separately below: nothing was DECLARED,
    // which is a different fact from claiming an optimizer this crate cannot
    // run.
    let algo_label = algo.label();
    if let Some(declared) = algo.declared() {
        if !ALGO_WHITELIST.contains(&declared) {
            eprintln!(
                "[ledger] R5-REJECT write: canon_name={canon_name} executed algo={declared:?} \
                 not in {ALGO_WHITELIST:?}. Refusing silent-fallback write -- see \
                 trios#777 / migration 0006."
            );
            note_rejected();
            return LedgerWrite::Rejected;
        }
    } else {
        // Not a rejection. Nothing declared this row's optimizer: either the
        // caller passed "" (smoke_train, bpb_smoke, tjepa_train, hybrid_train
        // and the default train_loop canon today), or `bpb_sample` fell back to
        // the canon suffix, which is a label somebody typed and not evidence of
        // what ran. Rejecting here would silently stop every legacy trainer's
        // ledger; saying out loud that the whitelist could not be applied -- and
        // that the label below is unconfirmed -- is the honest half of the guard.
        eprintln!(
            "[ledger] WARN: canon_name={canon_name} publishes with NO declared \
             optimizer (bind_executed_optimizer never called): ALGO_WHITELIST \
             cannot be applied to this row, and algo={algo_label:?} is an \
             unconfirmed label carried from the canon name."
        );
    }

    let Some(conn) = db() else {
        let status = db_status();
        let outcome = skipped_write_outcome(status);
        eprintln!(
            "[ledger] no connection ({}) - skipping bpb_sample, recorded as {}",
            status.as_str(),
            outcome.as_str()
        );
        note_dropped();
        return outcome;
    };

    let ts_now = chrono::Utc::now();
    use sea_orm::Statement;

    if let Some((format, suffix_algo, hidden)) = parsed {
        // Only a DECLARED optimizer is admissible on the right-hand side. When
        // the value itself came out of the canon suffix this comparison used to
        // be the suffix against itself, which is why a `-muon` canon on an
        // AdamW run reached ssot.bpb_samples labelled muon while the checkpoint
        // sidecar honestly recorded adamw.
        match canon_verdict(&suffix_algo, algo) {
            CanonVerdict::Contradicted => {
                eprintln!(
                    "[ledger] REJECT bpb_sample: canon_name={canon_name} claims algo \
                     '{suffix_algo}' but '{algo_label}' executed"
                );
                note_rejected();
                return LedgerWrite::Rejected;
            }
            // Warned about above, by name, before any connection was opened.
            CanonVerdict::Undeclared => {}
            CanonVerdict::Confirmed => {}
        }

        // Canonical IGLA path -> ssot.bpb_samples (the SoT for leaderboards).
        // `sha` and `run_id` are nullable TEXT: an absent value is written as
        // SQL NULL, never as '' (see `env_nonempty`).
        let sha = env_nonempty("GIT_SHA");
        let run_id = env_nonempty("RAILWAY_DEPLOYMENT_ID");
        let stmt = Statement::from_sql_and_values(
            conn.get_database_backend(),
            "INSERT INTO ssot.bpb_samples \
             (canon_name, format, algo, hidden, seed, step, bpb, sha, run_id, ts) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) \
             ON CONFLICT DO NOTHING",
            [
                seed_val(canon_name),
                seed_val(&format),
                seed_val(algo_label),
                hidden.into(),
                (seed as i64).into(),
                step.into(),
                (bpb as f64).into(),
                opt_text_val(sha.as_deref()),
                opt_text_val(run_id.as_deref()),
                ts_now.into(),
            ],
        );
        return match rt().block_on(conn.execute(stmt)) {
            // ON CONFLICT DO NOTHING can accept the statement and insert zero
            // rows. Reporting "ok" for that is how a run with a duplicate
            // (canon, seed, step) key believed it had published.
            Ok(r) if r.rows_affected() == 0 => {
                eprintln!(
                    "[ledger] bpb_sample (ssot) inserted 0 rows: {canon_name} seed={seed} \
                     step={step} -- duplicate key, nothing published"
                );
                note_dropped();
                LedgerWrite::Failed
            }
            Ok(_) => {
                // `ema` is printed, not stored: ssot.bpb_samples has no such
                // column. Saying so is cheaper than a caller assuming it landed.
                eprintln!(
                    "[ledger] bpb_sample ok (ssot): {canon_name} format={format} algo={algo_label} algo_declared={} hidden={hidden} seed={seed} step={step} bpb={bpb:.4} ema={} (not stored: no column in ssot)",
                    algo.declared().is_some(),
                    fmt_bpb(ema_bpb)
                );
                note_landed();
                LedgerWrite::Written
            }
            Err(e) => {
                eprintln!("[ledger] bpb_sample (ssot) failed: {e}");
                note_dropped();
                LedgerWrite::Failed
            }
        };
    }

    // Legacy/smoke fallback -> public.bpb_samples (Wave 29 PR-A idempotent upsert).
    if is_smoke {
        eprintln!(
            "[ledger] canon_name '{canon_name}' carries the reserved {SMOKE_CANON_PREFIX} \
             prefix: operator-supplied value, never routed to ssot.bpb_samples"
        );
    } else {
        eprintln!(
            "[ledger] canon_name '{canon_name}' not IGLA-shaped; falling back to public.bpb_samples"
        );
    }
    // `ema_bpb` was bound as a literal NULL, which made the column always NULL
    // and the CASE WHEN below dead code. It is now the caller's value, filtered
    // through the same guard as `bpb` so a sentinel EMA cannot ride in.
    let ema_val: Option<f64> = guarded_bpb("bpb_sample ema", canon_name, ema_bpb).map(|v| v as f64);

    // Last-write-wins; the rationale now lives on BPB_SAMPLE_LEGACY_UPSERT_SQL,
    // next to the statement it explains.
    let stmt = Statement::from_sql_and_values(
        conn.get_database_backend(),
        BPB_SAMPLE_LEGACY_UPSERT_SQL,
        [
            seed_val(canon_name),
            (seed as i64).into(),
            (step as i64).into(),
            (bpb as f64).into(),
            ema_val.into(),
            ts_now.into(),
        ],
    );
    match rt().block_on(conn.execute(stmt)) {
        Ok(r) if r.rows_affected() == 0 => {
            eprintln!(
                "[ledger] bpb_sample (public/legacy) inserted 0 rows: {canon_name} \
                 seed={seed} step={step} -- nothing published"
            );
            note_dropped();
            LedgerWrite::Failed
        }
        Ok(_) => {
            eprintln!(
                "[ledger] bpb_sample ok (public/legacy): {canon_name} seed={seed} \
                 step={step} stored bpb={bpb:.4} ema_bpb={}",
                fmt_bpb(ema_val.map(|v| v as f32))
            );
            note_landed();
            LedgerWrite::Written
        }
        Err(e) => {
            eprintln!("[ledger] bpb_sample (public/legacy) failed: {e}");
            note_dropped();
            LedgerWrite::Failed
        }
    }
}

/// Outcome of a ledger write, so the caller can record honestly instead of
/// assuming success. R5 still holds: this never panics and never propagates a
/// DB error - it only reports what happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerWrite {
    Written,
    /// No DSN in the environment at all: nothing was ever configured, so
    /// nothing was expected to be recorded.
    SkippedNoDsn,
    /// A DSN WAS configured and this run declined to use it, because
    /// [`LEDGER_WRITE_OPT_IN`] was not set to `1`.
    ///
    /// 2026-08-03: this variant exists because the sidecar was stating a cause
    /// it could not stand behind. `active_dsn` deliberately makes an
    /// un-opted-in run look, to every caller and to the exit status, exactly
    /// like a run with no DSN -- and that equivalence is right for BEHAVIOUR.
    /// It is wrong for the RECORD. A run launched with `DATABASE_URL` set and
    /// no opt-in wrote `ledger: "skipped-no-dsn"` into the schema-8 sidecar,
    /// which is the evidence document, and reads there as "nothing outside
    /// this checkout supplied a number because nothing was configured". The
    /// true statement is "a ledger was configured and this run chose not to
    /// touch it". The refusal was already announced on stderr by
    /// [`announce_missing_opt_in`], and stderr is not an artifact: neither
    /// `ckpt_replay` nor `interop/triosckp_reader.py` can see it. This is the
    /// same defect as the `dropped=3 when nothing was attempted` line one
    /// function up -- a counter or a field truthful about the code path and
    /// false about the world -- and it is the defect class this crate is
    /// supposed to catch, so it may not live in its own record.
    ///
    /// Only the CAUSE changes. `DbStatus` gains no state (the opt-in question
    /// is answered before connect, so no connection outcome describes it),
    /// the counters keep their meaning, and `ledger_exit_code` still returns 0
    /// -- refusing to write is not a failure of the run.
    SkippedNotOptedIn,
    Failed,
    /// The value or its provenance was refused. Not a database problem, and
    /// not retryable: nothing was lost, something wrong was stopped.
    Rejected,
}

impl LedgerWrite {
    /// The literal recorded in `CheckpointRecord::ledger`.
    pub fn as_str(self) -> &'static str {
        match self {
            LedgerWrite::Written => "written",
            LedgerWrite::SkippedNoDsn => "skipped-no-dsn",
            LedgerWrite::SkippedNotOptedIn => "skipped-not-opted-in",
            LedgerWrite::Failed => "failed",
            LedgerWrite::Rejected => "rejected",
        }
    }
}

/// The `ssot.checkpoints` upsert, verbatim, as a `const`.
///
/// R4-3 -- WHAT THIS STATEMENT DOES, STATED TRUTHFULLY.
///
/// The `DO UPDATE SET` list used to name only `path`, `sha256`, `bytes`, `bpb`
/// and `ts`, while the INSERT carried `algo`, `hidden`, `format_version`,
/// `data_synthetic`, `sha` and `run_id` as well. That is last-write-wins for
/// five columns and first-write-wins for six: a second run under the same
/// `(canon_name, seed, step)` key kept the FIRST run's architecture and
/// provenance while taking the SECOND run's artifact hash and metric. The
/// surviving row then described a run that never happened. Reproduced twice
/// against a live table: a row reading `hidden=64 | bytes=524592` (524592 is
/// the hidden=128 artifact size; hidden=64 is 442672), and a row reading
/// `algo='muon'` bound to muon-cwd's sha256 and muon-cwd's bpb. This is
/// reachable with the DEFAULT run name, which carries no recipe at all -
/// `ssot.checkpoints` already holds six live `trios-train-rng47` rows spanning
/// hidden 384, 64 and 32.
///
/// Every non-key column is now replaced from `EXCLUDED`: path, sha256, bytes,
/// algo, hidden, format_version, data_synthetic, bpb, sha, run_id, ts. The
/// surviving row therefore describes exactly one run - the last one - instead
/// of a composite of two. `data_synthetic` in particular can no longer stay
/// `false` underneath a replaced hash, which was the worst case: a synthetic
/// corpus wearing an earlier real run's honesty flag.
///
/// KNOWN STRUCTURAL LIMIT - this register is NOT append-only, and no widening
/// of `DO UPDATE SET` can make it so. A measurement register SHOULD keep every
/// row and mark supersession (`superseded_by BIGINT`, `retracted BOOLEAN`,
/// `retracted_reason TEXT`), because the earlier attempt is evidence: a
/// reproducibility claim that cannot show what the first run measured is not a
/// reproducibility claim. Last-write-wins destroys that history silently - the
/// only trace left is `ts` moving. Closing it needs a schema migration
/// (`migration/`) and a new insert path, not a wider update list. Named here as
/// a known limit rather than papered over.
///
/// FOLLOW-UP, not done here because it needs files outside this module: `sha`
/// is a bare commit string, while the sidecar written next to the same artifact
/// carries the full `checkpoint::GIT_PROVENANCE_VERIFIED` ("verified-local") /
/// `GIT_PROVENANCE_ASSERTED` ("asserted-by-environment") taxonomy plus
/// `git_dirty`. The ledger therefore cannot distinguish "git was queried" from
/// "GIT_SHA was exported and taken on trust", nor a clean tree from a dirty
/// one. Threading them in means two new columns in `migration/` and a wider
/// signature at the `src/train_loop.rs` call site; neither file is this one.
const CHECKPOINT_UPSERT_SQL: &str = "INSERT INTO ssot.checkpoints \
     (canon_name, seed, step, path, sha256, bytes, algo, hidden, format_version, \
      data_synthetic, bpb, sha, run_id, ts) \
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) \
     ON CONFLICT (canon_name, seed, step) DO UPDATE SET \
       algo = EXCLUDED.algo, \
       hidden = EXCLUDED.hidden, \
       format_version = EXCLUDED.format_version, \
       data_synthetic = EXCLUDED.data_synthetic, \
       path = EXCLUDED.path, \
       sha256 = EXCLUDED.sha256, \
       bytes = EXCLUDED.bytes, \
       bpb = EXCLUDED.bpb, \
       sha = EXCLUDED.sha, \
       run_id = EXCLUDED.run_id, \
       ts = EXCLUDED.ts";

/// Bind a checkpoint artifact to the run that produced it, in
/// `ssot.checkpoints`.
///
/// EPIC-446. Deliberately NOT gated by `parse_canon_name` or the
/// `ALGO_WHITELIST`: checkpoints go to `ssot.checkpoints` for EVERY canon
/// shape, including the default non-IGLA `trios-train-rng{seed}`, because the
/// join key is the `(canon_name, seed, step)` triple and works cross-schema
/// against both `ssot.bpb_samples` and `public.bpb_samples`. `algo` is the
/// optimizer that actually executed, which closes the hole where the whitelist
/// gates a string parsed out of `canon_name` (so `--optimizer=adamw` under
/// `TRIOS_CANON_NAME=IGLA-...-muon` records a lie).
///
/// That claim was true on INSERT and false on UPDATE until R4-3: the conflict
/// path did not replace `algo`, so the executed optimizer of the second run was
/// discarded and the first run's label kept. See [`CHECKPOINT_UPSERT_SQL`] for
/// exactly which columns are replaced, and for the append-only limit that this
/// upsert does not fix.
#[allow(clippy::too_many_arguments)]
pub fn checkpoint_record(
    canon_name: &str,
    seed: i32,
    step: i64,
    path: &str,
    sha256: &str,
    bytes: i64,
    algo: &str,
    hidden: i32,
    format_version: i32,
    data_synthetic: bool,
    bpb: Option<f64>,
) -> LedgerWrite {
    // See `bpb_sample_with_algo`: the per-kind tally is taken here, once.
    note_evidence(
        EvidenceKind::CheckpointRecord,
        checkpoint_record_inner(
            canon_name,
            seed,
            step,
            path,
            sha256,
            bytes,
            algo,
            hidden,
            format_version,
            data_synthetic,
            bpb,
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn checkpoint_record_inner(
    canon_name: &str,
    seed: i32,
    step: i64,
    path: &str,
    sha256: &str,
    bytes: i64,
    algo: &str,
    hidden: i32,
    format_version: i32,
    data_synthetic: bool,
    bpb: Option<f64>,
) -> LedgerWrite {
    let Some(conn) = db() else {
        let status = db_status();
        let outcome = skipped_write_outcome(status);
        eprintln!(
            "[ledger] no connection ({}) - skipping checkpoint_record, recorded as {}",
            status.as_str(),
            outcome.as_str()
        );
        note_dropped();
        return outcome;
    };

    use sea_orm::Statement;
    // Absent provenance is SQL NULL, not '': 16 of 17 live rows carried an
    // empty string, so `WHERE sha IS NULL` found nothing and the gap was
    // invisible to exactly the query written to find it.
    let sha = env_nonempty("GIT_SHA");
    let run_id = env_nonempty("RAILWAY_DEPLOYMENT_ID");
    let ts_now = chrono::Utc::now();

    let stmt = Statement::from_sql_and_values(
        conn.get_database_backend(),
        CHECKPOINT_UPSERT_SQL,
        [
            seed_val(canon_name),
            (seed as i64).into(),
            step.into(),
            seed_val(path),
            seed_val(sha256),
            bytes.into(),
            seed_val(algo),
            (hidden as i64).into(),
            (format_version as i64).into(),
            data_synthetic.into(),
            bpb.into(),
            opt_text_val(sha.as_deref()),
            opt_text_val(run_id.as_deref()),
            ts_now.into(),
        ],
    );
    match rt().block_on(conn.execute(stmt)) {
        Ok(r) if r.rows_affected() == 0 => {
            eprintln!(
                "[ledger] checkpoint_record inserted 0 rows: {canon_name} seed={seed} \
                 step={step} -- nothing published"
            );
            note_dropped();
            LedgerWrite::Failed
        }
        Ok(_) => {
            eprintln!(
                "[ledger] checkpoint_record ok: {canon_name} seed={seed} step={step} \
                 sha256={sha256} bytes={bytes}"
            );
            note_landed();
            LedgerWrite::Written
        }
        Err(e) => {
            eprintln!("[ledger] checkpoint_record failed: {e}");
            note_dropped();
            LedgerWrite::Failed
        }
    }
}

/// Helper: convert a &str into a sea_orm Value.
#[inline]
fn seed_val(s: &str) -> sea_orm::Value {
    sea_orm::Value::String(Some(Box::new(s.to_string())))
}

/// Helper: an optional text value, where absence is SQL NULL.
///
/// R4-3: `sha` and `run_id` were bound through [`seed_val`] after an
/// `unwrap_or_default()`, so a run built outside CI wrote `sha = ''`. Empty
/// string is a value: it satisfies `NOT NULL`, it joins, it groups, and
/// `WHERE sha IS NULL` - the query an auditor writes to find rows with no
/// provenance - returns nothing. 16 of 17 live `ssot.checkpoints` rows are in
/// that state. NULL says "not recorded"; '' says "recorded as nothing".
#[inline]
fn opt_text_val(s: Option<&str>) -> sea_orm::Value {
    sea_orm::Value::String(s.map(|v| Box::new(v.to_string())))
}

/// An environment variable that is set AND non-empty, else `None`.
///
/// `GIT_SHA=` exported empty by a CI step that failed to resolve the commit is
/// the same absence as `GIT_SHA` unset, and must reach the column as the same
/// NULL.
fn env_nonempty(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => Some(v.trim().to_string()),
        _ => None,
    }
}

/// Apply idempotent DDL: ensure `bpb_latest` column exists on `igla_race_trials`.
///
/// Old raw-SQL call:
///   ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS bpb_latest DOUBLE PRECISION
/// New: delegated to the SeaORM migration (Migrator::up runs at startup).
/// This stub is kept for callers that invoke it directly.
pub fn ensure_schema() {
    let Some(conn) = db() else {
        eprintln!(
            "[ledger] no connection ({}) - skipping ensure_schema",
            db_status().as_str()
        );
        note_dropped();
        return;
    };
    let res = rt().block_on(conn.execute_unprepared(
        "ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS bpb_latest DOUBLE PRECISION",
    ));
    // Deliberately asymmetric: a failed DDL is a dropped write, a successful
    // one is NOT a landed row. Counting it as landed would let a run whose
    // every sample was lost pass `ledger_exit_code` on the strength of an
    // ALTER TABLE.
    match res {
        Ok(_) => eprintln!("[ledger] ensure_schema ok"),
        Err(e) => {
            eprintln!("[ledger] ensure_schema failed: {e}");
            note_dropped();
        }
    }
}

/// Default checkpoint interval honoured by trainers writing to bpb_samples.
pub fn checkpoint_interval() -> usize {
    std::env::var("TRIOS_CHECKPOINT_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_postgres::types::{ToSql, Type};

    /// Serialise the tests that read `LANDED_WRITES` / `DROPPED_WRITES`.
    ///
    /// Those counters are process-wide, so two tests calling ledger functions
    /// in parallel observe each other's increments and an exact-delta
    /// assertion becomes a coin flip. Taking this lock keeps the deltas exact
    /// instead of loosening the assertions.
    static LEDGER_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Take the lock AND zero the per-kind evidence counters.
    ///
    /// The aggregate counters are left alone: existing tests assert deltas on
    /// them and a reset would change what they measure. The per-kind counters
    /// are different - `ledger_exit_code` reads them as an absolute state
    /// ("attempted at all, landed never"), so one earlier test that called
    /// `bpb_sample` without a DSN would otherwise make every later
    /// DSN-configured assertion in this file fire on the per-kind branch and
    /// stop exercising the branch it was written for.
    fn counter_guard() -> std::sync::MutexGuard<'static, ()> {
        let g = LEDGER_COUNTER_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for c in [&BPB_ATTEMPTED, &BPB_LANDED, &CKPT_ATTEMPTED, &CKPT_LANDED] {
            c.store(0, Ordering::Relaxed);
        }
        g
    }

    /// Clear every DSN alias AND the write opt-in so `db()` reports `NoDsn`
    /// deterministically.
    fn clear_dsn_env() {
        std::env::remove_var("DATABASE_URL");
        std::env::remove_var("TRIOS_NEON_DSN");
        std::env::remove_var("NEON_DATABASE_URL");
        std::env::remove_var("TRIOS_DATABASE_URL");
        std::env::remove_var(LEDGER_WRITE_OPT_IN);
    }

    /// Put this process in the "recorded run" state: a DSN under `alias` AND
    /// the explicit write opt-in.
    ///
    /// Both halves are required, so a test that wants the DSN-configured
    /// branch has to ask for the write the same way an operator does. Tests
    /// that set only the alias are testing the refusal, not the branch.
    fn set_dsn_env(alias: &str) {
        std::env::set_var(alias, "postgres://unused/never-connected");
        std::env::set_var(LEDGER_WRITE_OPT_IN, "1");
    }

    #[test]
    fn no_dsn_is_safe() {
        let _g = counter_guard();
        clear_dsn_env();
        trial_start("00000000-0000-0000-0000-000000000000", "{}", "TEST", "main");
        heartbeat("00000000-0000-0000-0000-000000000000", "TEST", Some(2.5), 1);
        trial_complete("00000000-0000-0000-0000-000000000000", Some(2.5));
    }

    // -- R3-3: a write that did not happen must be visible to ledger_exit_code --

    /// `trial_start`, `heartbeat`, `trial_complete` and `ensure_schema` all
    /// returned early on a missing connection without touching
    /// `DROPPED_WRITES`, so `ledger_exit_code()` could not fire for them and a
    /// run with no ledger at all exited 0.
    ///
    /// Asserted as a strict increase, not an exact delta: the counter is
    /// process-wide and other tests in this binary run concurrently.
    #[test]
    fn every_ledger_path_counts_a_dropped_write_without_a_connection() {
        let _g = counter_guard();
        clear_dsn_env();
        let id = "00000000-0000-0000-0000-000000000000";

        let before = dropped_writes();
        trial_start(id, "{}", "TEST", "main");
        assert!(
            dropped_writes() > before,
            "trial_start dropped a write silently"
        );

        let before = dropped_writes();
        heartbeat(id, "TEST", Some(2.5), 1);
        assert!(
            dropped_writes() > before,
            "heartbeat dropped a write silently"
        );

        let before = dropped_writes();
        trial_complete(id, Some(2.5));
        assert!(
            dropped_writes() > before,
            "trial_complete dropped a write silently"
        );

        let before = dropped_writes();
        ensure_schema();
        assert!(
            dropped_writes() > before,
            "ensure_schema dropped a write silently"
        );
    }

    /// A malformed UUID is an early return in all three trial paths.
    #[test]
    fn an_invalid_trial_uuid_counts_a_dropped_write() {
        let _g = counter_guard();
        clear_dsn_env();
        let before = dropped_writes();
        trial_start("not-a-uuid", "{}", "TEST", "main");
        heartbeat("not-a-uuid", "TEST", Some(2.5), 1);
        trial_complete("not-a-uuid", Some(2.5));
        assert!(dropped_writes() > before);
    }

    /// Malformed config JSON used to be replaced by `{}` and reported as
    /// "trial_start ok": a trial row describing no recipe at all.
    #[test]
    fn malformed_config_json_counts_a_dropped_write() {
        let _g = counter_guard();
        clear_dsn_env();
        let before = dropped_writes();
        trial_start(
            "00000000-0000-0000-0000-000000000000",
            "{not json",
            "TEST",
            "main",
        );
        assert!(dropped_writes() > before);
    }

    /// `rows_affected == 0` is what an `UPDATE ... WHERE trial_id = $1` returns
    /// when `trial_start` was dropped. Printing "ok: ... rows=0" for it is how
    /// a trial with no row believed it was recorded. Verifiable without a
    /// database because the decision is a pure function.
    #[test]
    fn zero_rows_affected_is_a_failure_not_an_ok() {
        assert_eq!(update_outcome(0), LedgerWrite::Failed);
        assert_eq!(update_outcome(0).as_str(), "failed");
        assert_eq!(update_outcome(1), LedgerWrite::Written);
        assert_eq!(update_outcome(7), LedgerWrite::Written);
    }

    /// The heartbeat/trial_complete BPB now goes through the same guard as
    /// `bpb_sample`, so `f32::MAX` (the old "could not measure" sentinel, and
    /// the initial value of `best_val_bpb` in tjepa_train) cannot reach a
    /// column.
    #[test]
    fn guarded_bpb_filters_sentinels_and_absence() {
        assert_eq!(guarded_bpb("t", "id", None), None);
        assert_eq!(guarded_bpb("t", "id", Some(f32::MAX)), None);
        assert_eq!(guarded_bpb("t", "id", Some(f32::NAN)), None);
        assert_eq!(guarded_bpb("t", "id", Some(0.0)), None);
        assert_eq!(guarded_bpb("t", "id", Some(2.6141)), Some(2.6141));
    }

    /// An absent measurement is never rendered as a number.
    #[test]
    fn fmt_bpb_never_invents_a_number() {
        assert_eq!(fmt_bpb(None), "unmeasured");
        assert_eq!(fmt_bpb(Some(2.6141)), "2.6141");
    }

    #[test]
    fn strip_channel_binding_removes_only_that_param() {
        let in_ = "postgresql://u:p@h/db?sslmode=require&channel_binding=require";
        assert_eq!(
            strip_channel_binding(in_),
            "postgresql://u:p@h/db?sslmode=require"
        );
    }

    #[test]
    fn strip_channel_binding_when_only_param() {
        let in_ = "postgresql://u:p@h/db?channel_binding=require";
        assert_eq!(strip_channel_binding(in_), "postgresql://u:p@h/db");
    }

    #[test]
    fn strip_channel_binding_passthrough_when_absent() {
        let in_ = "postgresql://u:p@h/db?sslmode=require";
        assert_eq!(strip_channel_binding(in_), in_);
    }

    #[test]
    fn strip_channel_binding_passthrough_no_query_string() {
        let in_ = "postgresql://u:p@h/db";
        assert_eq!(strip_channel_binding(in_), in_);
    }

    #[test]
    fn parse_canon_name_short_wave_matrix() {
        let c = "IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng1597-adamw";
        let (fmt, algo, hidden) = parse_canon_name(c).expect("parses");
        assert_eq!(fmt, "gf16");
        assert_eq!(algo, "adamw");
        assert_eq!(hidden, 128);
    }

    #[test]
    fn parse_canon_name_scarab_lane() {
        let c = "IGLA-SCARAB-ADAMW-binary16-h384-LR0001-rng123-adamw";
        let (fmt, algo, hidden) = parse_canon_name(c).expect("parses");
        assert_eq!(fmt, "binary16");
        assert_eq!(algo, "adamw");
        assert_eq!(hidden, 384);
    }

    #[test]
    fn parse_canon_name_muon_cwd_dashed_algo() {
        // muon-cwd suffix -- algo = everything AFTER the rng token,
        // joined with '-' to preserve multi-token algos.
        let c = "IGLA-SHORT-WAVE-MATRIX-fp16-h128-LR0.0001-rng47-muon-cwd";
        let (fmt, algo, hidden) = parse_canon_name(c).expect("parses");
        assert_eq!(fmt, "fp16");
        assert_eq!(algo, "muon-cwd");
        assert_eq!(hidden, 128);
    }

    #[test]
    fn parse_canon_name_rejects_legacy_scarab() {
        // Legacy `scarab-*` names (pre-2026-05-12) should NOT parse,
        // so they go down the public.bpb_samples fallback path.
        assert!(parse_canon_name("scarab-adamw-rng123").is_none());
        assert!(parse_canon_name("random-name").is_none());
        assert!(parse_canon_name("").is_none());
    }

    #[test]
    fn parse_canon_name_rejects_missing_h_token() {
        // No `h{N}` token -> hidden cannot be derived -> reject.
        assert!(parse_canon_name("IGLA-FAKE-gf16-LR0.0001-rng1597-adamw").is_none());
    }

    /// Regression test: seed and step parameters for bpb_sample must be bound
    /// as INT8 (i64) to match the BIGINT columns in public.bpb_samples (#114).
    #[test]
    fn bpb_sample_uses_i64_bind() {
        let seed: i32 = 43;
        let step: i32 = 200;

        let seed_i64: i64 = seed as i64;
        let step_i64: i64 = step as i64;

        // i64 must be accepted by Postgres INT8 (BIGINT).
        assert!(
            <i64 as ToSql>::accepts(&Type::INT8),
            "i64 must be accepted by Postgres INT8"
        );
        // i32 must NOT be accepted by INT8 -- documents the original bug.
        assert!(
            !<i32 as ToSql>::accepts(&Type::INT8),
            "i32 must NOT be accepted by Postgres INT8"
        );

        let _: i64 = seed_i64;
        let _: i64 = step_i64;
        assert_eq!(seed_i64, 43i64);
        assert_eq!(step_i64, 200i64);
    }

    /// B-22 write-side ALGO_WHITELIST guard -- invariants only (we don't have a
    /// live DSN here so we can't exercise bpb_sample directly; this documents
    /// the canonical whitelist so a future widening can't happen by accident).
    #[test]
    fn write_side_algo_whitelist_is_canonical() {
        // Canonical: only these three are real-trainer-backed. Asserted against
        // the shipping constant, not a copy of it, so widening the real list
        // fails here instead of passing quietly.
        const CANONICAL: &[&str] = &["adamw", "muon", "muon-cwd"];
        assert_eq!(
            ALGO_WHITELIST, CANONICAL,
            "the shipping whitelist drifted from the canonical set"
        );
        assert_eq!(CANONICAL.len(), 3, "whitelist must be exactly 3 entries");
        for &name in CANONICAL {
            let canon = format!("IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng47-{}", name);
            let (_fmt, algo, _hidden) =
                parse_canon_name(&canon).expect("canonical algo must parse");
            assert_eq!(
                algo, name,
                "parse_canon_name must round-trip canonical algos"
            );
        }
        for &fake in &[
            "soap",
            "lamb",
            "prodigy",
            "lion",
            "tiger",
            "adafactor",
            "sgdm",
        ] {
            assert!(
                !CANONICAL.contains(&fake),
                "fake algo leaked into whitelist: {fake}"
            );
        }
    }

    // -- B-no-unmeasured-number-reaches-the-ledger ----------------------------

    /// S2: `f32::MAX` was the "could not measure" sentinel returned by several
    /// evaluators, and every guard downstream tested `is_finite()` -- which it
    /// passes. A published row read
    /// `bpb=340282346638528859811704183484516925440.0000`.
    #[test]
    fn reject_bpb_refuses_the_f32_max_sentinel() {
        assert!(
            f32::MAX.is_finite(),
            "documents WHY is_finite() was not a guard"
        );
        assert!(reject_bpb(f32::MAX).is_some());
        assert!(reject_bpb(f32::INFINITY).is_some());
        assert!(reject_bpb(f32::NAN).is_some());
        assert!(reject_bpb(BPB_SENTINEL_CEILING).is_some());
    }

    /// A near-zero BPB is a degenerate eval corpus, never a perfect model.
    /// Measured: even a 100% verbatim train/val overlap only reaches ~2.6 on
    /// this architecture.
    #[test]
    fn reject_bpb_refuses_below_the_jepa_floor() {
        assert!(reject_bpb(0.0).is_some());
        assert!(reject_bpb(crate::race::victory::JEPA_PROXY_BPB_FLOOR as f32).is_some());
    }

    /// The two retracted figures that lie BELOW the publication floor must be
    /// refused by the BUILD, not by prose in a document nobody re-reads.
    ///
    /// 1.5492 is the "honest Gate-2 pass" of the PR's own
    /// `LEAK_INVESTIGATION.md`. It is better than the crate's own
    /// `BPB_CHAMPION` = 2.5193, and at the commit that produced it
    /// `train_loop.rs` contained no `bpb_sample` call at all (the wiring landed
    /// two days later), so the rows came from an out-of-repo stdout parser.
    ///
    /// 1.038 is the same shape one order lower: below `IGLA_TARGET_BPB` = 1.5
    /// on an architecture whose 100%-leak ceiling is ~2.71. Both used to clear
    /// every automated gate because the gate was the 0.1 proxy detector.
    ///
    /// RENAMED from `reject_bpb_refuses_the_retracted_champions`, which
    /// promised more than it delivered: `RETRACTION.md:59-61` grades 2.2393,
    /// 2.2111 and 2.1919 uncitable too, and this gate admits all three. See
    /// `the_publication_floor_does_not_catch_the_provenance_retractions`.
    #[test]
    fn reject_bpb_refuses_the_figures_below_the_publication_floor() {
        assert!(
            reject_bpb(1.5492).is_some(),
            "1.5492 is the retracted LEAK_INVESTIGATION.md Gate-2 pass"
        );
        assert!(
            reject_bpb(1.038).is_some(),
            "1.038 is below the 100%-leak ceiling of this architecture"
        );
        assert!(reject_bpb(crate::invariants::PUBLISHED_BPB_FLOOR).is_some());
    }

    /// The honest boundary of the value guard, stated instead of implied.
    ///
    /// `reject_bpb` is a plausibility test on a NUMBER. The 2.2393 / 2.2111 /
    /// 2.1919 family is uncitable for reasons a number cannot carry - no
    /// artifact, no corpus digest, seed 43 (forbidden under Canon #93), written
    /// outside the emit path - and all three sit comfortably above the floor.
    /// This test exists so nobody reads the gate above as covering them.
    ///
    /// Deliberately NOT fixed by adding those literals to a denylist here: the
    /// refusal is about PROVENANCE, not plausibility, so a future honest run
    /// that measures 2.2393 must be publishable. Enforcing provenance means
    /// requiring the artifact - the checkpoint sidecar and corpus digests that
    /// `checkpoint::save` now writes - not blacklisting three decimals.
    #[test]
    fn the_publication_floor_does_not_catch_the_provenance_retractions() {
        for v in [2.2393f32, 2.2111, 2.1919] {
            assert!(
                reject_bpb(v).is_none(),
                "documents that the value guard admits {v}: RETRACTION.md refuses \
                 it on provenance, and this gate cannot see provenance"
            );
        }
    }

    /// The honest calibration band must pass untouched: 7.00 at init, ~3.31 at
    /// step 1000, and the three artifact-backed step-12000 readings (2.6348
    /// headline, 2.6169 and 2.6141 archived sidecars).
    ///
    /// 0.5 used to sit in this list, certified as "a plausible measurement".
    /// It is unreachable on this architecture and outside the band the
    /// docstring names, so it made a green test named for honesty into the
    /// thing blocking the floor from ever being tightened.
    ///
    /// 2.5193 sat here too. `RETRACTION.md:58` grades it "No" - it pre-dates
    /// `checkpoint::save` doing any work, so no artifact exists behind it - and
    /// a test named for real measurements must not certify it as one. That the
    /// floor still admits it is asserted where it belongs, in
    /// `invariants::published_floor_sits_between_the_detector_and_the_champion`.
    #[test]
    fn reject_bpb_admits_real_measurements() {
        for v in [7.00f32, 3.31, 2.6348, 2.6169, 2.6141] {
            assert!(
                reject_bpb(v).is_none(),
                "rejected a plausible measurement: {v}"
            );
        }
    }

    /// S2 end to end: the sentinel is refused before any connection is looked
    /// at, so it can neither be written nor counted as a lost write.
    #[test]
    fn bpb_sample_rejects_the_sentinel_without_touching_the_db() {
        let _g = counter_guard();
        let before = dropped_writes();
        let out = bpb_sample("IGLA-TEST-gf16-h128-LR0.001-rng47-adamw", 47, 0, f32::MAX, None);
        assert_eq!(out, LedgerWrite::Rejected);
        assert_eq!(out.as_str(), "rejected");
        assert_eq!(
            dropped_writes(),
            before,
            "a rejected value is not a dropped write: nothing was lost"
        );
    }

    /// S9: no spelling of an operator-supplied canon reaches the IGLA
    /// namespace, and the prefix does not stack on repeated application.
    #[test]
    fn smoke_canon_name_pins_the_reserved_prefix() {
        assert_eq!(smoke_canon_name("bpb_smoke_test"), "SMOKE-bpb_smoke_test");
        assert_eq!(
            smoke_canon_name("IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng1597-adamw"),
            "SMOKE-IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng1597-adamw"
        );
        // Idempotent.
        let once = smoke_canon_name("x");
        assert_eq!(smoke_canon_name(&once), once);
        assert_eq!(smoke_canon_name(""), "SMOKE-UNNAMED");
        assert_eq!(smoke_canon_name("SMOKE-"), "SMOKE-UNNAMED");
    }

    /// A smoke canon can never satisfy `parse_canon_name`, which is the gate
    /// into `ssot.bpb_samples`.
    #[test]
    fn smoke_canon_never_routes_to_ssot() {
        let canon = smoke_canon_name("IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng1597-adamw");
        assert!(canon.starts_with(SMOKE_CANON_PREFIX));
        assert!(
            parse_canon_name(&canon).is_none(),
            "a SMOKE- canon must not parse into the ssot path"
        );
    }

    /// S10: the whitelist gated a string parsed out of the canon name, so a
    /// `--optimizer=adamw` run under a `-muon` canon was recorded as muon.
    /// The disagreement is now an error at process start.
    ///
    /// Only the error paths are exercised: the success path sets a process-wide
    /// `OnceLock` that other tests in this binary would inherit.
    #[test]
    fn bind_executed_optimizer_rejects_a_lying_canon_suffix() {
        let err = bind_executed_optimizer(
            "IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng47-muon",
            "adamw",
        )
        .expect_err("a canon claiming muon must not accept an adamw run");
        assert!(err.contains("muon") && err.contains("adamw"), "{err}");

        let err = bind_executed_optimizer("IGLA-X-gf16-h128-LR0.1-rng47-adamw", "soap")
            .expect_err("a non-whitelisted optimizer must not start a run");
        assert!(err.contains("soap"), "{err}");
    }

    /// The anti-mislabelling gate compared the canon suffix to a value that had
    /// been READ OUT OF that same suffix, so it could never fire.
    ///
    /// Verified live before the fix: `TRIOS_CANON_NAME=...-muon` on an AdamW run
    /// printed neither the "NO declared optimizer" WARN nor a REJECT, so with a
    /// DSN present the row reached `ssot.bpb_samples` labelled muon while the
    /// checkpoint sidecar recorded adamw. The artifact and the leaderboard
    /// contradicted each other and nothing said so.
    ///
    /// The fallback is kept -- legacy callers still publish -- but it arrives as
    /// `CanonSuffix`, which is not a declaration and therefore cannot confirm
    /// anything.
    #[test]
    fn a_canon_suffix_can_never_confirm_itself() {
        let canon = "IGLA-X-gf16-h16-LR0.003-rng47-muon";
        let (_format, suffix, _hidden) = parse_canon_name(canon).expect("canon parses");
        assert_eq!(suffix, "muon");

        // Nothing bound: the suffix is all there is, and it is a label.
        let resolved = AlgoSource::CanonSuffix(suffix.clone());
        assert_eq!(
            resolved.declared(),
            None,
            "a canon suffix is never a declaration"
        );
        assert_eq!(
            canon_verdict(&suffix, &resolved),
            CanonVerdict::Undeclared,
            "the suffix must not be able to confirm itself"
        );
        assert_eq!(
            canon_verdict(&suffix, &AlgoSource::Undeclared),
            CanonVerdict::Undeclared
        );

        // A real declaration still both confirms and contradicts, so the fix
        // disarmed nothing.
        assert_eq!(
            canon_verdict(&suffix, &AlgoSource::Declared("muon".to_string())),
            CanonVerdict::Confirmed
        );
        assert_eq!(
            canon_verdict(&suffix, &AlgoSource::Declared("adamw".to_string())),
            CanonVerdict::Contradicted,
            "an adamw run under a -muon canon must be refused"
        );
    }

    /// The legacy `bpb_sample` entry point resolves the suffix as a LABEL.
    ///
    /// Guarded on nothing having been bound in this test binary: the successful
    /// path of `bind_executed_optimizer` sets a process-wide `OnceLock`, which
    /// is why no test in this module takes it.
    #[test]
    fn resolve_algo_source_labels_the_fallback_as_a_suffix() {
        assert!(
            executed_optimizer().is_none(),
            "no test in this binary may bind the process-wide optimizer"
        );
        assert_eq!(
            resolve_algo_source("IGLA-X-gf16-h16-LR0.003-rng47-muon"),
            AlgoSource::CanonSuffix("muon".to_string())
        );
        assert_eq!(
            resolve_algo_source("trios-train-rng47"),
            AlgoSource::Undeclared,
            "a name that carries no suffix declares nothing"
        );
    }

    // -- R4-3: no ledger row may describe a run that never happened -----------

    /// Column names between the first `(` and the following `)`, i.e. the
    /// INSERT column list.
    fn insert_columns(sql: &str) -> Vec<String> {
        let open = sql.find('(').expect("an INSERT must open a column list");
        let close = open + sql[open..].find(')').expect("...and close it");
        split_columns(&sql[open + 1..close])
    }

    /// The `ON CONFLICT (...)` target columns.
    fn conflict_key(sql: &str) -> Vec<String> {
        let at = sql
            .find("ON CONFLICT (")
            .expect("an upsert must name its conflict key");
        let open = at + "ON CONFLICT ".len();
        let close = open + sql[open..].find(')').expect("conflict key closes");
        split_columns(&sql[open + 1..close])
    }

    /// Left-hand sides of the `DO UPDATE SET` assignment list.
    fn update_columns(sql: &str) -> Vec<String> {
        let at = sql
            .find("DO UPDATE SET")
            .expect("this statement has no update list");
        sql[at + "DO UPDATE SET".len()..]
            .split(',')
            .map(|assign| assign.split('=').next().unwrap_or("").trim().to_string())
            .filter(|c| !c.is_empty())
            .collect()
    }

    fn split_columns(list: &str) -> Vec<String> {
        list.split(',')
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect()
    }

    /// Columns the statement INSERTs but does not replace on conflict, minus
    /// the conflict key. Every one of these is a column that keeps the FIRST
    /// run's value while its neighbours take the SECOND run's - the chimera.
    fn uncovered_columns(sql: &str) -> Vec<String> {
        let key = conflict_key(sql);
        let updated = update_columns(sql);
        insert_columns(sql)
            .into_iter()
            .filter(|c| !key.contains(c) && !updated.contains(c))
            .collect()
    }

    /// The defect, asserted against the SQL string the code actually sends.
    ///
    /// `ssot.checkpoints` upserted `path, sha256, bytes, bpb, ts` while the
    /// INSERT also carried `algo, hidden, format_version, data_synthetic, sha,
    /// run_id`: last-write-wins for five columns, first-write-wins for six.
    /// Written as a set difference rather than a list of names on purpose - a
    /// recipe column added to the INSERT later and forgotten in the update list
    /// fails HERE, with no database and no live table needed.
    #[test]
    fn do_update_set_covers_every_non_key_column() {
        for (table, sql) in [
            ("ssot.checkpoints", CHECKPOINT_UPSERT_SQL),
            ("public.bpb_samples", BPB_SAMPLE_LEGACY_UPSERT_SQL),
        ] {
            let inserted = insert_columns(sql);
            let key = conflict_key(sql);
            let updated = update_columns(sql);
            assert!(!inserted.is_empty(), "{table}: parsed no INSERT columns");
            assert!(!key.is_empty(), "{table}: parsed no conflict key");

            let uncovered = uncovered_columns(sql);
            assert!(
                uncovered.is_empty(),
                "{table}: {uncovered:?} are INSERTed but not replaced on conflict. \
                 A second run under the same {key:?} would keep the first run's \
                 value for them while taking the second run's for the rest, and \
                 the surviving row would describe a run that never happened."
            );

            for col in &updated {
                assert!(
                    inserted.contains(col),
                    "{table}: DO UPDATE SET names {col}, which the INSERT does not carry"
                );
                assert!(
                    !key.contains(col),
                    "{table}: DO UPDATE SET rewrites the conflict key column {col}"
                );
            }
        }
    }

    /// Negative control: the audit above is only worth running if it fails on
    /// the shape it exists to catch. Without this, a parser bug that returned
    /// an empty list would make the check pass on anything.
    #[test]
    fn the_column_audit_catches_a_forgotten_column() {
        const CHIMERA: &str = "INSERT INTO ssot.checkpoints \
             (canon_name, seed, step, sha256, hidden, data_synthetic) \
             VALUES ($1, $2, $3, $4, $5, $6) \
             ON CONFLICT (canon_name, seed, step) DO UPDATE SET \
               sha256 = EXCLUDED.sha256";
        assert_eq!(
            uncovered_columns(CHIMERA),
            vec!["hidden".to_string(), "data_synthetic".to_string()],
            "the audit must name every column left behind by the update list"
        );
        assert_eq!(
            conflict_key(CHIMERA),
            vec![
                "canon_name".to_string(),
                "seed".to_string(),
                "step".to_string()
            ]
        );
    }

    /// The four columns whose omission was reproduced twice against a live
    /// table (`hidden=64 | bytes=524592`; `algo='muon'` on muon-cwd's sha256).
    /// Named explicitly as well as set-wise, so a restructure that keeps the
    /// set check happy by dropping them from the INSERT is still visible.
    #[test]
    fn checkpoint_upsert_replaces_recipe_and_provenance() {
        let inserted = insert_columns(CHECKPOINT_UPSERT_SQL);
        let updated = update_columns(CHECKPOINT_UPSERT_SQL);
        for col in [
            "algo",
            "hidden",
            "format_version",
            "data_synthetic",
            "sha",
            "run_id",
        ] {
            let col = col.to_string();
            assert!(inserted.contains(&col), "{col} left the INSERT list");
            assert!(
                updated.contains(&col),
                "{col} is INSERTed but not replaced on conflict"
            );
        }
        // The honesty flag specifically: a synthetic-corpus run must never be
        // able to inherit an earlier real run's `data_synthetic = false`.
        assert!(CHECKPOINT_UPSERT_SQL.contains("data_synthetic = EXCLUDED.data_synthetic"));
    }

    /// R4-3: the write-side whitelist was gated on `parsed.is_some()`, and
    /// `parse_canon_name` returns None for everything without the `IGLA-`
    /// prefix -- including the DEFAULT `trios-train-rng{seed}`. So the one
    /// guard that refuses an optimizer this crate cannot run was inert for the
    /// most common name in the ledger.
    #[test]
    fn a_non_conforming_canon_with_an_off_whitelist_algo_is_rejected() {
        let _g = counter_guard();
        clear_dsn_env();
        assert!(
            parse_canon_name("trios-train-rng47").is_none(),
            "the default canon carries no recipe, which is the point"
        );
        for fake in ["soap", "lamb", "prodigy", "lion"] {
            let before = dropped_writes();
            let out = bpb_sample_with_algo("trios-train-rng47", 47, 1000, 2.6141, None, fake);
            assert_eq!(
                out,
                LedgerWrite::Rejected,
                "a non-conforming canon executing {fake} must be refused"
            );
            assert_eq!(
                dropped_writes(),
                before,
                "a rejected value is not a dropped write: nothing was lost"
            );
        }
        // Still refused when the canon DOES parse (the pre-existing behaviour).
        assert_eq!(
            bpb_sample_with_algo(
                "IGLA-SHORT-WAVE-MATRIX-gf16-h128-LR0.0001-rng47-soap",
                47,
                1000,
                2.6141,
                None,
                "soap"
            ),
            LedgerWrite::Rejected
        );
    }

    /// The empty algo is a DIFFERENT fact from a fake one: it means no caller
    /// declared an optimizer. Those rows cannot reach `ssot.bpb_samples` (the
    /// canon does not parse), so they keep the quarantined public path and get
    /// a WARN line instead of a rejection -- rejecting them would silently stop
    /// every legacy trainer's ledger, which is not what this guard is for.
    #[test]
    fn an_undeclared_optimizer_is_warned_about_not_rejected() {
        let _g = counter_guard();
        clear_dsn_env();
        let out = bpb_sample_with_algo("trios-train-rng47", 47, 1000, 2.6141, None, "");
        assert_eq!(
            out,
            LedgerWrite::SkippedNoDsn,
            "an undeclared optimizer must reach the write path, not the R5 reject"
        );
    }

    /// Absent provenance must be SQL NULL. `''` satisfies `NOT NULL`, joins,
    /// groups, and hides from `WHERE sha IS NULL` -- the exact query an auditor
    /// writes to find rows with no provenance. 16 of 17 live rows were in that
    /// state.
    #[test]
    fn absent_git_provenance_binds_null_not_empty_string() {
        assert_eq!(opt_text_val(None), sea_orm::Value::String(None));
        assert_eq!(
            opt_text_val(Some("ef6f0887")),
            sea_orm::Value::String(Some(Box::new("ef6f0887".to_string())))
        );
        assert_ne!(
            opt_text_val(None),
            seed_val(""),
            "NULL and '' must not be the same binding"
        );
    }

    /// An env var exported empty by a CI step that failed to resolve the commit
    /// is the same absence as an unset one.
    #[test]
    fn env_nonempty_treats_blank_as_absent() {
        const KEY: &str = "TRIOS_R4_3_ENV_PROBE";
        std::env::remove_var(KEY);
        assert_eq!(env_nonempty(KEY), None);
        std::env::set_var(KEY, "");
        assert_eq!(env_nonempty(KEY), None);
        std::env::set_var(KEY, "   ");
        assert_eq!(env_nonempty(KEY), None, "whitespace is not a commit sha");
        std::env::set_var(KEY, " ef6f0887 ");
        assert_eq!(env_nonempty(KEY), Some("ef6f0887".to_string()));
        std::env::remove_var(KEY);
    }

    /// S11: with no DSN the exit code stays 0 -- there is nothing to be silent
    /// about. The FATAL path needs a configured DSN and is exercised by the
    /// live-ledger integration test.
    #[test]
    fn ledger_exit_code_is_zero_without_a_dsn() {
        std::env::remove_var("DATABASE_URL");
        std::env::remove_var("TRIOS_NEON_DSN");
        std::env::remove_var("NEON_DATABASE_URL");
        std::env::remove_var("TRIOS_DATABASE_URL");
        assert_eq!(ledger_exit_code(), 0);
    }

    /// A run whose every row was REFUSED must not exit 0.
    ///
    /// `LedgerWrite::Rejected` used to increment neither `LANDED_WRITES` nor
    /// `DROPPED_WRITES`, and `ledger_exit_code()` reads only those two. So a
    /// process in which 100% of BPB rows were refused as unpublishable
    /// reported `attempted=0 landed=0 dropped=0` and handed its supervisor a
    /// zero exit status -- a failure published as a clean run.
    ///
    /// No database is needed: the decision is a pure function of the counters
    /// and of whether a DSN is configured. `note_rejected` is driven directly
    /// because the three call sites are inside `bpb_sample_with_algo`, whose
    /// other branches would need a connection.
    #[test]
    fn a_run_whose_rows_were_all_refused_exits_non_zero() {
        let _g = counter_guard();
        clear_dsn_env();

        // Precondition, asserted rather than assumed: nothing lands in a unit
        // test binary, because landing requires a live connection.
        assert_eq!(
            landed_writes(),
            0,
            "this test reasons about landed == 0; a live ledger would invalidate it"
        );

        let before = rejected_writes();
        note_rejected();
        assert_eq!(
            rejected_writes(),
            before + 1,
            "a refusal must be counted somewhere"
        );

        // With no DSN there is nothing to be silent about; unchanged.
        assert_eq!(
            ledger_exit_code(),
            0,
            "no DSN configured: the run never claimed it would be recorded"
        );

        // With a DSN configured, refusal-only is FATAL.
        set_dsn_env("DATABASE_URL");
        let code = ledger_exit_code();
        clear_dsn_env();
        assert_eq!(
            code, 1,
            "every row refused and none landed must not exit 0"
        );
    }

    /// A landed heartbeat must not license a 0 exit for a run whose evidence
    /// rows all dropped.
    ///
    /// `ledger_exit_code` fired only on `landed == 0`, and `landed` counts
    /// heartbeats, `trial_start`, `trial_complete`, `bpb_sample` and
    /// `checkpoint_record` alike. One heartbeat through a flaky connection was
    /// therefore enough to report success for a run that recorded no BPB and
    /// no artifact -- the two rows that ARE the evidence. "The process was
    /// alive" is not "the run recorded itself".
    ///
    /// No database is needed: the decision is a pure function of the counters
    /// and of whether a DSN is configured, so the counters are driven directly.
    #[test]
    fn a_landed_heartbeat_does_not_license_dropped_evidence_rows() {
        let _g = counter_guard();
        clear_dsn_env();

        // The heartbeat got through: the aggregate `landed` is non-zero, which
        // is exactly what used to make this run look clean. The counter is
        // process-global and restored at the end of this test, because
        // `a_run_whose_rows_were_all_refused_exits_non_zero` asserts
        // `landed_writes() == 0` as an absolute precondition.
        let landed_before = landed_writes();
        note_landed();
        assert!(landed_writes() > 0, "the heartbeat must count as landed");

        // Both evidence kinds were attempted and both were lost.
        note_evidence(EvidenceKind::BpbSample, LedgerWrite::Failed);
        note_dropped();
        note_evidence(EvidenceKind::CheckpointRecord, LedgerWrite::Failed);
        note_dropped();
        assert_eq!(bpb_sample_writes(), (1, 0));
        assert_eq!(checkpoint_record_writes(), (1, 0));

        // No DSN: the run never claimed it would be recorded. Unchanged.
        assert_eq!(ledger_exit_code(), 0);

        set_dsn_env("DATABASE_URL");
        let code = ledger_exit_code();
        clear_dsn_env();
        LANDED_WRITES.store(landed_before, Ordering::Relaxed);
        assert_eq!(
            code, 1,
            "a landed heartbeat must not license a 0 when every bpb_sample and \
             every checkpoint_record dropped"
        );
    }

    /// The other half of the same rule: a kind that was never attempted cannot
    /// fail the run. A trainer built without checkpointing, or a smoke run
    /// that emits no artifact, must still exit 0 when its BPB rows land.
    #[test]
    fn an_unattempted_evidence_kind_does_not_fail_the_run() {
        let _g = counter_guard();
        clear_dsn_env();

        // A bpb_sample that landed increments both tallies, exactly as the
        // write path does. The aggregate one matters here: `DROPPED_WRITES` is
        // process-global and other tests in this binary have already added to
        // it, so a run with `landed == 0` would fail on the pre-existing
        // aggregate branch and this test would prove nothing about the new one.
        let landed_before = landed_writes();
        note_landed();
        note_evidence(EvidenceKind::BpbSample, LedgerWrite::Written);
        assert_eq!(bpb_sample_writes(), (1, 1));
        assert_eq!(
            checkpoint_record_writes(),
            (0, 0),
            "checkpoint_record was never called"
        );

        set_dsn_env("DATABASE_URL");
        let code = ledger_exit_code();
        clear_dsn_env();
        LANDED_WRITES.store(landed_before, Ordering::Relaxed);
        assert_eq!(code, 0, "a kind never attempted is not a kind that failed");
    }

    /// `dsn_configured` is the predicate a binary needs to tell "no ledger was
    /// asked for" apart from "the ledger lost the writes".
    #[test]
    fn dsn_configured_reports_the_absence_of_a_dsn() {
        let _g = counter_guard();
        clear_dsn_env();
        assert!(!dsn_configured(), "no alias set: no DSN configured");
        set_dsn_env("TRIOS_NEON_DSN");
        assert!(dsn_configured(), "an alias is a configured DSN");
        clear_dsn_env();
    }

    // -- the write opt-in ------------------------------------------------------

    /// An ambient DSN is configuration, not permission.
    ///
    /// `db()` used to connect on the presence of any alias, and a scouting run
    /// put 12 rows on the shared ledger before anyone noticed. The only guard
    /// anywhere was `env -i` inside a CI workflow, which protects CI and
    /// nothing else.
    ///
    /// Without the opt-in the module must be indistinguishable from a run that
    /// was never asked to record itself -- same `None`, same `NoDsn`, same
    /// `dsn_configured`, same exit code -- so that no caller and no supervisor
    /// needs to learn a new state.
    #[test]
    fn a_dsn_without_the_opt_in_writes_nothing() {
        let _g = counter_guard();
        clear_dsn_env();
        std::env::set_var("DATABASE_URL", "postgres://unused/never-connected");
        assert!(
            std::env::var(LEDGER_WRITE_OPT_IN).is_err(),
            "precondition: the opt-in is absent"
        );

        assert!(
            db().is_none(),
            "a DSN alone must not open a connection to shared state"
        );
        assert_eq!(
            db_status(),
            DbStatus::NoDsn,
            "an un-opted-in run must report the no-DSN shape"
        );
        assert_eq!(db_status().as_str(), "DSN unset");
        assert!(!dsn_configured());
        assert_eq!(
            ledger_exit_code(),
            0,
            "refusing to write is not a failure of the run"
        );
        clear_dsn_env();
    }

    /// The opt-in on its own configures nothing: it is permission to use a DSN,
    /// not a DSN. Asking to write without one must stay exactly as quiet as
    /// asking for nothing.
    #[test]
    fn the_opt_in_without_a_dsn_changes_nothing() {
        let _g = counter_guard();
        clear_dsn_env();
        std::env::set_var(LEDGER_WRITE_OPT_IN, "1");
        assert!(db().is_none());
        assert_eq!(db_status(), DbStatus::NoDsn);
        assert!(!dsn_configured());
        assert_eq!(ledger_exit_code(), 0);
        clear_dsn_env();
    }

    // -- the RECORDED CAUSE of a skipped write ---------------------------------

    /// Nothing configured: the record must say so, unchanged.
    #[test]
    fn no_dsn_at_all_is_recorded_as_skipped_no_dsn() {
        let _g = counter_guard();
        clear_dsn_env();
        assert_eq!(skipped_write_outcome(DbStatus::NoDsn), LedgerWrite::SkippedNoDsn);
        assert_eq!(
            bpb_sample_with_algo("trios-train-rng47", 47, 1000, 2.6141, None, "adamw"),
            LedgerWrite::SkippedNoDsn,
            "an empty environment is the one thing skipped-no-dsn may mean"
        );
        assert_eq!(LedgerWrite::SkippedNoDsn.as_str(), "skipped-no-dsn");
    }

    /// A DSN WAS configured and this run declined it. Reproduced live on
    /// 2026-08-03: `DATABASE_URL` set, `TRIOS_LEDGER_WRITE` unset, the trainer
    /// printed the refusal on stderr and then wrote `ledger: "skipped-no-dsn"`
    /// into the schema-8 sidecar -- a false cause in the evidence document,
    /// invisible to `ckpt_replay` and to `interop/triosckp_reader.py`, which
    /// read the artifact and not the console.
    #[test]
    fn a_configured_dsn_without_the_opt_in_is_recorded_as_not_opted_in() {
        let _g = counter_guard();
        clear_dsn_env();
        for alias in [
            "DATABASE_URL",
            "NEON_DATABASE_URL",
            "TRIOS_NEON_DSN",
            "TRIOS_DATABASE_URL",
        ] {
            clear_dsn_env();
            std::env::set_var(alias, "postgres://unused/never-connected");
            assert_eq!(
                skipped_write_outcome(DbStatus::NoDsn),
                LedgerWrite::SkippedNotOptedIn,
                "{alias} is a configured DSN, so the cause is the missing opt-in"
            );
            assert_eq!(
                bpb_sample_with_algo("trios-train-rng47", 47, 1000, 2.6141, None, "adamw"),
                LedgerWrite::SkippedNotOptedIn,
                "{alias} set without the opt-in must not be recorded as no-DSN"
            );
        }
        clear_dsn_env();
        assert_eq!(
            LedgerWrite::SkippedNotOptedIn.as_str(),
            "skipped-not-opted-in"
        );
    }

    /// The opted-in run is untouched by any of this.
    ///
    /// Asserted at the classifier and at `active_dsn`, not end to end: with the
    /// opt-in present `db()` really does try to connect, and a unit test must
    /// not depend on how long an unresolvable host takes to fail. What the
    /// change could plausibly break is reachability -- `active_dsn` is `Some`,
    /// so `db()` never sets `NoDsn`, so NEITHER skipped variant is on this
    /// path, and a lost write is still `Failed`.
    #[test]
    fn an_opted_in_dsn_keeps_its_existing_outcomes() {
        let _g = counter_guard();
        clear_dsn_env();
        set_dsn_env("DATABASE_URL");
        assert!(
            active_dsn().is_some(),
            "opted in: no code path can report this run as having no DSN"
        );
        assert_eq!(
            skipped_write_outcome(DbStatus::ConnectFailed),
            LedgerWrite::Failed,
            "a connect failure under an opted-in DSN is unchanged"
        );
        assert_eq!(skipped_write_outcome(DbStatus::Ready), LedgerWrite::Failed);
        clear_dsn_env();
    }

    /// Exactly `1`. A half-recognised spelling is how an opt-in silently
    /// becomes a default again.
    #[test]
    fn the_opt_in_is_the_literal_one_and_nothing_else() {
        let _g = counter_guard();
        clear_dsn_env();
        for spelling in ["", "0", "true", "yes", "on", "11", "1 1"] {
            std::env::set_var(LEDGER_WRITE_OPT_IN, spelling);
            assert!(
                !ledger_write_opted_in(),
                "{spelling:?} must not be read as an opt-in"
            );
        }
        for spelling in ["1", " 1", "1\n"] {
            std::env::set_var(LEDGER_WRITE_OPT_IN, spelling);
            assert!(
                ledger_write_opted_in(),
                "{spelling:?} is the opt-in, modulo surrounding whitespace"
            );
        }
        clear_dsn_env();
    }
}
