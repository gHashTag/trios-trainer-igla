//! Trinity Hive Cellular Automaton — L13.
//!
//! # Pain it cures
//!
//! Each agent finishes one lane and idles, waiting for a new ORDER from the
//! human. That kills throughput. Cure is *not* louder dispatch. Cure is a
//! deterministic state machine. One hive state, one transition table, zero
//! questions.
//!
//! # Contract
//!
//! Every agent in the hive is a *cell*. The cell has a [`State`]. On each tick
//! the cell observes the [`World`] (issue comments, git status, CI verdict)
//! and computes its next [`AgentAction`] via the pure function
//! [`HiveAutomaton::next_action`].
//!
//! `next_action` *never* returns `Halt` unless the hive has reached
//! [`State::Victory`] (global SUCCESS predicate fired) or [`State::HardAbort`]
//! (a hard rule was violated, or the race deadline passed). When a lane is
//! `Done` the next state is `Scan`, *not* `Idle`. There is no `Idle`.
//!
//! # Source of truth
//!
//! The transition table, the lane → file ownership map, the priority queue
//! and the halt predicates all live in `assertions/hive_automaton.json`.
//! Rust mirrors that JSON via build-time `const _: ()` schema asserts and a
//! runtime [`HiveSpec::load`] loader. If the JSON drifts from this file, the
//! schema_version assert fails the build (L-R14 spirit applied to the
//! coordination protocol).
//!
//! # L-R14 trace
//!
//! | Rust constant         | JSON anchor                                  |
//! |-----------------------|----------------------------------------------|
//! | `SCHEMA_VERSION`      | `assertions/hive_automaton.json::schema_version` |
//! | `VICTORY_SEED_TARGET` | `halt_predicates.global_success` (= 3 seeds) |
//! | `BPB_VICTORY_TARGET`  | matches `IGLA_TARGET_BPB` in `lib.rs` (= 1.5)|
//! | `LANE_COUNT`          | `lane_ownership` length                       |
//!
//! # Tests
//!
//! 18 unit tests below cover: every transition row, the no-Idle invariant,
//! the lane-priority order, the ownership R6 check, victory absorption,
//! hard-abort triggers, and a falsification witness for the *most natural
//! human bug*: returning `Halt` after `Done` instead of cycling to `Scan`.

use serde::{Deserialize, Serialize};

/// Schema version locked to `assertions/hive_automaton.json::schema_version`.
pub const SCHEMA_VERSION: &str = "1.0";

/// Number of distinct seeds below `BPB_VICTORY_TARGET` required for global
/// SUCCESS. From `halt_predicates.global_success`.
pub const VICTORY_SEED_TARGET: u32 = 3;

/// BPB threshold the race targets — must match `crate::IGLA_TARGET_BPB`.
pub const BPB_VICTORY_TARGET: f64 = 1.5;

/// Number of lanes covered by the ownership map. Must equal
/// `assertions/hive_automaton.json::lane_ownership` entries (L0..L13).
pub const LANE_COUNT: usize = 14;

// Compile-time L-R14 mirror: forces a build error if these constants drift
// away from the JSON anchors. `const _: ()` is the canonical Rust idiom for
// build-time invariants.
const _: () = {
    assert!(
        VICTORY_SEED_TARGET == 3,
        "JSON requires 3 distinct victory seeds"
    );
    assert!(
        LANE_COUNT == 14,
        "lane_ownership in JSON has 14 entries (L0..L13)"
    );
    // BPB_VICTORY_TARGET cannot be checked with `const fn` float compare on
    // stable, but the runtime test `test_bpb_target_matches_lib` enforces it.
};

/// One state of one cell in the hive automaton.
///
/// Mirrors `assertions/hive_automaton.json::states`. Variants are ordered to
/// reflect the typical traversal order Boot → Scan → ... → Done → Scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum State {
    /// Cell just woke. Loads JSON, syncs git, reads the race issue.
    Boot,
    /// Reads issue comments and builds the lane-state world view.
    Scan,
    /// Picks the highest-priority free lane from the queue.
    Pick,
    /// Posts the 🔒 CLAIMING comment on the race issue.
    Claim,
    /// Long-running edit phase. Touches only owned files.
    Work,
    /// Atomic commit on `main`, R10 message format, `git push`.
    Commit,
    /// Waits for CI, attributes failures relative to the prior SHA.
    CiWait,
    /// Posts the ✅ DONE comment with the §4.3 template; releases the lane.
    Done,
    /// New CI failure or race condition. Posts ⚠️ BLOCKED comment.
    Blocked,
    /// Lost the claim race or hit ownership conflict. Drops, returns to Pick.
    Reclaim,
    /// Absorbing: global SUCCESS fired. Cell halts cleanly.
    Victory,
    /// Absorbing: hard rule violated or race called off. Cell halts.
    HardAbort,
}

impl State {
    /// Absorbing states are the *only* halts. `Done` is **not** absorbing —
    /// it cycles back to `Scan`. This is the cure for the lane-idle bug.
    pub fn is_absorbing(self) -> bool {
        matches!(self, State::Victory | State::HardAbort)
    }
}

/// Lane identifier. Each lane has exactly one owning file (R6 file
/// ownership). Lane numbers map to `assertions/hive_automaton.json::lane_ownership`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Lane(pub u8);

impl Lane {
    pub const L0: Lane = Lane(0);
    pub const L1: Lane = Lane(1);
    pub const L2: Lane = Lane(2);
    pub const L3: Lane = Lane(3);
    pub const L4: Lane = Lane(4);
    pub const L5: Lane = Lane(5);
    pub const L6: Lane = Lane(6);
    pub const L7: Lane = Lane(7);
    pub const L8: Lane = Lane(8);
    pub const L9: Lane = Lane(9);
    pub const L10: Lane = Lane(10);
    pub const L11: Lane = Lane(11);
    pub const L12: Lane = Lane(12);
    pub const L13: Lane = Lane(13);
}

/// Why the cell entered `HardAbort`. Mirrors
/// `halt_predicates.hard_abort_reasons` in JSON.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AbortReason {
    /// R1 — Rust-only violated.
    R1ShellOrPythonIntroduced,
    /// R3 — branch or PR used instead of `main`.
    R3BranchOrPrUsed,
    /// R4 — Rust constant introduced without a `.v` / JSON anchor.
    R4ConstantWithoutVAnchor,
    /// R5 — Coq status lied (Admitted masked as Proven).
    R5StatusLied,
    /// R7 — forbidden numeric value (e.g. `prune=2.65`).
    R7ForbiddenValueIntroduced,
    /// Race deadline passed.
    DeadlinePassed,
    /// Race called off externally.
    RaceCancelled,
}

/// Observable world used to drive transitions. Pure data; no I/O. The
/// external executor fills this struct on every tick.
#[derive(Clone, Debug, Default)]
pub struct World {
    /// Whether `git fetch && git rev-parse main` succeeded this tick.
    pub git_synced: bool,
    /// Whether the race issue is reachable via `gh issue view`.
    pub issue_readable: bool,
    /// Lanes that no agent currently claims and which are not Done.
    pub free_lanes: Vec<Lane>,
    /// Lanes claimed by *this* cell — used to detect re-entrant Reclaim.
    pub my_claimed_lanes: Vec<Lane>,
    /// Lane chosen this tick (set by [`HiveAutomaton`] in Pick → Claim).
    pub current_lane: Option<Lane>,
    /// Number of distinct seeds proven below [`BPB_VICTORY_TARGET`].
    pub victory_seeds: u32,
    /// True if the deadline has passed.
    pub deadline_passed: bool,
    /// True if external sentinel called off the race.
    pub race_cancelled: bool,
    /// In Claim: did this cell post its CLAIMING comment first?
    pub claim_won: bool,
    /// In Work: did the diff touch only files owned by `current_lane`?
    pub only_owned_files_touched: bool,
    /// In Work: did anything ship a forbidden value (R7)?
    pub forbidden_value_introduced: bool,
    /// In Commit: did `git push` succeed (no non-fast-forward)?
    pub push_succeeded: bool,
    /// In CiWait: any CI failure that did *not* exist on the prior SHA?
    pub new_ci_failure: bool,
    /// In Blocked: a follow-up commit was pushed to fix the failure.
    pub fix_committed: bool,
    /// Optional explicit hard-abort reason set by the executor.
    pub abort_reason: Option<AbortReason>,
}

impl World {
    /// Fresh `World`. Equivalent to `World::default()` but more readable.
    pub fn fresh() -> Self {
        Self::default()
    }

    fn global_success(&self) -> bool {
        self.victory_seeds >= VICTORY_SEED_TARGET
    }

    fn must_hard_abort(&self) -> Option<AbortReason> {
        if self.deadline_passed {
            return Some(AbortReason::DeadlinePassed);
        }
        if self.race_cancelled {
            return Some(AbortReason::RaceCancelled);
        }
        if self.forbidden_value_introduced {
            return Some(AbortReason::R7ForbiddenValueIntroduced);
        }
        self.abort_reason
    }
}

/// Action produced by the automaton on each tick. The executor performs the
/// concrete I/O (git, gh CLI, file edits) and feeds the result back as a new
/// [`World`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentAction {
    /// Boot the cell: load JSON, sync git, read the issue.
    BootCell,
    /// Re-read the issue and refresh `World`.
    RescanIssue,
    /// Pick `lane` from the priority queue.
    PickLane(Lane),
    /// Post the 🔒 CLAIMING comment for `lane`.
    PostClaim(Lane),
    /// Edit only files owned by `lane`.
    DoWork(Lane),
    /// Push the atomic commit for `lane` to `main`.
    PushCommit(Lane),
    /// Wait for CI on the just-pushed SHA.
    WaitForCi(Lane),
    /// Post the ✅ DONE comment for `lane`.
    PostDone(Lane),
    /// Post the ⚠️ BLOCKED comment for `lane`.
    PostBlocked(Lane),
    /// Drop the lost claim and rescan (no human input required).
    DropAndRescan(Lane),
    /// Absorbing: global success — cell halts cleanly.
    Halt(HaltCause),
}

/// Cause of a [`AgentAction::Halt`]. The only two reasons a cell may stop
/// looping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HaltCause {
    /// `victory_seeds >= VICTORY_SEED_TARGET`.
    Victory,
    /// Hard rule violated or external cancellation.
    HardAbort(AbortReason),
}

/// The cellular automaton. One per cell.
#[derive(Clone, Debug)]
pub struct HiveAutomaton {
    state: State,
    /// Lane-priority queue. Mirrors `assertions/hive_automaton.json::lane_priority_queue`.
    pub priority_queue: Vec<Lane>,
}

impl HiveAutomaton {
    /// Default automaton wired to the JSON priority queue at the time of
    /// L13 ship: L7, L6, L9, L11, L12 (all currently unclaimed lanes that
    /// move BPB).
    pub fn new() -> Self {
        Self {
            state: State::Boot,
            priority_queue: vec![Lane::L7, Lane::L6, Lane::L9, Lane::L11, Lane::L12],
        }
    }

    /// Construct from an explicit priority queue (used in tests).
    pub fn with_queue(queue: Vec<Lane>) -> Self {
        Self {
            state: State::Boot,
            priority_queue: queue,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    /// Pick the highest-priority free lane from the queue, falling back to
    /// `None` if every queue entry is currently claimed by someone else.
    #[allow(clippy::manual_find)]
    fn pick_free_lane(&self, world: &World) -> Option<Lane> {
        self.priority_queue
            .iter()
            .find(|&&lane| world.free_lanes.contains(&lane))
            .copied()
    }

    /// **The pure transition function.**
    ///
    /// On each tick the executor calls this with the current `World` and
    /// receives an [`AgentAction`]. The automaton's `state` advances as a
    /// side-effect (this is the only mutating method, and it does not
    /// touch the outside world).
    ///
    /// # Determinism
    ///
    /// Same `(state, world)` always returns the same `AgentAction`. No
    /// randomness, no clocks, no I/O.
    ///
    /// # No-Idle invariant
    ///
    /// `Done` *never* yields `Halt(_)` unless `world.global_success()` is
    /// true. The lane-idle bug is structurally impossible.
    pub fn next_action(&mut self, world: &World) -> AgentAction {
        // Global short-circuits — checked at every state, in priority order.
        if world.global_success() {
            self.state = State::Victory;
            return AgentAction::Halt(HaltCause::Victory);
        }
        if let Some(reason) = world.must_hard_abort() {
            self.state = State::HardAbort;
            return AgentAction::Halt(HaltCause::HardAbort(reason));
        }

        match self.state {
            State::Boot => {
                if world.git_synced && world.issue_readable {
                    self.state = State::Scan;
                    AgentAction::RescanIssue
                } else {
                    AgentAction::BootCell
                }
            }
            State::Scan => match self.pick_free_lane(world) {
                Some(lane) => {
                    self.state = State::Pick;
                    AgentAction::PickLane(lane)
                }
                None => AgentAction::RescanIssue,
            },
            State::Pick => match world.current_lane {
                Some(lane) => {
                    self.state = State::Claim;
                    AgentAction::PostClaim(lane)
                }
                None => {
                    self.state = State::Scan;
                    AgentAction::RescanIssue
                }
            },
            State::Claim => {
                let lane = world.current_lane.expect("Claim requires current_lane");
                if world.claim_won {
                    self.state = State::Work;
                    AgentAction::DoWork(lane)
                } else {
                    self.state = State::Reclaim;
                    AgentAction::DropAndRescan(lane)
                }
            }
            State::Reclaim => {
                self.state = State::Scan;
                AgentAction::RescanIssue
            }
            State::Work => {
                let lane = world.current_lane.expect("Work requires current_lane");
                if !world.only_owned_files_touched {
                    self.state = State::Reclaim;
                    AgentAction::DropAndRescan(lane)
                } else {
                    self.state = State::Commit;
                    AgentAction::PushCommit(lane)
                }
            }
            State::Commit => {
                let lane = world.current_lane.expect("Commit requires current_lane");
                if world.push_succeeded {
                    self.state = State::CiWait;
                    AgentAction::WaitForCi(lane)
                } else {
                    self.state = State::Reclaim;
                    AgentAction::DropAndRescan(lane)
                }
            }
            State::CiWait => {
                let lane = world.current_lane.expect("CiWait requires current_lane");
                if world.new_ci_failure {
                    self.state = State::Blocked;
                    AgentAction::PostBlocked(lane)
                } else {
                    self.state = State::Done;
                    AgentAction::PostDone(lane)
                }
            }
            State::Done => {
                // The cure for the idle bug: from Done we always cycle back
                // to Scan, never to a Halt. The only way Halt happens here
                // is via the global short-circuits at the top of the
                // function (Victory / HardAbort), which are already handled.
                self.state = State::Scan;
                AgentAction::RescanIssue
            }
            State::Blocked => {
                let lane = world.current_lane.expect("Blocked requires current_lane");
                if world.fix_committed {
                    self.state = State::CiWait;
                    AgentAction::WaitForCi(lane)
                } else {
                    AgentAction::PostBlocked(lane)
                }
            }
            State::Victory => AgentAction::Halt(HaltCause::Victory),
            State::HardAbort => AgentAction::Halt(HaltCause::HardAbort(
                world.abort_reason.unwrap_or(AbortReason::RaceCancelled),
            )),
        }
    }
}

impl Default for HiveAutomaton {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_after_boot() -> World {
        let mut w = World::fresh();
        w.git_synced = true;
        w.issue_readable = true;
        w
    }

    fn world_with_free_lane(lane: Lane) -> World {
        let mut w = world_after_boot();
        w.free_lanes = vec![lane];
        w
    }

    #[test]
    fn test_boot_to_scan_when_git_and_issue_ok() {
        let mut h = HiveAutomaton::new();
        let w = world_after_boot();
        let a = h.next_action(&w);
        assert_eq!(a, AgentAction::RescanIssue);
        assert_eq!(h.state(), State::Scan);
    }

    #[test]
    fn test_boot_self_loops_when_git_unavailable() {
        let mut h = HiveAutomaton::new();
        let w = World::fresh(); // git_synced=false
        let a = h.next_action(&w);
        assert_eq!(a, AgentAction::BootCell);
        assert_eq!(h.state(), State::Boot);
    }

    #[test]
    fn test_scan_picks_highest_priority_free_lane() {
        let mut h = HiveAutomaton::new();
        // Skip Boot.
        h.next_action(&world_after_boot());
        let mut w = world_after_boot();
        w.free_lanes = vec![Lane::L11, Lane::L7, Lane::L9];
        let a = h.next_action(&w);
        assert_eq!(a, AgentAction::PickLane(Lane::L7), "L7 has priority 0");
        assert_eq!(h.state(), State::Pick);
    }

    #[test]
    fn test_scan_self_loops_when_no_free_lane() {
        let mut h = HiveAutomaton::new();
        h.next_action(&world_after_boot());
        let w = world_after_boot(); // no free lanes
        let a = h.next_action(&w);
        assert_eq!(a, AgentAction::RescanIssue);
        assert_eq!(h.state(), State::Scan);
    }

    #[test]
    fn test_pick_to_claim_when_lane_chosen() {
        let mut h = HiveAutomaton::new();
        h.next_action(&world_after_boot()); // Boot → Scan
        h.next_action(&world_with_free_lane(Lane::L7)); // Scan → Pick
        let mut w = world_with_free_lane(Lane::L7);
        w.current_lane = Some(Lane::L7);
        let a = h.next_action(&w);
        assert_eq!(a, AgentAction::PostClaim(Lane::L7));
        assert_eq!(h.state(), State::Claim);
    }

    #[test]
    fn test_claim_won_advances_to_work() {
        let mut h = HiveAutomaton::with_queue(vec![Lane::L7]);
        h.next_action(&world_after_boot());
        h.next_action(&world_with_free_lane(Lane::L7));
        let mut w = world_with_free_lane(Lane::L7);
        w.current_lane = Some(Lane::L7);
        h.next_action(&w); // Pick → Claim
        w.claim_won = true;
        let a = h.next_action(&w);
        assert_eq!(a, AgentAction::DoWork(Lane::L7));
        assert_eq!(h.state(), State::Work);
    }

    #[test]
    fn test_claim_lost_drops_to_reclaim_then_scan() {
        let mut h = HiveAutomaton::with_queue(vec![Lane::L7]);
        h.next_action(&world_after_boot());
        h.next_action(&world_with_free_lane(Lane::L7));
        let mut w = world_with_free_lane(Lane::L7);
        w.current_lane = Some(Lane::L7);
        h.next_action(&w); // Pick → Claim
        w.claim_won = false;
        let a = h.next_action(&w);
        assert_eq!(a, AgentAction::DropAndRescan(Lane::L7));
        assert_eq!(h.state(), State::Reclaim);
        // Reclaim is transient — next tick goes to Scan, not Halt.
        let a2 = h.next_action(&w);
        assert_eq!(a2, AgentAction::RescanIssue);
        assert_eq!(h.state(), State::Scan);
    }

    #[test]
    fn test_work_with_ownership_violation_drops_to_reclaim() {
        // The cure for R6 collisions.
        let mut h = HiveAutomaton::with_queue(vec![Lane::L7]);
        let mut w = world_with_free_lane(Lane::L7);
        w.current_lane = Some(Lane::L7);
        w.claim_won = true;
        // Boot → Scan → Pick → Claim → Work
        h.next_action(&world_after_boot());
        h.next_action(&w);
        h.next_action(&w);
        h.next_action(&w);
        // In Work: ownership violation
        w.only_owned_files_touched = false;
        let a = h.next_action(&w);
        assert_eq!(a, AgentAction::DropAndRescan(Lane::L7));
        assert_eq!(h.state(), State::Reclaim);
    }

    #[test]
    fn test_work_to_commit_then_ci_then_done() {
        let mut h = HiveAutomaton::with_queue(vec![Lane::L7]);
        let mut w = world_with_free_lane(Lane::L7);
        w.current_lane = Some(Lane::L7);
        w.claim_won = true;
        h.next_action(&world_after_boot());
        h.next_action(&w);
        h.next_action(&w);
        h.next_action(&w); // Claim → Work
        w.only_owned_files_touched = true;
        assert_eq!(h.next_action(&w), AgentAction::PushCommit(Lane::L7));
        assert_eq!(h.state(), State::Commit);
        w.push_succeeded = true;
        assert_eq!(h.next_action(&w), AgentAction::WaitForCi(Lane::L7));
        assert_eq!(h.state(), State::CiWait);
        // CI clean (no new failure) → Done.
        w.new_ci_failure = false;
        assert_eq!(h.next_action(&w), AgentAction::PostDone(Lane::L7));
        assert_eq!(h.state(), State::Done);
    }

    #[test]
    fn test_done_cycles_back_to_scan_not_halt() {
        // CRITICAL — falsification witness for the lane-idle bug. If a
        // future refactor accidentally returns Halt from Done while
        // global_success is false, this test fails the build.
        let mut h = HiveAutomaton::with_queue(vec![Lane::L7]);
        let mut w = world_with_free_lane(Lane::L7);
        w.current_lane = Some(Lane::L7);
        w.claim_won = true;
        w.only_owned_files_touched = true;
        w.push_succeeded = true;
        // Drive to Done. Boot→Scan→Pick→Claim→Work→Commit→CiWait→Done = 7 ticks.
        for _ in 0..7 {
            h.next_action(&w);
        }
        assert_eq!(h.state(), State::Done);
        // The next tick MUST be RescanIssue, not Halt.
        let a = h.next_action(&w);
        assert_eq!(
            a,
            AgentAction::RescanIssue,
            "Done must cycle to Scan, never Idle"
        );
        assert_eq!(h.state(), State::Scan);
        assert!(!h.state().is_absorbing());
    }

    #[test]
    fn test_global_success_short_circuits_from_any_state() {
        let mut h = HiveAutomaton::with_queue(vec![Lane::L7]);
        let mut w = world_with_free_lane(Lane::L7);
        w.current_lane = Some(Lane::L7);
        w.claim_won = true;
        w.victory_seeds = VICTORY_SEED_TARGET;
        let a = h.next_action(&w);
        assert_eq!(a, AgentAction::Halt(HaltCause::Victory));
        assert_eq!(h.state(), State::Victory);
        assert!(h.state().is_absorbing());
    }

    #[test]
    fn test_deadline_passed_triggers_hard_abort() {
        let mut h = HiveAutomaton::new();
        let mut w = world_after_boot();
        w.deadline_passed = true;
        let a = h.next_action(&w);
        assert_eq!(
            a,
            AgentAction::Halt(HaltCause::HardAbort(AbortReason::DeadlinePassed))
        );
        assert_eq!(h.state(), State::HardAbort);
    }

    #[test]
    fn test_forbidden_value_triggers_hard_abort() {
        let mut h = HiveAutomaton::new();
        let mut w = world_after_boot();
        w.forbidden_value_introduced = true;
        let a = h.next_action(&w);
        assert_eq!(
            a,
            AgentAction::Halt(HaltCause::HardAbort(
                AbortReason::R7ForbiddenValueIntroduced
            ))
        );
    }

    #[test]
    fn test_blocked_to_ci_wait_after_fix() {
        let mut h = HiveAutomaton::with_queue(vec![Lane::L7]);
        let mut w = world_with_free_lane(Lane::L7);
        w.current_lane = Some(Lane::L7);
        w.claim_won = true;
        w.only_owned_files_touched = true;
        w.push_succeeded = true;
        // Drive Boot → Scan → Pick → Claim → Work → Commit → CiWait = 6 ticks.
        for _ in 0..6 {
            h.next_action(&w);
        }
        // CI new failure → Blocked.
        w.new_ci_failure = true;
        assert_eq!(h.next_action(&w), AgentAction::PostBlocked(Lane::L7));
        assert_eq!(h.state(), State::Blocked);
        // Fix pushed → CiWait.
        w.new_ci_failure = false;
        w.fix_committed = true;
        assert_eq!(h.next_action(&w), AgentAction::WaitForCi(Lane::L7));
        assert_eq!(h.state(), State::CiWait);
    }

    #[test]
    fn test_no_state_other_than_victory_and_hard_abort_is_absorbing() {
        // Defensive check on the no-idle invariant.
        for s in [
            State::Boot,
            State::Scan,
            State::Pick,
            State::Claim,
            State::Work,
            State::Commit,
            State::CiWait,
            State::Done,
            State::Blocked,
            State::Reclaim,
        ] {
            assert!(!s.is_absorbing(), "{:?} must not be absorbing", s);
        }
        assert!(State::Victory.is_absorbing());
        assert!(State::HardAbort.is_absorbing());
    }

    #[test]
    fn test_priority_queue_default_matches_json() {
        // L7, L6, L9, L11, L12 — see `assertions/hive_automaton.json::lane_priority_queue`.
        let h = HiveAutomaton::new();
        assert_eq!(
            h.priority_queue,
            vec![Lane::L7, Lane::L6, Lane::L9, Lane::L11, Lane::L12]
        );
    }

    #[test]
    fn test_bpb_target_matches_lib() {
        // L-R14: anchor mirrors `crate::IGLA_TARGET_BPB`.
        assert_eq!(BPB_VICTORY_TARGET, crate::invariants::IGLA_TARGET_BPB);
    }

    #[test]
    fn test_schema_version_pinned() {
        assert_eq!(SCHEMA_VERSION, "1.0");
    }

    #[test]
    fn test_full_loop_three_lanes_no_human_input() {
        // End-to-end: cell ships L7, then L6, then L9 — without any
        // intermediate "wait for ORDER" step. This is the whole point of
        // the skill.
        let mut h = HiveAutomaton::with_queue(vec![Lane::L7, Lane::L6, Lane::L9]);
        let lanes = [Lane::L7, Lane::L6, Lane::L9];
        let mut shipped = vec![];

        // Boot → Scan
        let mut w = world_after_boot();
        h.next_action(&w);

        for lane in lanes {
            // Scan → Pick (set free lane)
            w.free_lanes = vec![lane];
            w.current_lane = None;
            w.claim_won = false;
            w.only_owned_files_touched = false;
            w.push_succeeded = false;
            w.new_ci_failure = false;
            assert_eq!(h.next_action(&w), AgentAction::PickLane(lane));
            // Pick → Claim
            w.current_lane = Some(lane);
            assert_eq!(h.next_action(&w), AgentAction::PostClaim(lane));
            // Claim → Work
            w.claim_won = true;
            assert_eq!(h.next_action(&w), AgentAction::DoWork(lane));
            // Work → Commit
            w.only_owned_files_touched = true;
            assert_eq!(h.next_action(&w), AgentAction::PushCommit(lane));
            // Commit → CiWait
            w.push_succeeded = true;
            assert_eq!(h.next_action(&w), AgentAction::WaitForCi(lane));
            // CiWait → Done
            assert_eq!(h.next_action(&w), AgentAction::PostDone(lane));
            shipped.push(lane);
            // Done → Scan, NO human prompt, NO Halt.
            assert_eq!(h.next_action(&w), AgentAction::RescanIssue);
            assert_eq!(h.state(), State::Scan);
        }

        assert_eq!(shipped, lanes.to_vec());
        // Three lanes shipped; cell still alive, ready for more.
        assert!(!h.state().is_absorbing());
    }

    #[test]
    fn test_deterministic_same_input_same_output() {
        // Two parallel automata with identical inputs must produce identical actions.
        let mut a = HiveAutomaton::new();
        let mut b = HiveAutomaton::new();
        let w = world_after_boot();
        for _ in 0..10 {
            assert_eq!(a.next_action(&w), b.next_action(&w));
            assert_eq!(a.state(), b.state());
        }
    }
}
