# Training Flow v2 — Gate-2 Decomposed Plan

> Status: draft proposal. Issue: [#24](https://github.com/gHashTag/trios-trainer-igla/issues/24).
> Anchor: `phi^2 + phi^-2 = 3` ([Zenodo 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)).
> Companion CLI: [`tri railway`](https://github.com/gHashTag/t27/pull/544) ONE SHOT.

## TL;DR

The champion sits at **BPB=2.2393** (sha `2446855`, seed 43, step 27000). Gate-2 demands **BPB<1.85 on 3 seeds (43, 44, 45) with step >= 4000** before deadline `2026-04-30 23:59 UTC`. The current single-config / single-optimizer flow has not closed the 0.39 BPB gap on any branch in this repo.

This plan decomposes the chase into **6 phases** (P0..P5), each with one falsifiable hypothesis, one exit criterion, and one owner. The plan combines four 2025 ablation-validated levers — **Muon optimizer**, **muP hyper-parameter transfer**, **Schedule-Free AdamW + WSD**, and **post-hoc EMA** — under one R5-honest ledger contract.

```
P0 Audit  ->  P1 OptLab  ->  P2 muP Transfer  ->  P3 SF+WSD  ->  P4 MultiObj+EMA  ->  P5 Gate-2 Push
   audit       muon vs adam    8M -> 24M HPs       schedule-free    JEPA + NCA + EMA   3 seeds, ledger
```

Every phase MUST satisfy the standing rules:

- **R5** — no DONE without merged PR + green CI + ledger row.
- **R7** — every emit carries `BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>`.
- **R8** — ledger row only valid for `step >= 4000`.
- **R9** — embargo (`assertions/embargo.txt`) checked before any `ledger::emit_row`.

## Why these levers (2025 evidence)

| Lever | Reported gain on small LM | Source |
|---|---|---|
| **Muon (orthogonalized momentum)** | -2.88% relative train loss vs AdamW @ same compute; ~1/3 fewer steps to convergence on 1B; half the optimizer-state memory of AdamW | [Shah et al. 2025 (IMU-1)](https://arxiv.org/abs/2602.02522), [PredNext "Why try Muon"](https://prednext.com/en/blog/optimizer-muon-2025/) |
| **muP / Maximal Update Param** | Optimal LR found at 8M transfers verbatim to >=10x width; DiT-XL muP -> 2.9x faster convergence | [Cerebras muP Practitioner](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization), [Zheng et al. 2025 muP-DiT](https://arxiv.org/abs/2505.15270) |
| **Schedule-Free AdamW** | MLCommons 2024 AlgoPerf winning entry (Self-Tuning track); matches or beats heavily tuned cosine | [Defazio et al. 2024 (Meta AI)](https://ai.meta.com/research/publications/the-road-less-scheduled/), [Emergent Mind survey](https://www.emergentmind.com/topics/schedule-free-adamw) |
| **Cautious Weight Decay** | Additional -0.97% relative on top of NorMuon | [IMU-1 §Optim ablation](https://arxiv.org/html/2602.02522v1) |
| **WSD (Warmup-Stable-Decay)** | Decouples decay timing from total-step commitment; better anytime curves | [Wen et al. 2024 WSD](https://arxiv.org/abs/2404.06395) |
| **Post-hoc EMA of checkpoints** | Free generalization gain at zero training cost | [Sanyal et al. 2024](https://arxiv.org/abs/2411.18583) |

These levers are independent — additivity is the working hypothesis, falsified by P1/P3 ablations.

---

## P0 — Audit and Reproduce Champion

**Pre-conditions**: clean checkout of `main`, FineWeb mirrors at `/data/fineweb_{train,val}.bin`, `cargo test --release` green.

**Hypothesis**: `configs/champion.toml --seed 43` reproduces `BPB = 2.2393 +/- 0.01 @ step 27000` on a fresh machine.

**Tasks**

1. Re-run `tests/champion_reproduction.rs` with `--ignored`.
2. Capture wall-clock + memory profile in `assertions/baseline_profile.json`.
3. Snapshot HEAD SHA into `docs/audit/P0_seed43.md` (full commit, full triplet).
4. Diff `src/train_loop.rs` against `gHashTag/trios@2446855::trios-igla-trainer/src/train_loop.rs`. Document any drift in `docs/audit/P0_drift.md` -- drift is allowed only if accompanied by a passing diff-test.
5. Lock the floor: append `champion@<sha>` to `assertions/champion_lock.txt`.

**Exit criterion**: ledger emits `BPB=2.2393 +/- 0.01 @ step=27000 seed=43 sha=<HEAD7> jsonl_row=<L> gate_status=below_target_evidence` and the row passes R8 + R9.

**Falsification**: BPB drift > 0.05 -> bisect against `gHashTag/trios@2446855` before any other phase.

**Owner**: `repro-auditor`.

---

## P1 — Optimizer Lab (AdamW vs Muon vs Muon+CWD)

**Pre-conditions**: P0 ledger row exists.

**Hypothesis**: at the champion architecture (256 d, 2L, 4H), Muon with `eta_2D=0.0235, eta_1D=0.007, momentum=0.95` reduces final BPB by **>=0.05** vs AdamW at the same step budget.

**Tasks**

1. Add `src/optimizer/muon.rs` -- Newton-Schulz orthogonalization, 7 NS steps (Polar-Express constants).
2. Extend `OptimizerKind::Muon { eta_2d, eta_1d, momentum, ns_steps }` and `OptimizerKind::MuonCwd { ..., cwd_lambda }`.
3. New configs:
   - `configs/lab/p1-adamw.toml` (control)
   - `configs/lab/p1-muon.toml`
   - `configs/lab/p1-muon-cwd.toml`
4. Each config: 12K steps, seed 43 only (lab phase, NOT a Gate-2 row).
5. CI gate: `cargo test --release optimizer::muon::ortho_invariant -- --exact` -- assert post-NS update `||W^T W - I||_F <= 1e-3`.

**Exit criterion**: lab leaderboard `assertions/lab/p1_leaderboard.jsonl` written with at least 3 rows (one per optimizer); winner declared by `argmin(bpb_final)` with margin >= 0.05.

**Falsification**: Muon does not beat AdamW by >=0.05 on this corpus -> proceed with AdamW for P2 and document the null result in `docs/audit/P1_null.md` (do not pretend gain).

**Owner**: `optim-lab`.

---

## P2 — muP Transfer (8M -> 24M -> Gate-2 Width)

**Pre-conditions**: P1 winner pinned.

**Hypothesis**: at the muP-anchored LR found at the 8M proxy, the same scalar LR transfers to 24M and to the Gate-2 candidate (~70M with `d=384, 4L`) with **<= 5% degradation** vs an LR-swept baseline at the larger size.

**Tasks**

1. Add `src/mup.rs`:
   - input/output multiplier scaling
   - attention QK 1/d_head scaling
   - per-parameter-group LR scaling (`embedding_mult`, `output_mult`, `attn_mult`)
2. Configs: `configs/lab/p2-proxy-8m.toml`, `p2-proxy-24m.toml`, `p2-target-70m.toml`.
3. LR sweep on 8M proxy across `{1e-3, 2e-3, 4e-3, 8e-3, 16e-3}` -> pick `lr_star`.
4. Apply `lr_star` to 24M and 70M with NO further sweep -> measure BPB at 12K steps each.
5. Validate INV-8 (`lr in [1e-3, 1e-2]`) at every sweep point.

**Exit criterion**: `assertions/lab/p2_transfer.jsonl` shows the 70M run hits within 5% of an LR-swept baseline at 70M; muP-transfer log saved to `docs/audit/P2_mup.md` with the chosen `lr_star`.

**Falsification**: 70M shows >10% degradation vs swept baseline -> debug muP scaling factors (most likely the `attn_mult` or initialization scale) before P3.

**Owner**: `mup-prover`.

---

## P3 — Schedule-Free AdamW + WSD (or Muon equivalent)

**Pre-conditions**: P1 + P2 winners frozen.

**Hypothesis**: replacing the cosine `phi-schedule` with **Schedule-Free** (or WSD with decay tail = 20%) yields **>= 0.04 BPB** improvement at 30K steps **AND** a strictly better anytime curve (BPB_t monotone-dominates the cosine baseline for `t >= 8K`).

**Tasks**

1. Implement Schedule-Free interpolation in `src/optimizer.rs::schedule_free`:
   - `y_t = (1 - beta1) * z_t + beta1 * x_t`
   - mixing coeff `c_{t+1} = 1/(t+1)`
2. Implement WSD: warmup (1K), stable (24K), decay (5K cosine).
3. Configs:
   - `configs/lab/p3-cosine.toml` (control, current)
   - `configs/lab/p3-sf.toml`
   - `configs/lab/p3-wsd.toml`
4. Eval every 500 steps, dump full curve to `assertions/lab/p3_curves.jsonl`.
5. Report anytime metric: `area_under_bpb_curve` (lower is better) per config.

**Exit criterion**: winning schedule beats cosine by >= 0.04 BPB AND anytime AUC drop >= 5%.

**Falsification**: neither SF nor WSD strictly dominate cosine -> stick with cosine, document the null in `docs/audit/P3_null.md`. Do NOT carry false-positive results into P4.

**Owner**: `schedule-bench`.

---

## P4 — Multi-Objective + Post-hoc EMA

**Pre-conditions**: P3 winner frozen. `gate2-attempt.toml` weights as floor.

**Hypothesis**: weighted CE + JEPA + NCA loss with `(w_ce, w_jepa, w_nca)` searched on the 8M proxy AND post-hoc EMA over the last `N=10` checkpoints removes another **>= 0.03 BPB** at no extra training cost.

**Tasks**

1. `src/objective.rs` — already houses CE+JEPA+NCA; add per-loss gradient scaling option.
2. Sweep `(w_jepa, w_nca)` on 4 settings: `{(0.0, 0.0), (0.5, 0.0), (0.5, 0.1), (0.7, 0.15)}`.
3. Post-hoc EMA in `src/checkpoint.rs::ema_average` -- weighted by training step (later checkpoints heavier).
4. Config: `configs/lab/p4-objective.toml` (chosen weights) + `configs/lab/p4-ema.toml` (EMA sweep `N in {3, 5, 10, 20}`).
5. Exit if BPB delta > +0.02 -- i.e. EMA may not regress.

**Exit criterion**: `assertions/lab/p4_objective.jsonl` shows >= 0.03 BPB drop, no row regresses below champion floor.

**Falsification**: EMA regresses BPB on >=2 of 4 settings -> drop EMA from the Gate-2 plan.

**Owner**: `objective-jeweller`.

---

## P5 — Gate-2 Push (3-Seed ONE SHOT)

**Pre-conditions**: P0..P4 ledger rows merged on `main`. `configs/gate2-final.toml` baked from the P1+P2+P3+P4 winners.

**Hypothesis**: with the P1..P4 winners stacked, all three seeds in `{43, 44, 45}` yield `BPB < 1.85` at `step >= 4000` before `2026-04-30 23:59 UTC`.

**Tasks**

1. Pin `configs/gate2-final.toml` (winning optimizer + LR + schedule + objective weights + EMA params).
2. Run the [`tri railway`](https://github.com/gHashTag/t27/pull/544) ONE SHOT (`up --confirm`) -- print the GraphQL bodies.
3. Operator POSTs to Railway; three services come up: `trainer-seed-43/44/45`.
4. Each service emits R7 triplets every 500 steps.
5. `assertions/seed_results.jsonl` accumulates rows; `tri railway gate2` reports verdict.
6. Stop condition: 3 distinct seeds with `BPB < 1.85 AND step >= 4000` OR deadline hit.

**Exit criterion**: 3 ledger rows with `gate_status="victory_candidate"` AND merged `feat: Gate-2 victory` PR. R5 honesty gate.

**Falsification**: deadline hit without quorum -> publish the post-mortem in `docs/audit/P5_postmortem.md`. Champion floor (2.2393) remains the public number; no DONE is claimed.

**Owner**: `gate2-pilot`.

---

## Decision Matrix (pre-registered)

This is the falsification table. Filled in only by future PRs after each phase closes.

| Phase | Hypothesis margin | Outcome (BPB delta) | Decision | PR |
|---|---|---|---|---|
| P0 | reproduce 2.2393 +/- 0.01 | _pending_ | _pending_ | _pending_ |
| P1 | Muon - AdamW <= -0.05 | _pending_ | _pending_ | _pending_ |
| P2 | muP transfer < 5% deg | _pending_ | _pending_ | _pending_ |
| P3 | SF/WSD - cosine <= -0.04 | _pending_ | _pending_ | _pending_ |
| P4 | objective+EMA <= -0.03 | _pending_ | _pending_ | _pending_ |
| P5 | 3 seeds < 1.85 | _pending_ | _pending_ | _pending_ |

## Lab vs Ledger discipline (R7/R8 hygiene)

- **Lab rows** live under `assertions/lab/*.jsonl`. They are NOT R7-validated triplets and MAY have step < 4000. They are for local decisions only and never roll up to Gate-2.
- **Ledger rows** live in `assertions/seed_results.jsonl`. They MUST satisfy R7 + R8 + R9. Only P0 and P5 are allowed to write here.
- A phase that wants to "promote" a lab row to a ledger row MUST run a full P5-style 3-seed verification.

## Concrete code touchpoints

| Phase | New files | Modified |
|---|---|---|
| P0 | `docs/audit/P0_seed43.md`, `assertions/baseline_profile.json`, `assertions/champion_lock.txt` | `tests/champion_reproduction.rs` |
| P1 | `src/optimizer/muon.rs`, `configs/lab/p1-*.toml` | `src/optimizer.rs`, `src/config.rs` |
| P2 | `src/mup.rs`, `configs/lab/p2-*.toml` | `src/model.rs` (per-group LR), `src/optimizer.rs` |
| P3 | _none_ | `src/optimizer.rs::schedule_free`, `src/optimizer.rs::wsd_lr` |
| P4 | `configs/lab/p4-*.toml` | `src/objective.rs`, `src/checkpoint.rs::ema_average` |
| P5 | `configs/gate2-final.toml`, `docs/audit/P5_*.md` | _none, by design_ |

## How to start P0 today

```bash
git checkout -b feat/p0-audit-25 main
cargo test --release reproduce_champion -- --ignored
git diff --no-index gHashTag/trios@2446855::trios-igla-trainer/src/train_loop.rs src/train_loop.rs > docs/audit/P0_drift.md
# run, capture, commit, R5-honest report
```

Then submit a PR titled `feat(p0): audit + champion reproduction (closes #N)`.

## Anchor

Mathematical foundation: `phi^2 + phi^-2 = 3` ([Zenodo 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)). Every phase MUST preserve this invariant in any modified numeric or scheduling code.
