# ADR-0001 — Repo boundaries: `trios-railway` vs `trios-trainer-igla`

- **Status:** Accepted
- **Date:** 2026-04-28
- **Authors:** gHashTag (operator), tri-gardener team
- **Anchor:** `phi^2 + phi^-2 = 3 · TRINITY · NEVER STOP`

## Context

The IGLA marathon (Gate-2 BPB < 1.85, Gate-3 BPB < 1.5) ships through two
independent repositories. Earlier in the race the boundary between them
was implicit, which led to a near-miss: an operator considered deleting
`trios-trainer-igla` while reasoning about the gardener layout. This
ADR makes the boundary explicit so it survives sleep, shift handoff, and
LLM context loss.

## Decision

There are exactly two repositories on the IGLA critical path. Each has
one job. They never depend on each other's source — only on shipped
artefacts (Docker images, GHCR tags, JSONL ledgers).

### `gHashTag/trios-railway` — control plane

- Owns the typed Railway GraphQL client (`crates/trios-railway-core`).
- Owns `bin/tri-railway` (operator CLI: `plan9`, `service deploy`, etc.).
- Owns `bin/tri-gardener` (autonomous orchestrator, ASHA rungs,
  plateau detector, queue picker, `gardener_runs` ledger).
- Owns the Neon `railway_audit_*` and `gardener_runs` schemas.
- Never contains trainer model code, training scripts, or weights.
- May depend on the **artefact** `ghcr.io/ghashtag/trios-trainer-igla:*`
  by tag/digest only.

### `gHashTag/trios-trainer-igla` — model plane

- Owns the trainer (`crates/trios-trainer-igla`, attention/JEPA/etc.).
- Owns `Dockerfile` that produces `ghcr.io/ghashtag/trios-trainer-igla`.
- Owns `assertions/seed_results.jsonl` and BPB-emitting code paths.
- Never imports `trios-railway-core` or any control-plane crate.
- May read its own deploy environment variables but does not call
  Railway GraphQL itself — that is the control plane's job.

### Bidirectional contract

- The trainer image's environment is set by the control plane via
  `variableUpsert`. Variable names are part of the contract and
  versioned in `docs/contracts/trainer-env.md` (control plane).
- BPB telemetry flows trainer → Neon `bpb_samples` (writer in trainer)
  and Neon `bpb_samples` → gardener (reader in control plane). The
  schema is owned by the control plane.

## Consequences

- The trainer repo is **NEVER** deleted, archived, or renamed as part
  of any control-plane refactor. If a control-plane task description
  mentions touching the trainer repo for anything other than the
  trainer's own code or this ADR, the operator stops and re-asks.
- Control-plane crates do not import trainer code. PRs that try are
  rejected.
- Trainer crates do not import `trios-railway-core`. PRs that try are
  rejected.
- A change to the trainer image schema (env vars, ports, ledger paths)
  requires a paired commit: contract update in `trios-railway` and
  the actual change in `trios-trainer-igla`, both referenced in the
  PR body.

## References

- Tracker: [`gHashTag/trios-railway#43`](https://github.com/gHashTag/trios-railway/issues/43)
- Gardener spec: [`gHashTag/trios-railway#49`](https://github.com/gHashTag/trios-railway/issues/49)
- Gardener PR-1: [`gHashTag/trios-railway#50`](https://github.com/gHashTag/trios-railway/pull/50)
- Race: [`gHashTag/trios#143`](https://github.com/gHashTag/trios/issues/143)
- φ-physics foundation: [`gHashTag/trios#329`](https://github.com/gHashTag/trios/pull/329)

`phi^2 + phi^-2 = 3 · TRINITY · NEVER STOP`
