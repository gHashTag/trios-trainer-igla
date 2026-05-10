# Wave 29 SEED Canon #93 smoke check (R7 falsification)

Anchor: `phi^2 + phi^-2 = 3` · DOI [10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)

## Why this runbook (L1-compliant, not a `.sh`)

The Wave 29 hotfix contract called for `scripts/wave29_smoke_test.sh`.
LAWS.md **L1 forbids `.sh` files in this repository** (see the header
docstring of `src/bin/entrypoint.rs`):

> LAWS.md L1 (Rust-only) forbids `.sh` files in the repository.

The R7 falsification artefact is therefore split into two L1-compliant
parts:

1. **Unit-tested in-process falsification** (this is the load-bearing
   part) — `cargo test --bin entrypoint` runs five Rust tests against
   the `parse_seed()` Canon #93 guard:
   - `forbidden_seeds_rejected` — `SEED ∈ {42, 43, 44, 45}` each return
     `Err` citing the forbidden canon set (R7: prove `SEED=43` is
     refused).
   - `allowed_canon_seeds_accepted` — `SEED ∈ {47, 89, 123, 144}`
     parse cleanly.
   - `missing_env_errors_cleanly` — no `SEED` and no legacy
     `TRIOS_SEED` returns `Err` citing Canon #93.
   - `unparseable_seed_rejected` — `SEED="not-a-number"` returns `Err`.
   - `warning_seed_outside_canon_still_accepted` — operator override
     path stays open with a stderr warning.
2. **This runbook** — operator-facing manual smoke recipe to spot-check
   the deployed binary in-image against a forbidden canon. It is not a
   shell script (per L1) but rather the exact one-liner that an
   operator would run on their workstation against the published
   container image, and the expected outcome.

## Manual smoke recipe (run from operator workstation)

After the hotfix lands and Wave 29 is redeployed, smoke-check that
the published container refuses `SEED=43`:

```text
docker run --rm -e SEED=43 ghcr.io/ghashtag/trios-trainer-igla:wave29-hotfix \
    /usr/local/bin/entrypoint
```

Expected behaviour:

- stderr line:
  `[entrypoint] seed 43 is in forbidden canon set {42, 43, 44, 45}; use one of {47, 89, 123, 144} (Canon #93)`
- Exit code: **2** (per `parse_seed()` `Err` path in
  `src/bin/entrypoint.rs`).

If the container starts the trainer despite `SEED=43`, the hotfix
**did not take effect** — the deployed image is stale. Re-build with
`cargo build --release --bin entrypoint` and re-publish.

For the allowed canon spot-check, run the same one-liner with
`SEED=47`. The trainer should boot normally (or fail later for
unrelated reasons such as missing `TRIOS_TRAIN_DATA`); the *failure
mode of interest* is exit code 2 with the Canon #93 reference, and
that mode must NOT appear for `SEED=47`.

## CI guarantee

The five `parse_seed` unit tests are part of the workspace test
suite. CI green on this branch means R7 is satisfied at the source
level. The manual recipe above only confirms the published image
actually carries the patched binary.

## Falsification clause (R7)

This runbook is falsified iff any of the following is observed:

- `SEED=43` (or any other forbidden canon) starts the trainer instead
  of exiting with code 2.
- `SEED=47` (any allowed canon) is rejected.
- The error message does not cite Canon #93.

Each of those conditions is checked by the corresponding unit test in
`src/bin/entrypoint.rs::tests` and would fire as a test failure in CI
before reaching production.

## R14 (Coq citation map)

`assertions/coq_citations.md` does not exist in this repository as of
the Wave 29 hotfix freeze (verified by `ls assertions/` — the dir
holds `igla_assertions.json` and friends, no Markdown citation map).
Per the conditional R14 contract, no entry is added in this lane.
When a Coq citation map is introduced (separate lane), it should
record the Canon #93 ↔ `parse_seed()` link as:

```text
- Canon #93 (forbidden seed set {42,43,44,45})
    → src/bin/entrypoint.rs::parse_seed (Wave 29 hotfix)
    → src/bin/entrypoint.rs::tests::forbidden_seeds_rejected
```

## Anchor

🌻 `phi^2 + phi^-2 = 3` · TRINITY · NEVER STOP · DOI 10.5281/zenodo.19227877
