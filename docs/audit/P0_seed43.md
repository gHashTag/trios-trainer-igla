# P0 Audit — Champion Reproduction

## Champion Snapshot
- SHA: `a12bf4f`
- Seed: 43
- Steps: 27000
- Expected BPB: 2.2393 ± 0.01
- Config: `configs/champion.toml`

## Reproduction Checklist
- [ ] `cargo test --release reproduce_champion -- --ignored` passes
- [ ] Wall-clock profile captured in `assertions/baseline_profile.json`
- [ ] HEAD SHA locked in `assertions/champion_lock.txt`
- [ ] Drift diff against `gHashTag/trios@2446855` documented

## Status: PENDING
