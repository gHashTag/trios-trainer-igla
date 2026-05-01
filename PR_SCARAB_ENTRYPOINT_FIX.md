# PR: Fix scarab entrypoint crash-loop

## Root Cause

entrypoint.rs has a whitelist that EXCLUDES `scarab`:

```rust
// entrypoint.rs:31
if !matches!(trainer.as_str(), "trios-train" | "gf16_test" | "ngram_train_gf16") {
    eprintln!("[entrypoint] {trainer:?} is not in the allowed set {{trios-train, gf16_test, ngram_train_gf16}}");
    std::process::exit(2);
}
```

**scarab-acc0 service (acc3 project)** uses main Dockerfile:
- `ENTRYPOINT ["/usr/local/bin/entrypoint"]`
- entrypoint checks whitelist → rejects scarab → CRASH LOOP

**Evidence:**
- Service 29aaf272 logs show: "[entrypoint] scarab not in allowed set"
- Container restarts, crashes, restarts...

## Solution Options

### Option A: Quick Fix (Whitelist)

Add `scarab` to whitelist:

```rust
// entrypoint.rs:31
if !matches!(trainer.as_str(), "trios-train" | "gf16_test" | "ngram_train_gf16" | "scarab") {
    eprintln!("[entrypoint] {trainer:?} is not in the allowed set {{trios-train, gf16_test, ngram_train_gf16, scarab}}");
    std::process::exit(2);
}
```

**Impact:** Minimal, 1-line change.
**Risk:** None, just allows scarab to run.

---

### Option B: Dockerfile Path Override (Recommended)

Configure Railway service to use `Dockerfile.scarab`:

Railway dashboard → acc0 → scarab-acc0 → Settings → Build → Dockerfile Path: `Dockerfile.scarab`

This bypasses entrypoint entirely since scarab is started directly via `CMD ["/usr/local/bin/scarab"]`.

**Impact:** No code changes needed, just config.
**Risk:** None, proper architecture (entrypoint for other services, direct binary for scarab).

---

### Option C: Clean PR (Ultimate)

1. **Remove Dockerfile.scarab entirely**
2. **Build scarab binary into main Dockerfile**
3. **Add ENTRYPOINT for scarab** (or keep trios-train with whitelist)
4. **Remove entrypoint.rs whitelist check** (scarab is now first-class citizen)

```dockerfile
# Main Dockerfile - scarab built-in as ENTRYPOINT
FROM debian:bookworm-slim
# ... install dependencies ...

COPY --from=builder /build/target/release/scarab /usr/local/bin/scarab
COPY --from=builder /build/target/release/trios-train /usr/local/bin/trios-train

# Scarab as ENTRYPOINT, trios-train via CMD
ENTRYPOINT ["/usr/local/bin/scarab"]
CMD []
```

```rust
// entrypoint.rs - REMOVE whitelist check for scarab
// scarab is now first-class citizen, runs directly without entrypoint
```

**Impact:** Removes architectural debt, cleaner.
**Risk:** Medium, more files changed.

---

### Option D: Fix via TRIOS_TRAINER_BIN Env (Minimal)

Make entrypoint understand `TRIOS_TRAINER_BIN=scarab` as "run scarab binary directly" instead of exec:

```rust
// entrypoint.rs
match trainer.as_str() {
    "trios-train" | "gf16_test" | "ngram_train_gf16" => {
        // existing logic
    }
    "scarab" => {
        // Bypass entrypoint, run scarab directly
        let scarab_path = "/usr/local/bin/scarab";
        let mut cmd = Command::new(scarab_path);
        // Forward all env vars but DO NOT add --seed/--steps args
        // scarab manages its own args from config_json
        // scarab itself will spawn trios-train with correct args
        cmd.exec();
    }
    _ => {
        // reject others
    }
}
```

**Impact:** Minimal change, no Dockerfile changes needed.
**Risk:** None, correct semantics (scarab runs scarab, not entrypoint for scarab).

---

## Recommended Path

1. **Short-term (Option A):** Patch whitelist + force-rebuild → verify probe 2048
2. **Long-term (Option B):** Configure Dockerfile Path for scarab-* services → redeploy

## Testing

After fix, verify with:

```bash
# Insert probe
INSERT INTO strategy_queue (...) VALUES ('IGLA-PROBE-VERIFY-FIX-seed1597', ...);

# After 5 min, check
SELECT id, canon_name, claimed_at, final_step, final_bpb
FROM strategy_queue WHERE canon_name = 'IGLA-PROBE-VERIFY-FIX-seed1597';

# Verify bpb_samples
SELECT COUNT(*) FROM bpb_samples WHERE canon_name = 'IGLA-PROBE-VERIFY-FIX-seed1597';
```

Expected results:
- claimed_at NOT NULL
- final_step >= 1
- final_bpb NOT NULL
- bpb_samples COUNT >= 1
