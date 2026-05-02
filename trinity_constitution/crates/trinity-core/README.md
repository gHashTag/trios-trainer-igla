# trinity-core — L0+L1: PhD Invariants

**Constitutional mandate (Law 5):** All invariants encoded as `const fn`.

## Modules

- `invariants.rs` — INV-1..INV-9 as compile-time constants
- `phi.rs` — φ, φ², φ⁻², Fibonacci sequence
- `bpb.rs` — honest BPB calculation (nats → bits)

## Key Constants

```rust
PHI = 1.618...                // Golden ratio
PHI_SQUARED + PHI_INVERSE_SQUARED = 3  // Identity
INV_1_MIN_STEPS = 5000       // Minimum training steps
INV_6_FLOOR_BPB = 2.0        // "Worse than random" threshold
```

## Usage

```rust
use trinity_core::{PHI, calculate_bpb};
use trinity_core::invariants::{is_valid_seed, is_valid_steps_budget};

assert!((PHI * PHI + 1.0 / (PHI * PHI) - 3.0).abs() < 1e-10);
assert!(is_valid_seed(1597));  // Valid Fibonacci seed
let bpb = calculate_bpb(2.0);  // Convert nats to bits
```

## Tests

```bash
cargo test -p trinity-core
```

All tests must pass before any PR using this crate.
