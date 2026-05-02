# trinity-trainer — L1: Pure Train Function

**Constitutional mandate (Law 3):** Pure function, no side effects.

## API

```rust
pub fn train(config: Config) -> Result<RunOutcome, TrainError>
```

## Config

```rust
pub struct Config {
    pub seed: u64,
    pub hidden: u32,
    pub lr: f32,
    pub steps: usize,
    pub format: String,
    pub corpus: String,
    pub train_path: String,
    pub val_path: String,
}
```

## RunOutcome

```rust
pub struct RunOutcome {
    pub status: TrainStatus,      // Success / Failed / Timeout
    pub final_step: usize,
    pub final_bpb: Option<f32>,
    pub bpb_curve: Vec<BpbPoint>, // [{step, bpb}, ...]
    pub error: Option<String>,
}
```

## Usage

```rust
use trinity_trainer::{train, Config};
use trinity_experiments::ExperimentConfig;

let config = Config::from(&experiment_config);
let outcome = train(config)?;

match outcome.status {
    TrainStatus::Success => println!("BPB: {:?}", outcome.final_bpb),
    TrainStatus::Failed => println!("Error: {:?}", outcome.error),
}
```

## Invariants

- **Law 3 (Rust-only):** All logic in Rust, no I/O
- **Law 4 (immutable):** Pure function, no external state mutation
- **Law 5 (φ-physics):** All values validated against INV-1..INV-9

## Tests

```bash
cargo test -p trinity-trainer
```
