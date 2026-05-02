# 🌻 Trinity — IGLA RACE Training System

**Constitutional monorepo with O(1) design — touch once, encode invariant, exit.**

## 📐 Architecture

7 crates, 7 layers, 7×O(1):

| Crate | Layer | Status |
|-------|-------|--------|
| `trinity-core` | L0+L1 | ✅ PR-O2 — invariants, φ-physics, BPB |
| `trinity-experiments` | L0 | ✅ PR-O3 — single SoT table, repo |
| `trinity-trainer` | L1 | ✅ PR-O4 — pure train function |
| `trinity-runner` | L2 | ✅ PR-O5 — claim loop |
| `trinity-orchestrator` | L3+L4+L5 | ⏳ PR-O6 — declarative deploy |
| `trinity-gardener` | L6 | ⏳ PR-O7 — heartbeat + ratify |

## 🚀 Quick Start

```bash
# Install Rust (if not already)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/gHashTag/trinity
cd trinity
cargo build --release

# Run runner (local test)
NEON_DATABASE_URL="postgres://..." cargo run --release --bin trinity-runner

# Run gardener (health endpoint)
PORT=8080 cargo run --release --bin trinity-gardener
```

## ⚖ Constitution

See [CONSTITUTION.md](CONSTITUTION.md) for the immutable laws:
- Law O(1) — each layer touched exactly once
- Law 1 (R5-honest) — no claim without verified source
- Law 2 (single SoT) — strategy_experiments only table
- Law 3 (Rust-only) — all layers in Rust
- Law 4 (immutable) — append-only, no UPDATE for history
- Law 5 (φ-physics) — INV-1..INV-9 encoded

## 📜 Roadmap

- [x] PR-O1 — Constitution + repo skeleton
- [x] PR-O2 — trinity-core invariants
- [x] PR-O3 — trinity-experiments DB layer
- [x] PR-O4 — trinity-trainer pure function
- [x] PR-O5 — trinity-runner claim loop
- [ ] PR-O6 — trinity-orchestrator declarative deploy
- [ ] PR-O7 — trinity-gardener heartbeat + ratify

## 🛡 Safety Guarantees

- **Idempotent claim** — `FOR UPDATE SKIP LOCKED` prevents double-claim
- **R5-honest** — final_bpb XOR last_error on done/failed
- **Fail-loud** — NaN/infinity BPB returns None, panics on invariant violation
- **Self-healing** — Railway healthcheck → auto-restart on failure

## 🌻 Mantra

```
φ² + φ⁻² = 3 · TRINITY · O(1) FOREVER

Touch once.
Encode invariant.
Exit clean.
```
