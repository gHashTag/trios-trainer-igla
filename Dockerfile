# trios-trainer — portable training image with DYNAMIC seed via TRIOS_SEED env.
#
# Per-seed Railway containers: each service sets TRIOS_SEED (e.g. 100/101/102)
# in its Railway environment; the entrypoint reads it at runtime so the same
# image runs every seed without rebuilds. New seed sequence (after 42–45):
#   - 100, 101, 102 (Gate-2 fleet attempt-2)
# Champion (seed=43, BPB=2.2393 @ 27K) stays available for reproduction runs.
#
# Data is pulled at runtime by `trios-train` itself via ureq (no .sh helpers,
# R1: no Python/Bash data scripts in the image).
FROM rust:1.90-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
RUN cargo build --release --bin trios-train

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/target/release/trios-train /usr/local/bin/trios-train

COPY configs /configs
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV RUST_LOG=info
ENV TRIOS_STEPS=81000
ENV TRIOS_HIDDEN=828
ENV TRIOS_LR=0.003
ENV TRIOS_TARGET_BPB=1.50
ENV TRIOS_CONFIG=/configs/gate2-attempt.toml
ENV TRIOS_ATTN_LAYERS=2
ENV TRIOS_EVAL_EVERY=1000
ENV TRIOS_OPTIMIZER=adamw

# DYNAMIC SEED: each Railway service sets TRIOS_SEED in its env (100/101/102).
# Default 100 is for the new fleet — overridden per service.
ENV TRIOS_SEED=100

# Entrypoint = dynamic seed wrapper for trios-train (Gate-2 attempt-2 fleet).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Railway cache invalidation: Mon Apr 27 12:25 +07 2026 (seed fleet 100/101/102)
