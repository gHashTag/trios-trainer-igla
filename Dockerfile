# trios-trainer — portable training image with DYNAMIC seed via TRIOS_SEED env.
#
# Per-seed Railway containers: each service sets TRIOS_SEED (e.g. 100/101/102)
# in its Railway environment; the entrypoint reads it at runtime so the same
# image runs every seed without rebuilds. New seed sequence (after 42–45):
#   - 100, 101, 102 (Gate-2 fleet attempt-2)
# Champion (seed=43, BPB=2.2393 @ 27K) stays available for reproduction runs.
FROM rust:1.90-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
RUN cargo build --release --bin trios-train --bin hybrid_train -p trios-trainer

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/target/release/trios-train /usr/local/bin/trios-train
COPY --from=builder /build/target/release/hybrid_train /usr/local/bin/hybrid_train
COPY --from=builder /build/src/bin/railway_start.sh /usr/local/bin/railway_start.sh
RUN chmod +x /usr/local/bin/railway_start.sh

COPY configs /configs
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Generate stub data if not present (avoids .dockerignore issues)
RUN mkdir -p /data && \
    echo "The quick brown fox jumps over the lazy dog. " | tr ' ' '\n' | head -1000 > /data/fineweb_train.bin && \
    echo "The brown fox jumped. " | tr ' ' '\n' > /data/fineweb_val.bin
# TODO: Replace stub data with real FineWeb dataset

# tiny_shakespeare data for hybrid_train (champion reproduction path).
RUN mkdir -p /work/data && \
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > /work/data/tiny_shakespeare.txt && \
    head -c 100000 /work/data/tiny_shakespeare.txt > /work/data/tiny_shakespeare_val.txt

ENV RUST_LOG=info
ENV TRIOS_STEPS=81000
ENV TRIOS_LR=0.003
ENV TRIOS_TARGET_BPB=1.50
ENV TRIOS_CONFIG=/configs/gate2-attempt.toml

# Champion reproduction params (used by hybrid_train via railway_start.sh).
ENV TRIOS_HIDDEN=384
ENV TRIOS_ATTN_LAYERS=2
ENV TRIOS_OPTIMIZER=adamw
ENV TRIOS_EVAL_EVERY=1000

# DYNAMIC SEED: each Railway service sets TRIOS_SEED in its env (100/101/102).
# Default 100 is for the new fleet — overridden per service.
ENV TRIOS_SEED=100

# Entrypoint = dynamic seed wrapper for trios-train (Gate-2 attempt-2 fleet).
# For champion reproduction (hybrid_train + tiny_shakespeare), override
# Railway startCommand to /usr/local/bin/railway_start.sh.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Railway cache invalidation: Mon Apr 27 11:55 +07 2026 (seed fleet 100/101/102)
