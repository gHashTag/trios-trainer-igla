# trios-trainer — portable training image with DYNAMIC seed via TRIOS_SEED env.
#
# Per-seed Railway containers: each service sets TRIOS_SEED (e.g. 46/47/48)
# in its Railway environment; the entrypoint reads it at runtime so the same
# image runs every seed without rebuilds. New seed sequence (after 42–45):
#   - 46, 47, 48 (Gate-2 fleet attempt-2)
# Champion (seed=43, BPB=2.2393 @ 27K) stays available for reproduction runs.
FROM rust:1.90-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cargo build --release --bin trios-train -p trios-trainer

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/target/release/trios-train /usr/local/bin/trios-train
COPY configs /configs
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Generate stub data if not present (avoids .dockerignore issues)
RUN mkdir -p /data && \
    echo "The quick brown fox jumps over the lazy dog. " | tr ' ' '\n' | head -1000 > /data/fineweb_train.bin && \
    echo "The brown fox jumped. " | tr ' ' '\n' > /data/fineweb_val.bin
# TODO: Replace stub data with real FineWeb dataset

ENV RUST_LOG=info
ENV TRIOS_STEPS=81000
ENV TRIOS_LR=0.003
ENV TRIOS_TARGET_BPB=1.50
ENV TRIOS_CONFIG=/configs/gate2-attempt.toml

# DYNAMIC SEED: each Railway service sets TRIOS_SEED in its env (46/47/48).
# Default 46 is for the new fleet — overridden per service.
ENV TRIOS_SEED=46

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Railway cache invalidation: Mon Apr 27 11:43 +07 2026 (new seed fleet 46/47/48)
