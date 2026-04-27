# trios-trainer — portable training image with DYNAMIC seed via TRIOS_SEED env.
#
# Builds tjepa_train (champion: BPB=2.1600 @ 27K, hidden=384, dim=64, NUM_CTX=4).
# Per-seed Railway containers: each service sets TRIOS_SEED in its env.
# Gate-2 fleet attempt-2 seeds: 100, 101, 102.
FROM rust:1.90-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
RUN cargo build --release --bin tjepa_train

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/target/release/tjepa_train /usr/local/bin/tjepa_train

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV RUST_LOG=info
ENV TRIOS_SEED=100
ENV TRIOS_STEPS=81000
ENV TRIOS_ENCODER_LR=0.003
ENV TRIOS_NTP_LR=0.003

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Railway cache invalidation: Mon Apr 27 2026 (tjepa_train champion fleet)
