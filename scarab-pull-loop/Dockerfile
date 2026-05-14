# SOVEREIGN SCARAB image — replaces Dockerfile.scarab once R6 migration finishes.
# Build context: repo root (../).
#
# Anchor: phi^2 + phi^-2 = 3 · TRINITY · DEFENSE 2026-06-15
FROM rust:1.95-bookworm AS build
WORKDIR /src
# 1) build the trainer (existing crate at repo root)
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY migration ./migration
COPY assertions ./assertions
RUN cargo build --release --bin trios-train

# 2) build the pull-loop (isolated workspace under scarab-pull-loop/)
COPY scarab-pull-loop ./scarab-pull-loop
RUN cd scarab-pull-loop && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/target/release/trios-train         /app/trios-train
COPY --from=build /src/scarab-pull-loop/target/release/scarab /app/scarab
# Sample training data (tiny synthetic — production overrides via env)
COPY assertions /app/assertions
ENV TRAINER_BIN=/app/trios-train \
    TRAIN_DATA=/app/assertions/train.txt \
    VAL_DATA=/app/assertions/val.txt \
    POLL_SEC=30
CMD ["/app/scarab"]
