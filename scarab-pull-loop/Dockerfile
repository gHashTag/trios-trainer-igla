# SOVEREIGN SCARAB v4 — multi-stage distroless image
# Anchor: phi^2 + phi^-2 = 3 · TRINITY · DEFENSE 2026-06-15

# ---------- Stage 1: builder ----------
FROM rust:1.85-slim-bookworm AS builder
WORKDIR /build

# Cache dependency compilation. Cargo.lock is generated on first build if
# absent (the crate's parent workspace ignores it), so we COPY it best-effort.
COPY Cargo.toml ./
COPY Cargo.lock* ./
RUN mkdir -p src && \
    echo 'fn main() {}' > src/main.rs && \
    echo '// dummy' > src/lib.rs && \
    cargo build --release --bin scarab && \
    rm -rf src \
           target/release/deps/scarab* \
           target/release/deps/libscarab_pull_loop* \
           target/release/deps/scarab_pull_loop* \
           target/release/libscarab_pull_loop* \
           target/release/scarab \
           target/release/.fingerprint/scarab-pull-loop-*

# Real sources
COPY src/ ./src/
# Force Cargo to detect changed sources (mtime is newer than cached fingerprint).
RUN cargo build --release --bin scarab && \
    strip target/release/scarab

# ---------- Stage 2: runtime (distroless · cc base for libpq-free TLS) -------
FROM gcr.io/distroless/cc-debian12:nonroot
WORKDIR /app

LABEL org.opencontainers.image.title="sovereign-scarab"
LABEL org.opencontainers.image.description="Eternal Rust agent — pulls strategy from ssot.scarab_strategy, runs trios-train, writes heartbeat + result. DB is control plane."
LABEL org.opencontainers.image.source="https://github.com/gHashTag/trios-trainer-igla"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL trinity.anchor="phi^2 + phi^-2 = 3"

COPY --from=builder /build/target/release/scarab /app/scarab

# Defaults — Railway will inject DATABASE_URL and RAILWAY_SERVICE_ID
ENV POLL_SEC=10
ENV GRACE_MS=10000
ENV LISTEN_NOTIFY=1
ENV TRAINER_BIN=/usr/local/bin/trios-train
ENV TRAIN_DATA=/data/train.txt
ENV VAL_DATA=/data/val.txt

USER nonroot
ENTRYPOINT ["/app/scarab"]
