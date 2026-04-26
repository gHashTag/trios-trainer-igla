# trios-trainer — portable training image.
# Build:   docker build -t ghcr.io/ghashtag/trios-trainer-igla:latest .
# Run:     docker run --rm -e TRIOS_SEED=43 -v $PWD/assertions:/work/assertions ghcr.io/ghashtag/trios-trainer-igla

# ---------- builder ----------
# rust:1.90-slim-bookworm — bookworm base so libc matches the runtime stage.
# (`is_multiple_of` for unsigned ints stabilized in 1.87.)
FROM rust:1.90-slim-bookworm AS builder

# git is required at runtime for ledger row push; we install in builder for cargo + final stage gets binary only
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# Copy the entire workspace — Cargo.toml + Cargo.lock + crates/*
COPY . .

RUN cargo build --release --bin trios-train -p trios-trainer

# ---------- runtime ----------
# Same Debian release as the builder — GLIBC must match.
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/target/release/trios-train /usr/local/bin/trios-train
COPY configs /configs

ENV RUST_LOG=info
ENV TRIOS_CONFIG=/configs/gate2-attempt.toml
ENV TRIOS_SEED=43
ENV TRIOS_TARGET_BPB=1.50
ENV TRIOS_LEDGER_PUSH=0

ENTRYPOINT ["/usr/local/bin/trios-train"]
CMD ["--config", "/configs/gate2-attempt.toml"]
