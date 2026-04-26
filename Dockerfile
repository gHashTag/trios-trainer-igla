# trios-trainer — portable training image.
# Build:   docker build -t ghcr.io/ghashtag/trios-trainer-igla:latest .
# Run:     docker run --rm -e TRIOS_SEED=43 -v $PWD/assertions:/work/assertions ghcr.io/ghashtag/trios-trainer-igla

# ---------- builder ----------
FROM rust:1.86-slim AS builder

# git is required at runtime for ledger row push; we install in builder for cargo + final stage gets binary only
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# Copy the entire workspace — Cargo.toml + Cargo.lock + crates/*
COPY . .

RUN cargo build --release --bin trios-train -p trios-trainer

# ---------- runtime ----------
# distroless is too minimal (no git for ledger push). Use slim Debian.
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
