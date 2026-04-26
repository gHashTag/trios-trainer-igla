# trios-trainer-igla — portable training image.
# Repo:    https://github.com/gHashTag/trios-trainer-igla
# Build:   docker build -t ghcr.io/ghashtag/trios-trainer-igla:latest .
# Run:     docker run --rm -e TRIOS_SEED=43 -v $PWD/assertions:/work/assertions \
#              ghcr.io/ghashtag/trios-trainer-igla:latest

# ---------- builder ----------
FROM rust:1.86-slim AS builder

# git required at runtime for ledger row push; install in builder for cargo,
# final stage receives only the static binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cargo build --release --bin trios-train

# ---------- runtime ----------
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
ENV TRIOS_TARGET_BPB=1.85
ENV TRIOS_LEDGER_PUSH=0

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/trios-train"]
CMD ["--config", "/configs/gate2-attempt.toml"]
