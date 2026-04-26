# trios-trainer — portable training image.
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
COPY data /data
# TODO: Replace stub data with real FineWeb dataset

ENV RUST_LOG=info
ENV TRIOS_SEED=43
ENV TRIOS_STEPS=81000
ENV TRIOS_LR=0.003
ENV TRIOS_TARGET_BPB=1.50

ENV TRIOS_CONFIG=/configs/gate2-attempt.toml

ENTRYPOINT ["/usr/local/bin/trios-train"]
CMD ["--config", "/configs/gate2-attempt.toml"]
