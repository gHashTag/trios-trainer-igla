FROM rust:1.90-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
RUN cargo build --release --bin tjepa_train -p trios-trainer

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/target/release/tjepa_train /usr/local/bin/tjepa_train
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN mkdir -p /work/data && \
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > /work/data/tiny_shakespeare.txt && \
    head -c 100000 /work/data/tiny_shakespeare.txt > /work/data/tiny_shakespeare_val.txt

ENV RUST_LOG=info
ENV TRIOS_SEED=100
ENV TRIOS_STEPS=81000
ENV TRIOS_ENCODER_LR=0.003
ENV TRIOS_NTP_LR=0.003

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# v3: tjepa_train champion + pre-baked TinyShakespeare data
