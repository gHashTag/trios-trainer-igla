FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default 1.91

WORKDIR /build
COPY . .
RUN cargo build --release \
        --bin entrypoint \
        --bin trios-train \
        --bin gf16_test \
        --bin ngram_train_gf16 \
        --bin bpb_smoke \
        -p trios-trainer

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/target/release/entrypoint /usr/local/bin/entrypoint
COPY --from=builder /build/target/release/trios-train /usr/local/bin/trios-train
COPY --from=builder /build/target/release/gf16_test /usr/local/bin/gf16_test
COPY --from=builder /build/target/release/ngram_train_gf16 /usr/local/bin/ngram_train_gf16
COPY --from=builder /build/target/release/bpb_smoke /usr/local/bin/bpb_smoke

RUN mkdir -p /work/data && \
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > /work/data/tiny_shakespeare.txt && \
    head -c 100000 /work/data/tiny_shakespeare.txt > /work/data/tiny_shakespeare_val.txt

ENV RUST_LOG=info
ENV TRIOS_SEED=43
ENV TRIOS_STEPS=81000
ENV TRIOS_LR=0.003
ENV TRIOS_HIDDEN=384
ENV TRIOS_OPTIMIZER=adamw

ENTRYPOINT ["/usr/local/bin/entrypoint"]
