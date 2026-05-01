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
        --bin scarab \
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
COPY --from=builder /build/target/release/scarab /usr/local/bin/scarab

# Byte-disjoint train/val split. The previous version ran
#   head -c 100000 tiny_shakespeare.txt > tiny_shakespeare_val.txt
# which made val a strict prefix of train and produced BPB ≈ 0 on every
# experiment after enough steps to memorise 100 KB. Fixed in
# trios-trainer-igla#60: train = first (size-100000) bytes, val = last
# 100000 bytes, strictly disjoint.
RUN mkdir -p /work/data && \
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > /tmp/full.txt && \
    SIZE=$(stat -c%s /tmp/full.txt) && \
    HEAD=$((SIZE - 100000)) && \
    head -c $HEAD /tmp/full.txt > /work/data/tiny_shakespeare.txt && \
    tail -c 100000 /tmp/full.txt > /work/data/tiny_shakespeare_val.txt && \
    rm /tmp/full.txt && \
    echo "[corpus-split] train=$(stat -c%s /work/data/tiny_shakespeare.txt) bytes  val=$(stat -c%s /work/data/tiny_shakespeare_val.txt) bytes"

ENV RUST_LOG=info
ENV TRIOS_SEED=43
ENV TRIOS_STEPS=81000
ENV TRIOS_LR=0.003
ENV TRIOS_HIDDEN=384
ENV TRIOS_OPTIMIZER=adamw

# Dual-mode ENTRYPOINT: supports both entrypoint and scarab
# When TRIOS_TRAINER_BIN=scarab → runs scarab directly
# When TRIOS_TRAINER_BIN=trios-train (default) → runs entrypoint → trios-train
ENTRYPOINT ["/usr/local/bin/entrypoint", "/usr/local/bin/scarab"]
