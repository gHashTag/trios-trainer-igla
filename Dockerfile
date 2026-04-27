FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default 1.91

WORKDIR /build
COPY . .
RUN cargo build --release --bin trios-train -p trios-trainer

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/target/release/trios-train /usr/local/bin/trios-train
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Bake corpus into image at build time (issue #13).
# No runtime curl — eliminates silent tinyshakespeare fallback (R5 + R8).
RUN mkdir -p /work/data
COPY data/fineweb_train.bin /work/data/fineweb_train.bin
COPY data/fineweb_val.bin   /work/data/fineweb_val.bin

# Guard: fail the build if corpus is missing or empty.
RUN test -s /work/data/fineweb_train.bin || (echo "ERROR: fineweb_train.bin missing or empty" && exit 1)
RUN test -s /work/data/fineweb_val.bin   || (echo "ERROR: fineweb_val.bin missing or empty"   && exit 1)

ENV RUST_LOG=info
ENV TRIOS_SEED=43
ENV TRIOS_STEPS=81000
ENV TRIOS_LR=0.003
ENV TRIOS_HIDDEN=384
ENV TRIOS_OPTIMIZER=adamw
# R8: forbid synthetic fallback — trainer must abort if corpus unavailable
ENV L_R8_SYNTHETIC_FALLBACK=FORBID

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
