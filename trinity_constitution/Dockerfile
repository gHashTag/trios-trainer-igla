# Trinity — Multi-stage Docker image
# Constitutional: One image, all binaries, O(1) build

FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default stable

WORKDIR /build
COPY . .

# Build all binaries in one pass
RUN cargo build --release --bins

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy binaries
COPY --from=builder /build/target/release/trinity-runner /usr/local/bin/trinity-runner
COPY --from=builder /build/target/release/trinity-reconcile /usr/local/bin/trinity-reconcile
COPY --from=builder /build/target/release/trinity-gardener /usr/local/bin/trinity-gardener

# Download data (same as in Dockerfile.scarab)
RUN mkdir -p /work/data && \
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > /tmp/full.txt && \
    SIZE=$(stat -c%s /tmp/full.txt) && \
    HEAD=$((SIZE - 100000)) && \
    head -c $HEAD /tmp/full.txt > /work/data/tiny_shakespeare.txt && \
    tail -c 100000 /tmp/full.txt > /work/data/tiny_shakespeare_val.txt && \
    rm /tmp/full.txt && \
    echo "[trinity] data ready: train=$(stat -c%s /work/data/tiny_shakespeare.txt) val=$(stat -c%s /work/data/tiny_shakespeare_val.txt)"

ENV RUST_LOG=info

# Default to runner (can be overridden in Railway config)
ENTRYPOINT ["/usr/local/bin/trinity-runner"]
