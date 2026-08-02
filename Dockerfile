FROM debian:bookworm-slim AS builder

# Optional Cargo feature flags (e.g. gf16). Default empty = baseline build.
# Pass --build-arg CARGO_FEATURES=gf16 to enable Wave-32-B GF16 kernel.
ARG CARGO_FEATURES=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pkg-config build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default 1.91

WORKDIR /build
ARG CACHE_BUST=1
COPY . .
RUN set -e; \
    FEATURES_FLAG=""; \
    if [ -n "${CARGO_FEATURES}" ]; then FEATURES_FLAG="--features ${CARGO_FEATURES}"; fi; \
    cargo build --release --locked \
        --bin entrypoint \
        --bin trios-train \
        --bin railway-sweep \
        --bin scarab \
        --bin gf16_test \
        --bin ngram_train_gf16 \
        --bin bpb_smoke \
        --bin tjepa_train \
        --bin hybrid_train \
        -p trios-trainer \
        ${FEATURES_FLAG}

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY --from=builder /build/target/release/entrypoint /usr/local/bin/entrypoint
COPY --from=builder /build/target/release/trios-train /usr/local/bin/trios-train
COPY --from=builder /build/target/release/railway-sweep /usr/local/bin/railway-sweep
COPY --from=builder /build/target/release/scarab /usr/local/bin/scarab
COPY --from=builder /build/target/release/gf16_test /usr/local/bin/gf16_test
COPY --from=builder /build/target/release/ngram_train_gf16 /usr/local/bin/ngram_train_gf16
COPY --from=builder /build/target/release/bpb_smoke /usr/local/bin/bpb_smoke
# Arch breakthrough (2026-05-15): JEPA-T + NCA + Hybrid trainers reachable from
# Railway services via TRIOS_TRAINER_BIN={tjepa_train,hybrid_train}. Required
# to break the gf256-h256 NTP-only plateau at BPB=2.5719 (champion lock).
# Multi-objective L = w_CE*NTP + w_JEPA*JEPA + w_NCA*NCA, see
# crates/trios-railway-mcp/src/tools.rs::gate2-final template for canonical
# weights (w_CE=1.0, w_JEPA=0.15, w_NCA=0.10).
COPY --from=builder /build/target/release/tjepa_train /usr/local/bin/tjepa_train
COPY --from=builder /build/target/release/hybrid_train /usr/local/bin/hybrid_train

# Byte-disjoint train/val split. The previous version ran
#   head -c 100000 tiny_shakespeare.txt > tiny_shakespeare_val.txt
# which made val a strict prefix of train and produced BPB near 0 on every
# experiment after enough steps to memorise 100 KB. Fixed in
# trios-trainer-igla#60: train = first (size-100000) bytes, val = last
# 100000 bytes, strictly disjoint.
#
# The corpus is pinned by checksum. Every BPB recorded in this repo is measured
# against these exact bytes, so an unpinned third-party URL would silently
# invalidate the whole ledger the day upstream re-encodes a newline. The three
# digests below are consistent by construction: train (the first SIZE-100000
# bytes) concatenated with val (the last 100000 bytes) reconstructs the
# canonical corpus byte for byte --
#   cat tiny_shakespeare.txt tiny_shakespeare_val.txt == full.txt
#     sha256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
#     bytes  1115394 = 1015394 (train) + 100000 (val)
# Verified 2026-08-02 against the two files committed under data/.
RUN mkdir -p /work/data && \
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > /tmp/full.txt && \
    echo "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  /tmp/full.txt" | sha256sum -c - && \
    SIZE=$(stat -c%s /tmp/full.txt) && \
    HEAD=$((SIZE - 100000)) && \
    head -c $HEAD /tmp/full.txt > /work/data/tiny_shakespeare.txt && \
    tail -c 100000 /tmp/full.txt > /work/data/tiny_shakespeare_val.txt && \
    echo "1a5aead1db78653f48ee799c4145ef71265f6aadd2c79ebffc9f0260cac1fb0d  /work/data/tiny_shakespeare.txt" | sha256sum -c - && \
    echo "2088af36b1c7831083ef22c0f6e1999b1dece15b9fbe2d4695364e95d497d502  /work/data/tiny_shakespeare_val.txt" | sha256sum -c - && \
    rm /tmp/full.txt && \
    echo "[corpus-split] train=$(stat -c%s /work/data/tiny_shakespeare.txt) bytes  val=$(stat -c%s /work/data/tiny_shakespeare_val.txt) bytes"

ENV RUST_LOG=info
# Wave-29 PR-A.1 (Canon #93): no baked-in seed default. The previous
# `ENV TRIOS_SEED` pinned a *forbidden* canon value -- any service deployed
# without an explicit override would inherit it and write rows under a
# forbidden seed, which Canon #93 rejects (forbidden set: {42, 43, 44, 45};
# allowed: {47, 89, 123, 144}). Service-level env vars still set the
# seed at deploy time; absent that, the trainer's CLI default of 47
# wins. The `parse_seed()` Canon #93 guard in entrypoint+trios-train
# rejects any forbidden value at process start.
# Wave-33B: defaults moved into the Rust binary (`entrypoint_env::resolve_env_alias`).
# Baking `ENV TRIOS_STEPS=81000` here meant the canonical key was *always*
# present at runtime, so an operator-set un-prefixed alias (e.g.
# `STEPS=200000`) lost the precedence race and the override silently
# regressed to 81000. Removing the baked ENVs lets the alias actually
# win when no `TRIOS_<KEY>` is supplied. The binary still defaults to
# 81000 / 0.003 / 384 / adamw if neither is set, so behavior is
# unchanged for services that didn't set anything. See PR #130 for the
# Rust-side resolution logic and `[entrypoint-trace]` line.

ENTRYPOINT ["/usr/local/bin/entrypoint"]
