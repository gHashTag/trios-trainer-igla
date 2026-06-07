#!/usr/bin/env python3
# Encode the t27 parallel compiler-verified corpus into byte-level .bin shards
# for the igla_coder CPU trainer. English+ASCII-only source.
#
# Vocab contract (matches src/bin/igla_coder.rs VOCAB=263):
#   0..=255  raw bytes
#   256      <BOS>   begin-of-document
#   257      <EOS>   end-of-document
#   258      <PRE>   FIM prefix sentinel  (Bavarian et al. 2022)
#   259      <SUF>   FIM suffix sentinel
#   260      <MID>   FIM middle sentinel
#   261      <LANG>  language tag sentinel (followed by 1 byte lang id)
#   262      <PAD>   reserved
#
# .bin format (matches load_bin): 1024-byte header, magic u32=20240520 at [0..4],
# token count u32 at [8..12], then count little-endian u16 tokens.
#
# Usage:
#   python3 prep_t27_corpus.py --corpus <t27_corpus dir> --out data \
#       [--val-frac 0.1] [--fim-frac 0.5] [--seed 42]

import argparse
import json
import os
import random
import struct

MAGIC = 20240520
BOS, EOS, PRE, SUF, MID, LANG, PAD = 256, 257, 258, 259, 260, 261, 262

# Stable single-byte language ids (only the low byte is emitted after <LANG>).
LANG_ID = {
    "c": 1, "cpp": 2, "rust": 3, "go": 4, "zig": 5, "python": 6,
    "typescript": 7, "kotlin": 8, "java": 9, "swift": 10, "ocaml": 11,
    "haskell": 12, "julia": 13, "verilog": 14, "json": 15, "markdown": 16,
}


def doc_tokens(lang, text, fim, rng):
    """Return a token list for one document, optionally FIM-transformed (PSM)."""
    body = list(text.encode("utf-8", "replace"))
    head = [BOS, LANG, LANG_ID.get(lang, 0)]
    if not fim or len(body) < 24:
        return head + body + [EOS]
    # PSM ordering: <PRE> prefix <SUF> suffix <MID> middle <EOS>
    a = rng.randint(1, len(body) - 2)
    b = rng.randint(a + 1, len(body) - 1)
    prefix, middle, suffix = body[:a], body[a:b], body[b:]
    return head + [PRE] + prefix + [SUF] + suffix + [MID] + middle + [EOS]


def write_bin(path, tokens):
    header = bytearray(1024)
    struct.pack_into("<I", header, 0, MAGIC)
    struct.pack_into("<I", header, 8, len(tokens))
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"".join(struct.pack("<H", t) for t in tokens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="t27_corpus dir with corpus.jsonl")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--fim-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    with open(os.path.join(args.corpus, "corpus.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng.shuffle(rows)

    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    def encode(rows, fim_frac):
        out = []
        for r in rows:
            fim = rng.random() < fim_frac
            out.extend(doc_tokens(r.get("lang", "c"), r.get("text", ""), fim, rng))
        return out

    train_tokens = encode(train_rows, args.fim_frac)
    val_tokens = encode(val_rows, 0.0)  # validation = plain NTP, no FIM

    os.makedirs(args.out, exist_ok=True)
    write_bin(os.path.join(args.out, "code_train.bin"), train_tokens)
    write_bin(os.path.join(args.out, "code_val.bin"), val_tokens)

    langs = sorted({r.get("lang", "c") for r in rows})
    print("docs train=%d val=%d" % (len(train_rows), len(val_rows)))
    print("tokens train=%d val=%d" % (len(train_tokens), len(val_tokens)))
    print("langs=%s" % ",".join(langs))
    print("fim_frac=%.2f vocab=263 anchor: phi^2 + phi^-2 = 3" % args.fim_frac)


if __name__ == "__main__":
    main()
