#!/usr/bin/env python3
# Rebuild the t27 parallel compiler-verified corpus as corpus.jsonl, mirroring
# the P1 extractor (epic #1032 / issue #1034). English+ASCII source. The two
# compiler-verified translation signals:
#   1) gen/numeric/formats_catalog.*  -- ONE module emitted into 16 languages
#      (a 16-way parallel module: the spec->many-langs translation signal).
#   2) gen/c/** <-> gen/verilog/**     -- 33 matched C<->Verilog module pairs.
# Plus gen/rust/** as additional single-language code volume.
#
# Output: <out>/corpus.jsonl with one {"lang":..., "text":...} row per file.
# Consumed by prep_t27_corpus.py to produce data/code_{train,val}.bin.
#
# Anchor: phi^2 + phi^-2 = 3

import argparse
import json
import os

# Map file extension / basename -> the LANG_ID keys prep_t27_corpus.py knows.
EXT_LANG = {
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".zig": "zig",
    ".py": "python",
    ".ts": "typescript",
    ".kt": "kotlin",
    ".java": "java",
    ".swift": "swift",
    ".ml": "ocaml",
    ".hs": "haskell",
    ".jl": "julia",
    ".v": "verilog", ".vh": "verilog", ".sv": "verilog",
    ".json": "json",
    ".md": "markdown",
}


def is_ascii(s):
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def collect(root, rel_dirs):
    rows = []
    skipped_non_ascii = 0
    for rel in rel_dirs:
        base = os.path.join(root, rel)
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            for fn in sorted(files):
                ext = os.path.splitext(fn)[1].lower()
                lang = EXT_LANG.get(ext)
                if lang is None:
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if not text.strip():
                    continue
                # L3 ASCII discipline: drop any non-ASCII source defensively.
                if not is_ascii(text):
                    skipped_non_ascii += 1
                    continue
                rows.append({
                    "lang": lang,
                    "path": os.path.relpath(path, root),
                    "text": text,
                })
    return rows, skipped_non_ascii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t27", required=True, help="path to gHashTag/t27 checkout")
    ap.add_argument("--out", default="t27_corpus")
    args = ap.parse_args()

    # The differentiator dirs: the parallel/translation-signal code.
    rel_dirs = [
        "gen/c",
        "gen/verilog",
        "gen/numeric",
        "gen/rust",
    ]
    rows, skipped = collect(args.t27, rel_dirs)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "corpus.jsonl")
    with open(out_path, "w", encoding="ascii") as f:
        for r in rows:
            f.write(json.dumps({"lang": r["lang"], "text": r["text"]}) + "\n")

    # Provenance / data card.
    by_lang = {}
    total_bytes = 0
    for r in rows:
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + 1
        total_bytes += len(r["text"].encode("utf-8"))
    print("docs=%d total_source_bytes=%d skipped_non_ascii=%d"
          % (len(rows), total_bytes, skipped))
    print("by_lang=%s" % json.dumps(dict(sorted(by_lang.items()))))
    print("out=%s anchor: phi^2 + phi^-2 = 3" % out_path)


if __name__ == "__main__":
    main()
