"""Self-contained F2 long-form CSV schema validator (stands in for the absent
f2_provenance_check / f2_csv_validate binaries, which are not on this branch).
Checks the contract from the f2-mediation-loop skill:
 - W3C-PROV preamble line present (# W3C-PROV: source = ...)
 - INPUT STRATUM banner present (# INPUT STRATUM = <s>)
 - header row is ASCII-only snake_case
 - every data row has the same width as the header (no nested cells)
 - declared stratum is one of the known strata
"""
import re
import sys

KNOWN_STRATA = {"canonical", "wd0", "warmup0", "mom_std"}
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def validate(path):
    errs, warns = [], []
    has_prov = has_stratum = False
    stratum = None
    header = None
    width = None
    nrows = 0
    with open(path, encoding="ascii", errors="strict") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if line.startswith("#"):
                if line.startswith("# W3C-PROV:"):
                    has_prov = True
                if line.startswith("# INPUT STRATUM"):
                    has_stratum = True
                    m = re.search(r"INPUT STRATUM\s*=\s*(\S+)", line)
                    if m:
                        stratum = m.group(1)
                continue
            cols = line.split(",")
            if header is None:
                header = cols
                width = len(cols)
                for h in header:
                    if not SNAKE.match(h):
                        errs.append(f"header col not snake_case ASCII: {h!r}")
                continue
            nrows += 1
            if len(cols) != width:
                errs.append(f"row {ln}: width {len(cols)} != header {width}")
    if not has_prov:
        errs.append("missing '# W3C-PROV:' preamble line")
    if not has_stratum:
        errs.append("missing '# INPUT STRATUM =' banner")
    if stratum and stratum not in KNOWN_STRATA:
        errs.append(f"unknown stratum {stratum!r} (expected one of {KNOWN_STRATA})")

    print(f"file: {path}")
    print(f"  prov_preamble={has_prov}  stratum={stratum}  "
          f"header_cols={width}  data_rows={nrows}")
    if errs:
        print("  SCHEMA: FAIL")
        for e in errs:
            print(f"    - {e}")
        return 1
    print("  SCHEMA: PASS (F2 long-form contract satisfied)")
    return 0


if __name__ == "__main__":
    sys.exit(validate(sys.argv[1] if len(sys.argv) > 1 else "coder_ablation_f2.csv"))
