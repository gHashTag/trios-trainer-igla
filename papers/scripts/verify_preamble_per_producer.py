#!/usr/bin/env python3
"""verify_preamble_per_producer.py — gate the §5.1 per-producer preamble
field list against actual binary source emissions.

Loop 114 C operationalizes the §5.1 claim that each producer binary
emits a specific set of `# prov:` fields. A static grep over each
binary's source compares the *actual* emission set against the
registered *expected* set; mismatches fail the gate.

Background:
- Loop 113 D rewrote §5.1 to scope the preamble field set per producer
  class (cell-level vs aggregator). The 37th adversarial pass (Loop
  114 A) was queued to verify the rewrite is accurate.
- This script discovers, statically, that some "aggregator" binaries
  emit NO preamble at all — §5.1's claim that aggregator CSVs carry
  the generatedAt/wasGeneratedBy/agent_git_sha/host/schema/version
  set is partially aspirational.

Registry entry shape: (binary_path, expected_fields).
- expected_fields = [] means "no preamble emitted; §5.1 must NOT
  claim a preamble for this binary".
- expected_fields = ["generatedAt", ...] means "binary must emit
  EXACTLY these fields (in some order) and §5.1 must list them".

Usage: papers/scripts/verify_preamble_per_producer.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]

# Pattern: `# prov:fieldname = ...` inside a writeln! call.
PROV_FIELD_RE = re.compile(r"#\s*prov:([a-zA-Z_]+)\s*=")


REGISTRY: list[tuple[str, list[str]]] = [
    # Cell-level CSV: full preamble per F2 protocol
    ("src/bin/f2_ablation_sweep.rs", [
        "generatedAt", "wasGeneratedBy", "agent_git_sha", "host",
        "trainer_internals_schema", "cargo_pkg_version",
    ]),
    # #1021 protocol contribution: same preamble set
    ("src/bin/f2_pairwise_perm.rs", [
        "generatedAt", "wasGeneratedBy", "agent_git_sha", "host",
        "phi_configs", "zoo_configs",
        "trainer_internals_schema", "cargo_pkg_version",
    ]),
    # Aggregator binaries — historically no preamble emitted; the §5.1
    # paragraph (post-Loop 114) must acknowledge this honestly.
    ("src/bin/f2_ablation_aggregate.rs", []),
    ("src/bin/f2_stratum_compare.rs", []),
    ("src/bin/f2_mediation_sensitivity.rs", []),
    ("src/bin/f2_dual_mediation.rs", []),
]


def emitted_fields(path: Path) -> list[str]:
    """Static grep: return the ordered list of prov:* field names the
    binary's source writes."""
    if not path.exists():
        return []
    text = path.read_text()
    return PROV_FIELD_RE.findall(text)


def check_entry(rel: str, expected: list[str]) -> list[str]:
    path = CRATE_ROOT / rel
    actual = emitted_fields(path)
    actual_set = set(actual)
    expected_set = set(expected)
    mismatches: list[str] = []
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    for f in sorted(missing):
        mismatches.append(
            f"{rel}: missing # prov:{f} (expected per §5.1 / Loop 114 registry)")
    for f in sorted(extra):
        mismatches.append(
            f"{rel}: emits # prov:{f} but registry didn't expect it "
            "(silent extra field — update §5.1 or the registry)")
    return mismatches


# Loop 115 C — runtime mode: for producers cheap to invoke from
# this script, run them on a synthetic input and parse the actual
# `# prov:` lines from stdout. Catches the class of bug where a
# writeln!(out, "# prov:foo = ...") is present in source but gated
# by a runtime condition that prevents emission. Static-only would
# miss this.
#
# For producers expensive to invoke (e.g. f2_ablation_sweep needs a
# full training run), runtime mode is N/A and we rely on the static
# grep + the smoke test in `smoke_f2_pairwise_perm.sh` (which already
# validates f2_pairwise_perm's runtime preamble via f2_provenance_check).

import os
import subprocess
import tempfile


RUNTIME_PRODUCERS: dict[str, dict] = {
    # binary_name → {input_csv_content, args, expected runtime # prov: keys}
    "f2_pairwise_perm": {
        "input_content": (
            "config,stratum,seed,val_bpb\n"
            "GFTernary,canonical,42,1.20\n"
            "GFTernary,canonical,43,1.18\n"
            "GFTernary,canonical,44,1.22\n"
            "GFTernary,canonical,45,1.19\n"
            "GFTernary,canonical,46,1.21\n"
            "bf16,canonical,42,1.40\n"
            "bf16,canonical,43,1.38\n"
            "bf16,canonical,44,1.42\n"
            "bf16,canonical,45,1.39\n"
            "bf16,canonical,46,1.41\n"
        ),
        "extra_args": ["--phi-configs", "GFTernary", "--zoo-configs", "bf16"],
        "expected_runtime_fields": [
            "generatedAt", "wasGeneratedBy", "agent_git_sha", "host",
            "phi_configs", "zoo_configs",
            "trainer_internals_schema", "cargo_pkg_version",
        ],
    },
}


def runtime_check_producer(binary: str, spec: dict) -> list[str]:
    """Invoke the producer on synthetic input; parse # prov: lines from output."""
    binary_path = CRATE_ROOT / "target" / "release" / binary
    if not binary_path.exists():
        # Build it (release profile to match the smoke test).
        subprocess.run(
            ["cargo", "build", "--release", "--bin", binary],
            cwd=CRATE_ROOT, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    if not binary_path.exists():
        return [f"{binary}: cannot build for runtime check"]

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as inf:
        inf.write(spec["input_content"])
        input_path = inf.name
    output_path = input_path.replace(".csv", "_out.csv")

    env = os.environ.copy()
    env.setdefault("F2_GIT_SHA", "runtime-check")
    env.setdefault("HOST", "runtime-check-host")

    try:
        result = subprocess.run(
            [str(binary_path), "--input", input_path, "--output", output_path,
             *spec["extra_args"]],
            env=env, capture_output=True, text=True, check=False, timeout=30,
        )
        if result.returncode != 0:
            return [f"{binary}: runtime invocation exited {result.returncode}"]
        text = Path(output_path).read_text()
        runtime_fields = PROV_FIELD_RE.findall(text)
        runtime_set = set(runtime_fields)
        expected_set = set(spec["expected_runtime_fields"])
        mismatches: list[str] = []
        for f in sorted(expected_set - runtime_set):
            mismatches.append(f"{binary}: runtime missing # prov:{f}")
        for f in sorted(runtime_set - expected_set):
            mismatches.append(f"{binary}: runtime emits unexpected # prov:{f}")
        return mismatches
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def main() -> int:
    all_mismatches: list[str] = []
    rows_ok = 0
    for rel, expected in REGISTRY:
        ms = check_entry(rel, expected)
        if ms:
            all_mismatches.extend(ms)
            print(f"# FAIL  (static)  {rel}", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            rows_ok += 1
            status = "no preamble (acknowledged)" if not expected \
                else f"{len(expected)} fields verified"
            print(f"# OK    (static)  {rel} — {status}")

    # Runtime mode for cheap-to-invoke producers.
    runtime_ok = 0
    for binary, spec in RUNTIME_PRODUCERS.items():
        ms = runtime_check_producer(binary, spec)
        if ms:
            all_mismatches.extend(ms)
            print(f"# FAIL  (runtime) {binary}", file=sys.stderr)
            for m in ms:
                print(f"  {m}", file=sys.stderr)
        else:
            runtime_ok += 1
            n = len(spec["expected_runtime_fields"])
            print(f"# OK    (runtime) {binary} — {n} fields parsed from live output")

    if all_mismatches:
        print(f"# verify_preamble_per_producer.py — "
              f"{len(all_mismatches)} mismatches across "
              f"{len(REGISTRY)} static + {len(RUNTIME_PRODUCERS)} runtime checks",
              file=sys.stderr)
        return 1
    print(f"# verify_preamble_per_producer.py — "
          f"{rows_ok}/{len(REGISTRY)} static + "
          f"{runtime_ok}/{len(RUNTIME_PRODUCERS)} runtime PASS, 0 drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
