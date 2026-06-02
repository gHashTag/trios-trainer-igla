#!/usr/bin/env python3
"""verify_dependency_graph.py — assert inter-gate import topology has
no cycles and respects STAGES execution order.

Loop 141 B operationalizes the dependency-graph class. Several gates
load sister gates via `_gate_utils.import_gate(name)`; if a future
loop accidentally introduces a cycle (gate A imports B which imports
A), the runtime behavior is undefined and the first-loader wins.
This gate parses each gate's source, extracts import_gate calls,
builds a DAG, and asserts:

  (a) No cycles. Detected via DFS with color marking.
  (b) Order. For every edge `A → B`, `STAGES.index(A) > STAGES.index(B)`
      so the dependent gate runs AFTER its dependency. This isn't
      required for correctness today (subprocess isolation), but
      it's a safety property for a future in-process orchestrator.

Usage: papers/scripts/verify_dependency_graph.py

Exit 0 if both invariants hold; 1 on cycle or order violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = CRATE_ROOT / "papers" / "scripts"
RUN_ALL_CHECKS = SCRIPTS_DIR / "run_all_checks.sh"

# Match calls of the form: import_gate("verify_FOO.py") or single-
# quoted variant. The pattern below is deliberately written to NOT
# match itself (Loop 141 — 64th-pass SEV-4 fix #3: prior wording
# said `import_gate("verify_X.py")` which itself matched the regex,
# producing a phantom self-edge that masked the off-by-one in edge
# counting).
_IMPORT_CALL_RE = re.compile(
    r'import_gate\(\s*[\'"](verify_[A-Za-z0-9_]+\.py)[\'"]'
)


def parse_imports() -> dict[str, set[str]]:
    """Return {gate_filename: {imported_filename, ...}} for all
    verify_*.py files in papers/scripts/."""
    graph: dict[str, set[str]] = {}
    for path in sorted(SCRIPTS_DIR.glob("verify_*.py")):
        text = path.read_text()
        imports = set(_IMPORT_CALL_RE.findall(text))
        # Self-reference (a gate testing its own loader) is excluded
        # since it's not a real dependency.
        imports.discard(path.name)
        graph[path.name] = imports
    return graph


def parse_stages_order() -> list[str] | str:
    """Return ordered list of stage gate filenames (from STAGES=()
    array in run_all_checks.sh). Returns error string on parse
    failure."""
    if not RUN_ALL_CHECKS.exists():
        return f"missing {RUN_ALL_CHECKS.relative_to(CRATE_ROOT)}"
    text = RUN_ALL_CHECKS.read_text()
    m = re.search(r"STAGES=\(\s*\n(.*?)\n\)", text, re.DOTALL)
    if not m:
        return "STAGES=( ... ) block not found"
    body = m.group(1)
    # Each STAGES entry has shape `name:python3 papers/scripts/X.py [args]`.
    # Extract X.py — only verify_*.py entries; others (shell, single
    # scripts) are out-of-scope for the dependency check.
    out: list[str] = []
    for line in body.splitlines():
        mm = re.search(r"papers/scripts/(verify_[A-Za-z0-9_]+\.py)", line)
        if mm:
            out.append(mm.group(1))
    return out


def find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """DFS cycle detection. Returns the first cycle as a path list,
    or None on DAG. Color: 0=white, 1=gray (on stack), 2=black."""
    color: dict[str, int] = {n: 0 for n in graph}
    parent: dict[str, str | None] = {n: None for n in graph}

    def dfs(start: str) -> list[str] | None:
        stack = [(start, iter(graph.get(start, ())))]
        color[start] = 1
        while stack:
            node, neighbors = stack[-1]
            try:
                nb = next(neighbors)
            except StopIteration:
                color[node] = 2
                stack.pop()
                continue
            if nb not in graph:
                # External node (e.g., gate not in papers/scripts/).
                continue
            if color[nb] == 1:
                # Found a back-edge → cycle. Reconstruct.
                cycle = [nb, node]
                while stack and parent.get(node):
                    node = parent[node]  # type: ignore
                    cycle.append(node)
                    if node == nb:
                        break
                return list(reversed(cycle))
            if color[nb] == 0:
                color[nb] = 1
                parent[nb] = node
                stack.append((nb, iter(graph.get(nb, ()))))
        return None

    for n in graph:
        if color[n] == 0:
            cy = dfs(n)
            if cy is not None:
                return cy
    return None


def main() -> int:
    graph = parse_imports()
    n_edges = sum(len(v) for v in graph.values())
    print(f"# verify_dependency_graph.py — {len(graph)} gates, "
          f"{n_edges} import edge(s)")

    mismatches: list[str] = []

    # (a) Cycle detection.
    cycle = find_cycle(graph)
    if cycle:
        mismatches.append(
            f"cycle detected: {' → '.join(cycle)}. The import_gate "
            "graph must be a DAG; refactor one of the edges to break "
            "the cycle.")
    else:
        print(f"# OK    DAG: no cycles in import_gate graph")

    # (b) STAGES order vs dependency direction.
    stages = parse_stages_order()
    if isinstance(stages, str):
        mismatches.append(f"STAGES parse: {stages}")
    else:
        stage_idx = {name: i for i, name in enumerate(stages)}
        n_order_ok = 0
        for gate, deps in graph.items():
            if gate not in stage_idx:
                continue  # gate isn't a registered stage; manual tool
            for dep in deps:
                if dep not in stage_idx:
                    continue
                if stage_idx[gate] <= stage_idx[dep]:
                    mismatches.append(
                        f"order violation: {gate} (stage "
                        f"{stage_idx[gate]}) imports {dep} (stage "
                        f"{stage_idx[dep]}) — dependent must run "
                        "after dependency.")
                else:
                    n_order_ok += 1
        # Loop 141 — 64th-pass SEV-4 fix #10: emit WARN when 0
        # stage→stage edges exist (a regression that drops all edges
        # would silently pass through to the final OK).
        if n_order_ok == 0 and not any(
            "order violation" in m for m in mismatches
        ):
            print(f"# WARN  order: 0 stage→stage import edges to "
                  "check; either every gate is independent (OK) or "
                  "the import scanner regressed (investigate).",
                  file=sys.stderr)
        elif n_order_ok > 0 and not any(
            "order violation" in m for m in mismatches
        ):
            print(f"# OK    order: all {n_order_ok} import edge(s) "
                  "respect STAGES order")

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"# verify_dependency_graph.py — "
              f"{len(mismatches)} dependency-graph violation(s)",
              file=sys.stderr)
        return 1
    print(f"# verify_dependency_graph.py — DAG + STAGES order both "
          "hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
