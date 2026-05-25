#!/bin/bash
# monitor_v2_sweep.sh — auto-collect BPB from v2 format sweep
# Usage: ./scripts/monitor_v2_sweep.sh

cd "$(dirname "$0")/.."
mkdir -p .trinity/results

SUMMARY=".trinity/v2_progress.json"

# Collect all v2 logs into JSON
{
  echo "{"
  echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"runs\": ["
  first=true
  for f in .trinity/results/v2_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    # Extract format, seed, steps from filename
    fmt=$(echo "$name" | sed 's/v2_//' | sed 's/_seed[0-9]*_.*//')
    seed=$(echo "$name" | grep -o 'seed[0-9]*' | sed 's/seed//')
    steps=$(echo "$name" | grep -o '[0-9]*k' | sed 's/k//')
    # Extract latest eval
    last_eval=$(grep "step=" "$f" | tail -1)
    if [ -n "$last_eval" ]; then
      last_step=$(echo "$last_eval" | awk '{print $2}' | sed 's/step=//')
      last_bpb=$(echo "$last_eval" | awk '{print $3}' | sed 's/val_bpb=//')
      best_bpb=$(echo "$last_eval" | awk '{print $5}' | sed 's/best=//')
    else
      last_step="0"
      last_bpb="null"
      best_bpb="null"
    fi
    # Check if DONE
    done_bpb=$(grep "DONE:" "$f" 2>/dev/null | sed 's/.*bpb=//' | awk '{print $1}')
    status="running"
    [ -n "$done_bpb" ] && status="done"
    [ -n "$done_bpb" ] && best_bpb="$done_bpb"

    $first || echo ","
    first=false
    echo -n "    {\"name\":\"$name\",\"format\":\"$fmt\",\"seed\":$seed,\"target_steps\":$steps,\"last_step\":$last_step,\"last_bpb\":$last_bpb,\"best_bpb\":$best_bpb,\"status\":\"$status\"}"
  done
  echo ""
  echo "  ]"
  echo "}"
} > "$SUMMARY"

echo "=== V2 SWEEP PROGRESS ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Summary table
python3 - "$SUMMARY" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
runs = data["runs"]

print(f"Total runs: {len(runs)}")
done = [r for r in runs if r["status"] == "done"]
running = [r for r in runs if r["status"] == "running"]
print(f"Done: {len(done)} | Running: {len(running)}")
print("")

# Group by format
from collections import defaultdict
by_fmt = defaultdict(list)
for r in runs:
    by_fmt[r["format"]].append(r)

print("--- Progress by Format ---")
print(f"{'Format':<12} {'Seeds':<8} {'Steps':<10} {'Best BPB':<10} {'Status':<8}")
print("-" * 55)
for fmt in sorted(by_fmt.keys()):
    rs = by_fmt[fmt]
    seeds = len(rs)
    max_step = max(r["last_step"] for r in rs)
    best_bpb = min((r["best_bpb"] for r in rs if r["best_bpb"] is not None), default=None)
    n_done = sum(1 for r in rs if r["status"] == "done")
    status = f"{n_done}/{seeds} done"
    bpb_str = f"{best_bpb:.4f}" if best_bpb is not None else "N/A"
    print(f"{fmt:<12} {seeds:<8} {max_step:<10} {bpb_str:<10} {status:<8}")

print("")
print("--- Done Runs (sorted by BPB) ---")
done_sorted = sorted(done, key=lambda r: r["best_bpb"] or 999)
for r in done_sorted[:20]:
    print(f"  {r['name']}: BPB={r['best_bpb']:.4f} steps={r['last_step']}")

print("")
print("--- Running Runs (sorted by step) ---")
run_sorted = sorted(running, key=lambda r: r["last_step"], reverse=True)
for r in run_sorted[:20]:
    bpb = f"BPB={r['best_bpb']:.4f}" if r['best_bpb'] is not None else "no eval yet"
    print(f"  {r['name']}: step={r['last_step']} {bpb}")
PY
