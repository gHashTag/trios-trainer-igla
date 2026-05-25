#!/bin/bash
cd "$(dirname "$0")/.."
echo "=== ALL SWEEPS STATUS ==="
echo "Timestamp: $(date)"
echo ""

# v2 runs
echo "--- v2 (Real QAT, normal priority) ---"
for f in .trinity/results/v2_*.log; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .log)
  last=$(grep "step=" "$f" | tail -1 | awk '{print $2, $4, $6}' 2>/dev/null)
  [ -n "$last" ] && echo "  $name: $last"
done | sort

echo ""
echo "--- v3 (Real QAT, nice -19, 10K) ---"
for f in .trinity/results/v3_*.log; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .log)
  last=$(grep "step=" "$f" | tail -1 | awk '{print $2, $4, $6}' 2>/dev/null)
  [ -n "$last" ] && echo "  $name: $last"
done | sort

echo ""
echo "--- v4 (Real QAT, nice -19, 20K) ---"
for f in .trinity/results/v4_*.log; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .log)
  last=$(grep "step=" "$f" | tail -1 | awk '{print $2, $4, $6}' 2>/dev/null)
  [ -n "$last" ] && echo "  $name: $last"
done | sort

echo ""
echo "--- Process counts ---"
v2_count=$(ps aux | grep "trios-train" | grep -v grep | grep -v "nice" | wc -l)
v3_count=$(ps aux | grep "trios-train" | grep -v grep | grep "nice" | wc -l)
echo "v2 (normal): $v2_count"
echo "v3/v4 (nice -19): $v3_count"
echo "Total: $(ps aux | grep trios-train | grep -v grep | wc -l)"
