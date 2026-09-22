#!/usr/bin/env bash
# Downstream of the T=40 headline run: significance testing and the ablation,
# both of which read results/main_T40.json and must not start before it exists.
set -u
cd "$(dirname "$0")"

echo "queue4: waiting for run_main_T40.py..."
while ! grep -q "^DONE" log_run_main_T40.log 2>/dev/null; do
  if ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q python.exe; then
    echo "queue4: no python alive and no DONE line; main T40 run died." >&2
    exit 1
  fi
  sleep 60
done
echo "queue4: main T40 finished, starting at $(date)"

for s in run_stats_T40.py run_ablation_T40.py; do
  echo ""
  echo "=========== $s  ($(date +%H:%M:%S)) ==========="
  start=$SECONDS
  if py "$s" > "log_${s%.py}.log" 2>&1; then
    echo "queue4: $s OK in $((SECONDS-start))s"
    tail -12 "log_${s%.py}.log"
  else
    echo "queue4: $s FAILED in $((SECONDS-start))s" >&2
    tail -20 "log_${s%.py}.log" >&2
  fi
done
echo ""
echo "queue4: all done at $(date)"
