#!/usr/bin/env bash
# Sequential experiment queue for the journal revision.
#
# Waits for run_main_ext.py to finish before starting, then runs the remaining
# experiments one at a time. Strictly sequential and never concurrent with the
# main run: this machine has limited free RAM and the belief matrix is
# (N x M*batch) float32, so two heavy jobs at once thrash rather than
# parallelise.
#
# Every script checkpoints its own JSON, so killing this queue at any point
# leaves the completed experiments intact.
set -u
cd "$(dirname "$0")"

echo "queue: waiting for run_main_ext.py to finish..."
while ! grep -q "^DONE" run_main_ext.log 2>/dev/null; do
  if ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q python.exe; then
    echo "queue: no python process alive and no DONE line; main run appears to have died." >&2
    break
  fi
  sleep 20
done
echo "queue: main run finished, starting queue at $(date)"

for s in run_stats.py run_domains.py run_budget.py run_eps_ensemble.py \
         run_lambda_sweep.py run_synthetic.py run_tier_corruption.py run_erasure.py; do
  echo ""
  echo "=========== $s  ($(date +%H:%M:%S)) ==========="
  start=$SECONDS
  if py "$s" > "log_${s%.py}.log" 2>&1; then
    echo "queue: $s OK in $((SECONDS-start))s"
    tail -4 "log_${s%.py}.log"
  else
    echo "queue: $s FAILED in $((SECONDS-start))s -- continuing with the rest" >&2
    tail -20 "log_${s%.py}.log" >&2
  fi
done

echo ""
echo "queue: all done at $(date)"
