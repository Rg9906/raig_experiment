#!/usr/bin/env bash
# Third and final experiment queue: the three conference-version experiments
# re-run under the journal protocol (batch_size=30, seeds 1000+s), so that
# every number in the paper comes from one configuration. See the module
# docstrings for why the originals are not comparable.
#
# Waits for run_queue2.sh to finish first: this machine has limited free RAM
# and the belief matrix is (N x M*batch) float32, so two heavy jobs at once
# thrash rather than parallelise.
set -u
cd "$(dirname "$0")"

echo "queue3: waiting for queue2 to finish..."
while ! grep -q "^queue: all done" run_queue2.log 2>/dev/null; do
  if ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q python.exe; then
    if ! grep -q "^queue: all done" run_queue2.log 2>/dev/null; then
      echo "queue3: no python alive and queue2 not finished; starting anyway." >&2
    fi
    break
  fi
  sleep 30
done
echo "queue3: starting at $(date)"

for s in run_ablation_ext.py run_correlated_ext.py run_latency_ext.py; do
  echo ""
  echo "=========== $s  ($(date +%H:%M:%S)) ==========="
  start=$SECONDS
  if py "$s" > "log_${s%.py}.log" 2>&1; then
    echo "queue3: $s OK in $((SECONDS-start))s"
    tail -8 "log_${s%.py}.log"
  else
    echo "queue3: $s FAILED in $((SECONDS-start))s -- continuing" >&2
    tail -20 "log_${s%.py}.log" >&2
  fi
done

echo ""
echo "queue3: all done at $(date)"
