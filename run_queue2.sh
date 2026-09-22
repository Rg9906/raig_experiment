#!/usr/bin/env bash
# Resumes the journal-revision queue from where the overnight run was killed.
# run_stats / run_domains / run_budget completed; run_eps_ensemble died after
# 10 of 60 draws (it has no resume, so it restarts). Strictly sequential for
# the same RAM reason as run_queue.sh.
set -u
cd "$(dirname "$0")"
for s in run_eps_ensemble.py run_lambda_sweep.py run_synthetic.py \
         run_tier_corruption.py run_erasure.py; do
  echo ""
  echo "=========== $s  ($(date +%H:%M:%S)) ==========="
  start=$SECONDS
  if py "$s" > "log_${s%.py}.log" 2>&1; then
    echo "queue: $s OK in $((SECONDS-start))s"
    tail -6 "log_${s%.py}.log"
  else
    echo "queue: $s FAILED in $((SECONDS-start))s -- continuing" >&2
    tail -20 "log_${s%.py}.log" >&2
  fi
done
echo ""
echo "queue: all done at $(date)"
