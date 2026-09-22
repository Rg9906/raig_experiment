#!/usr/bin/env bash
# Remaining T=40 work: the ablation at the new headline budget, then the other
# six noise conditions. The ablation is first because the paper needs it and it
# is cheap; the conditions are a long tail that the manuscript can survive
# without (the condition sweep is reported at T=20, the harder budget).
set -u
cd "$(dirname "$0")"
echo "queue5: start $(date)"

echo ""
echo "=========== run_ablation_T40.py ($(date +%H:%M:%S)) ==========="
if py run_ablation_T40.py > log_run_ablation_T40.log 2>&1; then
  echo "queue5: ablation OK"; cat log_run_ablation_T40.log
else
  echo "queue5: ablation FAILED" >&2; tail -20 log_run_ablation_T40.log >&2
fi

echo ""
echo "=========== run_main_T40.py (remaining conditions) ($(date +%H:%M:%S)) ==========="
py run_main_T40.py >> log_run_main_T40.log 2>&1 \
  && echo "queue5: conditions OK" || echo "queue5: conditions FAILED" >&2
tail -4 log_run_main_T40.log

echo ""
echo "queue5: all done at $(date)"
