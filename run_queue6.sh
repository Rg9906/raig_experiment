#!/usr/bin/env bash
# Fill in the (catalogue size x budget) grid. Ordered cheapest-first so that a
# partial run still leaves a complete rectangle of the table filled: T=60 at the
# small sizes first (seconds to minutes), then the two larger sizes.
#
# 79,803 is deliberately NOT run here. The full-catalogue row is already
# measured at n=900 by run_main_ext.py (T=20) and run_main_T40.py (T=40), which
# is a larger sample than this sweep uses; re-running it at n=600 would cost
# hours to produce a worse estimate of a number we already have.
set -u
cd "$(dirname "$0")"
echo "queue6: start $(date)"

for job in "1000 2500 5000 10000 T60" "20000 T20 T40" "40000 T20 T40" "20000 T60" "40000 T60"; do
  echo ""
  echo "=========== run_catalogue_size.py $job  ($(date +%H:%M:%S)) ==========="
  if py run_catalogue_size.py $job >> log_run_catalogue_size.log 2>&1; then
    echo "queue6: [$job] OK"
    tail -3 log_run_catalogue_size.log
  else
    echo "queue6: [$job] FAILED" >&2
    tail -15 log_run_catalogue_size.log >&2
  fi
done

echo ""
echo "queue6: all done at $(date)"
