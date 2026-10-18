#!/bin/zsh
# Watches a Salad run from the Mac: logs its status every minute and stops the group when the worker reports an
# end state (backup for the worker's own stop) or after a hard limit.  zsh scripts/salad/watch.sh <run> [limit_min]
cd /Users/youssefkaine/remotion
run=$1; limit=${2:-150}; t0=$(date +%s)
while true; do
  out=$(.venv/bin/python scripts/salad/salad.py status $run --lines 3 2>&1)
  echo "---- $(date +%T)"; echo "$out"
  if echo "$out" | head -1 | grep -qE "^== $run: (done|failed|crashed)"; then
    .venv/bin/python scripts/salad/salad.py stop $run; echo "STOPPED (finished)"; exit 0
  fi
  if [ $(( ($(date +%s) - t0) / 60 )) -ge $limit ]; then
    .venv/bin/python scripts/salad/salad.py stop $run; echo "STOPPED (time limit)"; exit 0
  fi
  .venv/bin/python scripts/salad/salad.py nudge $run 2>&1 | sed 's/^/   nudge: /'
  sleep 60
done
