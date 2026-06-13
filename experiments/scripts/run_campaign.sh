#!/usr/bin/env bash
# Scaling campaign: runs (qubits x regime) cells, two at a time.
# Usage: bash experiments/scripts/run_campaign.sh "4 6 8 10 12"
set -u
cd "$(dirname "$0")/../.."
SIZES="${1:-4 6 8 10 12}"
LOGDIR=experiments/results/scaling/logs
mkdir -p "$LOGDIR"

CELLS=""
for n in $SIZES; do
  for regime in miscal systematic nonlinear; do
    CELLS="$CELLS $n:$regime"
  done
done

echo "$CELLS" | tr ' ' '\n' | grep -v '^$' | xargs -P 2 -I{} bash -c '
  spec="{}"; n="${spec%%:*}"; regime="${spec##*:}"
  log="experiments/results/scaling/logs/${regime}_n${n}.log"
  if [ -f "experiments/results/scaling/${regime}_n${n}.json" ]; then
    echo "skip ${regime}_n${n} (exists)"; exit 0
  fi
  echo "[$(date +%H:%M:%S)] start ${regime}_n${n}"
  venv/bin/python experiments/scripts/scaling_cell.py \
    --qubits "$n" --regime "$regime" \
    > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] done ${regime}_n${n} (exit $?)"
'
echo "campaign finished"
