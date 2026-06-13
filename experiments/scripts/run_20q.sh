#!/usr/bin/env bash
# 20-qubit cells with trimmed budgets (launched after 16q cells finish).
set -u
cd "$(dirname "$0")/../.."
LOGDIR=experiments/results/scaling/logs
mkdir -p "$LOGDIR"
echo "miscal systematic nonlinear" | tr ' ' '\n' | xargs -P 2 -I{} bash -c '
  regime="{}"
  out="experiments/results/scaling/${regime}_n20.json"
  [ -f "$out" ] && { echo "skip ${regime}_n20"; exit 0; }
  echo "[$(date +%H:%M:%S)] start ${regime}_n20"
  venv/bin/python experiments/scripts/scaling_cell.py \
    --qubits 20 --regime "$regime" \
    --train-samples 2500 --val-samples 250 \
    --test-instances 100 --cdr-instances 30 \
    > "experiments/results/scaling/logs/${regime}_n20.log" 2>&1
  echo "[$(date +%H:%M:%S)] done ${regime}_n20 (exit $?)"
'
echo "20q campaign finished"
