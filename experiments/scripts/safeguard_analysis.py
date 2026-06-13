#!/usr/bin/env python3
"""Post-hoc safeguard analysis: ensemble-disagreement gating.

The scaling cells store per-instance predictions of all three NEM seeds.
The safeguarded estimator returns the ensemble mean when the seeds agree
(std below a threshold tau) and falls back to the raw noisy value when
they disagree — bounding worst-case behaviour without extra quantum
cost. tau is calibrated per cell as a quantile of the ensemble std on
the cell's own instances (no access to ideal values needed).

Also reports median / 95th-percentile absolute errors for every method,
exposing the unbounded tail risk of extrapolation-based ZNE.

Usage:
    python experiments/scripts/safeguard_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("experiments/results/scaling")
TAU_QUANTILE = 0.90  # gate the most-disagreeing 10% of instances


def analyze_cell(path):
    d = json.load(open(path))
    pi = d["per_instance"]
    ideal = np.array(pi["ideal"])
    noisy = np.array(pi["noisy"])
    seeds = [k for k in pi if k.startswith("neural_seed")]
    preds = np.stack([np.array(pi[k]) for k in sorted(seeds)])  # (S, N)

    ens_mean = preds.mean(axis=0)
    ens_std = preds.std(axis=0)
    tau = float(np.quantile(ens_std, TAU_QUANTILE))
    gated = np.where(ens_std <= tau, ens_mean, noisy)
    gate_rate = float((ens_std > tau).mean())

    def stats(pred, name):
        err = np.abs(np.asarray(pred) - ideal[: len(pred)])
        raw_err = np.abs(noisy[: len(pred)] - ideal[: len(pred)])
        return {
            "method": name,
            "mae": float(err.mean()),
            "median_ae": float(np.median(err)),
            "p95_ae": float(np.quantile(err, 0.95)),
            "max_ae": float(err.max()),
            "improvement_pct": float((raw_err.mean() - err.mean()) / raw_err.mean() * 100),
            "median_improvement_pct": float(
                (np.median(raw_err) - np.median(err)) / np.median(raw_err) * 100
            ),
            "worse_than_raw_rate": float((err > raw_err).mean()),
        }

    rows = [stats(noisy, "raw"),
            stats(pi["neural"], "neural(best-seed)"),
            stats(ens_mean, "neural(ensemble)"),
            stats(gated, "neural(safeguarded)")]
    for key in ("zne_richardson", "zne_exponential", "zne_adaptive", "cdr"):
        if key in pi:
            rows.append(stats(pi[key], key))
    return {"cell": d["cell"], "tau": tau, "gate_rate": gate_rate, "rows": rows}


def main():
    out = []
    for path in sorted(RESULTS_DIR.glob("*_n*.json")):
        out.append(analyze_cell(path))

    for cell in out:
        print(f"\n=== {cell['cell']} (tau={cell['tau']:.4f}, gated {cell['gate_rate']*100:.0f}%) ===")
        print(f"{'method':22s} {'MAE':>8s} {'med':>8s} {'p95':>8s} {'max':>9s} {'imp%':>6s} {'med-imp%':>8s} {'worse%':>6s}")
        for r in cell["rows"]:
            print(f"{r['method']:22s} {r['mae']:8.4f} {r['median_ae']:8.4f} "
                  f"{r['p95_ae']:8.4f} {r['max_ae']:9.3f} "
                  f"{r['improvement_pct']:+5.0f}% {r['median_improvement_pct']:+7.0f}% "
                  f"{r['worse_than_raw_rate']*100:5.0f}%")

    with open(RESULTS_DIR / "safeguard_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'safeguard_analysis.json'}")


if __name__ == "__main__":
    sys.exit(main())
