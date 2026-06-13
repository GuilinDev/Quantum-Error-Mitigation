#!/usr/bin/env python3
"""Generate publication figures from the experiment result JSONs.

Figures (PDF + PNG into paper/figures/):
  budget_pareto      — median |error| vs total shot budget per method,
                       two panels (systematic | miscal)
  phase_boundary     — median |error| vs coherent miscalibration delta
  scaling            — NEM / CDR / best-ZNE improvement vs qubit count,
                       three regime panels

Median + IQR are used throughout: extrapolation-based ZNE has unbounded
tail errors (single-instance MAE up to ~1e5) that make means
unreadable; tail statistics are reported separately in the safety table.

Usage:
    python experiments/scripts/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGDIR = ROOT / "paper" / "figures"

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "raw": "#666666",
    "neural": "#0072B2",
    "zne_exponential": "#E69F00",
    "zne_adaptive": "#D55E00",
    "zne_richardson": "#CC79A7",
    "cdr": "#009E73",
    "direct": "#999999",
}
LABELS = {
    "raw": "Unmitigated",
    "neural": "NEM (ours, 1 exec)",
    "zne_exponential": "ZNE-exp (3 execs)",
    "zne_adaptive": "ZNE-adaptive (4 execs)",
    "zne_richardson": "ZNE-Richardson (3 execs)",
    "cdr": "CDR (32 execs)",
    "direct": "Direct prediction",
}
MARKERS = {
    "raw": "s", "neural": "o", "zne_exponential": "^",
    "zne_adaptive": "v", "zne_richardson": "<", "cdr": "D",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def med_iqr(values, ideal):
    err = np.abs(np.asarray(values) - np.asarray(ideal)[: len(values)])
    return (float(np.median(err)),
            float(np.quantile(err, 0.25)),
            float(np.quantile(err, 0.75)))


def fig_budget_pareto():
    regimes = ["systematic", "miscal"]
    titles = {"systematic": "Stochastic device noise",
              "miscal": "Stochastic + coherent miscalibration"}
    methods = ["raw", "neural", "zne_exponential", "zne_adaptive", "cdr"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    for ax, regime in zip(axes, regimes):
        path = RESULTS / "budget" / f"budget_sweep_{regime}.json"
        d = json.load(open(path))
        budgets = d["budgets"]
        ideal = d["ideal"]
        for m in methods:
            med, lo, hi = [], [], []
            for B in budgets:
                vals = d["sweep"][str(B)]["per_instance"][m]
                a, b, c = med_iqr(vals, ideal)
                med.append(a), lo.append(b), hi.append(c)
            ax.plot(budgets, med, marker=MARKERS[m], ms=4, lw=1.4,
                    color=COLORS[m], label=LABELS[m])
            ax.fill_between(budgets, lo, hi, color=COLORS[m], alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Total shot budget $B$ per evaluation")
        ax.set_title(titles[regime])
        ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].set_ylabel("Median $|$error$|$ (IQR shaded)")
    axes[0].legend(loc="lower left", framealpha=0.9, handlelength=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"budget_pareto.{ext}")
    plt.close(fig)
    print("budget_pareto written")


def fig_phase_boundary():
    d = json.load(open(RESULTS / "boundary" / "phase_boundary.json"))
    deltas = d["delta_points"]
    methods = ["raw", "neural", "zne_exponential", "zne_adaptive", "cdr"]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for m in methods:
        med, lo, hi = [], [], []
        for delta in deltas:
            entry = d["sweep"][str(delta)]
            vals = entry["per_instance"][m]
            a, b, c = med_iqr(vals, entry["ideal"])
            med.append(a), lo.append(b), hi.append(c)
        ax.plot(deltas, med, marker=MARKERS[m], ms=4, lw=1.4,
                color=COLORS[m], label=LABELS[m])
        ax.fill_between(deltas, lo, hi, color=COLORS[m], alpha=0.12, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel(r"Coherent miscalibration bias $\delta$ (rad)")
    ax.set_ylabel("Median $|$error$|$ (IQR shaded)")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(loc="upper left", framealpha=0.9, handlelength=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"phase_boundary.{ext}")
    plt.close(fig)
    print("phase_boundary written")


def fig_scaling():
    regimes = ["systematic", "nonlinear", "miscal"]
    titles = {"systematic": "Stochastic device",
              "nonlinear": "Stochastic-correlated",
              "miscal": "+ Coherent miscalibration"}
    methods = ["neural", "zne_adaptive", "cdr"]

    cells = {}
    for path in (RESULTS / "scaling").glob("*_n*.json"):
        d = json.load(open(path))
        regime, n = d["cell"].rsplit("_n", 1)
        cells[(regime, int(n))] = d

    sizes = sorted({n for (_, n) in cells})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, regime in zip(axes, regimes):
        for m in methods:
            xs, med_imp = [], []
            for n in sizes:
                if (regime, n) not in cells:
                    continue
                d = cells[(regime, n)]
                pi = d["per_instance"]
                ideal = np.array(pi["ideal"])
                raw_err = np.abs(np.array(pi["noisy"]) - ideal)
                vals = pi[m]
                err = np.abs(np.array(vals) - ideal[: len(vals)])
                ref = np.median(raw_err[: len(vals)])
                xs.append(n)
                med_imp.append((ref - np.median(err)) / ref * 100)
            ax.plot(xs, med_imp, marker=MARKERS[m], ms=4.5, lw=1.4,
                    color=COLORS[m], label=LABELS[m])
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("Qubits")
        ax.set_title(titles[regime])
        ax.set_xticks(sizes)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Median error reduction (%)")
    axes[0].legend(loc="lower left", framealpha=0.9, handlelength=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"scaling.{ext}")
    plt.close(fig)
    print("scaling written")


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_budget_pareto()
    fig_phase_boundary()
    fig_scaling()
