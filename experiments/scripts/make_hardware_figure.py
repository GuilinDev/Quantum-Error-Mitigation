#!/usr/bin/env python3
"""Hardware figure: folding ZNE fails on a real IBM device.

Main-text panel pair (ibm_marrakesh, 4 qubits, 2 and 5 ansatz layers):
per-instance absolute error of unmitigated execution vs folding ZNE.
ZNE's mean error is several times the unmitigated error and exceeds it
on 80-88% of instances -- the failure mode the paper predicts, on real
silicon. The neural mitigator's behaviour (a sim-to-real overcorrection)
is reported separately in the appendix, not here.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
HW = ROOT / "experiments" / "results" / "hardware"
FIGDIR = ROOT / "paper" / "figures"

COLORS = {"raw": "#666666", "zne_richardson": "#E69F00"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def panel(ax, cell, title):
    pi = cell["per_instance"]
    ideal = np.array(pi["ideal"])
    raw_err = np.abs(np.array(pi["raw"]) - ideal)
    zne_err = np.abs(np.array(pi["zne_richardson"]) - ideal)
    x = np.arange(len(ideal))
    w = 0.38
    ax.bar(x - w / 2, raw_err, w, color=COLORS["raw"], label="Unmitigated")
    ax.bar(x + w / 2, zne_err, w, color=COLORS["zne_richardson"],
           label="ZNE-Richardson (3 exec)")
    ax.axhline(raw_err.mean(), color=COLORS["raw"], lw=1.0, ls="--")
    ax.axhline(zne_err.mean(), color=COLORS["zne_richardson"], lw=1.0, ls="--")
    worse = (zne_err > raw_err).mean() * 100
    ax.set_title(f"{title}\nZNE worse on {worse:.0f}% of instances "
                 f"(MAE {zne_err.mean()/raw_err.mean():.1f}$\\times$ raw)")
    ax.set_xlabel("Test instance")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3, lw=0.4)


def main():
    d2 = list(json.load(open(HW / "hardware_marrakesh_n4L2.json"))["cells"].values())[0]
    d5 = list(json.load(open(HW / "hardware_marrakesh_n4L5.json"))["cells"].values())[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    panel(axes[0], d2, "4 qubits, 2 layers")
    panel(axes[1], d5, "4 qubits, 5 layers")
    axes[0].set_ylabel("Absolute error $|\\langle H\\rangle - \\langle H\\rangle_{\\mathrm{ideal}}|$")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=7.5)
    fig.suptitle("ibm_marrakesh (real hardware, 156-qubit Heron r2)",
                 y=1.06, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"hardware_zne.{ext}")
    plt.close(fig)
    print("hardware_zne figure written")


if __name__ == "__main__":
    main()
