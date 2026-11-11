#!/usr/bin/env python3
"""Hardware figure: folding ZNE fails on a real IBM device.

Main-text panel pair (ibm_marrakesh, 4 qubits, 2 and 5 ansatz layers):
per-instance absolute error of unmitigated execution vs folding ZNE.
ZNE's mean error is several times the unmitigated error and exceeds it
on 80-88% of instances -- the failure mode the paper predicts, on real
silicon. The twin-trained neural mitigator from the same runs is shown
alongside (it does not transfer; see the appendix for on-device fine-tuning).
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

COLORS = {"raw": "#666666", "zne_richardson": "#E69F00", "neural": "#0072B2"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def panel(ax, cell, title):
    pi = cell["per_instance"]
    ideal = np.array(pi["ideal"])
    raw_err = np.abs(np.array(pi["raw"]) - ideal)
    zne_err = np.abs(np.array(pi["zne_richardson"]) - ideal)
    nem_err = np.abs(np.array(pi["neural"]) - ideal)
    x = np.arange(len(ideal))
    w = 0.27
    ax.bar(x - w, raw_err, w, color=COLORS["raw"], label="Unmitigated")
    ax.bar(x, zne_err, w, color=COLORS["zne_richardson"],
           label="ZNE-Richardson (3 exec)")
    ax.bar(x + w, nem_err, w, color=COLORS["neural"],
           label="NEM, twin-trained (1 exec)")
    for e, k in ((raw_err, "raw"), (zne_err, "zne_richardson"), (nem_err, "neural")):
        ax.axhline(e.mean(), color=COLORS[k], lw=1.0, ls="--")
    zw = (zne_err > raw_err).mean() * 100
    nw = (nem_err > raw_err).mean() * 100
    ax.set_title(f"{title}\nworse than raw: ZNE {zw:.0f}%, NEM (twin) {nw:.0f}%")
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
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=8)
    fig.suptitle("ibm_marrakesh (real hardware, 156-qubit Heron r2)",
                 y=1.06, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"hardware_zne.{ext}")
    plt.close(fig)
    print("hardware_zne figure written")


if __name__ == "__main__":
    main()
