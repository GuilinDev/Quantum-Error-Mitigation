"""Physical-range clipping baseline (reviewer request, QMI 2026-11).

The mean-magnetization observable lies in [-1, 1]; projecting any estimate onto
that interval can only reduce its absolute error and costs no extra
measurements.  This script post-processes the saved per-instance estimates
(no re-simulation) and reports, for every method, the mean error, the maximum
error and the worse-than-unmitigated rate before and after clipping.

Usage:  python experiments/scripts/clip_analysis.py
Writes: experiments/results/clip_analysis.json, paper/tables/clip_full.tex
"""
import json
from pathlib import Path

import numpy as np

RES = Path("experiments/results")
METHODS = ("neural", "zne_richardson", "zne_exponential", "zne_adaptive", "cdr")


def stats(pred, ideal, raw_err):
    pred = np.asarray(pred, dtype=float)
    n = len(pred)
    err = np.abs(pred - ideal[:n])
    re = raw_err[:n]
    return {
        "mae": float(err.mean()),
        "median_ae": float(np.median(err)),
        "max_ae": float(err.max()),
        "improvement_pct": float((re.mean() - err.mean()) / re.mean() * 100),
        "worse_than_raw_rate": float((err > re).mean()),
        "n": n,
    }


def analyze(pi_est, ideal, noisy):
    ideal = np.asarray(ideal, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    raw_err = np.abs(np.clip(noisy, -1, 1) - ideal)
    out = {}
    for m in METHODS:
        if m not in pi_est:
            continue
        p = np.asarray(pi_est[m], dtype=float)
        out[m] = {"unclipped": stats(p, ideal, raw_err),
                  "clipped": stats(np.clip(p, -1, 1), ideal, raw_err),
                  "frac_out_of_range": float((np.abs(p) > 1).mean())}
    return out


def main():
    result = {"scaling": {}, "budget": {}}
    for path in sorted((RES / "scaling").glob("*_n*.json")):
        d = json.load(open(path))
        pi = d["per_instance"]
        result["scaling"][d["cell"]] = analyze(pi, pi["ideal"], pi["noisy"])
    for regime in ("miscal", "systematic"):
        d = json.load(open(RES / "budget" / f"budget_sweep_{regime}.json"))
        result["budget"][regime] = {}
        for b, cell in d["sweep"].items():
            pi = cell["per_instance"]
            result["budget"][regime][b] = analyze(pi, d["ideal"], pi["raw"])
    (RES / "clip_analysis.json").write_text(json.dumps(result, indent=1))

    def row(tag, cell):
        parts = [f"{tag:18s}"]
        for m in ("neural", "zne_richardson", "zne_exponential", "zne_adaptive", "cdr"):
            if m in cell:
                u, c = cell[m]["unclipped"], cell[m]["clipped"]
                parts.append(f"{m[:8]:>8s} {u['improvement_pct']:+8.0f}->{c['improvement_pct']:+5.0f}% "
                             f"w{u['worse_than_raw_rate']*100:3.0f}->{c['worse_than_raw_rate']*100:3.0f} "
                             f"max{u['max_ae']:7.2f}->{c['max_ae']:4.2f}")
        print(" | ".join(parts))

    for k, v in result["scaling"].items():
        row(k, v)
    for r, cells in result["budget"].items():
        for b, v in cells.items():
            row(f"budget_{r}_{b}", v)


def latex_table():
    """Appendix table: ZNE before/after projection onto [-1, 1]."""
    r = json.load(open(RES / "clip_analysis.json"))
    L = [r"\begin{table}[t]",
         r"\caption{ZNE before and after projecting every estimate onto the physical range $[-1,1]$ "
         r"(structured-regime scaling cells; instances and shot counts as in Table~\ref{tab:scaling_full}). Projection "
         r"removes the divergent outliers (max error $\le 1.1$) but leaves the "
         r"worse-than-unmitigated rate unchanged in every cell (all out-of-range estimates were worse than "
         r"unmitigated before and after projection), and the mean improvement stays negative in most cells.}",
         r"\label{tab:clip_full}", r"\centering\small",
         r"\begin{tabular}{l|ccc|ccc|c}", r"\toprule",
         r" & \multicolumn{3}{c|}{ZNE-exponential} & \multicolumn{3}{c|}{ZNE-adaptive} & \\",
         r"Cell & raw fit & clipped & max & raw fit & clipped & max & worse (exp/ada) \\",
         r"\midrule"]
    names = {"miscal": "Coherent miscal.", "nonlinear": "Stochastic-corr."}
    neg = tot = 0
    for reg in ("miscal", "nonlinear"):
        cells = sorted((k for k in r["scaling"] if k.startswith(reg + "_n")),
                       key=lambda k: int(k.rsplit("_n", 1)[1]))
        for k in cells:
            c = r["scaling"][k]
            e, a = c["zne_exponential"], c["zne_adaptive"]
            def f(x):
                v = x["improvement_pct"]
                return f"${v:+.0f}\\%$" if abs(v) < 1e4 else r"$<\!-10^4\%$"
            for m in (e, a):
                tot += 1
                neg += m["clipped"]["improvement_pct"] < 0
            n = k.rsplit("_n", 1)[1]
            L.append(f"{names[reg]}, $n{{=}}{n}$ & {f(e['unclipped'])} & {f(e['clipped'])} & "
                     f"{e['clipped']['max_ae']:.2f} & {f(a['unclipped'])} & {f(a['clipped'])} & "
                     f"{a['clipped']['max_ae']:.2f} & "
                     f"{e['clipped']['worse_than_raw_rate']*100:.0f}/{a['clipped']['worse_than_raw_rate']*100:.0f}\\% \\\\")
        L.append(r"\midrule" if reg == "miscal" else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table}"]
    Path("paper/tables/clip_full.tex").write_text("\n".join(L) + "\n")
    print(f"clipped ZNE mean improvement negative in {neg}/{tot} structured (cell, variant) pairs")


if __name__ == "__main__":
    main()
    latex_table()
