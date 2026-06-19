#!/usr/bin/env python3
"""Emit LaTeX tables from experiment result JSONs into paper/tables/.

Tables:
  scaling_full.tex   — per-cell mean improvement for every method/size/regime
  budget_full.tex    — budget sweep means per method/budget/regime
  safety_full.tex    — tail statistics per structured-regime cell

Run after any experiment update; the paper \\input{}s these files.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "tables"

METHOD_ORDER = ["neural", "direct", "zne_richardson", "zne_exponential",
                "zne_adaptive", "cdr"]
METHOD_TEX = {
    "neural": "NEM (ours)",
    "direct": "Direct prediction",
    "zne_richardson": "ZNE-Richardson",
    "zne_exponential": "ZNE-exponential",
    "zne_adaptive": "ZNE-adaptive",
    "cdr": "CDR",
}
REGIME_TEX = {
    "systematic": "Stochastic device",
    "nonlinear": "Stochastic-correlated",
    "miscal": "Coherent miscalibration",
}


def fmt_pct(x):
    if abs(x) >= 10000:
        return r"$<-10^4$"
    return f"${x:+.0f}$"


def load_cells():
    cells = {}
    for path in (RESULTS / "scaling").glob("*_n*.json"):
        d = json.load(open(path))
        regime, n = d["cell"].rsplit("_n", 1)
        cells[(regime, int(n))] = d
    return cells


def scaling_table(cells):
    sizes = sorted({n for (_, n) in cells})
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Mean error reduction vs.\ unmitigated execution (\%) for every cell of the scaling study. "
        r"150 test instances per cell (100 at $n{=}20$); per-execution shots $S{=}8192$ ($n\le10$), "
        r"$1024$ ($n{=}12,16$), $512$ ($n{=}20$). Asterisks mark cells where the NEM improvement is "
        r"not significant at $p<0.05$ (paired $t$-test).}",
        r"\label{tab:scaling_full}",
        r"\centering\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "c" * len(sizes) + "}",
        r"\toprule",
        "Regime & Method & " + " & ".join(f"$n{{=}}{n}$" for n in sizes) + r" \\",
        r"\midrule",
    ]
    for regime in ["systematic", "nonlinear", "miscal"]:
        for i, m in enumerate(METHOD_ORDER):
            row = [REGIME_TEX[regime] if i == 0 else "", METHOD_TEX[m]]
            for n in sizes:
                d = cells.get((regime, n))
                if d is None or m not in d["results"]["improvement_pct"]:
                    row.append("---")
                    continue
                v = fmt_pct(d["results"]["improvement_pct"][m])
                if m == "neural" and d["results"]["p_vs_raw"]["neural"] >= 0.05:
                    v += r"$^{*}$"
                row.append(v)
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule" if regime != "miscal" else r"\bottomrule")
    lines += [r"\end{tabular}", r"}", r"\end{table*}"]
    return "\n".join(lines)


def budget_table():
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Equal-budget sweep at $n{=}8$ (100 instances): mean error reduction vs.\ "
        r"unmitigated execution (\%) when every method spends the same total budget $B$ per "
        r"evaluation point, split across its required executions.}",
        r"\label{tab:budget_full}",
        r"\centering\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Regime & Method & $B{=}2^{10}$ & $B{=}2^{12}$ & $B{=}2^{14}$ & $B{=}2^{16}$ \\",
        r"\midrule",
    ]
    for regime in ["systematic", "miscal"]:
        d = json.load(open(RESULTS / "budget" / f"budget_sweep_{regime}.json"))
        methods = ["neural", "zne_exponential", "zne_adaptive", "cdr"]
        for i, m in enumerate(methods):
            row = [REGIME_TEX[regime] if i == 0 else "", METHOD_TEX[m]]
            for B in d["budgets"]:
                row.append(fmt_pct(d["sweep"][str(B)]["improvement_pct"][m]))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule" if regime != "miscal" else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def safety_table():
    data = json.load(open(RESULTS / "scaling" / "safeguard_analysis.json"))
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Tail statistics across structured-regime cells: median and maximum absolute "
        r"error, and the fraction of instances on which each method returns a worse answer than "
        r"unmitigated execution.}",
        r"\label{tab:safety_full}",
        r"\centering\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccc|ccc|ccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c|}{NEM (ours)} & \multicolumn{3}{c|}{ZNE (best of three)} & \multicolumn{3}{c}{CDR} \\",
        r"Cell & Raw MAE & med & max & worse & med & max & worse & med & max & worse \\",
        r"\midrule",
    ]
    for cell in sorted(data, key=lambda c: c["cell"]):
        name = cell["cell"]
        if not name.startswith(("miscal", "nonlinear")):
            continue
        regime, n = name.rsplit("_n", 1)
        regime_tex = {"miscal": "Coherent miscal.",
                      "nonlinear": "Stochastic-corr."}[regime]
        label = f"{regime_tex}, $n{{=}}{n}$"
        rows = {r["method"]: r for r in cell["rows"]}
        raw = rows["raw"]
        nem = rows["neural(best-seed)"]
        znes = [rows[k] for k in ("zne_richardson", "zne_exponential", "zne_adaptive") if k in rows]
        zne = min(znes, key=lambda r: r["mae"])
        cdr = rows.get("cdr")
        def trio(r):
            return f"{r['median_ae']:.4f} & {r['max_ae']:.2f} & {r['worse_than_raw_rate']*100:.0f}\\%"
        lines.append(
            f"{label} & {raw['mae']:.4f} & "
            f"{trio(nem)} & {trio(zne)} & {trio(cdr)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    (OUT / "scaling_full.tex").write_text(scaling_table(cells) + "\n")
    (OUT / "budget_full.tex").write_text(budget_table() + "\n")
    (OUT / "safety_full.tex").write_text(safety_table() + "\n")
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables to {OUT}")


if __name__ == "__main__":
    main()
