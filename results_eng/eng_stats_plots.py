#!/usr/bin/env python3
# eng_stats_plots.py
"""
Compute mean ± std error per engineering problem × algorithm and
plot seaborn bar charts with std error bars.

Usage:
    python eng_stats_plots.py --in results_eng.csv --out eng_plots

The script tries to find columns flexibly:
- algorithm: one of {alg, algorithm, method, name}
- problem id: one of {fid, problem_id, pid, prob, problem}
- error: 'err' (preferred). If missing, uses (fbest - fopt) or 'fbest'.

Outputs:
- <out>/eng_mean_std_tidy.csv  (Problem, Algorithm, mean_err, std_err, runs)
- <out>/eng_mean_table.csv     (wide table of means)
- <out>/eng_std_table.csv      (wide table of stds)
- <out>/<fid>_<Problem>_error_bar.png  (seaborn barplot with std error bars)
- <out>/all_problems_grid.png  (optional small-multiples grid)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- CLI ----------
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--in", dest="inp", required=True, help="Input CSV with run-level results")
ap.add_argument("--out", dest="out", default="eng_plots", help="Output folder")
ap.add_argument("--style", default="whitegrid", help="seaborn style")
ap.add_argument("--palette", default="tab10", help="seaborn palette")
ap.add_argument("--order-by", choices=["mean", "name"], default="mean",
                help="Per-plot x-axis order: by ascending mean error or by algorithm name")
ap.add_argument("--dpi", type=int, default=200)
args = ap.parse_args()

outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
sns.set_theme(style=args.style)

# ---------- Load & detect columns ----------
df = pd.read_csv(args.inp)
lc = {c.lower(): c for c in df.columns}

def pick(cols):
    for c in cols:
        if c.lower() in lc: return lc[c.lower()]
    raise KeyError(f"Expected one of {cols} in columns {list(df.columns)}")

col_alg = pick(["alg","algorithm","method","name"])
col_fid = pick(["fid","problem_id","pid","prob","problem"])
# Prefer 'err'; else compute from fbest-fopt; else fall back to fbest
if "err" in lc:
    col_err = lc["err"]
elif "fbest" in lc and "fopt" in lc:
    df["__err"] = pd.to_numeric(df[lc["fbest"]], errors="coerce") - pd.to_numeric(df[lc["fopt"]], errors="coerce")
    col_err = "__err"
elif "fbest" in lc:
    col_err = lc["fbest"]
else:
    raise KeyError("Could not find error metric: need 'err' or 'fbest' (optionally 'fopt').")

# Clean types
df[col_alg] = df[col_alg].astype(str)
df[col_fid] = pd.to_numeric(df[col_fid], errors="coerce").astype(int)
df[col_err] = pd.to_numeric(df[col_err], errors="coerce")

# Optional mapping from fid -> human name
FID2NAME = {
    1: "Pressure Vessel",
    2: "Tension Spring",
    3: "Welded Beam",
    4: "Gear Box",
    5: "Speed Reducer",
    6: "Car Side Impact",
    7: "Hydrostatic Bearing",
    8: "Four-Bar Truss",
    9: "Ten-Bar Truss",
    10: "Cantilever (cont)",
    11: "Cantilever (disc)",
    12: "Stepped Column",
    13: "Machining Cost",
    14: "Heat Exchanger",
    15: "Thick Pressure Vessel",
    16: "Gear Train",
}

# ---------- Aggregate mean / std ----------
agg = (
    df.groupby([col_fid, col_alg])[col_err]
      .agg(mean_err="mean", std_err="std", runs="size")
      .reset_index()
)
agg["Problem"] = agg[col_fid].map(FID2NAME).fillna(agg[col_fid].astype(str))
agg.rename(columns={col_alg: "Algorithm"}, inplace=True)
agg = agg[["Problem", col_fid, "Algorithm", "mean_err", "std_err", "runs"]]

# Save tables
agg.sort_values([col_fid, "mean_err"]).to_csv(outdir/"eng_mean_std_tidy.csv", index=False)
mean_wide = agg.pivot_table(index="Problem", columns="Algorithm", values="mean_err")
std_wide  = agg.pivot_table(index="Problem", columns="Algorithm", values="std_err")
mean_wide.to_csv(outdir/"eng_mean_table.csv")
std_wide.to_csv(outdir/"eng_std_table.csv")

# ---------- Per-problem seaborn barplots with std error bars ----------
def draw_one(sub_df, fid, prob_name):
    # Order algorithms
    if args.order_by == "mean":
        order = sub_df.sort_values("mean_err")["Algorithm"].tolist()
    else:
        order = sorted(sub_df["Algorithm"].unique())
    # Base barplot (no automatic ci/err)
    plt.figure(figsize=(max(6, 0.6*len(order)), 4))
    ax = sns.barplot(data=sub_df, x="Algorithm", y="mean_err", order=order,
                     palette=args.palette, errorbar=None)
    # Add std error bars manually
    for i, alg in enumerate(order):
        row = sub_df[sub_df["Algorithm"] == alg].iloc[0]
        ax.errorbar(i, row["mean_err"], yerr=row["std_err"], fmt='none', ecolor='black', elinewidth=1, capsize=3)
        # Annotate value
        ax.text(i, row["mean_err"], f"{row['mean_err']:.2g}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Error (mean ± std)")
    ax.set_xlabel("")
    ax.set_title(f"{prob_name} — mean error ± std by algorithm")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out_png = outdir / f"{int(fid):02d}_{prob_name.replace(' ', '_').replace('/', '-')}_error_bar.png"
    plt.savefig(out_png, dpi=args.dpi)
    plt.close()
    return out_png

saved = []
for fid, sub in agg.groupby(col_fid):
    pname = sub["Problem"].iloc[0]
    saved.append(draw_one(sub, fid, pname))

# ---------- Optional: small-multiples grid (one panel per problem) ----------
# Works best when #problems <= 12–16
try:
    grid = agg.copy()
    # order algorithms globally by overall mean error
    global_order = (
        agg.groupby("Algorithm")["mean_err"].mean().sort_values().index.tolist()
    )
    g = sns.catplot(
        data=grid, x="Algorithm", y="mean_err", col="Problem",
        kind="bar", order=global_order, col_wrap=4, sharey=False,
        height=3.2, aspect=1.2, palette=args.palette, errorbar=None
    )
    # Manually add std error bars per panel
    for ax, (prob, sub) in zip(g.axes.flat, grid.groupby("Problem")):
        sub = sub.set_index("Algorithm").reindex(global_order).dropna(subset=["mean_err"])
        xs = np.arange(len(sub.index))
        ax.errorbar(xs, sub["mean_err"].values, yerr=sub["std_err"].values,
                    fmt='none', ecolor='black', elinewidth=1, capsize=2)
        ax.set_xlabel("")
        ax.set_ylabel("Error")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    g.figure.tight_layout()
    grid_png = outdir/"all_problems_grid.png"
    g.figure.savefig(grid_png, dpi=args.dpi)
except Exception as e:
    # If anything goes wrong with the grid (e.g., too many problems), just skip it
    grid_png = None

print(f"[OK] Saved tables to: {outdir}")
print(f"[OK] Saved per-problem plots ({len(saved)} files) to: {outdir}")
if grid_png:
    print(f"[OK] Saved grid figure: {grid_png}")
