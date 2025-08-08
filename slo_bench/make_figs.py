#!/usr/bin/env python3
# make_figs.py  ─ generate tables & plots for the engineering benchmark
# ---------------------------------------------------------------------
#   • Produces LaTeX table  :  table_summary.tex
#   • Saves figures (PNG + PDF) into --outdir
# ---------------------------------------------------------------------
import argparse, os, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None   # heat-map falls back to plain matplotlib

# ───────────────────────────── helpers ────────────────────────────────
def load_results(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "hit" not in df.columns:
        df["hit"] = df["err"] <= TOL
    return df

def make_latex_table(df: pd.DataFrame, out: pathlib.Path, tol: float):
    """One row per algorithm, aggregated over all problems & runs."""
    tab = (df.groupby("alg")
             .agg(mean_f=("f_best","mean"),
                  std_f =("f_best","std"),
                  suc_rate=("hit","mean"),
                  mean_ert=("nfe", lambda s: s.sum()/max(1,s[df.loc[s.index,"hit"]].sum())))
             .sort_values("mean_f"))
    tab["suc_rate"] = (100*tab["suc_rate"]).round(1).astype(str) + r"\%"
    tex = tab.to_latex(float_format="%.3g",
                       column_format="lcccc",
                       caption=f"Aggregate performance over all problems "
                               f"(tol=$\\le {tol:g}$).",
                       label="tab:summary",
                       bold_rows=True)
    out.write_text(tex)
    print(f"saved → {out}")

def fig_mean_best(df: pd.DataFrame, out_prefix: pathlib.Path):
    mean_tbl = (df.groupby(["prob","alg"])["f_best"]
                  .mean()
                  .unstack())
    mean_tbl.plot(kind="bar", figsize=(12,4))
    plt.ylabel("Mean penalised best $f$")
    plt.title("Mean best value – each problem and algorithm")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=300)
    plt.close()

def fig_ert(df: pd.DataFrame, tol: float, out_prefix: pathlib.Path):
    erts = []
    for (prob, alg), sub in df.groupby(["prob","alg"]):
        succ = sub["hit"].sum()
        tot  = sub["nfe"].sum()
        erts.append({"prob":prob, "alg":alg,
                     "ERT": tot/succ if succ else np.nan})
    ert_df = pd.DataFrame(erts).pivot(index="prob", columns="alg", values="ERT")
    ert_df.plot(kind="bar", figsize=(12,4))
    plt.ylabel("ERT (function evaluations)")
    plt.title(f"Expected runtime to reach tol ≤ {tol:g}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=300)
    plt.close()

def fig_success_heatmap(df: pd.DataFrame, tol: float, out_prefix: pathlib.Path):
    hit_tbl = (df.groupby(["prob","alg"])["hit"]
                 .mean()
                 .pivot(index="prob", columns="alg"))
    plt.figure(figsize=(8,5))
    if sns:
        sns.heatmap(hit_tbl*100, annot=True, fmt=".0f",
                    cmap="YlGnBu", cbar_kws={"label":"success %"},
                    vmin=0, vmax=100)
    else:                                       # fallback
        plt.imshow(hit_tbl*100, cmap="YlGnBu", vmin=0, vmax=100)
        plt.colorbar(label="success %")
        plt.xticks(range(len(hit_tbl.columns)), hit_tbl.columns, rotation=45, ha="right")
        plt.yticks(range(len(hit_tbl.index)), hit_tbl.index)
        for (i,j), val in np.ndenumerate(hit_tbl.values):
            plt.text(j+0.5,i+0.5,f"{val*100:.0f}",
                     ha="center",va="center",color="black")
    plt.title(f"Success rate (tol ≤ {tol:g})")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=300)
    plt.close()

# ─────────────────────────────── CLI ─────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv",    required=True, type=pathlib.Path,
                    help="CSV file produced by eng benchmark (raw_results.csv)")
    ap.add_argument("--outdir", default="article_figs", type=pathlib.Path)
    ap.add_argument("--tol",    default=1e-6, type=float,
                    help="tolerance used for success & ERT")
    args = ap.parse_args()

    CSV   = args.csv.resolve()
    OUT   = args.outdir.resolve(); OUT.mkdir(exist_ok=True)
    TOL   = args.tol

    df = load_results(CSV)
    print(f"Rows loaded: {len(df)} | algorithms: {df.alg.unique().tolist()} "
          f"| problems: {df.prob.nunique()}")

    # 1) LaTeX table
    make_latex_table(df, OUT/"table_summary.tex", TOL)

    # 2) Figures
    fig_mean_best(df, OUT/"fig_mean_best")
    fig_ert(df, TOL, OUT/"fig_ert")
    fig_success_heatmap(df, TOL, OUT/"fig_success_heatmap")

    print(f"all plots and table saved in → {OUT}")
