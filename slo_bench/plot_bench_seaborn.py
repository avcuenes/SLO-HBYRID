#!/usr/bin/env python3
import argparse, pathlib, sys, re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REQUIRED = {"alg","prob","dim","run","f_best","err","nfe","time_sec"}

def slugify(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()

def load_all(dirpath: pathlib.Path) -> pd.DataFrame:
    # read all per-problem CSVs (ignore known summaries)
    skip_names = {"raw_results.csv", "summary_all.csv"}
    files = [p for p in dirpath.glob("*.csv") if p.name not in skip_names and not p.name.endswith("_summary.csv") and "pivot_" not in p.name]
    if not files:
        print(f"[error] no CSV files found under {dirpath}", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for p in sorted(files):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] failed to read {p.name}: {e}", file=sys.stderr)
            continue
        miss = REQUIRED - set(df.columns)
        if miss:
            print(f"[warn] {p.name} missing {miss}; skipping", file=sys.stderr)
            continue
        # coerce numerics
        for c in ["dim","run","f_best","err","nfe","time_sec"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # drop rows with non-finite err or f_best
        ok = np.isfinite(df["err"].values) & np.isfinite(df["f_best"].values)
        df = df.loc[ok].copy()
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("[error] no valid CSVs after parsing", file=sys.stderr)
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)

def stats_per_alg(dfp: pd.DataFrame) -> pd.DataFrame:
    # mean/std across runs
    out = (dfp
           .groupby("alg", as_index=False)
           .agg(err_mean=("err","mean"),
                err_std =("err","std"),
                f_mean  =("f_best","mean"),
                f_std   =("f_best","std"),
                t_mean  =("time_sec","mean"),
                t_std   =("time_sec","std"),
                n_runs  =("run","nunique")))
    out[["err_std","f_std","t_std"]] = out[["err_std","f_std","t_std"]].fillna(0.0)
    # sort best→worst by error
    out = out.sort_values("err_mean", ascending=True)
    return out

def plot_err_bar(stats: pd.DataFrame, prob: str, outdir: pathlib.Path, dpi: int, fmts: list[str]):
    order = list(stats["alg"])
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 0.5*len(order)+1.5))
    sns.barplot(data=stats, x="err_mean", y="alg", order=order, ax=ax, color=sns.color_palette()[0], ci=None)
    # error bars (± std)
    ax.errorbar(stats["err_mean"], np.arange(len(order)),
                xerr=stats["err_std"], fmt="none", ecolor="black", elinewidth=1, capsize=3)
    ax.set_xlabel("Mean error (lower is better)")
    ax.set_ylabel("")
    ax.set_title(f"{prob} — mean ± std of error")
    ax.margins(x=0.02)
    for fmt in fmts:
        path = outdir / f"{slugify(prob)}_err_bar.{fmt}"
        fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[ok] wrote {path}")
    plt.close(fig)

def plot_err_box(dfp: pd.DataFrame, stats: pd.DataFrame, prob: str, outdir: pathlib.Path, dpi: int, fmts: list[str]):
    order = list(stats["alg"])
    sns.set_theme(context="paper", style="white", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 0.5*len(order)+1.5))
    sns.boxplot(data=dfp, x="err", y="alg", order=order, orient="h", showfliers=False, ax=ax)
    sns.stripplot(data=dfp, x="err", y="alg", order=order, orient="h", size=3, alpha=0.35, color="k", ax=ax)
    ax.set_xlabel("Error across runs")
    ax.set_ylabel("")
    ax.set_title(f"{prob} — distribution of error by algorithm")
    ax.margins(x=0.02)
    for fmt in fmts:
        path = outdir / f"{slugify(prob)}_err_box.{fmt}"
        fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[ok] wrote {path}")
    plt.close(fig)

def plot_time_bar(stats: pd.DataFrame, prob: str, outdir: pathlib.Path, dpi: int, fmts: list[str], logx: bool):
    stats_time = stats.dropna(subset=["t_mean"])
    if stats_time.empty:
        return
    order = list(stats_time.sort_values("t_mean")["alg"])
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 0.5*len(order)+1.5))
    sns.barplot(data=stats_time, x="t_mean", y="alg", order=order, ax=ax, color=sns.color_palette()[2], ci=None)
    ax.errorbar(stats_time.set_index("alg").loc[order]["t_mean"],
                np.arange(len(order)),
                xerr=stats_time.set_index("alg").loc[order]["t_std"].fillna(0.0),
                fmt="none", ecolor="black", elinewidth=1, capsize=3)
    ax.set_xlabel("Mean time per run (s)" + (" [log scale]" if logx else ""))
    ax.set_ylabel("")
    ax.set_title(f"{prob} — runtime mean ± std")
    if logx:
        ax.set_xscale("log")
    ax.margins(x=0.02)
    for fmt in fmts:
        path = outdir / f"{slugify(prob)}_time_bar.{fmt}"
        fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[ok] wrote {path}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Per-problem seaborn plots for benchmark results")
    ap.add_argument("--dir", default="eng_results", help="Folder with per-problem CSVs")
    ap.add_argument("--out", default="figs_per_problem", help="Output folder")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fmt", nargs="+", default=["png"], help="Image formats (e.g., png svg pdf)")
    ap.add_argument("--logtime", action="store_true", help="Use log scale for time plots")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    outroot = pathlib.Path(args.out)
    outroot.mkdir(parents=True, exist_ok=True)

    df = load_all(root)

    for prob, dfp in df.groupby("prob"):
        subdir = outroot / slugify(prob)
        subdir.mkdir(parents=True, exist_ok=True)
        stats = stats_per_alg(dfp)
        plot_err_bar(stats, prob, subdir, args.dpi, args.fmt)
        plot_err_box(dfp, stats, prob, subdir, args.dpi, args.fmt)
        plot_time_bar(stats, prob, subdir, args.dpi, args.fmt, logx=args.logtime)

if __name__ == "__main__":
    main()
