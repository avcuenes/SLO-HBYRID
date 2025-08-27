#!/usr/bin/env python3
# cec2022_mean_std.py — per-function × solver mean/std tables + plots
import argparse, pathlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ------------------------------- helpers ------------------------------------
def ensure_out(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def fmt_sci(x):
    if pd.isna(x): return ""
    return f"{x:.3e}"

def agg_mean_std(df: pd.DataFrame, metric: str, by=("fid","alg")):
    agg = (df.groupby(list(by))[metric]
             .agg(mean="mean", std="std", median="median", count="count")
             .reset_index()
             .sort_values(list(by)))
    mean_pivot = agg.pivot(index=by[0], columns=by[1], values="mean")
    std_pivot  = agg.pivot(index=by[0], columns=by[1], values="std")
    combo = mean_pivot.copy().astype(object)
    for i in combo.index:
        for j in combo.columns:
            m = mean_pivot.loc[i, j]
            s = std_pivot.loc[i, j]
            combo.loc[i, j] = f"{fmt_sci(m)} ± {fmt_sci(s)}" if pd.notna(m) and pd.notna(s) else ""
    return agg, mean_pivot, std_pivot, combo

def save_tables(mean_pivot, std_pivot, combo, out: pathlib.Path, tag=""):
    suffix = f"_{tag}" if tag else ""
    (out/f"err_mean_by_fid_alg{suffix}.csv").write_text(mean_pivot.to_csv())
    (out/f"err_std_by_fid_alg{suffix}.csv").write_text(std_pivot.to_csv())
    (out/f"err_mean_std_by_fid_alg{suffix}.csv").write_text(combo.to_csv())

    # LaTeX (note: pandas warns about future Styler API; this still works)
    (out/f"err_mean_by_fid_alg{suffix}.tex").write_text(
        mean_pivot.to_latex(float_format="%.3e",
                            caption=f"Mean metric per function{(' ('+tag+')') if tag else ''}",
                            label=f"tab:mean_{tag or 'all'}"))
    (out/f"err_std_by_fid_alg{suffix}.tex").write_text(
        std_pivot.to_latex(float_format="%.3e",
                           caption=f"Std dev per function{(' ('+tag+')') if tag else ''}",
                           label=f"tab:std_{tag or 'all'}"))
    (out/f"err_mean_std_by_fid_alg{suffix}.tex").write_text(
        combo.to_latex(escape=False,
                       caption=f"Mean ± Std per function{(' ('+tag+')') if tag else ''}",
                       label=f"tab:meanstd_{tag or 'all'}"))

def plot_heatmap_of_means(mean_pivot, out: pathlib.Path, tag="", log_eps=1e-16):
    vals = np.log10(mean_pivot + log_eps)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(vals.values, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(mean_pivot.columns)))
    ax.set_xticklabels(mean_pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mean_pivot.index)))
    ax.set_yticklabels([f"F{int(fid):02d}" for fid in mean_pivot.index])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(mean metric)")
    # annotate (small fonts)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = mean_pivot.iloc[i, j]
            if pd.notna(v): ax.text(j, i, f"{v:.1e}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Mean metric per function (log10){' - '+tag if tag else ''}")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(out/f"mean_heatmap{('_'+tag) if tag else ''}.{ext}", dpi=300)
    plt.close(fig)

def plot_per_function_bars(agg, out: pathlib.Path, tag=""):
    for fid, sub in agg.groupby("fid"):
        sub = sub.sort_values("mean")
        x = np.arange(len(sub["alg"]))
        y, e = sub["mean"].values, sub["std"].values
        fig, ax = plt.subplots(figsize=(6.5,4))
        ax.bar(x, y, yerr=e, capsize=3)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["alg"].tolist(), rotation=45, ha="right")
        ax.set_ylabel("metric (mean ± std)")
        ax.set_title(f"F{int(fid):02d}: mean ± std by solver{(' - '+tag) if tag else ''}")
        plt.tight_layout()
        for ext in ("png","pdf"):
            plt.savefig(out/f"bar_F{int(fid):02d}{('_'+tag) if tag else ''}.{ext}", dpi=300)
        plt.close(fig)

# ----------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=pathlib.Path,
                    help="CEC-2022 results CSV (needs columns: fid, alg, err or fbest, dim, run)")
    ap.add_argument("--out", default="appendix_figs", type=pathlib.Path)
    ap.add_argument("--metric", default="err", choices=["err","fbest"],
                    help="Which column to summarize")
    ap.add_argument("--dims", type=str, default="",
                    help="Optional comma-separated dimensions to include, e.g. 10,20")
    ap.add_argument("--split-by-dim", action="store_true",
                    help="If set, also produce per-dimension tables/plots")
    args = ap.parse_args()

    out = ensure_out(args.out)
    df = pd.read_csv(args.csv)

    # Select metric column
    metric = args.metric
    if metric not in df.columns:
        raise SystemExit(f"[ERROR] metric '{metric}' not found in CSV columns: {list(df.columns)}")

    # Optional filter by dims
    dims = None
    if args.dims:
        dims = [int(x) for x in args.dims.split(",") if x.strip()]
        df = df[df["dim"].isin(dims)]
        if df.empty:
            raise SystemExit("[ERROR] No rows left after filtering by --dims")

    # ---------- overall (all dims together) ----------
    agg, mean_pivot, std_pivot, combo = agg_mean_std(df, metric, by=("fid","alg"))
    save_tables(mean_pivot, std_pivot, combo, out, tag="")
    plot_heatmap_of_means(mean_pivot, out, tag="")
    plot_per_function_bars(agg, out, tag="")

    # ---------- per-dimension splits (optional) ----------
    if args.split_by_dim:
        for d, sub in df.groupby("dim"):
            tag = f"D{int(d)}"
            subdir = ensure_out(out/f"by_dim_{int(d)}")
            agg_d, mean_pivot_d, std_pivot_d, combo_d = agg_mean_std(sub, metric, by=("fid","alg"))
            save_tables(mean_pivot_d, std_pivot_d, combo_d, subdir, tag=tag)
            plot_heatmap_of_means(mean_pivot_d, subdir, tag=tag)
            plot_per_function_bars(agg_d, subdir, tag=tag)

    print("[OK] Wrote tables and figures to:", out.resolve())

if __name__ == "__main__":
    main()
