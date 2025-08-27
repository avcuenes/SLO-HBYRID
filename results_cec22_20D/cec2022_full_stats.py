#!/usr/bin/env python3
# cec2022_full_stats.py — full statistical analysis for CEC-2022 results
import argparse, pathlib, warnings, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

# ---------------- IO ----------------
def load_runs(csv: pathlib.Path, tol: float, dims=None) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if "hit" not in df.columns:
        df["hit"] = df["err"] <= tol
    if dims is not None:
        df = df[df["dim"].isin(dims)]
    # standardize evals column name
    if "evals" not in df.columns:
        if "nfe" in df.columns:
            df = df.rename(columns={"nfe": "evals"})
        else:
            warnings.warn("No 'evals' or 'nfe' column; convergence plots & budgets limited.")
    return df

def collapse_ert(ert_csv: pathlib.Path, dims=None) -> pd.DataFrame:
    ert = pd.read_csv(ert_csv)
    if dims is not None:
        ert = ert[ert["dim"].isin(dims)]
    return ert.groupby(["fid","alg"], as_index=False)["ERT"].mean()

# ---------------- Helpers ----------------
def holm_bonferroni(pvals_dict):
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    for i, ((a,b), p) in enumerate(items, start=1):
        p_adj = min(1.0, (m - i + 1) * p)
        out[(a,b)] = p_adj
    return out

def _compute_cd(mean_ranks: pd.Series, n_datasets: int, alpha=0.05) -> float:
    # Nemenyi CD = q_alpha * sqrt(k(k+1)/(6N)), q_alpha = q_{1-alpha, k, inf}/sqrt(2)
    k = len(mean_ranks)
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    return float(q_alpha * np.sqrt(k*(k+1)/(6.0*n_datasets)))

def _coverage_table(pivot: pd.DataFrame) -> pd.DataFrame:
    """How many functions each algorithm has values for."""
    return pivot.notna().sum(axis=0).rename("n_funcs").to_frame().sort_values("n_funcs", ascending=False)

def _largest_complete_submatrix(pivot: pd.DataFrame, min_algs: int = 2, min_fids: int = 2):
    """
    Greedily drop the worst-covered algorithms until at least `min_fids`
    complete rows exist among the remaining algorithms.
    Returns: (submatrix with no NaNs, dropped_algs list)
    """
    algs = list(pivot.columns)
    dropped = []
    while True:
        # rows that are complete for the *current* alg set
        rows = pivot.dropna(axis=0, subset=algs).index
        if len(rows) >= min_fids and len(algs) >= min_algs:
            # done: return the NaN-free submatrix
            return pivot.loc[rows, algs], dropped

        # can’t reduce further?
        if len(algs) <= min_algs:
            # no valid ≥2×2 submatrix
            return pivot.loc[[]], dropped

        # coverage ONLY over remaining algs (the bug was here)
        cov = pivot[algs].notna().sum(axis=0)

        # if everything is zero coverage (weird edge case), bail
        if cov.empty or cov.max() == 0:
            return pivot.loc[[]], dropped

        # drop the currently worst-covered alg
        worst = cov.idxmin()
        if worst in algs:
            algs.remove(worst)
            dropped.append(worst)
        else:
            # safety: shouldn’t happen now, but prevents infinite loop
            break


# ---------------- Friedman + CD ----------------
def friedman_cd(df: pd.DataFrame, out_prefix: pathlib.Path, alpha=0.05, strict: bool=False):
    # mean error per (fid, alg)
    raw = (df.groupby(["fid","alg"])["err"].mean().unstack())

    # Save coverage diagnostics
    cov = _coverage_table(raw)
    cov.to_csv(out_prefix.with_name("coverage_by_alg.csv"))
    raw.isna().to_csv(out_prefix.with_name("missing_mask.csv"))

    if strict:
        pivot = raw.dropna(axis=0, how="any")
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            raise ValueError(
                "Friedman (strict) requires ≥2 functions × ≥2 algorithms with complete data.\n"
                f"Coverage by algorithm:\n{cov.to_string(index=True)}"
            )
        dropped = []
    else:
        pivot, dropped = _largest_complete_submatrix(raw, min_algs=2, min_fids=2)
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            raise ValueError(
                "Friedman (relaxed) could not find a ≥2×2 complete submatrix.\n"
                f"Coverage by algorithm:\n{cov.to_string(index=True)}\n"
                "Tip: run with --dims to filter to a dimension with better coverage."
            )

    algs = list(pivot.columns)
    stat, p = friedmanchisquare(*[pivot[col] for col in algs])

    ranks_per_fid = pivot.rank(axis=1, method="average", ascending=True)
    mean_ranks = ranks_per_fid.mean().sort_values()
    cd = _compute_cd(mean_ranks, n_datasets=pivot.shape[0], alpha=alpha)

    # Nemenyi pairwise diffs
    pairs = {}
    ordered = list(mean_ranks.index)
    for i in range(len(ordered)):
        for j in range(i+1, len(ordered)):
            a, b = ordered[i], ordered[j]
            diff = abs(mean_ranks[b] - mean_ranks[a])
            pairs[(a,b)] = {"diff": float(diff), "significant": bool(diff >= cd)}

    # Plot CD on the subset actually used
    _plot_cd(mean_ranks, pivot.shape[0], out_prefix, alpha=alpha, cd=cd)

    # Save outputs
    mean_ranks.to_csv(out_prefix.with_name("ranks.csv"), header=["mean_rank"])
    ranks_per_fid.to_csv(out_prefix.with_name("ranks_per_function.csv"))
    pd.DataFrame([{"alg1":a, "alg2":b, **vals} for (a,b), vals in pairs.items()]) \
      .to_csv(out_prefix.with_name("nemenyi_pairs.csv"), index=False)

    (out_prefix.with_name("friedman_subset_info.txt")).write_text(
        f"Functions used: {pivot.shape[0]}\n"
        f"Algorithms used: {pivot.shape[1]}\n"
        f"Dropped algorithms (insufficient coverage): {dropped}\n"
        f"Coverage by algorithm:\n{cov.to_string(index=True)}\n"
    )

    return stat, p, mean_ranks, cd, ranks_per_fid, pairs

def _plot_cd(mean_ranks: pd.Series, n_datasets: int, out_prefix: pathlib.Path, alpha=0.05, cd=None):
    algs = list(mean_ranks.index)
    ranks = mean_ranks.values
    order = np.argsort(ranks)
    algs = [algs[i] for i in order]
    ranks = ranks[order]
    k = len(algs)
    if k < 2:
        raise ValueError("CD plot needs ≥2 algorithms.")
    if cd is None:
        cd = _compute_cd(mean_ranks, n_datasets, alpha=alpha)

    fig, ax = plt.subplots(figsize=(8.5, 2.5))
    ax.set_xlim(1-0.2, k+0.2)
    ax.set_ylim(0, 1)
    ax.get_yaxis().set_visible(False)

    # ruler
    ax.hlines(0.8, 1, k, colors="k", lw=1)
    for r in range(1, k+1):
        ax.vlines(r, 0.78, 0.82, colors="k", lw=1)
        ax.text(r, 0.84, f"{r}", ha="center", va="bottom", fontsize=9)

    # points and labels
    y0 = 0.55
    for name, r in zip(algs, ranks):
        ax.plot([r], [y0], "ko", ms=4)
        ax.text(r, y0-0.1, name, rotation=90, ha="center", va="top", fontsize=9)

    # CD bar
    ax.hlines(0.95, 1, 1+cd, colors="k", lw=2)
    ax.text(1+cd/2, 0.98, f"CD={cd:.2f}", ha="center", va="bottom", fontsize=9)

    # greedy significance cliques: connect adjacent whose span < CD
    y_bar = 0.68
    start = 0
    while start < k:
        end = start
        while end+1 < k and (ranks[end+1] - ranks[start]) < cd - 1e-12:
            end += 1
        if end > start:
            ax.hlines(y_bar, ranks[start], ranks[end], colors="k", lw=3)
            y_bar -= 0.07
        start = end + 1

    ax.set_title(f"Critical-Difference diagram (Friedman–Nemenyi, α={alpha})")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=300)
    plt.close()

# ---------------- Pairwise Wilcoxon (Holm) ----------------
def wilcoxon_on_ranks(ranks_per_fid: pd.DataFrame, out_path: pathlib.Path):
    algs = list(ranks_per_fid.columns)
    pvals = {}
    stats = {}
    for i in range(len(algs)):
        for j in range(i+1, len(algs)):
            a, b = algs[i], algs[j]
            ra = ranks_per_fid[a]
            rb = ranks_per_fid[b]
            d = ra - rb
            d = d[d != 0.0]
            if len(d) == 0:
                p = 1.0
                W = 0.0
            else:
                W, p = wilcoxon(d, alternative="two-sided", zero_method="pratt", correction=False)
            pvals[(a,b)] = p
            stats[(a,b)] = W

    p_adj = holm_bonferroni(pvals)
    rows = []
    for (a,b), p in pvals.items():
        rows.append({
            "alg1": a, "alg2": b, "W": stats[(a,b)],
            "p": p, "p_holm": p_adj[(a,b)],
            "reject_0.05": bool(p_adj[(a,b)] < 0.05)
        })
    df = pd.DataFrame(rows).sort_values("p_holm")
    df.to_csv(out_path, index=False)
    return df

# ---------------- Plots: heatmap, wins, convergence, performance profiles ----------------
def plot_rank_heatmap(ranks_per_fid: pd.DataFrame, out_prefix: pathlib.Path):
    A = ranks_per_fid.to_numpy()
    fig, ax = plt.subplots(figsize=(1.2+0.35*A.shape[1], 0.8+0.35*A.shape[0]))
    im = ax.imshow(A, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(A.shape[1])); ax.set_xticklabels(ranks_per_fid.columns, rotation=45, ha="right")
    ax.set_yticks(range(A.shape[0])); ax.set_yticklabels([f"F{fid:02d}" for fid in ranks_per_fid.index])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i,j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if im.norm(A[i,j])>0.5 else "black")
    ax.set_title("Per-function ranks (lower=better)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=300)
    plt.close()

def plot_wins_bar(df: pd.DataFrame, out_prefix: pathlib.Path):
    win_idx = (df.groupby(["fid","alg"])["err"].mean()
                 .reset_index()
                 .sort_values(["fid","err"])
                 .groupby("fid").first())
    counts = win_idx["alg"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,3.5))
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("# functions won"); ax.set_title("Wins per algorithm (lowest mean error per function)")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=300)
    plt.close()
    counts.to_csv(out_prefix.with_suffix(".csv"), header=["wins"])

def convergence_curves(df: pd.DataFrame, outdir: pathlib.Path):
    if "evals" not in df.columns:
        warnings.warn("No 'evals' column; skipping convergence plots.")
        return
    for fid, sub in df.groupby("fid"):
        plt.figure(figsize=(6,4))
        for alg, s in sub.groupby("alg"):
            med = s.groupby("evals")["err"].median()
            plt.plot(med.index, med.values, label=alg)
        plt.yscale("log"); plt.xlabel("evaluations"); plt.ylabel("|f - f*|")
        plt.title(f"F{fid:02d} — median convergence"); plt.legend()
        plt.tight_layout()
        for ext in ("pdf","png"):
            plt.savefig(outdir/f"conv_F{fid:02d}.{ext}", dpi=300)
        plt.close()

def performance_profile_ert(ert: pd.DataFrame, df_runs: pd.DataFrame, out_prefix: pathlib.Path,
                            penalty: float = 0.0):
    """
    Dolan–Moré profile on ERT across functions.
    If penalty > 0: fill NaN ERT with penalty × budget(fid),
    where budget(fid) = max evals observed for that fid in df_runs (fallback 40000).
    """
    mat = ert.pivot(index="fid", columns="alg", values="ERT")
    if penalty > 0:
        # infer per-fid budget
        if "evals" in df_runs.columns:
            budget_by_fid = df_runs.groupby("fid")["evals"].max()
        else:
            budget_by_fid = pd.Series(40000, index=mat.index).fillna(40000)
        for fid in mat.index:
            if pd.isna(mat.loc[fid]).any():
                b = float(budget_by_fid.get(fid, 40000))
                mat.loc[fid] = mat.loc[fid].fillna(penalty * b)
        mat = mat.dropna(axis=0, how="any")
    else:
        mat = mat.dropna(axis=0, how="any")

    if mat.empty:
        warnings.warn("ERT matrix empty after processing; skipping performance profile.")
        return

    R = mat.values / np.minimum.reduce(mat.values, axis=1, keepdims=True)
    taus = np.logspace(0, 3, 200)  # 1..1000
    fig, ax = plt.subplots(figsize=(6.5,4))
    for j, alg in enumerate(mat.columns):
        rj = R[:, j]
        rho = [(rj <= t).mean() for t in taus]
        ax.plot(taus, rho, label=alg)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(r"Performance ratio $\tau$")
    ax.set_ylabel(r"Profile $\rho_a(\tau)$")
    ttl = "Performance profile (ERT"
    ttl += f", penalty×budget={penalty:g}" if penalty > 0 else ""
    ttl += ")"
    ax.set_title(ttl)
    ax.legend(loc="lower right")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(f"{out_prefix}.{ext}", dpi=300)
    plt.close()

# ---------------- ERT tables ----------------
def write_ert_table(ert: pd.DataFrame, outdir: pathlib.Path):
    mat = ert.pivot(index="fid", columns="alg", values="ERT")
    (outdir/"ert_matrix.csv").write_text(mat.to_csv())
    tex = mat.to_latex(float_format="%.0f",
                       caption="ERT (evals) per CEC-2022 function.",
                       label="tab:ert_cec2022")
    (outdir/"ert_table.tex").write_text(tex)

# ---------------- Report ----------------
def write_report(outdir: pathlib.Path, friedman_stat, friedman_p, mean_ranks: pd.Series, cd: float, wilcox_df: pd.DataFrame):
    lines = []
    lines.append("# CEC-2022 Statistical Summary\n")
    lines.append(f"- **Friedman χ²** = {friedman_stat:.3f}, **p** = {friedman_p:.3g}\n")
    lines.append(f"- **Mean ranks** (lower=better):\n")
    for a, r in mean_ranks.items():
        lines.append(f"  - {a}: {r:.3f}")
    lines.append(f"\n- **Nemenyi Critical Distance** (α=0.05): **CD = {cd:.2f}**.\n")
    if wilcox_df is not None and not wilcox_df.empty:
        rej = wilcox_df[wilcox_df["reject_0.05"]]
        lines.append(f"- **Wilcoxon (Holm)**: {len(rej)}/{len(wilcox_df)} pairs significant at 0.05.\n")
    (outdir/"report.md").write_text("\n".join(lines))

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=pathlib.Path,
                    help="CEC2022 results CSV (alg,fid,dim,run,err,evals/nfe,...)")
    ap.add_argument("--ert", type=pathlib.Path,
                    help="ERT summary CSV (alg,fid,dim,ERT,...)")
    ap.add_argument("--out", default="appendix_figs", type=pathlib.Path)
    ap.add_argument("--tol", default=1e-8, type=float,
                    help="hit tolerance for success rate")
    ap.add_argument("--dims", type=str, default="",
                    help="comma-separated dims to filter, e.g. 10,20")
    ap.add_argument("--strict-friedman", action="store_true",
                    help="Require full coverage for Friedman (no relaxed submatrix).")
    ap.add_argument("--perf-profile-penalty", type=float, default=0.0,
                    help="If >0, fill missing ERT with penalty × per-fid budget before profiling.")
    args = ap.parse_args()

    dims = None
    if args.dims:
        dims = [int(x) for x in args.dims.split(",") if x]

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_runs(args.csv, args.tol, dims=dims)

    # 1) Friedman + CD + Wilcoxon(Holm) + rank heatmap + wins
    stat, p, mean_ranks, cd, ranks_per_fid, _pairs = friedman_cd(
        df, args.out/"fig_cd", alpha=0.05, strict=args.strict_friedman
    )
    wilcox = wilcoxon_on_ranks(ranks_per_fid, args.out/"wilcoxon_holm.csv")
    plot_rank_heatmap(ranks_per_fid, args.out/"rank_heatmap")
    plot_wins_bar(df, args.out/"wins_bar")

    # 2) Convergence per function
    convergence_curves(df, args.out)

    # 3) ERT matrix/table + performance profile
    if args.ert and args.ert.exists():
        ert = collapse_ert(args.ert, dims=dims)
        write_ert_table(ert, args.out)
        performance_profile_ert(
            ert, df, args.out/"perf_profile_ert",
            penalty=float(args.perf_profile_penalty)
        )
    else:
        print("No --ert CSV provided; skipping ERT table & performance profile.")

    # 4) Small markdown report
    write_report(args.out, stat, p, mean_ranks, cd, wilcox)

if __name__ == "__main__":
    main()
