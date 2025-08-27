#!/usr/bin/env python3
import argparse, pathlib, sys
import pandas as pd
import numpy as np

REQUIRED = ["alg","prob","dim","run","f_best","err","nfe","time_sec"]

def _coerce_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure numeric cols are numeric and drop non-finite rows for stats
    for c in ["dim","run","f_best","err","nfe","time_sec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # keep rows but mark non-finite metrics for filtering in stats
    return df

def _agg_stats(series: pd.Series):
    s = series[np.isfinite(series.values)]
    n = int(s.size)
    if n == 0:
        return pd.Series({"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan, "n": 0})
    return pd.Series({
        "min":  float(np.min(s)),
        "max":  float(np.max(s)),
        "mean": float(np.mean(s)),
        "std":  float(np.std(s, ddof=1)) if n > 1 else 0.0,
        "n":    n,
    })
def summarize_df(df):
    import pandas as pd
    grp  = df.groupby("alg", as_index=True)
    dims = grp["dim"].first().rename("dim")

    # Aggregate stats for f_best
    f_stats = grp["f_best"].agg(["min", "max", "mean", "std", "count"]).rename(
        columns={
            "min":   "f_best",
            "max":   "f_worst",
            "mean":  "f_mean",
            "std":   "f_std",
            "count": "f_n",
        }
    )

    # Aggregate stats for err
    err_stats = grp["err"].agg(["min", "max", "mean", "std", "count"]).rename(
        columns={
            "min":   "err_best",
            "max":   "err_worst",
            "mean":  "err_mean",
            "std":   "err_std",
            "count": "err_n",
        }
    )

    out = pd.concat([dims, f_stats, err_stats], axis=1).reset_index()
    # pandas std is NaN for single sample; make that 0 for readability
    out["f_std"]   = out["f_std"].fillna(0.0)
    out["err_std"] = out["err_std"].fillna(0.0)

    # Nice ordering: lower error first, then lower f
    out = out.sort_values(["err_mean", "f_mean", "alg"], ascending=[True, True, True])
    return out


def load_per_problem(dirpath: pathlib.Path, include_global: bool):
    files = sorted(p for p in dirpath.glob("*.csv") if p.name != "raw_results.csv")
    if include_global and (dirpath / "raw_results.csv").exists():
        files.append(dirpath / "raw_results.csv")
    if not files:
        print(f"[error] no CSV files found under {dirpath}", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] failed to read {p.name}: {e}", file=sys.stderr)
            continue
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            print(f"[warn] {p.name} missing columns {missing}; skipping", file=sys.stderr)
            continue
        df = _coerce_and_clean(df)
        dfs.append(df)

    if not dfs:
        print("[error] no valid CSVs after parsing", file=sys.stderr)
        sys.exit(1)

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

def write_outputs(dirpath: pathlib.Path, summaries_dir: pathlib.Path, combined: pd.DataFrame, make_latex: bool):
    summaries_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for prob, dfp in combined.groupby("prob"):
        summ = summarize_df(dfp)
        summ.insert(1, "prob", prob)
        out_csv = summaries_dir / f"{prob}_summary.csv"
        summ.to_csv(out_csv, index=False)
        print(f"[ok] wrote {out_csv}")
        if make_latex:
            tex = summaries_dir / f"{prob}_summary.tex"
            summ.to_latex(tex, index=False, float_format=lambda x: f"{x:.6g}")
        all_summaries.append(summ)

    if not all_summaries:
        print("[warn] no per-problem summaries produced", file=sys.stderr)
        return

    comb = pd.concat(all_summaries, ignore_index=True)
    comb_out = summaries_dir / "summary_all.csv"
    comb.to_csv(comb_out, index=False)
    print(f"[ok] wrote {comb_out}")
    if make_latex:
        (summaries_dir / "summary_all.tex").write_text(
            comb.to_latex(index=False, float_format=lambda x: f"{x:.6g}")
        )

    # Convenience pivot of err_mean for quick “who wins where”
    try:
        pivot_err = comb.pivot_table(index="alg", columns="prob", values="err_mean", aggfunc="first")
        pivot_err = pivot_err.sort_index()
        pv_out = summaries_dir / "pivot_err_mean.csv"
        pivot_err.to_csv(pv_out)
        print(f"[ok] wrote {pv_out}")
        if make_latex:
            (summaries_dir / "pivot_err_mean.tex").write_text(
                pivot_err.to_latex(float_format=lambda x: f"{x:.6g}")
            )
    except Exception as e:
        print(f"[warn] pivot failed: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Summarize per-problem benchmark CSVs")
    ap.add_argument("--dir", default="eng_results", help="Folder with per-problem CSVs")
    ap.add_argument("--summaries-dir", default="summaries", help="Subfolder name for outputs (under --dir)")
    ap.add_argument("--also-global", action="store_true", help="Also include raw_results.csv if present")
    ap.add_argument("--latex", action="store_true", help="Write LaTeX tables alongside CSVs")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    combined = load_per_problem(root, include_global=args.also_global)
    write_outputs(root, root / args.summaries_dir, combined, make_latex=args.latex)

if __name__ == "__main__":
    main()
