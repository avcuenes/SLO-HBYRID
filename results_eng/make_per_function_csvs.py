#!/usr/bin/env python3
# make_per_function_stats.py
"""
Split a run-level results CSV into one CSV per engineering function/problem.
For each problem (fid) it writes:
  - <out>/raw/F<fid>_<Problem>.csv                 (raw rows for that problem)
  - <out>/agg/F<fid>_<Problem>_stats.csv           (per-algorithm stats on the chosen metric)

Per-algorithm stats columns:
  Algorithm, mean, std, std_of_mean, best, worst, runs

Usage:
  python make_per_function_stats.py --in results_eng.csv --out per_function --metric fbest

If --metric is omitted, the script tries: fbest → err → value → first numeric column.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ---------- CLI ----------
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--in", dest="inp", required=True, help="Input CSV with run-level results")
ap.add_argument("--out", dest="out", default="per_function", help="Output directory")
ap.add_argument("--metric", dest="metric", default=None,
                help="Column to average (e.g., fbest, err, time_sec). If omitted, auto-detects.")
args = ap.parse_args()

inp = Path(args.inp)
out = Path(args.out)
out_raw = out / "raw"
out_agg = out / "agg"
out_raw.mkdir(parents=True, exist_ok=True)
out_agg.mkdir(parents=True, exist_ok=True)

# ---------- Load & detect columns ----------
df = pd.read_csv(inp)
lc = {c.lower(): c for c in df.columns}

def pick(cands, required=True):
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    if required:
        raise KeyError(f"Need one of {cands} in columns {list(df.columns)}")
    return None

col_alg = pick(["alg","algorithm","method","name"])
col_fid = pick(["fid","problem_id","pid","prob","problem"])
col_run = pick(["run","seed","trial"], required=False)

# Choose metric column
if args.metric:
    if args.metric.lower() not in lc:
        raise KeyError(f"--metric '{args.metric}' not found. Available columns: {list(df.columns)}")
    col_val = lc[args.metric.lower()]
else:
    if "fbest" in lc:
        col_val = lc["fbest"]
    elif "err" in lc:
        col_val = lc["err"]
    elif "value" in lc:
        col_val = lc["value"]
    else:
        # fallback: first numeric column that isn't id-like
        id_like = {col_alg.lower(), col_fid.lower(), (col_run or "").lower()}
        num_cols = [c for c in df.columns if c.lower() not in id_like and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise KeyError("No numeric column to aggregate—pass --metric <colname>.")
        col_val = num_cols[0]

# Optional context columns to keep in raw output if present
keep_optional = [x for x in ["suite","dim","fbest","fopt","nfe","time_sec","budget","hit","err"] if x in lc]
keep_cols_raw = [col for col in [col_alg, col_fid, col_run, col_val] if col] + \
                [lc[k] for k in keep_optional if lc[k] != col_val]

# ---------- Clean types ----------
df[col_alg] = df[col_alg].astype(str)
df[col_fid] = pd.to_numeric(df[col_fid], errors="coerce").astype(int)
df[col_val] = pd.to_numeric(df[col_val], errors="coerce")

# ---------- Problem names (for filenames) ----------
FID2NAME = {
    1: "Pressure_Vessel",
    2: "Tension_Spring",
    3: "Welded_Beam",
    4: "Gear_Box",
    5: "Speed_Reducer",
    6: "Car_Side_Impact",
    7: "Hydrostatic_Bearing",
    8: "Four_Bar_Truss",
    9: "Ten_Bar_Truss",
    10: "Cantilever_Cont",
    11: "Cantilever_Disc",
    12: "Stepped_Column",
    13: "Machining_Cost",
    14: "Heat_Exchanger",
    15: "Thick_Pressure_Vessel",
    16: "Gear_Train",
}
def pname(fid: int) -> str:
    s = FID2NAME.get(int(fid), f"Problem_{int(fid)}")
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)

# ---------- Write per-function CSVs ----------
count = 0
for fid, sub in df.groupby(col_fid, sort=True):
    name = pname(fid)

    # RAW rows for this problem
    raw = sub[keep_cols_raw].copy() if keep_cols_raw else sub.copy()
    sort_cols = [c for c in [col_alg, col_run] if c in raw.columns]
    if sort_cols:
        raw = raw.sort_values(sort_cols)
    raw_path = out_raw / f"F{int(fid):02d}_{name}.csv"
    raw.to_csv(raw_path, index=False)

    # Aggregated stats per algorithm for the chosen metric
    grp = sub.groupby(col_alg)[col_val]
    agg = grp.agg(mean="mean", std="std", best="min", worst="max", runs="size").reset_index()
    # std_of_mean = std / sqrt(runs)  (a.k.a. standard error of the mean)
    agg["std_of_mean"] = agg["std"] / np.sqrt(agg["runs"].clip(lower=1))

    # Reorder & rename
    agg = agg.rename(columns={col_alg: "Algorithm"})
    agg = agg[["Algorithm","mean","std","std_of_mean","best","worst","runs"]]
    agg = agg.sort_values("mean")

    agg_path = out_agg / f"F{int(fid):02d}_{name}_stats.csv"
    agg.to_csv(agg_path, index=False)

    count += 1

print(f"[OK] Processed {count} problems into:")
print(f" - raw CSVs: {out_raw}")
print(f" - stats CSVs: {out_agg}")
print(f"Metric aggregated: {col_val}")
