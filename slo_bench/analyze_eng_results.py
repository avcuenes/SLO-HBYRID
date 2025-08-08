#!/usr/bin/env python3
# analyze_eng_results.py  – summarise and plot *all* algorithms

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns          # comment-out if seaborn not installed
from pathlib import Path

CSV = Path("eng_results/raw_results.csv")      # adjust if you renamed it
TOL = 1e-6                                     # success threshold

# ── 1. load & basic sanity ─────────────────────────────────────────────
df = pd.read_csv(CSV)
print(f"Loaded {len(df)} rows from {CSV}")

# add a HIT column once so we can reuse
df["hit"] = df["err"] <= TOL

# ── 2. per-algorithm summary table ────────────────────────────────────
tbl = (df.groupby("alg")
         .agg(runs=("run", "count"),
              mean_best=("f_best","mean"),
              sd_best=("f_best","std"),
              success=("hit","mean"),
              mean_nfe=("nfe","mean"))
         .sort_values("mean_best"))
print("\n=== Summary over ALL problems & runs ===")
print(tbl.to_string(float_format="%.4g"))

# # ── 3. mean best PER PROBLEM plot ─────────────────────────────────────
# mean_per_prob = (df.groupby(["prob","alg"])["f_best"]
#                    .mean()
#                    .unstack())                  # rows=prob, cols=alg

# mean_per_prob.plot(kind="bar", figsize=(10,4))
# plt.ylabel("Mean penalised f_best")
# plt.title("Mean best value – each problem, all algorithms")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# ── 4. ERT bar plot (choose one problem) ──────────────────────────────
PROB = df["prob"].unique()[0]   # pick first; change as you like
sub  = df[df["prob"] == PROB]
ert  = (sub.groupby("alg")
          .apply(lambda d: d["nfe"].sum() / max(1, d["hit"].sum()))
          .sort_values())

ert.plot(kind="bar", figsize=(6,3))
plt.ylabel("ERT (evals to hit tol ≤ %.0e)" % TOL)
plt.title(f"{PROB}  –  Expected runtime")
plt.tight_layout()
plt.show()

# ── 5. box-plot of distributions for that problem ────────────────────
plt.figure(figsize=(8,3.5))
sns.boxplot(data=sub, x="alg", y="f_best")
plt.xticks(rotation=40, ha="right")
plt.ylabel("Penalised best value")
plt.title(f"{PROB}  –  dispersion over runs")
plt.tight_layout()
plt.show()
