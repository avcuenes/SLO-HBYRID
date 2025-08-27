#!/usr/bin/env python3
# viz_from_holm_numbers.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

ALPHA = 0.05
IN = Path(sys.argv[1] if len(sys.argv) > 1 else "wilcoxon_holm.csv")
OUT = Path("stats_out"); OUT.mkdir(parents=True, exist_ok=True)

# ---- load your exact columns ----
df = pd.read_csv(IN)
needed = ["alg1","alg2","p_holm","reject_0.05"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise SystemExit(f"CSV is missing columns: {missing}. Found: {list(df.columns)}")

df = df[["alg1","alg2","p_holm","reject_0.05"]].rename(
    columns={"alg1":"A","alg2":"B","p_holm":"p_adj","reject_0.05":"reject"}
)
df["A"] = df["A"].astype(str)
df["B"] = df["B"].astype(str)
df["p_adj"] = pd.to_numeric(df["p_adj"], errors="coerce").clip(lower=0, upper=1)
df["reject"] = df["reject"].astype(bool)

# ---- algorithms and pairwise matrix ----
algs = sorted(pd.unique(pd.concat([df["A"], df["B"]], ignore_index=True)))
P = pd.DataFrame(1.0, index=algs, columns=algs, dtype=float)
for _, r in df.dropna(subset=["p_adj"]).iterrows():
    P.loc[r["A"], r["B"]] = r["p_adj"]
    P.loc[r["B"], r["A"]] = r["p_adj"]

S = -np.log10(P.clip(lower=1e-300))  # for readability of tiny p-values
mask = np.triu(np.ones_like(S, dtype=bool))  # show lower triangle

def _draw_heatmap(label_mode="p"):
    """label_mode: 'p' (scientific p) or 'neglog10' (numeric strength)."""
    plt.figure(figsize=(0.6*len(algs)+3, 0.6*len(algs)+2))
    if HAVE_SNS:
        ax = sns.heatmap(S, mask=mask, cmap="viridis", linewidths=.5, square=True,
                         cbar_kws=dict(label="-log10(p_Holm)"))
    else:
        ax = plt.gca()
        im = ax.imshow(np.ma.array(S.values, mask=mask), interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("-log10(p_Holm)")
        ax.set_xticks(range(len(algs))); ax.set_xticklabels(algs, rotation=45, ha="right")
        ax.set_yticks(range(len(algs))); ax.set_yticklabels(algs)

    # annotate numbers in lower triangle
    for i, a in enumerate(algs):
        for j, b in enumerate(algs):
            if i > j:
                p = float(P.loc[a, b])
                s = float(S.loc[a, b])
                if label_mode == "p":
                    txt = f"{p:.1e}"  # scientific p
                    val_for_color = s
                else:
                    txt = f"{s:.2f}"  # -log10(p)
                    val_for_color = s
                # choose text color for contrast
                color = "white" if val_for_color >= 2.0 else "black"  # 2 ≈ p=0.01
                ax.text(j+0.5, i+0.5, txt, ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    suffix = "p" if label_mode == "p" else "neglog10"
    out_png = OUT/f"holm_heatmap_labels_{suffix}.png"
    plt.savefig(out_png, dpi=220); plt.close()
    print("Saved:", out_png)

# two versions: numbers as p, and as -log10(p)
_draw_heatmap("p")
_draw_heatmap("neglog10")

# ---- significant-pairs count per algorithm (direction-agnostic) ----
sig = df[df["p_adj"] < ALPHA]
counts = pd.Series(0, index=algs, dtype=int)
for _, r in sig.iterrows():
    counts[r["A"]] += 1
    counts[r["B"]] += 1

plt.figure(figsize=(max(6, 0.5*len(algs)), 3.2))
if HAVE_SNS:
    ax = sns.barplot(x=counts.index, y=counts.values)
else:
    ax = plt.gca(); ax.bar(counts.index, counts.values)
for i, v in enumerate(counts.values):
    ax.text(i, v + max(0.02*counts.max(), 0.1), str(int(v)), ha="center", va="bottom", fontsize=9)
plt.ylabel(f"Significant pairs (Holm, α={ALPHA})")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
bar_png = OUT/"holm_signif_pairs_bar.png"
plt.savefig(bar_png, dpi=220); plt.close()
print("Saved:", bar_png)

# ---- numeric pairwise table (no stars) ----
pair_rows = []
for _, r in df.sort_values("p_adj").iterrows():
    pair_rows.append([r["A"], r["B"], r["p_adj"], -np.log10(max(r["p_adj"], 1e-300))])
tbl = pd.DataFrame(pair_rows, columns=["Alg_A","Alg_B","p_Holm","neglog10_pHolm"])
tbl_csv = OUT/"holm_pairwise.csv"; tbl.to_csv(tbl_csv, index=False)
print("Saved:", tbl_csv)

latex = (
    "\\begin{table}[t]\\centering\\small\n"
    "\\begin{tabular}{l l r r}\\toprule\n"
    "Alg A & Alg B & $p_{\\text{Holm}}$ & $-\\log_{10}(p_{\\text{Holm}})$\\\\\\midrule\n"
)
for _, r in tbl.iterrows():
    latex += f"{r['Alg_A']} & {r['Alg_B']} & {r['p_Holm']:.3g} & {r['neglog10_pHolm']:.2f}\\\\\n"
latex += "\\bottomrule\\end{tabular}\n"
latex += "\\caption{Wilcoxon signed-rank pairwise tests with Holm correction. Numerical values only.}\n"
latex += "\\label{tab:wilcoxon-holm-numeric}\n\\end{table}\n"
(OUT/"holm_pairwise_numerical.tex").write_text(latex, encoding="utf-8")
print("Saved:", OUT/"holm_pairwise_numerical.tex")

# ---- brief summary text ----
summary = [
    f"File: {IN.name}",
    f"Algorithms: {', '.join(algs)}",
    f"Significant pairs (<{ALPHA}): {int((df['p_adj']<ALPHA).sum())}",
]
(OUT/"summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("Saved:", OUT/"summary.txt")
