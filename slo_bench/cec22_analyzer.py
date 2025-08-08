# cec22_analyzer.py
import argparse, csv, math, os
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt

THRESHOLDS = [1e-1, 1e-3, 1e-5, 1e-8]

def load_results(paths):
    rows=[]
    for p in paths:
        with open(p, newline="") as fh:
            rd = csv.DictReader(fh)
            for r in rd:
                rows.append(dict(
                    alg=r["alg"],
                    fid=int(r["fid"]),
                    dim=int(r["dim"]),
                    run=int(r["run"]),
                    f_best=float(r["f_best"]),
                    gap=float(r["gap"]),
                    evals=int(r["evals"]),
                ))
    return rows

def ecdf(values):
    x = np.sort(np.asarray(values, float))
    y = np.arange(1, len(x)+1, dtype=float)/len(x) if len(x) else np.array([])
    return x, y

def average_ranks(per_func_errors_by_alg):
    # per_func_errors_by_alg: dict[fid] -> dict[alg] -> list[error]
    avg_rank = defaultdict(list)
    for fid, m in per_func_errors_by_alg.items():
        # score per alg = median error (robust)
        meds = {alg: np.median(v) if len(v)>0 else np.inf for alg,v in m.items()}
        # smaller is better -> ranks
        ordered = sorted(meds.items(), key=lambda kv: kv[1])
        # handle ties: average the rank positions
        ranks = {}
        i = 0
        while i < len(ordered):
            j = i+1
            while j < len(ordered) and math.isclose(ordered[j][1], ordered[i][1], rel_tol=0, abs_tol=1e-15):
                j += 1
            avg = (i+1 + j)/2.0
            for k in range(i, j):
                ranks[ordered[k][0]] = avg
            i = j
        for alg,rnk in ranks.items():
            avg_rank[alg].append(rnk)
    return {alg: float(np.mean(rs)) for alg, rs in avg_rank.items()}

def make_plots(rows, outdir):
    os.makedirs(outdir, exist_ok=True)
    # group by dim
    dims = sorted(set(r["dim"] for r in rows))
    for D in dims:
        R = [r for r in rows if r["dim"]==D]
        # ECDF of log10(gap) at final budget
        by_alg = OrderedDict()
        per_func = defaultdict(lambda: defaultdict(list))
        for r in R:
            by_alg.setdefault(r["alg"], []).append(r["gap"])
            per_func[r["fid"]][r["alg"]].append(r["gap"])

        plt.figure()
        for alg, vals in by_alg.items():
            logg = [math.log10(max(1e-300, v)) for v in vals]  # avoid log(0)
            x, y = ecdf(logg)
            plt.plot(x, y, label=alg)
        plt.xlabel("log10(gap)  (gap = f_best - f*)")
        plt.ylabel("ECDF  (fraction of runs)")
        plt.title(f"CEC-2022 ECDF of log10(gap) at final budget (D={D})")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"ecdf_loggap_D{D}.png"), dpi=160)
        plt.close()

        # Success counts at thresholds
        plt.figure()
        algs = list(by_alg.keys())
        succ = []
        total = []
        for alg in algs:
            vals = by_alg[alg]
            total.append(len(vals))
            succ.append([sum(v<=t for v in vals) for t in THRESHOLDS])
        succ = np.array(succ, float)  # shape [A, T]
        width = 0.8 / len(algs)
        for i, alg in enumerate(algs):
            for j, t in enumerate(THRESHOLDS):
                plt.bar(j + i*width, succ[i, j] / max(1, total[i]), width=width)
        plt.xticks([j + 0.4 - 0.4/len(algs) for j in range(len(THRESHOLDS))],
                   [f"{t:.0e}" for t in THRESHOLDS])
        plt.ylim(0, 1.0)
        plt.ylabel("success rate (fraction)")
        plt.title(f"CEC-2022 Success Rate by Threshold (D={D})")
        plt.grid(True, axis="y", ls=":")
        plt.legend(algs, title="alg", loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"success_rates_D{D}.png"), dpi=160)
        plt.close()

        # Average ranks (per function) table
        ranks = average_ranks(per_func)
        with open(os.path.join(outdir, f"avg_ranks_D{D}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["alg","avg_rank"])
            for alg, rnk in sorted(ranks.items(), key=lambda kv: kv[1]):
                w.writerow([alg, f"{rnk:.3f}"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="CEC-2022 result CSVs (alg,fid,dim,run,f_best,gap,evals)")
    ap.add_argument("--outdir", default="cec22_figs")
    args = ap.parse_args()
    rows = load_results(args.csv)
    make_plots(rows, args.outdir)
    print("Wrote figures to", args.outdir)

if __name__ == "__main__":
    main()
