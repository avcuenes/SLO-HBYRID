# -*- coding: utf-8 -*-
"""
slo_bench.cec_run
-----------------
CEC-2014 / CEC-2022 benchmark runner for SLO-HBYRID (Spiral-LSHADE hybrid),
Spiral-NM, and baseline optimisers.

CSV per-run log  →  <outdir>/<suite>_results.csv
CSV ERT summary →  <outdir>/<suite>_ERT_summary.csv

Example
-------
python3 -m slo_bench.cec_run \
  --suite cec2022 --dims 10 20 --fids 1-12 --runs 30 \
  --algs SLO_HBYRID CMAES SciPyDE jSO L_SHADE LBFGSB \
  --budget-mult 20000 --target-tol 1e-8 \
  --seed0 0 --outdir results_cec
"""
from __future__ import annotations
import argparse, csv, os, sys, time, importlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------
#  Spiral-based hybrids
# ---------------------------------------------------------------------
from spiral_lshade import SpiralLSHADEParams, _slo_lshade_core

# ---------------------------------------------------------------------
#  Optional third-party libraries
# ---------------------------------------------------------------------
try:    import cma;                                               HAVE_CMA    = True
except Exception:                                                 HAVE_CMA    = False

try:
    from scipy.optimize import differential_evolution, minimize;  HAVE_SCIPY  = True
except Exception:                                                 HAVE_SCIPY  = False

try:
    import pyade.jso   as pyade_jso
    import pyade.lshade as pyade_lshade
    HAVE_PYADE = True
except Exception:
    HAVE_PYADE = False

# OPFUNU – CEC suites
try:    from opfunu.cec_based import cec2014 as ofu2014;           HAVE_2014   = True
except Exception:                                                 HAVE_2014   = False
try:    from opfunu.cec_based import cec2022 as ofu2022;           HAVE_2022   = True
except Exception:                                                 HAVE_2022   = False

# =====================================================================
#  Helpers
# =====================================================================
def set_seed(seed: int):
    np.random.seed(int(seed))

def project_box_open(x, lb, ub):
    x  = np.asarray(x,  dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    rng = ub - lb
    eps = np.maximum(1e-8 * np.maximum(rng, 1.0), 1e-12)
    eps = np.minimum(eps, 0.45 * np.maximum(rng, 1e-30))
    return np.minimum(ub - eps, np.maximum(lb + eps, x))

@dataclass
class Problem:
    fid: int; name: str; dim: int
    lower: np.ndarray; upper: np.ndarray
    fopt: float
    f: Callable[[np.ndarray], float]

def _to_array(v, default, size):
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        arr = np.asarray(v, dtype=float)
        if arr.size == 1:      return np.full(size, float(arr.item()))
        if arr.size == size:   return arr.astype(float)
    return np.full(size, float(default))

# ---------------------------------------------------------------------
#  Problem factory
# ---------------------------------------------------------------------
def build_problem(suite: str, fid: int, dim: int) -> Problem:
    suite = suite.lower()
    if suite == "cec2014":
        if not HAVE_2014:
            raise RuntimeError("opfunu cec2014 not available")
        fmap = {i: getattr(ofu2014, f"F{i}2014") for i in range(1, 31)}
    elif suite == "cec2022":
        if not HAVE_2022:
            raise RuntimeError("opfunu cec2022 not available")
        fmap = {i: getattr(ofu2022, f"F{i}2022") for i in range(1, 13)}
    else:
        raise ValueError("suite must be 'cec2014' or 'cec2022'")

    if fid not in fmap:
        raise ValueError(f"fid {fid} not in suite {suite}")
    fcls = fmap[fid]

    # constructor kwargs vary across OPFUNU versions
    fobj = None
    for k in ("ndim", "dimension", "problem_size", "n_dimensions", "dim"):
        try:
            fobj = fcls(**{k: dim})
            break
        except TypeError:
            continue
    if fobj is None:
        try:
            fobj = fcls(dim)
        except TypeError as e:
            raise RuntimeError(f"Cannot construct {fcls.__name__} for dim={dim}: {e}")

    lower = _to_array(getattr(fobj, "lb", -100.0), -100.0, dim)
    upper = _to_array(getattr(fobj, "ub",  100.0),  100.0, dim)
    # OPFUNU names the optimum bias either f_bias or f_global
    fopt  = float(getattr(fobj, "f_bias", getattr(fobj, "f_global", 0.0)))

    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if hasattr(fobj, "evaluate"):
            return float(fobj.evaluate(x))
        return float(fobj(x))

    return Problem(fid=fid, name=f"F{fid:02d}_{suite.upper()}",
                   dim=dim, lower=lower, upper=upper, fopt=fopt, f=f)

# ---------------------------------------------------------------------
#  Evaluation counter with early-hit
# ---------------------------------------------------------------------
class EvalCounter:
    def __init__(self, f: Callable[[np.ndarray], float],
                 fopt: float, target_tol: float):
        self.f = f
        self.fopt = float(fopt)
        self.target_tol = float(target_tol)
        self.nfe  = 0
        self.best = float("inf")
        self.hit  = False
    def __call__(self, x: np.ndarray) -> float:
        y = float(self.f(x)); self.nfe += 1
        if y < self.best:
            self.best = y
            if abs(self.best - self.fopt) <= self.target_tol:
                self.hit = True
        return y

# =====================================================================
#  Algorithm adapters
# =====================================================================
# --- Spiral-LSHADE hybrid (SLO-HBYRID) --------------------------------
def run_spiral_lshade_hybrid(f, lb, ub, budget, seed, kw=None):
    if isinstance(kw, SpiralLSHADEParams):
        kw = {"params": kw}
    kw  = dict(kw) if kw else {}
    p   = kw.get("params", SpiralLSHADEParams())
    rng = np.random.default_rng(int(kw.get("rng_seed", seed)))
    best_f, nfe = _slo_lshade_core(f, np.asarray(lb), np.asarray(ub),
                                   int(budget), rng, p)
    return float(best_f), int(nfe)

def run_slo_hbyrid(prob: Problem, budget: int, seed: int) -> Tuple[float, int]:
    evalf = EvalCounter(lambda z: prob.f(project_box_open(z, prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    f_best, nfe = run_spiral_lshade_hybrid(evalf, prob.lower, prob.upper,
                                           budget, seed, kw={})
    return float(f_best), int(nfe)



# --- CMA-ES -----------------------------------------------------------
def run_cmaes(prob: Problem, budget: int, seed: int):
    if not HAVE_CMA: raise RuntimeError("cma not installed")
    evalf = EvalCounter(lambda z: prob.f(project_box_open(z, prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    x0 = (prob.lower + prob.upper) / 2.0
    sigma0 = 0.3 * float(np.max(prob.upper - prob.lower))
    opts = dict(bounds=[prob.lower, prob.upper],
                maxfevals=int(budget), seed=int(seed),
                verbose=-9, CMA_active=True)
    _, f_best, _, nfe, *_ = cma.fmin(lambda x: evalf(np.asarray(x, float)),
                                     x0, sigma0, options=opts)
    return float(f_best), int(nfe)

# --- SciPy Differential Evolution ------------------------------------
def run_scipy_de(prob: Problem, budget: int, seed: int):
    if not HAVE_SCIPY:
        raise RuntimeError("scipy not installed")
    evalf = EvalCounter(lambda z: prob.f(project_box_open(z, prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    dim, pop = prob.dim, 15
    maxiter = max(1, budget // (pop * dim))
    res = differential_evolution(
        lambda x: evalf(np.asarray(x, float)),
        bounds=list(zip(prob.lower, prob.upper)),
        popsize=pop, maxiter=maxiter, seed=seed,
        polish=False, updating='deferred', workers=1)
    return float(res.fun), int(evalf.nfe)

# --- SciPy L-BFGS-B ---------------------------------------------------
def run_lbfgsb(prob: Problem, budget: int, seed: int):
    if not HAVE_SCIPY:
        raise RuntimeError("scipy not installed")
    evalf = EvalCounter(lambda z: prob.f(project_box_open(z, prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    x0 = prob.lower + np.random.default_rng(seed).random(prob.dim) * (prob.upper - prob.lower)
    minimize(lambda x: evalf(np.asarray(x, float)), x0, method="L-BFGS-B",
             bounds=list(zip(prob.lower, prob.upper)),
             options=dict(maxfun=int(budget), disp=False))
    return float(evalf.best), int(evalf.nfe)

# --- PyADE jSO / L-SHADE ---------------------------------------------
def run_pyade(prob: Problem, budget: int, seed: int, which: str):
    if not HAVE_PYADE:
        raise RuntimeError("pyade not installed or import failed")
    evalf = EvalCounter(lambda z: prob.f(project_box_open(z, prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    bounds = np.stack([prob.lower, prob.upper], axis=1)
    algo   = pyade_jso if which.lower() == "jso" else pyade_lshade
    params = algo.get_default_params(dim=prob.dim)
    params.update(bounds=bounds,
                  func=lambda x: evalf(np.asarray(x, float)),
                  max_evals=int(budget), max_fes=int(budget), seed=int(seed))
    pop = params.get("NP", max(20, 5 * prob.dim))
    params["iters"] = params.get("iters", max(1, budget // pop))
    sol, fit = algo.apply(**params)
    return float(fit), int(evalf.nfe)

# Mapping of CLI names to runner functions
ALG_MAP = {
    "SLO_HBYRID": run_slo_hbyrid,
    "CMAES":      run_cmaes,
    "SciPyDE":    run_scipy_de,
    "LBFGSB":     run_lbfgsb,
    "jSO":        lambda p,b,s: run_pyade(p, b, s, "jso"),
    "L_SHADE":    lambda p,b,s: run_pyade(p, b, s, "lshade"),
    "L-SHADE":    lambda p,b,s: run_pyade(p, b, s, "lshade"),
}

# =====================================================================
#  ERT summary helpers
# =====================================================================
@dataclass
class RunResult:
    alg: str; suite: str; fid: int; dim: int; run: int
    fbest: float; err: float; nfe: int; hit: int; time_sec: float; budget: int

def summarize_ert(rows: List[RunResult], budget_mult: int) -> List[Dict[str, float]]:
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r.alg, r.suite, r.fid, r.dim)].append(r)

    summary=[]
    for (alg, suite, fid, dim), lst in grouped.items():
        successes = sum(r.hit for r in lst)
        total_evals = sum(r.nfe if r.hit else dim * budget_mult for r in lst)
        ert = total_evals / successes if successes else float("nan")
        best_err = min(r.err for r in lst)
        mean_err = float(np.mean([r.err for r in lst]))
        summary.append(dict(alg=alg, suite=suite, fid=fid, dim=dim,
                            ERT=ert, succ=successes, runs=len(lst),
                            best_err=best_err, mean_err=mean_err))
    return summary

# =====================================================================
#  Main
# =====================================================================
def parse_ids(spec: str) -> List[int]:
    res=[]
    for part in str(spec).split(","):
        s = part.strip()
        if not s:
            continue
        if "-" in s:
            a, b = map(int, s.split("-"))
            lo, hi = (a, b) if a <= b else (b, a)
            res.extend(range(lo, hi+1))
        else:
            res.append(int(s))
    return sorted(set(res))

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--suite", choices=["cec2014", "cec2022"], default="cec2022")
    ap.add_argument("--dims",  type=int, nargs="+", default=[10, 20])
    ap.add_argument("--fids",  type=str, default=None, help="e.g. '1-12'")
    ap.add_argument("--runs",  type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--budget-mult", type=int, default=20000,
                    help="budget = budget_mult × dim")
    ap.add_argument("--target-tol", type=float, default=1e-8)
    ap.add_argument("--algs",  nargs="+", default=["SLO_HBYRID", "CMAES", "SciPyDE"])
    ap.add_argument("--outdir", type=str, default="results_cec")
    args = ap.parse_args()

    if args.fids is None:
        args.fids = "1-12" if args.suite == "cec2022" else "1-30"
    fids = parse_ids(args.fids)

    os.makedirs(args.outdir, exist_ok=True)
    results_csv = os.path.join(args.outdir, f"{args.suite}_results.csv")
    with open(results_csv, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["alg","suite","fid","dim","run","fbest","err","nfe","hit","time_sec"])

    all_rows: List[RunResult] = []
    for dim in args.dims:
        budget = args.budget_mult * dim
        for fid in fids:
            try:
                prob = build_problem(args.suite, fid, dim)
            except Exception as e:
                print(f"[SKIP] F{fid:02d} D{dim}: {e}", file=sys.stderr)
                continue
            for run_idx in range(args.runs):
                seed = args.seed0 + run_idx
                for alg in args.algs:
                    if alg not in ALG_MAP:
                        print(f"[SKIP] unknown alg {alg}", file=sys.stderr)
                        continue
                    runner = ALG_MAP[alg]
                    t0 = time.time()
                    try:
                        fbest, nfe = runner(prob, budget, seed)
                    except Exception as e:
                        print(f"[ERR] {alg} F{fid:02d}D{dim}: {e}", file=sys.stderr)
                        fbest, nfe = float("inf"), 0
                    t1 = time.time()
                    err = abs(fbest - prob.fopt)
                    hit = int(err <= args.target_tol)
                    row = RunResult(alg=alg, suite=args.suite, fid=fid, dim=dim,
                                    run=seed, fbest=fbest, err=err, nfe=nfe, hit=hit,
                                    time_sec=t1 - t0, budget=budget)
                    all_rows.append(row)
                    with open(results_csv, "a", newline="") as fh:
                        csv.writer(fh).writerow(
                            [row.alg,row.suite,row.fid,row.dim,row.run,
                             row.fbest,row.err,row.nfe,row.hit,row.time_sec])
                    print(f"{alg:10s}|{args.suite} F{fid:02d} D{dim:02d} run{run_idx:02d} "
                          f"| f_best={fbest:.3e} err={err:.1e} nfe={nfe:7d} {'HIT' if hit else ''}")

    # ------------- ERT summary -------------
    ert_csv = os.path.join(args.outdir, f"{args.suite}_ERT_summary.csv")
    ert_rows = summarize_ert(all_rows, budget_mult=args.budget_mult)
    with open(ert_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alg","suite","fid","dim","ERT","succ","runs","best_err","mean_err"])
        for d in sorted(ert_rows, key=lambda z: (z["alg"], z["dim"], z["fid"])):
            w.writerow([d["alg"], d["suite"], d["fid"], d["dim"],
                        d["ERT"], d["succ"], d["runs"], d["best_err"], d["mean_err"]])
    print("Saved:", results_csv)
    print("Saved:", ert_csv)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
