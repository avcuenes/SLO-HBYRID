# -*- coding: utf-8 -*-
"""
slo_bench.cec_run
-----------------
CEC2014 / CEC2022 benchmark runner for Spiral-NM and baselines.
• Uses opfunu.cec_based to construct problems.
• Records per-run CSV and an ERT summary CSV.
• Early-stop when |f_best - fopt| <= target_tol, otherwise stop at budget.

Example:
  python3 -m slo_bench.cec_run \
    --suite cec2022 --dims 10 20 --fids 1-12 --runs 10 \
    --algs NMStep2 CMAES SciPyDE jSO L_SHADE \
    --budget-mult 20000 --target-tol 1e-8 \
    --seed0 0 --outdir results_cec
"""
from __future__ import annotations
import argparse, csv, os, sys, time, importlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Spiral-NM
spiral_nm = importlib.import_module("slo_bench.spiral_nm")
assert hasattr(spiral_nm, "spiral_nm_step2"), "spiral_nm_step2 missing"
spiral_step = spiral_nm.spiral_nm_step2

from spiral_lshade import SpiralLSHADEParams, _slo_lshade_core
# Optional libraries
try:
    import cma
    HAVE_CMA = True
except Exception:
    HAVE_CMA = False

try:
    from scipy.optimize import differential_evolution, minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import pyade.jso as pyade_jso
    import pyade.lshade as pyade_lshade
    HAVE_PYADE = True
except Exception:
    HAVE_PYADE = False

# opfunu CEC
try:
    from opfunu.cec_based import cec2014 as ofu2014
    HAVE_2014 = True
except Exception:
    HAVE_2014 = False

try:
    from opfunu.cec_based import cec2022 as ofu2022
    HAVE_2022 = True
except Exception:
    HAVE_2022 = False


# ---------------- utilities ----------------

def set_seed(seed: int):
    np.random.seed(int(seed))

def project_box_open(x, lb, ub):
    x = np.asarray(x, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    rng = ub - lb
    eps = np.maximum(1e-8 * np.maximum(rng, 1.0), 1e-12)
    eps = np.minimum(eps, 0.45 * np.maximum(rng, 1e-30))
    return np.minimum(ub - eps, np.maximum(lb + eps, x))


@dataclass
class Problem:
    fid: int
    name: str
    dim: int
    lower: np.ndarray
    upper: np.ndarray
    fopt: float
    f: Callable[[np.ndarray], float]


def _to_array(v, default, size):
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        arr = np.asarray(v, dtype=np.float64)
        if arr.size == 1:  return np.full(size, float(arr.item()))
        if arr.size == size: return arr.astype(np.float64)
        return np.full(size, float(default))
    else:
        try: val = float(v)
        except Exception: val = float(default)
        return np.full(size, val)


def build_problem(suite: str, fid: int, dim: int) -> Problem:
    suite = suite.lower()
    if suite == "cec2014":
        if not HAVE_2014: raise RuntimeError("opfunu cec2014 not available")
        # map 1..30
        fmap = {
            1: ofu2014.F12014, 2: ofu2014.F22014, 3: ofu2014.F32014, 4: ofu2014.F42014, 5: ofu2014.F52014,
            6: ofu2014.F62014, 7: ofu2014.F72014, 8: ofu2014.F82014, 9: ofu2014.F92014, 10: ofu2014.F102014,
            11: ofu2014.F112014, 12: ofu2014.F12014, 13: ofu2014.F132014, 14: ofu2014.F142014, 15: ofu2014.F152014,
            16: ofu2014.F162014, 17: ofu2014.F172014, 18: ofu2014.F182014, 19: ofu2014.F192014, 20: ofu2014.F202014,
            21: ofu2014.F212014, 22: ofu2014.F22014, 23: ofu2014.F222014, 24: ofu2014.F232014, 25: ofu2014.F242014,
            26: ofu2014.F252014, 27: ofu2014.F262014, 28: ofu2014.F272014, 29: ofu2014.F282014, 30: ofu2014.F292014
        }
    elif suite == "cec2022":
        if not HAVE_2022: raise RuntimeError("opfunu cec2022 not available")
        fmap = {
            1: ofu2022.F12022, 2: ofu2022.F22022, 3: ofu2022.F32022, 4: ofu2022.F42022, 5: ofu2022.F52022, 6: ofu2022.F62022,
            7: ofu2022.F72022, 8: ofu2022.F82022, 9: ofu2022.F92022, 10: ofu2022.F102022, 11: ofu2022.F112022, 12: ofu2022.F122022
        }
    else:
        raise ValueError("suite must be one of {'cec2014','cec2022'}")

    if fid not in fmap: raise ValueError(f"fid {fid} not in suite {suite}")
    fcls = fmap[fid]

    fobj = None
    for k in ("ndim", "dimension", "problem_size", "n_dimensions", "dim"):
        try:
            fobj = fcls(**{k: dim})
            break
        except TypeError:
            continue
    if fobj is None:
        try: fobj = fcls(dim)
        except TypeError as e:
            raise RuntimeError(f"Cannot construct {fcls.__name__} for dim={dim}: {e}")

    lb_attr = getattr(fobj, "lb", -100.0)
    ub_attr = getattr(fobj, "ub", 100.0)
    lower = _to_array(lb_attr, -100.0, dim)
    upper = _to_array(ub_attr, 100.0, dim)
    fopt  = float(getattr(fobj, "f_bias", 0.0))

    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if hasattr(fobj, "evaluate"):
            return float(fobj.evaluate(x))
        return float(fobj(x))

    return Problem(fid=fid, name=f"F{fid:02d}_{suite.upper()}", dim=dim,
                   lower=lower, upper=upper, fopt=fopt, f=f)


class EvalCounter:
    def __init__(self, f: Callable[[np.ndarray], float], fopt: float, target_tol: float):
        self.f = f; self.fopt=float(fopt); self.target_tol=float(target_tol)
        self.nfe=0; self.best=float("inf"); self.hit=False
    def __call__(self, x: np.ndarray)->float:
        y=float(self.f(x)); self.nfe+=1
        if y<self.best:
            self.best=y
            if abs(self.best-self.fopt)<=self.target_tol: self.hit=True
        return y


# ---------------- algorithms ----------------


def run_spiral_lshade_hybrid(
    f: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
    kw: Optional[Dict] = None,
) -> Tuple[float, int]:
    """Adapter matching (f, lb, ub, budget, seed, kw) → (f_best, n_evals)."""
    # normalise kw handling
    if isinstance(kw, SpiralLSHADEParams):
        kw = {"params": kw}
    kw = dict(kw) if kw else {}

    # pull params & rng seed
    p = kw.get("params", SpiralLSHADEParams())
    rng_seed_val = kw.pop("rng_seed", seed)
    if isinstance(rng_seed_val, np.random.Generator):
        rng = rng_seed_val
    else:
        rng = np.random.default_rng(int(rng_seed_val))

    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    best_f, evals = _slo_lshade_core(f, lb, ub, budget, rng, p)
    return float(best_f), int(evals)

def _project_box_open(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return np.minimum(np.maximum(x, lb + eps), ub - eps)
def run_spiral_lshade(
    prob,               # an object with .dim .lower .upper .f  (and optionally .fopt)
    budget: int,
    seed: int,
    ):
    """
    Wrapper so you can call the Spiral-LSHADE hybrid the same way you call run_spiral_nm.
    Returns
    -------
    f_best : float   best objective value found
    n_evals: int     number of evaluations performed
    """
    rng = np.random.default_rng(seed)
    d = prob.dim
    lb = np.asarray(prob.lower, float)
    ub = np.asarray(prob.upper, float)

    # starting point (uniform open box, like your run_spiral_nm)
    x0 = lb + rng.random(d) * (ub - lb)

    # evaluation counter
    eval_counter = EvalCounter(
        lambda z: prob.f(_project_box_open(z, lb, ub)),
        getattr(prob, "fopt", None),
        target_tol=1e-8
    )

    # call the hybrid optimiser
    f_best, n_evals = run_spiral_lshade_hybrid(
        f=eval_counter,
        lb=lb,
        ub=ub,
        budget=budget,
        seed=seed,
        kw={"params": SpiralLSHADEParams()}   # customise params if you like
    )

    return float(f_best), int(n_evals)
def run_spiral_nm(prob: Problem, budget: int, seed: int)->Tuple[float,int]:
    d=prob.dim; x0=prob.lower+np.random.default_rng(seed).random(d)*(prob.upper-prob.lower)
    evalf=EvalCounter(lambda z: prob.f(project_box_open(z,prob.lower,prob.upper)),
                      prob.fopt, target_tol=1e-8)
    xb, fb, ne, _ = spiral_step(evalf, x0, prob.lower, prob.upper, budget=budget, random_state=seed)
    return float(fb), int(ne)

def run_cmaes(prob: Problem, budget: int, seed: int)->Tuple[float,int]:
    if not HAVE_CMA: raise RuntimeError("cma not installed")
    evalf=EvalCounter(lambda z: prob.f(project_box_open(z,prob.lower,prob.upper)),
                      prob.fopt, target_tol=1e-8)
    x0=(prob.lower+prob.upper)/2.0
    sigma0=0.3*float(np.max(prob.upper-prob.lower))
    opts={'bounds':[prob.lower,prob.upper],'maxfevals':int(budget),'seed':int(seed),'verbose':-9,'CMA_active':True}
    _, fbest, _, nfe, *_ = cma.fmin(lambda x: evalf(np.asarray(x,dtype=np.float64)), x0, sigma0, options=opts)
    return float(fbest), int(nfe)

def run_scipy_de(prob: Problem, budget: int, seed: int)->Tuple[float,int]:
    if not HAVE_SCIPY: raise RuntimeError("scipy not installed")
    evalf=EvalCounter(lambda z: prob.f(project_box_open(z,prob.lower,prob.upper)),
                      prob.fopt, target_tol=1e-8)
    dim=prob.dim; pop=15; maxiter=max(1,budget//(pop*dim))
    res=differential_evolution(lambda x: evalf(np.asarray(x,dtype=np.float64)),
                               bounds=list(zip(prob.lower,prob.upper)),
                               popsize=pop, maxiter=maxiter, seed=seed,
                               polish=False, updating='deferred', workers=1)
    return float(res.fun), int(evalf.nfe)

def run_lbfgsb(prob: Problem, budget: int, seed: int)->Tuple[float,int]:
    if not HAVE_SCIPY: raise RuntimeError("scipy not installed")
    evalf=EvalCounter(lambda z: prob.f(project_box_open(z,prob.lower,prob.upper)),
                      prob.fopt, target_tol=1e-8)
    x0=prob.lower+np.random.default_rng(seed).random(prob.dim)*(prob.upper-prob.lower)
    res=minimize(lambda x: evalf(np.asarray(x,dtype=np.float64)), x0, method="L-BFGS-B",
                 bounds=list(zip(prob.lower,prob.upper)), options={'maxfun':int(budget),'disp':False})
    return float(res.fun), int(evalf.nfe)

def run_pyade(prob: Problem, budget: int, seed: int, which:str)->Tuple[float,int]:
    if not HAVE_PYADE: raise RuntimeError("pyade-python not installed")
    evalf=EvalCounter(lambda z: prob.f(project_box_open(z,prob.lower,prob.upper)),
                      prob.fopt, target_tol=1e-8)
    bounds=np.stack([prob.lower,prob.upper],axis=1)
    if which.lower()=="jso": algo=pyade_jso
    else: algo=pyade_lshade
    params=algo.get_default_params(dim=prob.dim)
    params["bounds"]=bounds; params["func"]=lambda x: evalf(np.asarray(x,dtype=np.float64))
    pop=params.get("NP", max(20,5*prob.dim)); iters=max(1,budget//max(pop,1))
    if "max_iters" in params: params["max_iters"]=iters
    elif "iters" in params:   params["iters"]=iters
    if "seed" in params: params["seed"]=seed
    sol, fit=algo.apply(**params)
    return float(fit), int(evalf.nfe)


ALG_MAP = {
    "SLO_HBYRID" : run_spiral_lshade,
    "NMStep2": run_spiral_nm,
    "CMAES":   run_cmaes,
    "SciPyDE": run_scipy_de,
    "LBFGSB":  run_lbfgsb,
    "jSO":     lambda prob,bud,seed: run_pyade(prob,bud,seed,"jso"),
    "L_SHADE": lambda prob,bud,seed: run_pyade(prob,bud,seed,"lshade"),
    "L-SHADE": lambda prob,bud,seed: run_pyade(prob,bud,seed,"lshade"),
}


# ---------------- ERT summary ----------------

@dataclass
class RunResult:
    alg: str; suite: str; fid: int; dim: int; run: int
    fbest: float; err: float; nfe: int; hit: int; time_sec: float

def summarize_ert(rows: List[RunResult]) -> List[Dict[str, float]]:
    out=[]
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows: groups[(r.alg,r.suite,r.fid,r.dim)].append(r)
    for key,lst in groups.items():
        alg,suite,fid,dim=key
        succ=sum(1 for r in lst if r.hit)
        total=sum(r.nfe for r in lst)
        ert=(total/succ) if succ>0 else float("nan")
        best_err=min(r.err for r in lst)
        mean_err=float(np.mean([r.err for r in lst]))
        out.append({"alg":alg,"suite":suite,"fid":fid,"dim":dim,"ERT":ert,
                    "succ":succ,"runs":len(lst),"best_err":best_err,"mean_err":mean_err})
    return out


# ---------------- main ----------------

def parse_ids(spec: str)->List[int]:
    res=[]
    for part in str(spec).split(","):
        s=part.strip()
        if not s: continue
        if "-" in s:
            a,b = map(int, s.split("-"))
            lo,hi=(a,b) if a<=b else (b,a)
            res.extend(range(lo,hi+1))
        else:
            res.append(int(s))
    return sorted(set(res))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, default="cec2022", choices=["cec2014","cec2022"])
    ap.add_argument("--dims",  type=int, nargs="+", default=[10,20])
    ap.add_argument("--fids",  type=str, default=None, help="Ranges like '1-12' (2022) or '1-30' (2014)")
    ap.add_argument("--runs",  type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--budget-mult", type=int, default=20000, help="Total evals = budget_mult * dim")
    ap.add_argument("--target-tol", type=float, default=1e-8)
    ap.add_argument("--algs",  nargs="+", default=["NMStep2","CMAES","SciPyDE"])
    ap.add_argument("--outdir", type=str, default="results_cec")
    args=ap.parse_args()

    if args.fids is None:
        args.fids = "1-12" if args.suite.lower()=="cec2022" else "1-30"
    fids=parse_ids(args.fids)

    os.makedirs(args.outdir, exist_ok=True)
    # Combined CSV
    results_csv=os.path.join(args.outdir, f"{args.suite}_results.csv")
    with open(results_csv,"w",newline="") as fh:
        csv.writer(fh).writerow(["alg","suite","fid","dim","run","fbest","err","nfe","hit","time_sec"])

    all_rows: List[RunResult]=[]
    for dim in args.dims:
        budget=args.budget_mult*dim
        for fid in fids:
            # build problem
            try:
                prob=build_problem(args.suite, fid, dim)
            except Exception as e:
                print(f"[SKIP] F{fid:02d} dim={dim}: {e}", file=sys.stderr)
                continue
            for r in range(args.runs):
                seed=args.seed0 + r
                for alg in args.algs:
                    if alg not in ALG_MAP:
                        print(f"[SKIP] unknown alg {alg}", file=sys.stderr); continue
                    runner=ALG_MAP[alg]
                    t0=time.time()
                    try:
                        fbest,nfe=runner(prob, budget, seed)
                    except Exception as e:
                        print(f"[ERR] {alg} F{fid:02d}D{dim}: {e}", file=sys.stderr)
                        fbest,nfe=float("inf"),0
                    t1=time.time()
                    err=abs(fbest-prob.fopt); hit=int(err<=args.target_tol)
                    row=RunResult(alg=alg, suite=args.suite, fid=fid, dim=dim, run=seed,
                                  fbest=fbest, err=err, nfe=nfe, hit=hit, time_sec=(t1-t0))
                    all_rows.append(row)
                    with open(results_csv,"a",newline="") as fh:
                        csv.writer(fh).writerow([row.alg,row.suite,row.fid,row.dim,row.run,row.fbest,row.err,row.nfe,row.hit,row.time_sec])
                    print(f"{alg:10s}|{args.suite} F{fid:02d} d{dim:02d} run{r:02d} | f*={fbest:.3e} err={err:.1e} nfe={nfe:7d} {'HIT' if hit else ''}")

    # ERT summary
    ert_rows = summarize_ert(all_rows)
    ert_csv = os.path.join(args.outdir, f"{args.suite}_ERT_summary.csv")
    with open(ert_csv,"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["alg","suite","fid","dim","ERT","succ","runs","best_err","mean_err"])
        for d in sorted(ert_rows, key=lambda z:(z["alg"],z["dim"],z["fid"])):
            w.writerow([d["alg"], d["suite"], d["fid"], d["dim"], d["ERT"], d["succ"], d["runs"], d["best_err"], d["mean_err"]])
    print("Saved:", results_csv)
    print("Saved:", ert_csv)

if __name__=="__main__":
    main()
