# [snip: header docstring omitted for brevity — identical features as before]
from __future__ import annotations
import math, sys, json
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Optional, List
import numpy as np

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _reflect(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    side = ub - lb
    y = (x - lb) % (2.0 * side)
    y = np.where(y <= side, y, 2.0 * side - y)
    return lb + np.clip(y, 0.0, side)

_GOLDEN = math.pi * (3.0 - math.sqrt(5.0))

def _pop_spread(pop: np.ndarray) -> float:
    return float(np.mean(np.std(pop, axis=0))) if len(pop) else 0.0

@dataclass
class SpiralLSHADEParams:
    # SHA / population
    H: int = 6
    pop_init: int = 18           # NP0 = pop_init * D
    pop_min: int = 6
    p_init: float = 1.0
    p_min: float = 0.05
    arc_rate: float = 1.8

    # spiral drift
    spiral_alpha: float = 0.08
    spiral_gamma: float = 0.02
    spiral_eps: float = 0.02

    # early-stopping / polishing
    stop_on_target: bool = True
    target_value: float = 0.0
    target_tol: float = 1e-8

    stall_gens_stop: int = 60
    nm_stall_gens: int = 20
    nm_radius_frac: float = 5e-4
    enable_nm: bool = True

    # diversity-based stop
    spread_stop_frac: float = 1e-6

    # Eigen-DE
    eigen_every: int = 10
    eigen_on_stall: bool = True
    eigen_prob: float = 0.35

    # verbosity
    verbose: bool = False

def _sample_F_CR(rng: np.random.Generator, M_F: np.ndarray, M_CR: np.ndarray):
    H = len(M_F); r = int(rng.integers(H))
    F = rng.standard_cauchy() * 0.1 + M_F[r]
    while F <= 0: F = rng.standard_cauchy() * 0.1 + M_F[r]
    F = min(F, 1.0)
    CR = float(np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0))
    return F, CR, r

def _mutate_cur_to_pbest(x_i, x_pbest, pop, archive, rng, F):
    NP = len(pop)
    r1 = int(rng.integers(NP))
    while np.all(pop[r1] == x_i): r1 = int(rng.integers(NP))
    union = NP + len(archive)
    r2_idx = int(rng.integers(union))
    while r2_idx == r1: r2_idx = int(rng.integers(union))
    r2 = pop[r2_idx] if r2_idx < NP else archive[r2_idx - NP]
    return x_i + F * (x_pbest - x_i + pop[r1] - r2)

def _eigenframe(pop: np.ndarray, ridge: float = 1e-12):
    mu = np.mean(pop, axis=0)
    X = pop - mu
    C = (X.T @ X) / max(1, len(pop) - 1) + ridge * np.eye(X.shape[1])
    vals, vecs = np.linalg.eigh(C)
    Q = vecs[:, np.argsort(vals)[::-1]]
    return mu, Q

def _hit_target_allowed(evals: int, min_evals_stop: int,
                        best_f: float, target_value: float, tol: float) -> bool:
    # allow early stop only after some work was done
    if evals < min_evals_stop:
        return False
    # robust: negatives (from logger/wrapper) count as "at target"
    err = max(0.0, best_f - target_value)
    return err <= tol

def _slo_lshade_core(
    f: Callable[[np.ndarray], float],
    lb: np.ndarray, ub: np.ndarray,
    budget: int, rng: np.random.Generator, p: SpiralLSHADEParams
) -> Tuple[float, int]:
    D = lb.size; side = ub - lb
    NP0 = int(p.pop_init * D)
    NP_min = int(p.pop_min); NP = NP0

    # minimum evals before we allow target-based early stop
    min_evals_stop = NP0 + max(50, 2*D)

    M_F = np.full(p.H, 0.5); M_CR = np.full(p.H, 0.5); k_mem = 0

    pop = lb + rng.random((NP, D)) * side
    fitness = np.array([f(ind) for ind in pop], dtype=float)
    evals = NP
    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy(); best_f = float(fitness[best_idx])
    last_improve_gen = 0

    theta = 0.0
    radius = p.spiral_alpha * float(np.min(side))
    archive: List[np.ndarray] = []

    mu_eig: Optional[np.ndarray] = None
    Q_eig: Optional[np.ndarray] = None

    gen = 0
    while evals < budget and NP >= NP_min:
        gen += 1

        if p.stop_on_target and _hit_target_allowed(evals, min_evals_stop, best_f, p.target_value, p.target_tol):
            break

        # linear population reduction
        NP_target = int(round(NP0 - (NP0 - NP_min) * (evals / max(1, budget))))
        if NP_target < NP:
            keep = np.argsort(fitness)[:NP_target]
            pop, fitness = pop[keep], fitness[keep]; NP = NP_target
            if len(archive) > int(p.arc_rate * NP): archive = archive[: int(p.arc_rate * NP)]

        # p-best pool
        p_best_frac = p.p_init - (p.p_init - p.p_min) * (evals / max(1, budget))
        p_best_size = max(2, int(math.ceil(p_best_frac * NP)))
        p_best_idx = np.argsort(fitness)[:p_best_size]

        # eigenframe periodically or when stalled
        stalled_gens = gen - last_improve_gen
        do_eig = (gen % max(1, p.eigen_every) == 0) or (p.eigen_on_stall and stalled_gens >= p.nm_stall_gens)
        if do_eig and NP >= max(3, D + 1):
            mu_eig, Q_eig = _eigenframe(pop)

        S_F, S_CR, S_df = [], [], []
        new_pop = pop.copy(); new_fit = fitness.copy()

        use_eigen = (Q_eig is not None) and (rng.random() < p.eigen_prob or (p.eigen_on_stall and stalled_gens >= p.nm_stall_gens))

        for i in range(NP):
            F, CR, _ = _sample_F_CR(rng, M_F, M_CR)
            if radius < 5e-3 * np.min(side): CR = max(CR, 0.9)

            x_i = pop[i]; x_pbest = pop[int(rng.choice(p_best_idx))]
            v = _mutate_cur_to_pbest(x_i, x_pbest, pop, archive, rng, F)

            # rand/1 rescue if step collapses
            if np.linalg.norm(v - x_i) < 1e-8 * np.max(side):
                a, b, c = rng.choice(len(pop), 3, replace=False)
                v = pop[a] + F * (pop[b] - pop[c])

            if use_eigen:
                xi_ = (x_i - mu_eig) @ Q_eig; v_ = (v - mu_eig) @ Q_eig
                j_rand = int(rng.integers(D))
                cross = rng.random(D) < CR; cross[j_rand] = True
                u_ = np.where(cross, v_, xi_); u = mu_eig + u_ @ Q_eig.T
            else:
                j_rand = int(rng.integers(D))
                cross = rng.random(D) < CR; cross[j_rand] = True
                u = np.where(cross, v, x_i)

            u = _reflect(u, lb, ub)
            fu = float(f(u)); evals += 1

            if fu < fitness[i]:
                new_pop[i] = u; new_fit[i] = fu
                archive.append(x_i.copy())
                if len(archive) > int(p.arc_rate * NP):
                    archive.pop(int(rng.integers(len(archive))))
                S_F.append(F); S_CR.append(CR); S_df.append(fitness[i] - fu)
                if fu < best_f:
                    best_f = fu; best_x = u.copy(); last_improve_gen = gen

            if p.stop_on_target and _hit_target_allowed(evals, min_evals_stop, best_f, p.target_value, p.target_tol):
                pop, fitness = new_pop, new_fit
                return best_f, evals

        pop, fitness = new_pop, new_fit

        # update memories
        if S_F:
            df = np.array(S_df, float); w = df / (np.sum(df) + 1e-16)
            M_F[k_mem] = float(np.sum(w * np.array(S_F)))
            M_CR[k_mem] = float(np.sum(w * np.array(S_CR)))
            k_mem = (k_mem + 1) % p.H

        # isotropic spiral drift (off when very close)
        theta = 0.0  # angle not used for isotropic; keep for compatibility if needed
        radius *= (1.0 - p.spiral_gamma)
        axis = rng.standard_normal(D); axis /= (np.linalg.norm(axis) + 1e-12)
        if radius < 1e-3 * np.min(side): axis[:] = 0.0
        best_x = _reflect(best_x + axis * radius, lb, ub)
        fb_move = float(f(best_x)); evals += 1
        if fb_move < best_f:
            best_f = fb_move; last_improve_gen = gen

        if p.verbose and (gen % 25 == 0):
            print(json.dumps({
                "gen": gen, "evals": evals, "best": best_f,
                "NP": NP, "radius": radius, "spread": _pop_spread(pop),
                "MF": float(M_F.mean()), "MCR": float(M_CR.mean())
            }), file=sys.stderr)

        tiny_radius = radius < p.nm_radius_frac * np.min(side)
        stalled_gens = gen - last_improve_gen
        if p.enable_nm and _HAS_SCIPY and (tiny_radius or stalled_gens >= p.nm_stall_gens):
            budget_left = budget - evals
            if budget_left > 20:
                res = minimize(f, best_x, method="Nelder-Mead",
                               options={"maxfev": min(1000, budget_left),
                                        "xatol": 1e-8, "fatol": 1e-12})
                evals += res.nfev
                if res.fun < best_f:
                    best_f = float(res.fun); best_x = res.x.copy()
                last_improve_gen = gen

        # in-place restart if long stall and diversity collapsed
        if (stalled_gens >= 120) and (_pop_spread(pop) < 1e-6 * np.max(side)):
            half = max(2, len(pop)//2)
            pop[:half] = _reflect(best_x + 0.2*np.min(side)*rng.standard_normal((half, D)), lb, ub)
            pop[half:] = lb + rng.random((len(pop)-half, D)) * side
            fitness[:half] = np.array([f(ind) for ind in pop[:half]])
            fitness[half:] = np.array([f(ind) for ind in pop[half:]])
            evals += len(pop); radius = 0.2 * np.min(side); last_improve_gen = gen

        if tiny_radius and stalled_gens >= p.stall_gens_stop: break
        if stalled_gens >= p.stall_gens_stop and _pop_spread(pop) < p.spread_stop_frac * float(np.max(side)): break

    return float(best_f), int(evals)

def run_spiral_lshade_hybrid(f, lb, ub, budget, seed, kw: Optional[Dict]=None) -> Tuple[float, int]:
    if isinstance(kw, SpiralLSHADEParams): kw = {"params": kw}
    kw = dict(kw) if kw else {}
    p: SpiralLSHADEParams = kw.get("params", SpiralLSHADEParams())
    seed_val = kw.get("rng_seed", seed)
    rng = seed_val if isinstance(seed_val, np.random.Generator) else np.random.default_rng(int(seed_val))
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    return _slo_lshade_core(f, lb, ub, int(budget), rng, p)

run_slo_lshade = run_spiral_lshade_hybrid

if __name__ == "__main__":
    def rastrigin(x): 
        return float(10*len(x) + np.sum(x*x - 10*np.cos(2*math.pi*x)))
    D=2; lb=-5.12*np.ones(D); ub=5.12*np.ones(D)
    params = SpiralLSHADEParams(
        pop_init=12, pop_min=4, p_init=0.9, p_min=0.05,
        arc_rate=2.0, spiral_alpha=0.06, spiral_gamma=0.03, spiral_eps=0.02,
        stop_on_target=True, target_value=0.0, target_tol=1e-8,
        enable_nm=True, nm_stall_gens=20, nm_radius_frac=5e-4,
        verbose=True
    )
    fb, ne = run_spiral_lshade_hybrid(rastrigin, lb, ub, 20000, 0, kw={"params": params})
    print("[demo] best", fb, "evals", ne)
