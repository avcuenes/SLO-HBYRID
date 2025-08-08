#!/usr/bin/env python3
# all_eng_bench.py  –  SINGLE-FILE benchmark: problems + all adapters + runner
# ---------------------------------------------------------------------------
# Example:
#   python all_eng_bench.py \
#          --algs SLO_HBYRID CMAES SciPyDE LBFGSB jSO L_SHADE \
#          --budget-mult 5000 --runs 20 --seed0 0
# ---------------------------------------------------------------------------

from __future__ import annotations
import argparse, csv, math, time, pathlib, sys, importlib
from typing import Callable, Dict
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# 1.  FOUR engineering problems  (static penalty α·Σ violation², β=2)
# ═══════════════════════════════════════════════════════════════════════
ALPHA = 1e7  # global penalty weight
BETA  = 2    # power in Σ violation^β          

def _viol(x): return np.maximum(x, 0.0)

class _Problem:
    name:  str
    dim:   int
    lower: np.ndarray
    upper: np.ndarray
    fopt:  float | None
    def f_raw(self, x): ...    # objective
    def g(self, x): ...        # vector of g≤0
    def f(self, x):
        v = np.sum(_viol(self.g(x))**BETA)
        return self.f_raw(x) + ALPHA * v

# --- 1.1 Pressure-vessel -----------------------------------------------------
class PressureVessel(_Problem):
    name = "pressure_vessel"
    lower = np.array([0.0, 0.0, 10.0, 10.0])
    upper = np.array([99. , 99. , 200. , 200. ])
    dim   = 4
    fopt  = 6059.714
    def f_raw(self,x):
        return (0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]**2 +
                3.1661*x[0]**2*x[3]   + 19.84*x[0]**2*x[2])
    def g(self,x):
        g1 = -x[0] + 0.0193*x[2]
        g2 = -x[1] + 0.00954*x[2]
        g3 = -math.pi*x[2]**2*x[3] - 4/3*math.pi*x[2]**3 + 750*1728
        g4 =  x[3] - 240
        return np.array([g1,g2,g3,g4])

# --- 1.2 Tension / compression spring ---------------------------------------
class Spring(_Problem):
    name="tension_spring"
    lower=np.array([0.05,0.25,11.0]); upper=np.array([2.0,1.3,12.0]); dim=3
    fopt=2.6254
    def f_raw(self,x): return (x[2]+2)*x[1]*x[0]**2
    def g(self,x):
        g1 = 1 - (x[1]**3*x[2])/(71785*x[0]**4)
        g2 = (4*x[1]**2 - x[0]*x[1])/(12566*(x[1]*x[0]**3 - x[0]**4)) + 1/(5108*x[0]**2) - 1
        g3 = 1 - 140.45*x[0]/(x[1]**2*x[2])
        g4 = (x[1]+x[0])/1.5 - 1
        return np.array([g1,g2,g3,g4])

# --- 1.3 Welded-beam ---------------------------------------------------------
P,L,E,G = 6000.,14.,30e6,12e6
tau_max,sig_max,delta_max = 13600.,30000.,0.25
class WeldedBeam(_Problem):
    name="welded_beam"; lower=np.array([0.1,0.1,0.1,0.1])
    upper=np.array([2.0,3.5,10.0,2.0]); dim=4; fopt=1.724852
    def f_raw(self,x): return 1.10471*x[0]**2*x[1] + 0.04811*x[2]*x[3]*(14+x[1])
    def _tau(self,x):
        tau1=P/(math.sqrt(2)*x[0]*x[1])
        M=P*(L+x[1]/2); R=math.sqrt(x[1]**2/4+((x[0]+x[2])/2)**2)
        J=2*((x[0]*x[1]*math.sqrt(2))*(x[1]**2/12+((x[0]+x[2])/2)**2))
        tau2=M*R/J
        return math.sqrt(tau1**2+2*tau1*tau2*x[1]/(2*R)+tau2**2)
    def _delta(self,x): return 4*P*L**3/(E*x[2]**3*x[3])
    def _sigma(self,x): return 6*P*L/(x[3]*x[2]**2)
    def _Pc(self,x):
        return (4.013*E*math.sqrt(x[2]**2*x[3]**6/36)/L**2) * \
               (1 - x[2]/(2*L)*math.sqrt(E/(4*G)))
    def g(self,x):
        g = np.zeros(7)
        g[0]= self._tau(x)-tau_max
        g[1]= self._sigma(x)-sig_max
        g[2]= x[0]-x[3]
        g[3]= 0.10471*x[0]**2+0.04811*x[2]*x[3]*(14+x[1])-5
        g[4]= 0.125-x[0]
        g[5]= self._delta(x)-delta_max
        g[6]= P-self._Pc(x)
        return g

# --- 1.4 Gear-box (5-D) ------------------------------------------------------
i,rho,n,sigma_y = 4,8,6,294.3
y_,b2_,Kv,Kw,N1,Pwr = 0.102,0.193,0.389,0.8,1500,7.5
class GearBox(_Problem):
    name="gear_box"; lower=np.array([20,10,30,18,2.75])
    upper=np.array([32,30,40,25,4]); dim=5; fopt=None
    def f_raw(self,x):
        b,d1,d2,Z1,m=x
        Dr=m*(i*Z1-2.5); lw=2.5*m; Di=Dr-2*lw; bw=3.5*m
        d0=d2+25; dp=0.25*(Di-d0)
        return (math.pi/4)*(rho/1000)*(b*m**2*Z1**2*(i**2+1)
              -(Di**2-d0**2)*(1-bw) - n*dp**2*bw - (d1+d2)*b)
    def g(self,x):
        b,d1,d2,Z1,m=x
        Dr=m*(i*Z1-2.5); lw=2.5*m; Di=Dr-2*lw; d0=d2+25
        D1=m*Z1; D2=i*m*Z1; Z2=Z1*D2/D1
        v=math.pi*D1*N1/60000; b1=102*Pwr/v; b3=4.97e6*Pwr/(N1*2)
        Fs=math.pi*Kv*Kw*sigma_y*b*m*y_; Fp=2*Kv*Kw*D1*b*Z2/(Z1+Z2)
        g1=-Fs+b1; g2=-(Fs/Fp)+b2_; g3=-(d1**3)+b3
        return np.array([g1,g2,g3])


# ======================================================================
# 1. Speed-reducer  (7 D)  – Arora & Haugen 1988
# ======================================================================
class SpeedReducer(_Problem):
    name  = "speed_reducer"
    lower = np.array([2.6 ,0.7 ,17. ,7.3 ,7.3 ,2.9 ,5.0])
    upper = np.array([3.6 ,0.8 ,28. ,8.3 ,8.3 ,3.9 ,5.5])
    dim   = 7
    fopt  = 2994.0
    def f_raw(self, x):
        x1,x2,x3,x4,x5,x6,x7 = x
        return (0.785*x1*x2**2*(10*x3**2/3 + 14.933*x3 - 43.093)
              - 1.508*x1*(x6**2 + x7**2)
              + 7.477*(x6**3 + x7**3)
              + 0.785*x4*(x6**2 + x7**2))
    def g(self, x):
        x1,x2,x3,x4,x5,x6,x7 = x
        g = np.zeros(11)
        g[0]  = 27/(x1*x2**2*x3) - 1
        g[1]  = 397.5/(x1*x2**2*x3**2) - 1
        g[2]  = 1.93*x4**3/(x2*x3*x6**4) - 1
        g[3]  = 1.93*x5**3/(x2*x3*x7**4) - 1
        g[4]  = math.sqrt(745*x4/(x2*x3)) / (0.1*x6**3) - 1
        g[5]  = math.sqrt(745*x5/(x2*x3)) / (0.1*x7**3) - 1
        g[6]  = x2*x3 - 40
        g[7]  = x1/x2 - 12
        g[8]  = 5 - x1/x2
        g[9]  = 1.9 - x4 + 1.5*x6
        g[10] = 1.9 - x5 + 1.1*x7
        return g

# ======================================================================
# 2. Car side-impact beam  (11 D)  – Fourie & Heyns 1999
# ======================================================================
class CarSideImpact(_Problem):
    name  = "car_side_impact"
    lower = np.array([0.5 ,0.45 ,0.5 ,0.5 ,0.875 ,0.4 ,0.4 ,0.5 ,0.5 ,0.875 ,0.4])
    upper = np.array([1.5 ,1.35 ,1.5 ,1.5 ,2.625,1.2 ,1.2 ,1.5 ,1.5 ,2.625 ,1.2])
    dim   = 11
    fopt  = None
    def f_raw(self, x):
        return (1.98 + 4.9*x[0] + 6.67*x[1] + 6.98*x[2]
              + 4.01*x[3] + 1.78*x[4] + 2.73*x[5])
    def g(self, x):
        g = np.zeros(10)
        g[0] = 1.16 - 0.3717*x[1]*x[3] - 0.0092928*x[2]
        g[1] = 0.261 - 0.0159*x[1]*x[2] - 0.06486*x[0]
        g[2] = 0.214 + 0.00817*x[4] - 0.045195*x[1] - 0.0135168*x[1]
        g[3] = 0.74  - 0.698*x[4] - 0.173*x[5]
        g[4] = 0.68  - 0.2*x[2] - 0.051*x[3] - 0.102*x[0]
        g[5] = 1.0   - 0.1353*x[0] + 0.007283*x[3] + 0.009498*x[2]
        g[6] = (math.pi/4)*(x[0]**2 - x[6]**2) - 40
        g[7] = (math.pi/4)*(x[5]**2 - x[7]**2) - 20
        g[8] = (math.pi/4)*(x[4]**2*x[8] - x[5]**2*x[8])
        g[9] = x[9] - x[10]
        return g

# ======================================================================
# 3. Hydrostatic thrust bearing  (4 D)  – Akhtar & Tai 2002
# ======================================================================
class HydrostaticBearing(_Problem):
    name  = "hydrostatic_thrust_bearing"
    lower = np.array([0.5 ,0.35,17.0,7.0])
    upper = np.array([1.1 ,0.95,28.0,1.9])
    dim   = 4
    fopt  = None
    def f_raw(self, x):
        x1,x2,x3,x4 = x
        return 4.9e-5*x1**(-0.5)*x2**2*math.sqrt(x3) + 1.62e-3*x4/x3
    def g(self, x):
        x1,x2,x3,x4 = x
        return np.array([
            -x1 + 0.5,   x1 - 1.1,
            -x2 + 0.35,  x2 - 0.95,
            -x3 + 17.0,  x3 - 28.0,
            x4 - 1.9])

# ======================================================================
# 4. Four-bar truss  (3 D)  – Deb 1991
# ======================================================================
class FourBarTruss(_Problem):
    name="four_bar_truss"
    lower=np.array([1.,1.,1.])
    upper=np.array([5.,5.,5.])
    dim  =3
    fopt =1.0
    def f_raw(self,x): return x[0]*x[2]*(1+x[1])
    def g(self,x):
        g1=(x[0]+2*x[1])*x[2]-5
        g2=x[0]*x[1]-25
        g3=x[0]-2*x[1]
        return np.array([g1,g2,g3])

# ======================================================================
# 5. Ten-bar planar truss (continuous)  – Rajeev & Krishnamoorthy 1992
# ======================================================================
LEN  = np.array([1,1,math.sqrt(2),math.sqrt(2),1,
                 1,math.sqrt(2),math.sqrt(2),math.sqrt(2),math.sqrt(2)])
RHO  = 0.1
class TenBarTruss(_Problem):
    name  = "ten_bar_truss"
    lower = np.full(10,0.1)
    upper = np.full(10,35.0)
    dim   = 10
    fopt  = 505.0
    def f_raw(self,x): return RHO*np.dot(LEN,x)
    def g(self,x):
        σ = 5.0/x               # surrogate stress
        return σ/25.0 - 1       # ≤0


# ======================================================================
# 6. 25-bar space truss  (10 D surrogate)  – Farshi & Kelley 1977
# ======================================================================
_L25 = np.array([1,1,1,1, np.sqrt(2),np.sqrt(2),np.sqrt(2),np.sqrt(2),1,1])
_RHO = 0.1
class SpaceTruss25(_Problem):
    name="space_truss_25bar"
    lower=np.full(10,0.1); upper=np.full(10,3.0); dim=10; fopt=2.13e3
    def f_raw(self,A): return _RHO*np.dot(_L25,A)
    def g(self,A):      return A-0.1                    # A_min constraint

# ======================================================================
# 7. Cantilever beam (continuous)  – Bram & Glover 1997
# ======================================================================
class CantileverCont(_Problem):
    name="cantilever_cont"
    lower=np.array([0.01,0.01]); upper=np.array([0.2,0.2]); dim=2; fopt=1.33995
    _P,_L,_E = 500,200,2e5
    def f_raw(self,x):
        b,h=x; return self._P*self._L**3/(3*self._E*b*h**3)
    def g(self,x):
        b,h=x; σ=6*self._P*self._L/(b*h**2)
        return np.array([σ-140])

# ======================================================================
# 8. Cantilever beam (discrete-section)  – Degertekin 2008
# ======================================================================
_SECT=np.array([1,2,3,4,5,6])/10   # 0.1 in² … 0.6 in²
class CantileverDisc(_Problem):
    name="cantilever_disc"
    lower=np.zeros(5); upper=np.full(5,5); dim=5; fopt=2.475
    def _round_idx(self,idx): return np.clip(np.round(idx).astype(int),0,5)
    def f_raw(self,idx):
        A=_SECT[self._round_idx(idx)]
        return 100*np.sum(A)                              # weight surrogate
    def g(self,idx):
        A=_SECT[self._round_idx(idx)]
        σ=1000/A-140
        return np.array([σ.max()-1e-3])

# ======================================================================
# 9. Stepped-column buckling (10 D)  – Liang 2000
# ======================================================================
class SteppedColumn(_Problem):
    name="stepped_column"
    lower=np.full(10,0.1); upper=np.full(10,10.0); dim=10; fopt=1.119e-4
    def f_raw(self,A): return 0.1*np.sum(A)
    def g(self,A):
        Pcr=(math.pi**2)*2.1e5*np.sum(A)/(100**2)
        return np.array([1e5/Pcr-1])

# ======================================================================
# 10. Machining parameters  (4 D)  – Dimopoulos & Margaritis 2001
# ======================================================================
class MachiningCost(_Problem):
    name="machining_cost"
    lower=np.array([0.5,0.1,50,0.01]); upper=np.array([2.0,0.5,400,0.05])
    dim=4; fopt=2.385
    def f_raw(self,x):
        d,f,v,t=x
        return 0.8*d+0.01*f+0.0005*v+0.2*t
    def g(self,x):
        d,f,v,_=x
        g1=1-0.002*v*f
        g2=0.1-d+0.5*f
        return np.array([g1,g2])

# ======================================================================
# 11. Heat-exchanger design  (3 D)  – Rao 2011
# ======================================================================
class HeatExchanger(_Problem):
    name="heat_exchanger"
    lower=np.array([0.3,5,100]); upper=np.array([1.5,9,600]); dim=3; fopt=7176.1
    def f_raw(self,x):
        D,L,u=x
        return 0.6224*D*L*u+131*D+146.3
    def g(self,x):
        D,L,u=x
        Q=0.8*D*L*u; return np.array([60-Q])

# ======================================================================
# 12. Thick-wall pressure vessel (discrete Ts,Th)  – Sandgren 1990
# ======================================================================
_mult=0.0625
class ThickPressureVessel(_Problem):
    name="pressure_vessel_thick"
    lower=np.array([0.0625,0.0625,10,10])
    upper=np.array([2,2,100,240]); dim=4; fopt=5885.3
    def _snap(self,x): return np.round(x/_mult)*_mult
    def f_raw(self,x):
        Ts,Th,R,L=self._snap(x[0]),self._snap(x[1]),x[2],x[3]
        return (0.6224*Ts*R*L+1.7781*Th*R**2+3.1661*Ts**2*L+19.84*Ts**2*R)
    def g(self,x):
        Ts,Th,R,L=self._snap(x[0]),self._snap(x[1]),x[2],x[3]
        return np.array([
            -Ts+0.0625, -Th+0.0625, -R+10, R-100,
            -L+10, L-240,
            math.pi*R**2*L+4/3*math.pi*R**3-1.0e5])

# ======================================================================
# 13. Gear-train integer  (4 int)  – Goldberg 1989
# ======================================================================
_target=1/6.931
class GearTrain(_Problem):
    name="gear_train"
    lower=np.full(4,12); upper=np.full(4,60); dim=4; fopt=2.7e-4
    def _round(self,z): return np.clip(np.round(z).astype(int),12,60)
    def f_raw(self,z):
        z=self._round(z)
        ratio=(z[0]*z[1])/(z[2]*z[3])
        return abs(ratio-_target)
    def g(self,z): return np.zeros(1)       # no extra constraints

# gather problems
PROBLEMS = [PressureVessel(), Spring(), WeldedBeam(), GearBox(),SpeedReducer(),
    CarSideImpact(),
    HydrostaticBearing(),
    FourBarTruss(),
    TenBarTruss(),
    SpaceTruss25(), CantileverCont(), CantileverDisc(),
        SteppedColumn(), MachiningCost(), HeatExchanger(),
        ThickPressureVessel(), GearTrain()
]

# ═══════════════════════════════════════════════════════════════════════
# 2.  Utility: reflection to keep box constraints
# ═══════════════════════════════════════════════════════════════════════
def reflect(x, lb, ub):
    rng = ub - lb
    y   = (x - lb) % (2*rng)
    return np.where(y <= rng, lb + y, ub - (y - rng))

# ═══════════════════════════════════════════════════════════════════════
# 3.  Adapters for all optimisers
# ═══════════════════════════════════════════════════════════════════════
def run_slo_hbyrid(f, lb, ub, budget, seed, kw):
    from spiral_lshade import SpiralLSHADEParams, _slo_lshade_core
    rng=np.random.default_rng(seed)
    best, ne=_slo_lshade_core(f, lb, ub, int(budget), rng, SpiralLSHADEParams())
    return float(best), int(ne)

def run_cmaes(f, lb, ub, budget, seed, kw):
    import cma, numpy as np
    lb,ub=np.asarray(lb),np.asarray(ub)
    x0=(lb+ub)/2; sigma=0.3*np.ptp(ub)
    _f=lambda x: f(reflect(np.asarray(x),lb,ub))
    _, fb, _, ne,*_=cma.fmin(_f,x0,sigma,
            dict(bounds=[lb,ub],maxfevals=int(budget),seed=int(seed),verbose=-9))
    return float(fb), int(ne)

def run_scipyde(f, lb, ub, budget, seed, kw):
    from scipy.optimize import differential_evolution
    dim=len(lb); pop=15; iters=max(1,budget//(pop*dim))
    _f=lambda x: f(reflect(np.asarray(x),lb,ub))
    res=differential_evolution(_f, list(zip(lb,ub)),
                               popsize=pop,maxiter=iters,seed=seed,
                               polish=False, updating='deferred', workers=1)
    return float(res.fun), int(res.nfev)

def run_lbfgsb(f, lb, ub, budget, seed, kw):
    from scipy.optimize import minimize
    lb,ub=np.asarray(lb),np.asarray(ub)
    x0=lb+(ub-lb)*np.random.default_rng(seed).random(len(lb))
    ne=0
    def _f(x): nonlocal ne; ne+=1; return f(reflect(np.asarray(x),lb,ub))
    minimize(_f,x0,method="L-BFGS-B",
             bounds=list(zip(lb,ub)),options=dict(maxfun=int(budget),disp=False))
    return _f(x0), min(ne,budget)

def run_jso(f, lb, ub, budget, seed, kw):
    import pyade.jso as jso
    params=jso.get_default_params(dim=len(lb))
    params.update(max_evals=int(budget),seed=int(seed),
                  bounds=np.stack([lb,ub],1))
    _f=lambda x: f(reflect(np.asarray(x),lb,ub))
    _, fit=jso.apply(func=_f, bounds=params["bounds"], opts=params)
    return float(fit), int(params.get("nfe",budget))

def run_lshade_pyade(f, lb, ub, budget, seed, kw):
    import pyade.lshade as ls
    params=ls.get_default_params(dim=len(lb))
    params.update(max_evals=int(budget),seed=int(seed),
                  bounds=np.stack([lb,ub],1))
    _f=lambda x: f(reflect(np.asarray(x),lb,ub))
    _, fit=ls.apply(func=_f, bounds=params["bounds"], opts=params)
    return float(fit), int(params.get("nfe",budget))

# ---- MEALPY meta-heuristics helper ----------------------------------
def _mealpy(algo_cls, f, lb, ub, budget, seed):
    from mealpy import _Problem as MP, FloatVar
    pop=50; epoch=max(1,budget//pop)
    class P(MP):
        def __init__(self): super().__init__(bounds=FloatVar(lb,ub),minmax="min")
        def obj_func(self, sol): return float(f(reflect(np.asarray(sol),lb,ub)))
    algo=algo_cls(epoch=epoch,pop_size=pop,seed=seed)
    _,fit=algo.solve(P(),verbose=False)
    return float(fit), epoch*pop

def run_gwo(f,lb,ub,budget,seed,kw):
    from mealpy.swarm_based import GWO
    return _mealpy(GWO.OriginalGWO,f,lb,ub,budget,seed)
def run_pso(f,lb,ub,budget,seed,kw):
    from mealpy.swarm_based import PSO
    return _mealpy(PSO.OriginalPSO,f,lb,ub,budget,seed)
def run_woa(f,lb,ub,budget,seed,kw):
    from mealpy.swarm_based import WOA
    return _mealpy(WOA.OriginalWOA,f,lb,ub,budget,seed)
def run_ssa(f,lb,ub,budget,seed,kw):
    from mealpy.swarm_based import SSA
    return _mealpy(SSA.OriginalSSA,f,lb,ub,budget,seed)
def run_shade_mealpy(f,lb,ub,budget,seed,kw):
    from mealpy.evolutionary_based import SHADE
    return _mealpy(SHADE.OriginalSHADE,f,lb,ub,budget,seed)
def run_ga(f,lb,ub,budget,seed,kw):
    from mealpy.evolutionary_based import GA
    return _mealpy(GA.BaseGA,f,lb,ub,budget,seed)

# mapping
ADAPTERS: Dict[str,Callable]=dict(
    SLO_HBYRID=run_slo_hbyrid, CMAES=run_cmaes, SciPyDE=run_scipyde,
    LBFGSB=run_lbfgsb, jSO=run_jso, L_SHADE=run_lshade_pyade,
    GWO=run_gwo, PSO=run_pso, WOA=run_woa, SSA=run_ssa,
    SHADE=run_shade_mealpy, GA=run_ga)

# ═══════════════════════════════════════════════════════════════════════
# 4.  Runner
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--algs", nargs="+", required=True)
    ap.add_argument("--budget-mult", type=int, default=5000)
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--outdir", default="eng_results")
    args=ap.parse_args()

    out=pathlib.Path(args.outdir); out.mkdir(exist_ok=True)
    log=out/"raw_results.csv"
    with log.open("w",newline="") as fh:
        csv.writer(fh).writerow(
            ["alg","prob","dim","run","f_best","err","nfe","time_sec"])

    for prob in PROBLEMS:
        budget=args.budget_mult*prob.dim
        for r in range(args.runs):
            seed=args.seed0+r
            for alg in args.algs:
                if alg not in ADAPTERS:
                    print(f"[skip] unknown {alg}", file=sys.stderr); continue
                try:
                    fn=ADAPTERS[alg]
                except Exception as e:
                    print(f"[skip] load {alg}: {e}", file=sys.stderr); continue
                t0=time.time()
                try:
                    f_best,nfe=fn(prob.f, prob.lower, prob.upper,
                                   budget, seed, kw={})
                except Exception as e:
                    print(f"[{alg}] err: {e}", file=sys.stderr)
                    f_best, nfe = float("inf"), 0
                dt=time.time()-t0
                fopt=prob.fopt if prob.fopt is not None else 0.0
                err=abs(f_best - fopt)
                with log.open("a",newline="") as fh:
                    csv.writer(fh).writerow(
                        [alg, prob.name, prob.dim, r,
                         f_best, err, nfe, dt])
                print(f"{alg:10s}|{prob.name:18s}|run {r:02d}|"
                      f"f_best={f_best:.4g} err={err:.2e} nfe={nfe:6d}")

if __name__=="__main__":
    main()
