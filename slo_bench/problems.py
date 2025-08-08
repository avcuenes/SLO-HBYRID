# problems.py -----------------------------------------------------------------
"""
Five constrained engineering-design test functions
--------------------------------------------------
Implemented from the exact formulations supplied by the user.
Static penalty method:
    F_penalised = f(x) + α · Σ max(0, g_i(x))^β    with β = 2
You may change ALPHA below if you prefer another weight.
"""

from __future__ import annotations
import numpy as np, math
from dataclasses import dataclass
from typing import Callable

ALPHA = 1.0e7    # global penalty weight  (you can tune per problem if desired)
BETA  = 2        # power in Σ violation^β  (matches your code)

# ---------------- utility ----------------------------------------------------
def _violate(v: float) -> float:               # same helper as in your classes
    return max(0.0, v)

# ---------------- wrapper ----------------------------------------------------
@dataclass
class Problem:
    name:   str
    dim:    int
    lower:  np.ndarray
    upper:  np.ndarray
    fopt:   float
    _raw_f: Callable[[np.ndarray], float]      # original f(x)
    _g:     Callable[[np.ndarray], np.ndarray] # vector g(x)

    def f(self, x: np.ndarray) -> float:
        """Penalised objective (to be minimised)."""
        x = np.asarray(x, float)
        raw = self._raw_f(x)
        viol = np.sum(np.power(np.maximum(self._g(x), 0.0), BETA))
        return raw + ALPHA * viol

# ============================================================================
# 1. Pressure-vessel  (4-D)  --------------------------------------------------
# ============================================================================
def _pv_raw(x):
    return (0.6224*x[0]*x[2]*x[3] +
            1.7781*x[1]*x[2]**2    +
            3.1661*x[0]**2*x[3]    +
            19.84*x[0]**2*x[2])

def _pv_g(x):
    g1 = -x[0] + 0.0193*x[2]
    g2 = -x[1] + 0.00954*x[2]
    g3 = -math.pi*x[2]**2*x[3] - 4/3*math.pi*x[2]**3 + 750*1728
    g4 =  x[3] - 240
    return np.array([g1, g2, g3, g4], float)

pressure_vessel = Problem(
    name="pressure_vessel",
    dim=4,
    lower=np.array([0.0, 0.0, 10.0, 10.0]),
    upper=np.array([99.0, 99.0, 200.0, 200.0]),
    fopt=6059.714,          # literature optimum (feasible)
    _raw_f=_pv_raw, _g=_pv_g)

# ============================================================================
# 2. Tension / compression spring (3-D) --------------------------------------
# ============================================================================
def _spr_raw(x):
    return (x[2]+2)*x[1]*x[0]**2

def _spr_g(x):
    g1 = 1 -  (x[1]**3 * x[2])           / (71785 * x[0]**4)
    g2 = (4*x[1]**2 - x[0]*x[1])         / (12566*(x[1]*x[0]**3 - x[0]**4)) \
         + 1/(5108*x[0]**2) - 1
    g3 = 1 - 140.45*x[0] / (x[1]**2 * x[2])
    g4 = (x[1] + x[0]) / 1.5 - 1
    return np.array([g1,g2,g3,g4], float)

tension_spring = Problem(
    name="tension_compression_spring",
    dim=3,
    lower=np.array([0.05, 0.25, 11.0]),
    upper=np.array([2.00, 1.30, 12.0]),
    fopt=2.6254,
    _raw_f=_spr_raw, _g=_spr_g)

# ============================================================================
# 3. Welded-beam (4-D) -------------------------------------------------------
# ============================================================================
P, L, E, G  = 6000.0, 14.0, 30e6, 12e6
tau_max, sig_max, delta_max = 13600.0, 30000.0, 0.25

def _wb_raw(x):
    return 1.10471*x[0]**2*x[1] + 0.04811*x[2]*x[3]*(14. + x[1])

def _wb_tau(x):
    tau1 = P / (math.sqrt(2)*x[0]*x[1])
    M    = P * (L + x[1]/2)
    R    = math.sqrt(x[1]**2/4 + ((x[0]+x[2])/2)**2)
    J    = 2*((x[0]*x[1]*math.sqrt(2))*(x[1]**2/12 + ((x[0]+x[2])/2)**2))
    tau2 = M*R / J
    return math.sqrt(tau1**2 + 2*tau1*tau2*x[1]/(2*R) + tau2**2)

def _wb_delta(x):
    return 4*P*L**3/(E*x[2]**3*x[3])

def _wb_sigma(x):
    return 6*P*L/(x[3]*x[2]**2)

def _wb_Pc(x):
    return (4.013*E*math.sqrt(x[2]**2*x[3]**6/36)/L**2) * \
           (1 - x[2]/(2*L)*math.sqrt(E/(4*G)))

def _wb_g(x):
    g1 = _wb_tau(x)   - tau_max
    g2 = _wb_sigma(x) - sig_max
    g3 = x[0] - x[3]
    g4 = 0.10471*x[0]**2 + 0.04811*x[2]*x[3]*(14.+x[1]) - 5
    g5 = 0.125 - x[0]
    g6 = _wb_delta(x) - delta_max
    g7 = P - _wb_Pc(x)
    return np.array([g1,g2,g3,g4,g5,g6,g7], float)

welded_beam = Problem(
    name="welded_beam",
    dim=4,
    lower=np.array([0.1, 0.1, 0.1, 0.1]),
    upper=np.array([2.0, 3.5, 10.0, 2.0]),
    fopt=1.724852,
    _raw_f=_wb_raw, _g=_wb_g)

# ============================================================================
# 4. Gear-box (5-D)  ---------------------------------------------------------
# Formulation exactly as in your `gearbox.run`
i, rho, n, sigma_y = 4, 8, 6, 294.3
y_, b2_, Kv, Kw, N1, Pwr = 0.102, 0.193, 0.389, 0.8, 1500, 7.5

def _gb_raw(x):
    b,d1,d2,Z1,m = x
    Dr = m*(i*Z1 - 2.5)
    lw = 2.5*m
    Di = Dr - 2*lw
    bw = 3.5*m
    d0 = d2 + 25
    dp = 0.25*(Di - d0)
    return (math.pi/4)*(rho/1000)*(b*m**2*Z1**2*(i**2+1) -
            (Di**2-d0**2)*(1-bw) - n*dp**2*bw - (d1+d2)*b)

def _gb_g(x):
    b,d1,d2,Z1,m = x
    Dr = m*(i*Z1 - 2.5)
    lw = 2.5*m
    Di = Dr - 2*lw
    dp = 0.25*(Di - (d2+25))
    D1 = m*Z1
    D2 = i*m*Z1
    Z2 = Z1*D2/D1
    v  = math.pi*D1*N1/60000
    b1 = 102*Pwr/v
    b3 = 4.97e6*Pwr/(N1*2)
    Fs = math.pi*Kv*Kw*sigma_y*b*m*y_
    Fp = 2*Kv*Kw*D1*b*Z2/(Z1+Z2)
    g1 = -Fs + b1
    g2 = -(Fs/Fp) + b2_
    g3 = -(d1**3) + b3
    # g4 / g5 from original formulation omitted in user code
    return np.array([g1,g2,g3], float)

gear_box = Problem(
    name="gear_box",
    dim=5,
    lower=np.array([20, 10, 30, 18, 2.75]),
    upper=np.array([32, 30, 40, 25, 4.00]),
    fopt=None,
    _raw_f=_gb_raw, _g=_gb_g)

# ============================================================================
# Expose list used by benchmark harness
# ============================================================================
PROBLEMS = [pressure_vessel,
            tension_spring,
            welded_beam,
            gear_box]
