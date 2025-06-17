"""
cf.py
~~~~~
Characteristic functions for Student-t Lévy increments.

Exports
-------
phi_student_unit(u, nu)
phi_levy(u, h, nu, mu=0., sigma=1.)
"""
import numpy as np
from scipy.special import kv, gamma


def phi_student_unit(u: np.ndarray, nu: float, eps: float = 1e-12) -> np.ndarray:
    """
    Numerically safe characteristic function of *unit-scale* Student-t_ν.

    Parameters
    ----------
    u   : array-like, frequency points
    nu  : degrees of freedom (ν > 0)
    eps : threshold below which we use the analytic limit 1

    Returns
    -------
    φ₁(u) as a numpy array of dtype complex128
    """
    u = np.asarray(u, dtype=float)
    a = np.sqrt(nu) * np.abs(u)

    out = np.ones_like(a)            # φ(0)=1

    mask = a > eps                   # regular zone
    a_big = a[mask]
    if a_big.size:
        num = (a_big ** (nu / 2)) * kv(nu / 2, a_big)
        den = 2 ** (nu / 2 - 1) * gamma(nu / 2)
        out[mask] = num / den

    return out


def phi_levy(u: np.ndarray,
             h: float,
             nu: float,
             mu: float = 0.,
             sigma: float = 1.) -> np.ndarray:
    """
    Characteristic function of the Student-Lévy increment X_h.

    φ_h(u) = exp(i μ h u) · [ φ₁(σ u) ]^h
    """
    return np.exp(1j * mu * h * u) * np.power(phi_student_unit(sigma * u, nu), h)
