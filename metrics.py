"""
metrics.py
~~~~~~~~~~
Density support trimming & sample-vs-pdf quality metrics.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import kstest, wasserstein_distance


def auto_xlim(x: np.ndarray,
              f: np.ndarray,
              thresh: float = 0.01) -> tuple[float, float]:
    """
    Return (xmin, xmax) where f(x) ≥ thresh · max(f).
    """
    f = np.asarray(f, dtype=float)
    m = f.max()
    mask = f >= m * thresh
    return (x[mask][0], x[mask][-1]) if mask.any() else (x[0], x[-1])


# ---------------------------------------------------------------------
# quality metrics
# ---------------------------------------------------------------------
def ks_w1(samples: np.ndarray,
          x_grid: np.ndarray,
          pdf_grid: np.ndarray) -> tuple[float, float, float]:
    """
    Kolmogorov–Smirnov statistic, KS-p, and Wasserstein-1 distance.

    cdf_grid is built by trapezoidal cum-sum (works as step function
    because grid is dense).
    """
    samples = np.asarray(samples, dtype=float).ravel()
    dx = x_grid[1] - x_grid[0]
    cdf_grid = np.cumsum(pdf_grid) * dx
    interp_cdf = lambda z: np.interp(z, x_grid, cdf_grid, left=0.0, right=1.0)
    ks_stat, p_val = kstest(samples, interp_cdf)
    w1 = wasserstein_distance(samples,
                              np.random.choice(x_grid, size=len(samples),
                                               p=pdf_grid * dx))
    return ks_stat, p_val, w1


def avg_loglik(samples: np.ndarray,
               x_grid: np.ndarray,
               pdf_grid: np.ndarray,
               eps: float = 1e-12) -> float:
    """Average log-likelihood under given pdf grid."""
    pdf_interp = np.maximum(np.interp(samples, x_grid, pdf_grid),
                            eps)
    return np.log(pdf_interp).mean()
