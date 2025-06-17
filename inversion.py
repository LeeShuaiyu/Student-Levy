# inversion.py
# ~~~~~~~~~~~~
# Numerical inversion engines for Student-Lévy increments.

from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.stats import t

from cf import phi_levy
from windows import get_window


# ------------------------------------------------------------
# helper: smarter x_max for heavy-tail & tiny-h
# ------------------------------------------------------------
def _auto_xmax(h: float,
               nu: float,
               mu: float,
               sigma: float,
               p: float = 0.995,
               max_factor: float = 10.0) -> float:
    """
    Heuristic support radius:

    * ν ≤ 2  → approximate Cauchy p-quantile   tan((p-0.5)π)
    * ν  > 2 → Student-t p-quantile
    * capped by  max_factor · σ√h
    """
    sigma_h = sigma * np.sqrt(h)

    if nu <= 2:                         # heavy tail ~ Cauchy
        from numpy import tan, pi
        q = sigma_h * tan((p - 0.5) * np.pi)
    else:                               # light(er) tail
        q = sigma_h * t.ppf(p, df=nu)

    return mu * h + np.minimum(q, max_factor * sigma_h)


# ------------------------------------------------------------
# 1) FFT engine
# ------------------------------------------------------------
def density_fft(h: float,
                nu: float,
                *,
                mu: float = 0.,
                sigma: float = 1.,
                x_max_auto: bool = True,
                x_max: float | None = None,
                n_grid: int = 2 ** 15,
                window: str | None = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute density f_h(x) on an equally-spaced grid via IFFT.

    Returns
    -------
    x  : numpy array (len = n_grid)
    f  : density values, ∫f dx ≈ 1
    """
    if x_max_auto and x_max is None:
        x_max = _auto_xmax(h, nu, mu, sigma)

    dx = 2 * x_max / n_grid
    x  = np.linspace(-x_max + mu * h,  x_max + mu * h - dx, n_grid)

    dt = 2 * np.pi / (n_grid * dx)
    u  = np.linspace(-n_grid/2, n_grid/2 - 1, n_grid) * dt

    # characteristic function
    phi = phi_levy(u, h, nu, mu, sigma)

    # frequency-domain window & amplitude correction
    win = get_window(window, n_grid)          # ones if window=None
    phi *= win
    phi /= win[n_grid // 2]                   # ensure φ(0)=1, compensate only center bin

    # IFFT with correct scaling
    f = np.real(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(phi))))
    f *= n_grid * dt / (2 * np.pi)            # multiply back the 1/N factor
    f[f < 0] = 0.0                            # clip negatives
    return x, f


# ------------------------------------------------------------
# 2) Quadrature engine
# ------------------------------------------------------------
def density_quad(h: float,
                 nu: float,
                 x: np.ndarray,
                 *,
                 mu: float = 0.,
                 sigma: float = 1.,
                 U: float = 80.,
                 N_u: int = 4097,
                 window: str | None = None
                 ) -> np.ndarray:
    """
    Evaluate f_h(x) at arbitrary points via finite-interval integration.
    """
    u   = np.linspace(-U, U, N_u)
    phi = phi_levy(u, h, nu, mu, sigma)

    w   = get_window(window, N_u)
    phi *= w
    norm = np.mean(w)

    integrand = phi[:, None] * np.exp(-1j * np.outer(u, x))
    dens = np.real(np.trapz(integrand, u, axis=0)) / (2 * np.pi * norm)
    dens[dens < 0] = 0.0
    return dens
