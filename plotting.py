"""
plotting.py
~~~~~~~~~~~
Helper functions for visualisation.
"""
import numpy as np
import matplotlib.pyplot as plt
from metrics import auto_xlim


def plot_density(x: np.ndarray,
                 f: np.ndarray,
                 *,
                 ax=None,
                 label: str = '',
                 adaptive: bool = False,
                 thresh: float = 0.01,
                 **plot_kwargs):
    """Draw a single density curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, f, label=label, **plot_kwargs)
    if adaptive:
        ax.set_xlim(*auto_xlim(x, f, thresh))
    ax.set_xlabel("x"); ax.set_ylabel("density"); ax.grid(True)
    if label:
        ax.legend()


def compare_with_limit(x: np.ndarray,
                       f: np.ndarray,
                       pdf_limit,
                       *,
                       ax=None,
                       lbl_fft: str = 'FFT',
                       lbl_lim: str = 'Limit',
                       **plot_kwargs):
    """Overlay FFT density with an analytic limit density."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, f, label=lbl_fft, **plot_kwargs)
    ax.plot(x, pdf_limit(x), 'r--', label=lbl_lim)
    ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.grid(True); ax.legend()
