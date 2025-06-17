"""
windows.py
~~~~~~~~~~
Utility to fetch common window functions for frequency-domain tapering.
"""
import numpy as np


def get_window(name: str | None, N: int) -> np.ndarray:
    """
    Parameters
    ----------
    name : 'hann' | 'hanning' | 'blackman' | None
    N    : length

    Returns
    -------
    1-D numpy array of length N (dtype float64)
    """
    if name is None:
        return np.ones(N, dtype=float)

    name = name.lower()
    if name in {'hann', 'hanning'}:
        return np.hanning(N)
    elif name == 'blackman':
        return np.blackman(N)
    else:
        raise ValueError(f"unknown window '{name}'")
