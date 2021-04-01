#!/usr/bin/env python3
"""Calculate pdf of a Gaussian distribution"""


import numpy as np


def pdf(X, m, S):
    """Calculate pdf of a Gaussian distribution"""
    if ((type(X) != type(m) or type(m) != type(S) or type(X) is not np.ndarray
         or X.ndim != 2 or m.ndim != 1 or S.ndim != 2)):
        return None
    d = X.shape[1]
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None
    try:
        deter = np.linalg.det(S)
        if deter == 0:
            return None
        xmean = (X - m).T
        numer = (xmean * np.matmul(np.linalg.inv(S), xmean)).sum(axis=0)
        numer = np.exp(numer / -2)
        denom = np.sqrt(np.power(2 * np.pi, d) * deter)
        result = (numer / denom)
        result = np.maximum(result, 1e-300)
        return result
    except Exception:
        return None
