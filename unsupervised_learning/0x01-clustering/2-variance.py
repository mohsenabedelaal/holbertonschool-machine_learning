#!/usr/bin/env python3
"""Calculate total intracluster variance of a data set"""


import numpy as np


def variance(X, C):
    """Calculate total intracluster variance of a data set"""
    if ((type(X) is not np.ndarray or X.ndim != 2 or
         type(C) is not np.ndarray or C.ndim != 2)):
        return None
    try:
        return (np.square(np.apply_along_axis(np.subtract, 1, X, C))
                .sum(axis=2).min(axis=1).sum())
    except Exception:
        return None
