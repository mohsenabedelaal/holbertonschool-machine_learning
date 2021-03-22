#!/usr/bin/env python3
"""Calculate a correlation matrix given a covariance matrix"""


import numpy as np


def correlation(C):
    """Calculate a correlation matrix given a covariance matrix"""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    corrs = np.ndarray(C.shape)
    for x in range(C.shape[0]):
        for y in range(C.shape[1]):
            corrs[x][y] = C[x][y] / np.sqrt(C[x][x]) / np.sqrt(C[y][y])
    return corrs
