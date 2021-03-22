#!/usr/bin/env python3
"""Return means and covariance matrix of a multivariate data set"""


import numpy as np


def mean_cov(X):
    """Return means and covariance matrix of a multivariate data set"""
    if type(X) != np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    means = np.mean(X, axis=0, keepdims=True)
    covmat = np.ndarray((X.shape[1], X.shape[1]))
    for x in range(X.shape[1]):
        for y in range(X.shape[1]):
            covmat[x][y] = (((X[:, x] - means[:, x]) *
                             (X[:, y] - means[:, y])).sum() /
                            (X.shape[0] - 1))
    return means, covmat
