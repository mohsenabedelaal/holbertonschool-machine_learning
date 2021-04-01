#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize variables for a Gaussian Mixture Model"""
    if type(k) is not int or k < 1 or type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    pi = np.full((k,), 1 / k)
    m = kmeans(X, k)[0]
    S = np.full((k, X.shape[1], X.shape[1]), np.identity(X.shape[1]))
    return pi, m, S
