#!/usr/bin/env python3
"""Initialize cluster centroids for kmeans"""


import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for kmeans"""
    if type(k) is not int or k <= 0:
        return None
    try:
        return np.random.uniform(np.amin(X, axis=0),
                                 np.amax(X, axis=0), (k, X.shape[1]))
    except Exception:
        return None
