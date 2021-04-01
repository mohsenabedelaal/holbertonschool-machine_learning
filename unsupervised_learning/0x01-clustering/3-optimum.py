#!/usr/bin/env python3
"""Test for optimum number of k clusters"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test for optimum number of k clusters"""
    if type(X) is not np.ndarray:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if ((X.ndim != 2 or type(kmin) is not int
         or kmin < 1 or type(iterations) is not int or iterations < 1
         or type(kmax) is not int or kmax <= kmin)):
        return None, None
    results = [kmeans(X, kmin, iterations)]
    firstvar = variance(X, results[0][0])
    d_vars = [0]
    idx = 0
    kmin += 1
    while kmin <= kmax:
        centroids, assigns = kmeans(X, kmin, iterations)
        vari = variance(X, centroids)
        results.append((centroids, assigns))
        d_vars.append(firstvar - vari)
        idx += 1
        kmin += 1
    return results, d_vars
