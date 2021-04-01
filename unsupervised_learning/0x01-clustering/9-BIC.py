#!/usr/bin/env python3
"""Find best number of clusters using Bayesian Information Criterion."""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find best number of clusters using Bayesian Information Criterion."""
    if ((type(X) is not np.ndarray or X.ndim != 2 or type(kmin) is not int
         or kmin < 1 or type(verbose) is not bool or type(tol) is not float
         or tol < 0 or type(iterations) is not int or iterations < 1)):
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if ((type(kmax) is not int or kmax <= kmin)):
        return None, None, None, None
    best = None
    bics = []
    loglikes = []
    for k in range(kmin, kmax + 1):
        pi, m, S, _, loglike = expectation_maximization(X, k,
                                                        iterations=iterations,
                                                        tol=tol,
                                                        verbose=verbose)
        bic = (6 * k - 1) * np.log(X.shape[0]) - 2 * loglike
        if best is None or bics[best - kmin] > bic:
            best = k
            bestres = (pi, m, S)
        bics.append(bic)
        loglikes.append(loglike)
    return best, bestres, np.asarray(loglikes), np.asarray(bics)
