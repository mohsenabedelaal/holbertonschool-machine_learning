#!/usr/bin/env python3
"""Perform expectation-maximization for GMM"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Perform expectation-maximization for GMM"""
    if ((type(X) is not np.ndarray or X.ndim != 2 or type(k) is not int
         or k < 1 or type(verbose) is not bool or type(tol) is not float
         or tol < 0 or type(iterations) is not int or iterations < 1)):
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    loglikeprev = 0
    i = 0
    while i < iterations:
        expects, loglike = expectation(X, pi, m, S)
        if verbose and not i % 10:
            print("Log Likelihood after {} iterations: {}"
                  .format(i, loglike.round(5)))
        if abs(loglike - loglikeprev) < tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}"
                      .format(i, loglike.round(5)))
            return pi, m, S, expects, loglike
        pi, m, S = maximization(X, expects)
        i += 1
        loglikeprev = loglike
    expects, loglike = expectation(X, pi, m, S)
    if verbose:
        print("Log Likelihood after {} iterations: {}"
              .format(i, loglike.round(5)))
    return pi, m, S, expects, loglike
