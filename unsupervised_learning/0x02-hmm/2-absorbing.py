#!/usr/bin/env python3
"""Check if a transition matrix is absorbing"""


import numpy as np


def absorbing(P):
    """Check if a transition matrix is absorbing"""
    if ((type(P) is not np.ndarray or P.ndim != 2 or
         P.shape[0] != P.shape[1] or np.any(P < 0)
         or not np.all(np.isclose(P.sum(axis=1), 1)))):
        return None
    P = P.copy()
    absorbers = np.ndarray(P.shape[0])
    while True:
        prev = absorbers.copy()
        absorbers = np.any(P == 1, axis=0)
        if absorbers.all():
            return True
        if np.all(absorbers == prev):
            return False
        absorbed = np.any(P[:, absorbers], axis=1)
        P[absorbed, absorbed] = 1
