#!/usr/bin/env python3
"""
 Shuffle Data
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way:
    Args:
        - X is the first numpy.ndarray of shape (m, nx) to shuffle
            - m is the number of data points
            - nx is the number of features in X
        - Y is the second numpy.ndarray of shape (m, ny) to shuffle
            - m is the same number of data points as in X
            - ny is the number of features in Y
    Return:
        The shuffled X and Y matrices
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)

    return X[shuffle], Y[shuffle]
