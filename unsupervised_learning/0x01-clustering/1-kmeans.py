#!/usr/bin/env python3
"""Perform kmeans on some data"""


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


def kmeans(X, k, iterations=1000):
    """Perform kmeans on some data"""
    if type(iterations) is not int or iterations < 1:
        return None, None
    centroids = initialize(X, k)
    if centroids is None:
        return None, None
    assigns = None
    while iterations:
        iterations -= 1
        prev = centroids.copy()
        assigns = np.apply_along_axis(np.subtract, 1, X, centroids)
        assigns = np.argmin(np.square(assigns).sum(axis=2), axis=1)
        for cent in range(centroids.shape[0]):
            Xs = np.argwhere(assigns == cent)
            if Xs.shape[0] == 0:
                centroids[cent] = initialize(X, 1)
            else:
                centroids[cent] = np.mean(X[Xs], axis=0)
        if np.all(prev == centroids):
            return centroids, assigns
    assigns = np.apply_along_axis(np.subtract, 1, X, centroids)
    assigns = np.argmin(np.square(assigns).sum(axis=2), axis=1)
    return centroids, assigns
