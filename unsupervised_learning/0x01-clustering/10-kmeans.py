#!/usr/bin/env python3
"""Do kmeans using sklearn"""


import sklearn.cluster


def kmeans(X, k):
    """Do kmeans using sklearn"""
    kmeans = sklearn.cluster.KMeans(k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
