#!/usr/bin/env python3
""" Calcuoate normalization constants for matrix"""


import numpy as np


def normalization_constants(X):
    """ Calculates normalization constants for matrix """
    return X.mean(axis=0), X.std(axis=0)
