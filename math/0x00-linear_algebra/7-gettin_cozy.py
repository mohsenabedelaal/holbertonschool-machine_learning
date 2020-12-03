#!/usr/bin/env python3
"""
module
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    function
    """
    mat3 = []

    if axis == 0:
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    for i in range(len(mat1a)):
        mat3.append(mat1a[i] + mat2a[i])
    return mat3
