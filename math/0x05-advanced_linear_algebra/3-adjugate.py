#!/usr/bin/env python3
""" Compute the adjugate of a simetric matrix"""


def adjugate(matrix):
    """ FUnction Adjugate"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([isinstance(row, list) for row in matrix]):
        raise TypeError("matrix must be a list of lists")
    if not all([len(row) == len(matrix) for row in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")
    cof = cofactor(matrix)
    return [[row[i] for row in cof] for i in range(len(cof[0]))]


def cofactor(matrix):
    """
    matrix is a list of lists whose cofactor matrix should be calculated
    Returns: the cofactor matrix of matrix
    """
    cofactor_m = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            cofactor_m[i][j] = (-1) ** (i + j) * cofactor_m[i][j]
    return cofactor_m


def minor(matrix):
    """
    matrix is a list of lists whose determinant should be calculated
    Returns: the minor of matrix
    """
    if len(matrix) is 1:
        return [[1]]
    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]
    minor_m = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[i])):
            minor_row.append(determinant(getMatrixMinor(matrix, i, j)))
        minor_m.append(minor_row)
    return minor_m


def getMatrixMinor(m, i, j):
    """selects the minor of a squared matrix"""
    return [row[:j] + row[j+1:] for row in (m[:i] + m[i+1:])]


def determinant(matrix):
    """
    matrix is a list of lists whose determinant should be calculated
    Returns: the determinant of matrix
    """
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) is 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    deter = 0
    for c in range(len(matrix)):
        deter += ((-1) ** c) * matrix[0][c] *\
                 determinant(getMatrixMinor(matrix, 0, c))
    return deter
