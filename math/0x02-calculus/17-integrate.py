#!/usr/bin/env python3
"""
17-integrate.py
"""


def poly_integral(poly, C=0):
    """
    Module integral
    """
    if type(poly) != list or len(poly) == 0 or type(C) != int:
        return None
    result = [C]
    if len(poly) == 0:
        return result
    for i in range(0, len(poly)):
        add = poly[i] / (i+1)
        if add.is_integer():
            add = int(add)
        result.append(add)
    return result
