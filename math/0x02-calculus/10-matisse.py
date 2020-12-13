#!/usr/bin/env python3
"""derivative poly"""


def poly_derivative(poly):
    """Module for der"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    if len(poly) == 2:
        return [poly(1)]
    else:
        der = []
        for i in range(1, len(poly)):
            if isinstance(poly[i], (int, float)):
                result = i * poly[i]
                der.append(result)
            else:
                return None
        return der
