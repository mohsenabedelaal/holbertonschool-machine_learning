#!/usr/bin/env python3
"""derivative poly"""


def poly_integral(poly, C=0):
    """Module for integral"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    if len(poly) == 2:
        return [poly(1)]
    else:
        integral = []
        for i in range(0, len(poly)):
            if isinstance(poly[i], (int, float)):
                if i == 0:
                    integral.append(0)
                if poly[i] % (i + 1) == 0:
                    result = int((1/(i+1)) * poly[i])
                else:
                    result = (1/(i+1)) * poly[i]
                integral.append(result)
            else:
                return None
        return integral
