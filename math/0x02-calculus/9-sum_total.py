#!/usr/bin/env python3
"""summation i squared"""


def summation_i_squared(n):
    """Module for sum"""
    if type(n) != int or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
