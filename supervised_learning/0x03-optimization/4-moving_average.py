#!/usr/bin/env python3
"""
Moving average
"""

import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set:
    Args:
        - data: is the list of data to calculate the moving average of
        - beta: is the weight used for the moving average
        Your moving average calculation should use bias correction
    Return:
        A list containing the moving averages of data
    """

    beta
    # F(t) = β * F(t - 1) + (1 - β) * a(t)
    # a = real data
    # F = forcasted data
    # bias_correction = 1 - (β ** (i + 1))

    new_list = []
    moving_avg = 0
    for i in range(len(data)):
        moving_avg = ((moving_avg * beta) + ((1 - beta) * data[i]))
        bias_correction = 1 - (beta ** (i + 1))
        new_list.append(moving_avg / bias_correction)
    return new_list
