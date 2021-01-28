#!/usr/bin/env python3

"""Performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images
        - images is a numpy.ndarray with shape (m, h, w)
         containing multiple grayscale images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
        - kernel is a numpy.ndarray with shape (kh, kw)
         containing the kernel for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
        You are only allowed to use two for loops; any
         other loops of any kind are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    output_h = int(h - kh + 1)
    output_w = int(w - kw + 1)
    output_d = m

    output = np.zeros((output_d, output_h, output_w))

    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * images[:,
                                               y: y + kh,
                                               x: x + kw])\
                                               .sum(axis=(1, 2))

    return output
