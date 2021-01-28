#!/usr/bin/env python3

"""Performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
        - images is a numpy.ndarray with shape (m, h, w)
         containing multiple grayscale images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
        - kernel is a numpy.ndarray with shape (kh, kw)
         containing the kernel for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
        - if necessary, the image should be padded with 0’s
        You are only allowed to use two for loops; any
         other loops of any kind are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    output = np.zeros((m, h, w))

    ph = max(int((kh - 1) / 2), int(kh / 2))
    pw = max(int((kw - 1) / 2), int(kw / 2))

    pad_size = ((0, 0), (ph, ph), (pw, pw))
    pad_images = np.pad(images, pad_width=pad_size, mode='constant',
                        constant_values=0)

    for x in range(w):
        for y in range(h):
            output[:, y, x] = (kernel * pad_images[:,
                                                   y: y + kh,
                                                   x: x + kw])\
                                                   .sum(axis=(1, 2))

    return output
