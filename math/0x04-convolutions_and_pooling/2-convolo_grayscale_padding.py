#!/usr/bin/env python3

"""Performs a convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images
     with custom padding
        - images is a numpy.ndarray with shape (m, h, w)
         containing multiple grayscale images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
        - kernel is a numpy.ndarray with shape (kh, kw)
         containing the kernel for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
        - padding is a tuple of (ph, pw)
            - ph is the padding for the height of the image
            - pw is the padding for the width of the image
        You are only allowed to use two for loops; any
         other loops of any kind are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    ph = padding[0]
    pw = padding[1]

    pad_size = ((0, 0), (ph, ph), (pw, pw))
    pad_images = np.pad(images, pad_width=pad_size, mode='constant',
                        constant_values=0)

    output_h = int(pad_images.shape[1] - kh + 1)
    output_w = int(pad_images.shape[2] - kw + 1)
    output = np.zeros((m, output_h, output_w))

    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * pad_images[:,
                                                   y: y + kh,
                                                   x: x + kw])\
                                                   .sum(axis=(1, 2))

    return output
