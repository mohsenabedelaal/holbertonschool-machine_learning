#!/usr/bin/env python3

"""Performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images
        - images is a numpy.ndarray with shape (m, h, w, c)
         containing multiple images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
            - c is the number of channels in the image
        - kernel is a numpy.ndarray with shape (kh, kw, c)
         containing the kernel for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
        - padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
            - if ‘same’, performs a same convolution
            - if ‘valid’, performs a valid convolution
            - if a tuple:
                - ph is the padding for the height of the image
                - pw is the padding for the width of the image
            - the image should be padded with 0’s
        - stride is a tuple of (sh, sw)
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image
        You are only allowed to use two for loops;
         any other loops of any kind are not allowed
          Hint: loop over i and j
        You are only allowed to use two for loops; any
         other loops of any kind are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    sh = stride[0]
    sw = stride[1]

    if type(padding) is tuple:
        ph = padding[0]
        pw = padding[1]
    elif padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    else:
        ph = 0
        pw = 0

    pad_size = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    pad_images = np.pad(images, pad_width=pad_size, mode='constant',
                        constant_values=0)

    output_h = int((pad_images.shape[1] - kh) / sh + 1)
    output_w = int((pad_images.shape[2] - kw) / sw + 1)
    output = np.zeros((m, output_h, output_w))

    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * pad_images[:,
                                                   y * sh: y * sh + kh,
                                                   x * sw: x * sw + kw,
                                                   :]).sum(axis=(1, 2, 3))

    return output
