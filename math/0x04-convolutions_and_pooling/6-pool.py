#!/usr/bin/env python3

"""Performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
        - images is a numpy.ndarray with shape (m, h, w, c)
         containing multiple images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
            - c is the number of channels in the image
        - kernel_shape is a tuple of (kh, kw)
         containing the kernel shape for the pooling
            - kh is the height of the kernel
            - kw is the width of the kernel
        - stride is a tuple of (sh, sw)
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image
        - mode indicates the type of pooling
            - max indicates max pooling
            - avg indicates average pooling
        You are only allowed to use two for loops;
         any other loops of any kind are not allowed
        Returns: a numpy.ndarray containing the pooled images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    output_h = int(((h - kh) / sh) + 1)
    output_w = int(((w - kw) / sw) + 1)
    output = np.zeros((m, output_h, output_w, c))

    for x in range(output_w):
        for y in range(output_h):
            if mode == 'avg':
                output[:, y, x, :] = np.mean(images[:,
                                                    y * sh: y * sh + kh,
                                                    x * sw: x * sw + kw],
                                             axis=(1, 2))
            if mode == 'max':
                output[:, y, x, :] = np.max(images[:,
                                                   y * sh: y * sh + kh,
                                                   x * sw: x * sw + kw],
                                            axis=(1, 2))

    return output
