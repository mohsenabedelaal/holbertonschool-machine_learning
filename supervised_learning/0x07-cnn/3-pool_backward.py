#!/usr/bin/env python3

"""Performs back propagation over a pooling layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network
        - dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
         containing the partial derivatives with respect to the output
          of the pooling layer
            - m is the number of examples
            - h_new is the height of the output
            - w_new is the width of the output
            - c is the number of channels
        - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
         containing the output of the previous layer
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
        - kernel_shape is a tuple of (kh, kw) containing the
         size of the kernel for the pooling
            - kh is the kernel height
            - kw is the kernel width
        - stride is a tuple of (sh, sw) containing the strides for the pooling
            - sh is the stride for the height
            - sw is the stride for the width
        - mode is a string containing either max or avg, indicating
         whether to perform maximum or average pooling, respectively
        - you may import numpy as np
        Returns: the partial derivatives with respect to the previous
         layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)

    for z in range(m):
        for y in range(h_new):
            for x in range(w_new):
                for k in range(c_new):
                    pool = A_prev[z, y * sh:(kh+y*sh), x * sw:(kw+x*sw), k]
                    dA_aux = dA[z, y, x, k]
                    if mode == 'avg':
                        avg = dA_aux / kh / kw
                        o_mask = np.ones(kernel_shape)
                        dA_prev[z, y * sh:(kh + y * sh),
                                x * sw:(kw+x*sw), k] += o_mask * avg
                    if mode == 'max':
                        z_mask = np.zeros(kernel_shape)
                        _max = np.amax(pool)
                        np.place(z_mask, pool == _max, 1)
                        dA_prev[z, y * sh:(kh + y * sh),
                                x * sw:(kw+x*sw), k] += z_mask * dA_aux

    return dA_prev
