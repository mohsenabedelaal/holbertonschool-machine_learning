#!/usr/bin/env python3
"""Initialize and forward prop for a bidirectional rnn cell"""


import numpy as np


class BidirectionalCell:
    """Bidirectional cell class"""
    def __init__(self, i, h, o):
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h * 2, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Forward prop the foreward direction"""
        concat = np.concatenate((h_prev, x_t), 1)
        return np.tanh(np.matmul(concat, self.Whf) + self.bhf)

    def backward(self, h_prev, x_t):
        """Forward prop the backward direction"""
        concat = np.concatenate((h_prev, x_t), 1)
        return np.tanh(np.matmul(concat, self.Whb) + self.bhb)

    def output(self, H):
        """Return output from cell."""
        out = np.exp(np.matmul(H, self.Wy) + self.by,)
        return out / out.sum(axis=2, keepdims=True)
