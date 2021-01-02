#!/usr/bin/env python3
""" Module to Create a neuron
"""
import numpy as np


class Neuron:
    """Neuron Class"""
    def __init__(self, nx):
        """constructor"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """get function"""
        return self.__W

    @property
    def b(self):
        """get function"""
        return self.__b

    @property
    def A(self):
        """get function"""
        return self.__A

    def forward_prop(self, X):
        """forward_prop function"""
        ax = np.dot(self.__W, X) + self.__b
        self.__A = 1.0 / (1.0 + np.exp(-ax))
        return self.__A

    def cost(self, Y, A):
        """Calculate Cost Function"""
        summ = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = summ / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluate neuron"""
        A = np.ndarray((1, X.shape[1]))
        A[0] = self.forwrad_prop(X)
        return np.round(A).astype(int), self.cost(Y, A)
