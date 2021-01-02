#ython3
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
        # print(ax.shape)
        self.__A = 1.0 / (1.0 + np.exp(-ax))
        # print(self.__A.shape)
        return self.__A
