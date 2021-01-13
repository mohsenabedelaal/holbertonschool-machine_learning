#!/usr/bin/env python3
""" Creates a deep neural network. """
import numpy as np


class DeepNeuralNetwork:
    """ Deep neural network class. """

    def __init__(self, nx, layers):
        """ Initializer for the deep neural network. """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or not layers:
            raise TypeError('layers must be a list of positive integers')
        self.__L = 0
        self.__cache = {}
        self.__weights = {}
        rand = np.random.randn
        for idx, neurons in enumerate(layers):
            if type(neurons) != int or neurons <= 0:
                raise TypeError('layers must be a list of positive integers')
            if idx == 0:
                self.__weights['W1'] = rand(neurons, nx) * np.sqrt(2 / nx)
            else:
                p = layers[idx - 1]
                r = rand(neurons, p)
                self.__weights["W{}".format(idx + 1)] = r * np.sqrt(2 / p)
            self.__L += 1
            self.__weights["b{}".format(idx + 1)] = np.zeros((neurons, 1))

    @property
    def L(self):
        """ Getter for L (Number of layers). """
        return self.__L

    @property
    def cache(self):
        """ Getter for cache. """
        return self.__cache

    @property
    def weights(self):
        """ Getter for weights. """
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation of the network. """
        self.__cache['A0'] = X
        out = np.matmul(self.weights['W1'], X) + self.weights['b1']
        A = 1 / (1 + np.exp(-out))
        self.__cache['A1'] = A
        for i in range(1, self.L):
            w = self.weights['W{}'.format(i + 1)]
            b = self.weights['b{}'.format(i + 1)]
            out = np.matmul(w, A) + b
            A = 1 / (1 + np.exp(-out))
            self.__cache['A{}'.format(i + 1)] = A
        return (A, self.cache)

    def cost(self, Y, A):
        """ Calculates the cost of the network. """
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        c = np.sum(loss[0]) / Y.shape[1]
        return c
