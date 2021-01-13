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
        self.L = 0
        self.cache = {}
        self.weights = {}
        rand = np.random.randn
        for idx, neurons in enumerate(layers):
            if type(neurons) != int or neurons <= 0:
                raise TypeError('layers must be a list of positive integers')
            if idx == 0:
                self.weights['W1'] = rand(neurons, nx) * np.sqrt(2 / nx)
            else:
                p = layers[idx - 1]
                r = rand(neurons, p)
                self.weights["W{}".format(idx + 1)] = r * np.sqrt(2 / p)
            self.L += 1
            self.weights["b{}".format(idx + 1)] = np.zeros((neurons, 1))
