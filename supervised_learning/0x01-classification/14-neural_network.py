#!/usr/bin/env python3
""" Creates a neural network. """
import numpy as np


class NeuralNetwork:
    """ Neural network class. """
    def __init__(self, nx, nodes):
        """ Initializer for the neural network. """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) != int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Getter for W1. """
        return self.__W1

    @property
    def b1(self):
        """ Getter for b1. """
        return self.__b1

    @property
    def A1(self):
        """ Getter for A1. """
        return self.__A1

    @property
    def W2(self):
        """ Getter for W2. """
        return self.__W2

    @property
    def b2(self):
        """ Getter for b2. """
        return self.__b2

    @property
    def A2(self):
        """ Getter for A2. """
        return self.__A2

    def forward_prop(self, X):
        """ Forward propagation for the neural network. """
        self.__A1 = 1/(1 + np.exp(-(np.matmul(self.W1, X) + self.b1)))
        self.__A2 = 1/(1 + np.exp(-(np.matmul(self.W2, self.A1) + self.b2)))
        return (self.A1, self.A2)

    def cost(self, Y, A):
        """ Calculates the cost of the network. """
        m = Y.shape[1]
        c = np.sum(-Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)) / m
        return c

    def evaluate(self, X, Y):
        """ Evaluates the output of the network. """
        A_h, A = self.forward_prop(X)
        c = self.cost(Y, A)
        return (np.where(A >= 0.5, 1, 0), c)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates the gradient descent. """
        dA1 = np.matmul(self.W2.T, (A2 - Y))
        self.__W1 = self.W1 - alpha * np.matmul(dA1 * A1 * (1 - A1), X.T) /\
            A2.shape[1]
        self.__b1 = self.b1 - alpha * np.sum(dA1 * A1 * (1 - A1), axis=1,
                                             keepdims=True) / A2.shape[1]
        self.__W2 = self.W2 - alpha * np.matmul(A2 - Y, A1.T) / A2.shape[1]
        self.__b2 = self.b2 - alpha * np.sum(A2 - Y, axis=1, keepdims=True) /\
            A2.shape[1]

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the network. """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        return self.evaluate(X, Y)
