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
            """ Evaluates the neuron’s predictions"""
            A = self.forward_prop(X)
            cost = self.cost(Y, A)
            A = np.where(A >= 0.5, 1, 0)
            return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """gradient descent"""
        dz = A - Y
        dw = (np.matmul(dz, X.T)) / Y.shape[1]
        db = (np.sum(dz)) / Y.shape[1]
        self.__W -= alpha*dw
        self.__b -= alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.5):
        """Train the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be a float")
        while iterations > 0:
            self.gradient_descent(X, Y, self.forward_prop(X), alpha)
            iterations -= 1
        return self.evaluate(X, Y)
