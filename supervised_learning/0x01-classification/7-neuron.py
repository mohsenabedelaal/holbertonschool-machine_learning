#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification.
"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """ Neuron class. """
    def __init__(self, nx):
        """ Initializer for the neuron. """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for W. """
        return self.__W

    @property
    def b(self):
        """ Getter for b. """
        return self.__b

    @property
    def A(self):
        """ Getter for A. """
        return self.__A

    def forward_prop(self, X):
        """ Forward propagation algorithm using sigmoid function. """
        out = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-out))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the neuron. """
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        c = np.sum(loss[0]) / Y.shape[1]
        return c

    def evaluate(self, X, Y):
        """ Evaluates the output of the neuron. """
        A = self.forward_prop(X)
        c = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return (A, c)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates the gradient descent. """
        self.__W = self.__W - alpha * np.matmul(A - Y, X.T) / A.shape[1]
        self.__b = self.__b - alpha * np.sum(A - Y) / A.shape[1]

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the neuron. """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose or step:
            if type(step) != int:
                raise TypeError('step must be an integer')
            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        iteration = 0
        x_data = np.arange(0, iterations, step)
        y_data = []
        for i in range(iterations):
            A, c = self.evaluate(X, Y)
            if iteration % step == 0:
                if graph:
                    y_data.append(c)
                if verbose:
                    print('Cost after {} iterations: {}'.format(iteration, c))
            iteration += 1
            self.gradient_descent(X, Y, self.A, alpha)
        A, c = self.evaluate(X, Y)
        if verbose:
            print('Cost after {} iterations: {}'.format(iteration, c))
        if graph:
            y_data[-1] = c
            plt.plot(x_data, y_data, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return (A, c)
