#!/usr/bin/env python3
""" Module to Create a neuron
"""
import numpy as np


class Neuron:
    """ Class that models a Neuron"""
    def __init__(self, nx):
        """ Class constructor for Neuron"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for weights"""
        return self.__W

    @property
    def b(self):
        """ Getter for bias"""
        return self.__b

    @property
    def A(self):
        """ Getter for activated output"""
        return self.__A
