#!/usr/bin/env python3
"""Contains the one_hot function"""

import tensorflow.keras as keras


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix
    :param labels:
    :param classes:
    :return: the one-hot matrix
    """
    return keras.utils.to_categorical(
        labels, classes)
