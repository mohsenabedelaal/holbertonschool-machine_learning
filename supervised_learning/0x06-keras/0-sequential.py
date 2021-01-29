#!/usr/bin/env python3
"""Contains the build_model function"""

import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library
    :param nx: number of input features to the network
    :param layers: list containing the number of nodes
        in each layer of the network
    :param activations: list containing the activation
        functions used for each layer of the network
    :param lambtha: L2 regularization parameter
    :param keep_prob: probability that a node will be kept for dropout
    :return: keras model
    """
    # model is a stack of layers
    network = keras.Sequential()

    # regularization scheme
    reg = keras.regularizers.L1L2(l2=lambtha)

    # first densely-connected layer
    network.add(keras.layers.Dense(units=layers[0],
                                   activation=activations[0],
                                   kernel_regularizer=reg,
                                   input_shape=(nx,),
                                   ))

    # subsequent densely-connected layers:
    for i in range(1, len(layers)):
        network.add(keras.layers.Dropout(1 - keep_prob))
        network.add(keras.layers.Dense(units=layers[i],
                                       activation=activations[i],
                                       kernel_regularizer=reg,
                                       ))

    return network
