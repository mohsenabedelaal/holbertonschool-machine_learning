#!/usr/bin/env python3

"""Builds an identity block as described in
     Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds an projection block as described in
     Deep Residual Learning for Image Recognition (2015)
        - X is the output from the previous layer
        - nb_filters is an integer representing the number
         of filters in X
        - growth_rate is the growth rate for the dense block
        - layers is the number of layers in the dense block
        - You should use the bottleneck layers used for DenseNet-B
        - All weights should use he normal initialization
        - All convolutions should be preceded by Batch Normalization
         and a rectified linear activation (ReLU), respectively
        Returns: The concatenated output of each layer within the
         Dense Block and the number of filters within the concatenated
          outputs, respectively
    """
    initializer = K.initializers.he_normal(seed=None)

    for _ in range(layers):
        layer = K.layers.BatchNormalization()(X)

        layer = K.layers.Activation('relu')(layer)

        layer = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=(1, 1),
                                padding='same',
                                kernel_initializer=initializer,
                                )(layer)

        layer = K.layers.BatchNormalization()(layer)

        layer = K.layers.Activation('relu')(layer)

        layer = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding='same',
                                kernel_initializer=initializer,
                                )(layer)

        X = K.layers.concatenate([X, layer])
        nb_filters += growth_rate

    return X, nb_filters
