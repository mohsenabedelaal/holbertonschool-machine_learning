#!/usr/bin/env python3

"""Builds a transition layer as described in
 Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    Densely Connected Convolutional Networks
X is the output from the previous layer
nb_filters is an integer representing the number of filters in X
compression is the compression factor for the transition layer
Your code should implement compression as used in DenseNet-C
All weights should use he normal initialization
All convolutions should be preceded by Batch Normalization and
 a rectified linear activation (ReLU), respectively
Returns: The output of the transition layer and the number of
 filters within the output, respectively
    """
    initializer = K.initializers.he_normal(seed=None)

    layer = K.layers.BatchNormalization()(X)

    layer = K.layers.Activation('relu')(layer)

    nb_filters *= compression
    nb_filters = int(nb_filters)

    layer = K.layers.Conv2D(filters=nb_filters,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=initializer,
                            )(layer)

    layer = K.layers.AveragePooling2D(pool_size=(2, 2),
                                      strides=(2, 2),
                                      padding='same')(layer)

    return layer, nb_filters
