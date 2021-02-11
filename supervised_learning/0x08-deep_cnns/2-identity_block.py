#!/usr/bin/env python3

"""Builds an identity block as described in
     Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
     Deep Residual Learning for Image Recognition (2015)
        - A_prev is the output from the previous layer
        - filters is a tuple or list containing F11, F3, F12, respectively:
            - F11 is the number of filters in the first 1x1 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F12 is the number of filters in the second 1x1 convolution
        - All convolutions inside the block should be followed by batch
         normalization along the channels axis and a rectified linear
          activation (ReLU), respectively.
        - All weights should use he normal initialization
        Returns: the activated output of the identity block
    """
    initializer = K.initializers.he_normal(seed=None)

    F11 = K.layers.Conv2D(filters=filters[0],
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=initializer,
                          )(A_prev)

    b_norm = K.layers.BatchNormalization(axis=3)(F11)

    relu = K.layers.Activation('relu')(b_norm)

    F3 = K.layers.Conv2D(filters=filters[1],
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=initializer,
                         )(relu)

    b_norm = K.layers.BatchNormalization(axis=3)(F3)

    relu = K.layers.Activation('relu')(b_norm)

    F12 = K.layers.Conv2D(filters=filters[2],
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=initializer,
                          )(relu)

    b_norm = K.layers.BatchNormalization(axis=3)(F12)

    add = K.layers.Add()([b_norm, A_prev])

    relu = K.layers.Activation('relu')(add)

    return relu
