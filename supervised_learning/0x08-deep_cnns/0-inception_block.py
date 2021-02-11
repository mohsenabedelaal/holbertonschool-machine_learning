#!/usr/bin/env python3

"""Builds an inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
     Going Deeper with Convolutions (2014)
        - A_prev is the output from the previous layer
            - filters is a tuple or list containing
             F1, F3R, F3,F5R, F5, FPP, respectively:
            - F1 is the number of filters in the 1x1 convolution
            - F3R is the number of filters in the 1x1 convolution
             before the 3x3 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F5R is the number of filters in the 1x1 convolution
             before the 5x5 convolution
            - F5 is the number of filters in the 5x5 convolution
            - FPP is the number of filters in the 1x1 convolution
             after the max pooling
        - All convolutions inside the inception block should use
         a rectified linear activation (ReLU)
        Returns: the concatenated output of the inception block
    """
    initializer = K.initializers.he_normal(seed=None)

    F1 = K.layers.Conv2D(filters=filters[0],
                         kernel_size=(1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer=initializer,
                         )(A_prev)

    F3R = K.layers.Conv2D(filters=filters[1],
                          kernel_size=(1, 1),
                          padding='same',
                          activation='relu',
                          kernel_initializer=initializer,
                          )(A_prev)

    F3 = K.layers.Conv2D(filters=filters[2],
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         kernel_initializer=initializer,
                         )(F3R)

    F5R = K.layers.Conv2D(filters=filters[3],
                          kernel_size=(1, 1),
                          padding='same',
                          activation='relu',
                          kernel_initializer=initializer,
                          )(A_prev)

    F5 = K.layers.Conv2D(filters=filters[4],
                         kernel_size=(5, 5),
                         padding='same',
                         activation='relu',
                         kernel_initializer=initializer,
                         )(F5R)

    MP = K.layers.MaxPool2D(pool_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(A_prev)

    FPP = K.layers.Conv2D(filters=filters[5],
                          kernel_size=(1, 1),
                          padding='same',
                          activation='relu',
                          kernel_initializer=initializer,
                          )(MP)

    output = K.layers.concatenate([F1, F3, F5, FPP], axis=-1)
    return output
