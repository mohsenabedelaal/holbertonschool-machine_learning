#!/usr/bin/env python3

"""Builds an identity block as described in
     Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds an projection block as described in
     Deep Residual Learning for Image Recognition (2015)
        - A_prev is the output from the previous layer
        - filters is a tuple or list containing F11, F3, F12, respectively:
            - F11 is the number of filters in the first 1x1 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F12 is the number of filters in the second 1x1 convolution
             as well as the 1x1 convolution in the shortcut connection
            - s is the stride of the first convolution in both the main
             path and the shortcut connection
        - All convolutions inside the block should be followed by
         batch normalization along the channels axis and a
          rectified linear activation (ReLU), respectively.
        - All weights should use he normal initialization
        Returns: the activated output of the projection block
    """
    initializer = K.initializers.he_normal(seed=None)

    F11, F3, F12 = filters

    conv = K.layers.Conv2D(filters=F11,
                           kernel_size=(1, 1),
                           strides=(s, s),
                           padding='same',
                           kernel_initializer=initializer,
                           )(A_prev)

    b_norm = K.layers.BatchNormalization(axis=3)(conv)

    relu = K.layers.Activation('relu')(b_norm)

    conv = K.layers.Conv2D(filters=F3,
                           kernel_size=(3, 3),
                           padding='same',
                           kernel_initializer=initializer,
                           )(relu)

    b_norm = K.layers.BatchNormalization(axis=3)(conv)

    relu = K.layers.Activation('relu')(b_norm)

    conv = K.layers.Conv2D(filters=F12,
                           kernel_size=(1, 1),
                           padding='same',
                           kernel_initializer=initializer,
                           )(relu)

    b_norm = K.layers.BatchNormalization(axis=3)(conv)

    # shortcut connection

    conv_2 = K.layers.Conv2D(filters=F12,
                             kernel_size=(1, 1),
                             strides=(s, s),
                             padding='same',
                             kernel_initializer=initializer,
                             )(A_prev)

    b_norm_2 = K.layers.BatchNormalization(axis=3)(conv_2)

    add = K.layers.Add()([b_norm, b_norm_2])

    relu = K.layers.Activation('relu')(add)

    return relu
