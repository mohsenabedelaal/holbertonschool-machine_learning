#!/usr/bin/env python3
"""Builds the inception network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in
     Going Deeper with Convolutions (2014)
        - You can assume the input data will
         have shape (224, 224, 3)
        - All convolutions inside and outside the
         inception block should use a rectified linear activation (ReLU)
        - You may use inception_block = __import__('0-inception_block')
        .inception_block
        Returns: the keras model
    """
    initializer = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding='same',
                             activation='relu',
                             kernel_initializer=initializer,
                             )(X)

    pool_1 = K.layers.MaxPool2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(conv_1)

    conv_2_1 = K.layers.Conv2D(filters=64,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(pool_1)

    conv_2_2 = K.layers.Conv2D(filters=192,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(conv_2_1)

    pool_2 = K.layers.MaxPool2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(conv_2_2)

    # inception 3
    Y = inception_block(pool_2, [64, 96, 128, 16, 32, 32])
    Y = inception_block(Y, [128, 128, 192, 32, 96, 64])

    pool_3 = K.layers.MaxPool2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(Y)
    # inception 4
    Y = inception_block(pool_3, [192, 96, 208, 16, 48, 64])
    Y = inception_block(Y, [160, 112, 224, 24, 64, 64])
    Y = inception_block(Y, [128, 128, 256, 24, 64, 64])
    Y = inception_block(Y, [112, 144, 288, 32, 64, 64])
    Y = inception_block(Y, [256, 160, 320, 32, 128, 128])

    pool_4 = K.layers.MaxPool2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(Y)

    # inception 5
    Y = inception_block(pool_4, [256, 160, 320, 32, 128, 128])
    Y = inception_block(Y, [384, 192, 384, 48, 128, 128])

    pool_5 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                       padding='same')(Y)

    dropout = K.layers.Dropout(0.4)(pool_5)

    linear = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=initializer,
                            )(dropout)

    model = K.models.Model(inputs=X, outputs=linear)
    return model
