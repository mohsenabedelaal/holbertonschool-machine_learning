#!/usr/bin/env python3

"""Builds an identity block as described in
     Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds an projection block as described in
     Deep Residual Learning for Image Recognition (2015)
        - You can assume the input data will have shape (224, 224, 3)
        - All convolutions inside and outside the blocks should be
         followed by batch normalization along the channels axis
          and a rectified linear activation (ReLU), respectively.
        - All weights should use he normal initialization
        You may use:
        - identity_block = __import__('2-identity_block').identity_block
        - projection_block = __import__('3-projection_block').projection_block
        Returns: the keras model
    """
    initializer = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    # conv1
    layer = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer,
                            )(X)

    layer = K.layers.BatchNormalization(axis=3)(layer)

    layer = K.layers.Activation('relu')(layer)

    # conv2_x
    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(layer)

    layer = projection_block(layer, [64, 64, 256], 1)
    for _ in range(2):
        layer = identity_block(layer, [64, 64, 256])

    # conv3_x
    layer = projection_block(layer, [128, 128, 512])
    for _ in range(3):
        layer = identity_block(layer, [128, 128, 512])

    # conv4_x
    layer = projection_block(layer, [256, 256, 1024])
    for _ in range(5):
        layer = identity_block(layer, [256, 256, 1024])

    # conv5_x
    layer = projection_block(layer, [512, 512, 2048])
    for _ in range(2):
        layer = identity_block(layer, [512, 512, 2048])

    layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(layer)

    layer = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=initializer,
                           )(layer)

    model = K.models.Model(inputs=X, outputs=layer)
    return model
