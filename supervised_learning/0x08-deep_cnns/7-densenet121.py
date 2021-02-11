#!/usr/bin/env python3

"""Builds the DenseNet-121 architecture as described in
 Densely Connected Convolutional Networks
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
     Densely Connected Convolutional Networks
        - growth_rate is the growth rate
        - compression is the compression factor
        - You can assume the input data will have shape (224, 224, 3)
        - All convolutions should be preceded by Batch Normalization and
         a rectified linear activation (ReLU), respectively
        - All weights should use he normal initialization
        You may use:
        dense_block = __import__('5-dense_block').dense_block
        transition_layer = __import__('6-transition_layer').transition_layer
        Returns: the keras model
    """
    initializer = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    layer = K.layers.BatchNormalization(axis=3)(X)

    layer = K.layers.Activation('relu')(layer)

    # conv
    layer = K.layers.Conv2D(filters=2 * growth_rate,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer,
                            )(layer)

    # pool
    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(layer)

    layers = [6, 12, 24]
    nb_filters = 2 * growth_rate

    # dense-transition
    for i in range(3):
        layer, nb_filters = dense_block(
            layer, nb_filters, growth_rate, layers[i])

        layer, nb_filters = transition_layer(layer, nb_filters, compression)

    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 16)

    layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(layer)

    layer = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=initializer,
                           )(layer)

    model = K.models.Model(inputs=X, outputs=layer)
    return model
