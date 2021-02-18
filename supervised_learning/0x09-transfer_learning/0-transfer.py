#!/usr/bin/env python3

"""Use transfer learning with DenseNet-169 to predict CIFAR-10 dataset"""
import tensorflow as tf
import tensorflow.keras as K


def preprocess_data(X, Y):
    """Pre-processes the data for the model
        - X is a numpy.ndarray of shape (m, 32, 32, 3) containing
         the CIFAR 10 data, where m is the number of data points
        - Y is a numpy.ndarray of shape (m,) containing
         the CIFAR 10 labels for X
        Returns: X_p, Y_p
        - X_p is a numpy.ndarray containing the preprocessed X
        - Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.densenet.preprocess_input(X)

    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


if __name__ == '__main__':

    (X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)

    X_val, Y_val = preprocess_data(X_val, Y_val)

    initializer = K.initializers.he_normal(seed=None)

    input_shape_densenet = (224, 224, 3)

    densenet_model = K.applications.DenseNet169(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape_densenet,
        pooling=None
    )

    densenet_model.trainable = True

    for layer in densenet_model.layers:
        if 'conv5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    input = K.Input(shape=(32, 32, 3))

    preprocess = K.layers.Lambda(
        lambda x: tf.image.resize_images(x, (224, 224)), name='lamb')(input)

    layer = densenet_model(inputs=preprocess)

    layer = K.layers.Flatten()(layer)

    layer = K.layers.BatchNormalization()(layer)

    layer = K.layers.Dense(units=256,
                           activation='relu',
                           kernel_initializer=initializer
                           )(layer)

    layer = K.layers.Dropout(0.4)(layer)

    layer = K.layers.BatchNormalization()(layer)

    layer = K.layers.Dense(units=128,
                           activation='relu',
                           kernel_initializer=initializer
                           )(layer)

    layer = K.layers.Dropout(0.4)(layer)

    layer = K.layers.Dense(units=10,
                           activation='softmax',
                           kernel_initializer=initializer
                           )(layer)

    model = K.models.Model(inputs=input, outputs=layer)

    model.compile(loss='binary_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, Y_train, epochs=20, validation_data=(
        X_val, Y_val), batch_size=32, verbose=1)

    model.save('cifar10.h5')
