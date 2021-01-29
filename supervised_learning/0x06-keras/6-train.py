#!/usr/bin/env python3
"""Contains the train_model function"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent
    :param network: model to train
    :param data: numpy.ndarray of shape (m, nx) containing the input data
    :param labels: one-hot numpy.ndarray of shape (m, classes)
        containing the labels of data
    :param batch_size: size of the batch used for mini-batch grad desc
    :param epochs: number of passes through data for mini-batch grad desc
    :param validation_data:  data to validate the model with, if not None
    :param early_stopping: boolean that indicates whether
        early stopping should be used
    :param patience: patience used for early stopping
    :param verbose: boolean that determines if output should be
        printed during training
    :param shuffle: boolean that determines whether to shuffle
        the batches every epoch.
    :return: History object generated after training the model
    """
    callback_list = []

    # early stopping callback
    if validation_data and early_stopping:
        es = K.callbacks.EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       patience=patience)
        callback_list.append(es)

    # training
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=callback_list)

    return history
