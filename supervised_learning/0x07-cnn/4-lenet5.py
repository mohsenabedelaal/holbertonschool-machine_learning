#!/usr/bin/env python3
""" module"""
import tensorflow as tf


def lenet5(x, y):
    """function"""
    w = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding="same",
        kernel_initializer=w,
        activation=tf.nn.relu)(x)
    max_pool1 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2)(conv1)
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="valid",
        nitializer=w,
        activation=tf.nn.relu)(max_pool1)
    max_pool2 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2)(conv2)
    flat_layer = tf.layers.Flatten()(max_pool2)
    full_layer1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                                  kernel_initializer=w)(flat_layer)
    full_layer2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                                  kernel_initializer=w)(full_layer1)
    full_layer3 = tf.layers.Dense(units=10, kernel_initializer=w)(full_layer2)
    y_pred = tf.nn.softmax(full_layer3)
    equal = tf.equal(tf.argmax(full_layer3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, full_layer3)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    return (y_pred, train, loss, accuracy)
