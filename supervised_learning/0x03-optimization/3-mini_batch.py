#!/usr/bin/env python3
"""
Mini-Batch
"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batch_size=32,
                     epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.
    Args:
        - X_train: is a numpy.ndarray of shape (m, 784) containing the training
          data
            - m: is the number of data points
            - 784: is the number of input features
        - Y_train: is a one-hot numpy.ndarray of shape (m, 10) containing the
          training labels
            - 10: is the number of classes the model should classify
        - X_valid: is a numpy.ndarray of shape (m, 784) containing the
          validation data
        - Y_valid: is a one-hot numpy.ndarray of shape (m, 10) containing the
          validation labels
        - batch_size: is the number of data points in a batch
        - epochs: is the number of times the training should pass through the
          whole dataset
        - load_path: is the path from which to load the model
        - save_path: is the path to where the model should be saved after
          training
    Returns:
        The path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        m = X_train.shape[0]
        steps = m // batch_size
        if steps % batch_size != 0:
            steps = steps + 1
            flag = True
        else:
            flag = False

        # x is a placeholder for the input data
        x = tf.get_collection("x")[0]
        # y is a placeholder for the labels
        y = tf.get_collection("y")[0]
        # accuracy is an op to calculate the model's accuracy
        accuracy = tf.get_collection("accuracy")[0]
        # loss is an op to calculate the model's cost
        loss = tf.get_collection("loss")[0]
        # train_op is an op to perform one pass of model's gradient descent
        train_op = tf.get_collection("train_op")[0]

        feed_dict_t = {x: X_train, y: Y_train}
        feed_dict_v = {x: X_valid, y: Y_valid}

        for epoch in range(epochs + 1):
            # Calculate cost and accuracy for training set
            train_cost, train_accuracy = sess.run([loss, accuracy],
                                                  feed_dict_t)
            # Calculate cost and accuracy for validation set
            valid_cost, valid_accuracy = sess.run([loss, accuracy],
                                                  feed_dict_v)
            # The current epoch
            print("After {} epochs:".format(epoch))
            # The model's cost on the entire training set
            print("\tTraining Cost: {}".format(train_cost))
            # The model's accuracy on the entire training set
            print("\tTraining Accuracy: {}".format(train_accuracy))
            # The model's cost on the entire validation set
            print("\tValidation Cost: {}".format(valid_cost))
            # The model's accuracy on the entire validation set
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                # Shuffle the training data
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                for step_number in range(steps):
                    # Slice the data for mini-batches
                    start = step_number * batch_size
                    if step_number == steps - 1 and flag:
                        end = X_train.shape[0]
                    else:
                        end = (batch_size * step_number) + batch_size

                    feed_dict_minib = {x: X_shuffled[start:end],
                                       y: Y_shuffled[start:end]}
                    # Run the training by step
                    sess.run(train_op, feed_dict_minib)

                    if step_number != 0 and (step_number + 1) % 100 == 0:
                        # The number of times gradient descent has been run in
                        # the current epoch
                        print("\tStep {}:".format(step_number + 1))
                        # Calculate model's cost on the current mini-batch
                        step_cost = sess.run(loss, feed_dict_minib)
                        print("\t\tCost: {}".format(step_cost))
                        # Calculate model's accuracy on the current mini-batch
                        step_accuracy = sess.run(accuracy, feed_dict_minib)
                        print("\t\tAccuracy: {}".format(step_accuracy))

        save_path = saver.save(sess, save_path)
    return save_path
