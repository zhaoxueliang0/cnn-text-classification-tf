import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W,
                                                         self.input_x)  # return a ndarray of shape [batch size, sentence length, embedding dim]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,
                                                          -1)  # return a ndarray of shape [batch size, sentence length, embedding dim, 1]

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded, # [batch, height, width, in_channels]
                    W,  # convolution kernel, with shape [filter_height, filter_width, in_channels, out_channels]
                    strides=[1, 1, 1, 1],  # the step size of each dimension
                    padding="VALID",
                    name="conv")  # return a ndarray of shape [batch, sentence length - filter size + 1, 1, channels(num_channels)]
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    # ignore pooling of "batch" and "channels", with shape [1, height, width, 1]
                    strides=[1, 1, 1, 1],
                    # the step that window slides on each dimension, with shape [1, stride,stride, 1],
                    padding='VALID',
                    name="pool")  # return a ndarray of shape [batch, height(1), width(1), channels(num_filters)]
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)  # connect to a very long array
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # [batch, num_filters_total]

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")  # ndarray with shape [batch, num_classes]
            self.predictions = tf.argmax(self.scores, 1, name="predictions")  # ndarray with shape [batch,]

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)  # ndarray with shape [batch,]
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss  # float

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))  # ndarray with shape [batch,]
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")  # float
