import os
import tensorflow as tf
import numpy as np
import time
import inspect
import config


VGG_MEAN = config.VGG_MEAN
IMAGE_PIXELS = config.IMAGE_PIXELS
CATEGORIES = config.CATEGORIES


def get_conv_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="weight")


def get_conv_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, filter_size, in_channels, out_channels, name):
    with tf.variable_scope(name):
        w = get_conv_weight([filter_size, filter_size, in_channels, out_channels])
        b = get_conv_bias([out_channels])
        conv = tf.nn.conv2d(bottom, w, strides=[1, 1, 1, 1], padding='SAME') + b
        relu = tf.nn.relu(conv)
        return relu


def fc_layer(bottom, in_size, out_size, name):
    with tf.variable_scope(name):
        w = get_conv_weight([in_size, out_size])
        b = get_conv_bias([out_size])
        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.matmul(x, w) + b
        return fc


class VGG19(object):

    def __init__(self, trainable=True):
        print("initialization")
        self.trainable = trainable

    def build(self, train_mode=None):
        """
            :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        start_time = time.time()
        print("build model started")
        images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS, IMAGE_PIXELS, 3])
        labels = tf.placeholder(tf.float32, [None, CATEGORIES])
        keep_prob = tf.placeholder(tf.float32)

        self.conv1_1 = conv_layer(images, 3, 3, 64, "conv1_1")
        self.conv1_2 = conv_layer(self.conv1_1, 3, 64, 64, "conv1_2")
        self.pool1 = max_pool(self.conv1_2, "pool1")

        self.conv2_1 = conv_layer(self.pool1, 3, 64, 128, "conv2_1")
        self.conv2_2 = conv_layer(self.conv2_1, 3, 128, 128, "conv2_2")
        self.pool2 = max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = conv_layer(self.pool2, 3, 128, 256, "conv3_1")
        self.conv3_2 = conv_layer(self.conv3_1, 3, 256, 256, "conv3_2")
        self.conv3_3 = conv_layer(self.conv3_2, 3, 256, 256, "conv3_3")
        self.conv3_4 = conv_layer(self.conv3_3, 3, 256, 256, "conv3_4")
        self.pool3 = max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = conv_layer(self.pool3, 3, 256, 512, "conv4_1")
        self.conv4_2 = conv_layer(self.conv4_1, 3, 512, 512, "conv4_2")
        self.conv4_3 = conv_layer(self.conv4_2, 3, 512, 512, "conv4_3")
        self.conv4_4 = conv_layer(self.conv4_3, 3, 512, 512, "conv4_4")
        self.pool4 = max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = conv_layer(self.pool4, 3, 512, 512, "conv5_1")
        self.conv5_2 = conv_layer(self.conv5_1, 3, 512, 512, "conv5_2")
        self.conv5_3 = conv_layer(self.conv5_2, 3, 512, 512, "conv5_3")
        self.conv5_4 = conv_layer(self.conv5_3, 3, 512, 512,"conv5_4")
        self.pool5 = max_pool(self.conv5_4, 'pool5')

        self.fc6 = fc_layer(self.pool5, 25088, 4096, "fc6") # 25088 = ((IMAGE_PIXELS // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, keep_prob), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, keep_prob)

        self.fc7 = fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, keep_prob), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, keep_prob)

        # change the top layer.
        self.fc8 = fc_layer(self.relu7, 4096, CATEGORIES, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.images = images
        self.labels = labels
        self.keep_prob = keep_prob
