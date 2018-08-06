from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def l1_normalize(x, axis=None, epsilon=1e-12, name=None):
    x_norm = tf.maximum(tf.reduce_sum(tf.abs(x), axis, keepdims=True), epsilon)
    return tf.divide(x, x_norm, name=name)