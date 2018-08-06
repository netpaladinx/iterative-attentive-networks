from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


class Model(object):

    def __init__(self, hparams):
        self.hparams = hparams

        self.x = tf.placeholder(tf.float32, [None, self.hparams.n_input], name='input')
        self.y = tf.placeholder(tf.float32, [None, self.hparams.n_output], name='label')
        self.batch_size = tf.shape(self.x)[0]

        self.reuse_h_network = False
        self.reuse_g_network = False

        out = self.build_network()

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=self.y))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(self.y, 1)), tf.float32))

        global_step = tf.train.create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        self.init = tf.global_variables_initializer()

        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Accuracy', self.accuracy)
        self.summary_op = tf.summary.merge_all()

    def build_network(self):
        W1 = self.weight_network(self.x, self.hparams.n_input, self.hparams.n_hidden[0])
        h1 = tf.nn.relu(self.fc_layer(self.x, W1, self.hparams.n_hidden[0]))
        W2 = self.weight_network(h1, self.hparams.n_hidden[0], self.hparams.n_hidden[1])
        h2 = tf.nn.relu(self.fc_layer(h1, W2, self.hparams.n_hidden[1]))
        W3 = self.weight_network(h2, self.hparams.n_hidden[1], self.hparams.n_output)
        out = self.fc_layer(h2, W3, self.hparams.n_output)
        return out

    def weight_network(self, v, n_cur, n_suc):
        state_1 = tf.zeros(shape=[self.batch_size, n_cur, self.hparams.n_wn_state], name='state_1')
        r = tf.random_normal(shape=[self.batch_size, n_suc], name='suc_layer_val')
        state_2 = tf.zeros(shape=[self.batch_size, n_suc, self.hparams.n_wn_state], name='state_2')

        # new_s, _ = self.weight_network_cell(v, state_1, r, state_2, n_cur, n_suc)
        # new_c, _ = self.weight_network_cell(r, new_s, v, state_1, n_suc, n_cur)
        # _, A = self.weight_network_cell(v, new_c, r, new_s, n_cur, n_suc)
        _, A = self.weight_network_cell(v, state_1, r, state_2, n_cur, n_suc)

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        W = tf.Variable(xavier_initializer([n_cur, n_suc]), name='weight')
        return tf.multiply(W, A)

    def weight_network_cell(self, v, c, r, s, n_cur, n_suc):
        v_reshaped = tf.reshape(v, [self.batch_size, n_cur, 1])
        v_state = tf.concat([v_reshaped, c], axis=2, name='cur_layer_val_with_state')
        v_state_reshaped = tf.reshape(v_state, [self.batch_size * n_cur, self.hparams.n_wn_state + 1])
        h1 = self.h_network(v_state_reshaped)
        h1_reshaped = tf.reshape(h1, [self.batch_size, n_cur, self.hparams.n_hn_output])

        r_reshaped = tf.reshape(r, [self.batch_size, n_suc, 1])
        r_state = tf.concat([r_reshaped, s], axis=2, name='suc_layer_val_with_state')
        r_state_reshaped = tf.reshape(r_state, [self.batch_size * n_suc, self.hparams.n_wn_state + 1])
        h2 = self.h_network(r_state_reshaped)
        h2_reshaped = tf.reshape(h2, [self.batch_size, n_suc, self.hparams.n_hn_output])

        A = tf.nn.softmax(tf.matmul(h2_reshaped, h1_reshaped, transpose_b=True), axis=2)

        i = tf.matmul(A, h1_reshaped)
        i_old_state = tf.concat([i, s], axis=2)
        i_old_state_reshaped = tf.reshape(i_old_state, [self.batch_size * n_suc,
                                                        self.hparams.n_hn_output + self.hparams.n_wn_state])
        new_state = tf.reshape(self.g_network(i_old_state_reshaped),
                               [self.batch_size, n_suc, self.hparams.n_wn_state])
        return new_state, tf.transpose(A, perm=[0, 2, 1])

    def h_network(self, x_state):
        with tf.variable_scope('h-network') as scope:
            if self.reuse_h_network:
                scope.reuse_variables()
            else:
                self.reuse_h_network = True

            hidden = fully_connected(x_state, num_outputs=self.hparams.n_hn_hidden, activation_fn=tf.nn.relu)
            h = fully_connected(hidden, num_outputs=self.hparams.n_hn_output, activation_fn=tf.nn.tanh)
            return h

    def g_network(self, i_old_state):
        with tf.variable_scope('g-network') as scope:
            if self.reuse_g_network:
                scope.reuse_variables()
            else:
                self.reuse_g_network = True

            hidden = fully_connected(i_old_state, num_outputs=self.hparams.n_gn_hidden, activation_fn=tf.nn.relu)
            g = fully_connected(hidden, num_outputs=self.hparams.n_wn_state, activation_fn=tf.nn.relu)
            return g

    def fc_layer(self, x, W, n_suc):
        with tf.variable_scope('fc-layer'):
            b = tf.Variable(tf.zeros(shape=[n_suc]), name='bias')
            x_reshaped = tf.reshape(x, [self.batch_size, 1, tf.shape(x)[1]])
            mul = tf.matmul(x_reshaped, W)
            return tf.add(tf.reshape(mul, [self.batch_size, tf.shape(mul)[2]]), b)
            # return tf.add(tf.matmul(x, W), b)


# Hyper parameters
default_hparams = tf.contrib.training.HParams(
    n_wn_state=3,
    n_hn_hidden=4,
    n_hn_output=4,
    n_gn_hidden=8,
    n_input=784,
    n_output=10,
    n_hidden=[256, 256],
    learning_rate=0.001,
)
