from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import ops


class IAN(object):
    '''Iterative Attention Networks'''

    def __init__(self, hparams):
        self.hparams = hparams

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self._build_network()
            self.init_op = tf.global_variables_initializer()

            #tf.summary.scalar('loss', self.loss)
            #tf.summary.scalar('accuracy', self.accuracy)
            #self.summary_op = tf.summary.merge_all()

        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(graph=self.tf_graph, config=self.tf_config)

    def _create_inputs(self):
        self.x_pl = tf.placeholder(tf.float32, [None, self.hparams.n_input], name='input')
        self.y_pl = tf.placeholder(tf.float32, [None, self.hparams.n_output], name='label')
        self.batch_size = tf.shape(self.x_pl)[0]


    def _create_IAN(self, acti, n_in, n_out, n_dims, scope='IAN'):
        '''acti: batch_size x n_in
        '''
        with tf.variable_scope(scope):
            phi_in = tf.get_variable('phi_in', shape=[n_in, n_dims], initializer=tf.variance_scaling_initializer())
            phi_out = tf.get_variable('phi_out', shape=[n_out, n_dims], initializer=tf.variance_scaling_initializer())

            state_in = tf.multiply(tf.expand_dims(acti, axis=2), phi_in, name='state_in_0')  # batch_size x n_in x n_dim
            #state_in = tf.tile(tf.expand_dims(phi_in, axis=0), [self.batch_size, 1, 1], name='state_in_0')
            state_out = tf.identity(phi_out, name='state_out_0')  # n_out x n_dim
            dot_in2out = tf.tensordot(state_in, state_out, axes=[[2],[1]], name='dot_in2out_0')  # batch_size x n_in x n_out
            #dot_in2out = tf.nn.relu(dot_in2out)
            #att_in2out = tf.nn.softmax(dot_in2out, axis=1, name='att_in2out_0')  # batch_size x n_in(normalized) x n_out
            att_in2out = ops.l1_normalize(dot_in2out, axis=1, name='att_in2out_0')
            update_out = tf.matmul(att_in2out, state_in, transpose_a=True, name='update_out_0')  # batch_size x n_out x n_dim

            state_in = tf.identity(state_in, name='state_in_1')  # batch_size x n_in x n_dim
            state_out = tf.identity(update_out, name='state_out_1')  # batch_size x n_out x n_dim
            dot_out2in = tf.matmul(state_out, state_in, transpose_b=True, name='dot_out2in_1')  # batch_size x n_out x n_in
            #dot_out2in = tf.nn.relu(dot_out2in)
            #att_out2in = tf.nn.softmax(dot_out2in, axis=1, name='att_out2in_1')  # batch_size x n_out(normalized) x n_in
            att_out2in = ops.l1_normalize(dot_out2in, axis=1, name='att_out2in_1')
            update_in = tf.matmul(att_out2in, state_out, transpose_a=True, name='update_in_1')

            state_in = tf.identity(update_in, name='state_in_2')  # batch_size x n_in x n_dim
            state_out = tf.identity(state_out, name='state_out_2')  # batch_size x n_out x n_dim
            dot_in2out = tf.matmul(state_in, state_out, transpose_b=True, name='dot_in2out_2')  # batch_size x n_in x n_out
            #dot_in2out = tf.nn.relu(dot_in2out)
            #att_in2out = tf.nn.softmax(dot_in2out, axis=1, name='att_in2out_2')  # batch_size x n_in(normalized) x n_out
            att_in2out = ops.l1_normalize(dot_in2out, axis=1, name='att_in2out_2')

            return att_in2out

    def _create_layer(self, acti, n_in, n_out, activation_fn=tf.nn.relu, scope='layer'):
        with tf.variable_scope(scope):
            weight = tf.get_variable('weight', shape=[n_in, n_out], initializer=tf.variance_scaling_initializer())
            bias = tf.get_variable('bias', shape=[n_out], initializer=tf.zeros_initializer())

            att = self._create_IAN(acti, n_in, n_out, self.hparams.n_dims)

            att_weight = tf.multiply(att, weight, name='att_weight')
            output = tf.reshape(tf.matmul(tf.expand_dims(acti, axis=1), att_weight) + bias, [self.batch_size, -1])
            #output = tf.matmul(acti, weight) + bias

            if activation_fn is not None:
                output = activation_fn(output)
            output = tf.identity(output, name='output')
            return output

    def _create_optimizer(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y_pl))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output, 1), tf.arg_max(self.y_pl, 1)), tf.float32))
        self.global_step = tf.train.create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.learning_rate)
        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def _build_network(self):
        self._create_inputs()
        with tf.variable_scope('FeedForward') as scope:
            self.hidden1 = self._create_layer(self.x_pl, self.hparams.n_input, self.hparams.n_hidden, scope='hidden_1')
            #self.hidden2 = self._create_layer(self.hidden1, self.hparams.n_hidden, self.hparams.n_hidden, scope='hidden_2')
            self.output = self._create_layer(self.hidden1, self.hparams.n_hidden, self.hparams.n_output, activation_fn=None, scope='output')
        self._create_optimizer()

    def fit(self, FLAGS):
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

        with self.tf_session as sess:
            #summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

            sess.run(self.init_op)

            for i in xrange(FLAGS.epoch_num):
                total_batch = int(mnist.train.num_examples / FLAGS.batch_size)

                for j in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)
                    _, loss, accuracy = sess.run([self.opt_op, self.loss, self.accuracy],
                                                 feed_dict={self.x_pl: batch_x, self.y_pl: batch_y})
                    if (i * total_batch + j) % FLAGS.display_steps == 0:
                        print('Train --- Epoch {}, Step {}: Loss {}, Accuracy {}'.format(i, j, loss, accuracy))
                        #summary_writer.add_summary(summary, global_step=i*total_batch+j)

                #loss, accuracy = sess.run([self.loss, self.accuracy],
                #                          feed_dict={self.x_pl: mnist.test.images, self.y_pl: mnist.test.labels})
                #print('Eval --- Epoch {}: Loss {}, Accuracy {}'.format(i, loss, accuracy))


# Hyper parameters
default_hparams = tf.contrib.training.HParams(
    n_input=784,
    n_output=10,
    n_hidden=128,
    n_dims=32,
    learning_rate=0.001
)