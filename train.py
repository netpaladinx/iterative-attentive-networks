from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def train(model_cls, hparams, FLAGS):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    model = model_cls(hparams)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        sess.run(model.init)

        for i in range(FLAGS.epoch_num):
            total_batch = int(mnist.train.num_examples / FLAGS.batch_size)

            for j in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)

                _, loss, accuracy, summary = sess.run([model.train_op, model.loss, model.accuracy, model.summary_op],
                                                      feed_dict={model.x: batch_x, model.y: batch_y})
                if (i * total_batch + j) % FLAGS.display_steps == 0:
                    print('Train --- Epoch {}, Step {}: Loss {}, Accuracy {}'.format(i, j, loss, accuracy))
                    summary_writer.add_summary(summary, global_step=i * total_batch + j)

            loss, accuracy = sess.run([model.loss, model.accuracy],
                                      feed_dict={model.x: mnist.test.images, model.y: mnist.test.labels})
            print('Eval --- Epoch {}: Loss {}, Accuracy {}'.format(i, loss, accuracy))
