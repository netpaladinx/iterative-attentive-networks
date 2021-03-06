from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import models


def main(args):
    hparams = models.default_hparams
    hparams.parse(FLAGS.hparams)
    print(hparams.values())

    ian = models.IAN(hparams)
    ian.fit(FLAGS)

if __name__ == '__main__':
    tf.flags.DEFINE_string("hparams", "", """Comma separated list of name=value pairs.""")
    tf.flags.DEFINE_boolean("debug", False, """Enabling debug for producing consistent results.""")
    tf.flags.DEFINE_string("data_dir", "./data", """Store downloaded data""")
    tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint", "")
    tf.flags.DEFINE_string("summary_dir", "./summary", "")
    tf.flags.DEFINE_integer("save_checkpoint_steps", 100, "")
    tf.flags.DEFINE_integer("save_summaries_steps", 10, "")
    tf.flags.DEFINE_integer("display_steps", 10, "")
    tf.flags.DEFINE_integer("epoch_num", 30, "")
    tf.flags.DEFINE_integer("batch_size", 64, "")

    FLAGS = tf.flags.FLAGS
    if FLAGS.debug:
        np.random.seed(0)
        tf.set_random_seed(0)

    tf.app.run()