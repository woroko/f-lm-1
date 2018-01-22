# -*- coding: utf-8 -*-

"""
Entry point for training and eval
"""
import os

import tensorflow as tf

import predict
from data_utils import Vocabulary, Dataset
from language_model import LM
#from prediction import sentence_ppl
from run_utils import run_train, run_eval

tf.flags.DEFINE_string("logdir", "lm1b", "Logging directory.")
tf.flags.DEFINE_string("datadir", None, "Logging directory.")
tf.flags.DEFINE_string("mode", "train", "Whether to run 'train' or 'eval' model.")
tf.flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
tf.flags.DEFINE_integer("num_gpus", 8, "Number of GPUs used.")
tf.flags.DEFINE_integer("eval_steps", 50, "Number of eval steps.")
tf.flags.DEFINE_integer("num_sen", 100, "Number of sentences to generate.")

FLAGS = tf.flags.FLAGS

def main(_):
    """
    Start either train or eval. Note hardcoded parts of path for training and eval data
    """
    hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
    hps._set("num_gpus", FLAGS.num_gpus)
    print ('*****HYPER PARAMETERS*****')
    print (hps)
    print ('**************************')

    vocab = Vocabulary.from_file(os.path.join(FLAGS.datadir, "vocabulary.txt"))

    if FLAGS.mode == "train":
        #hps.batch_size = 256
        dataset = Dataset(vocab, os.path.join(FLAGS.datadir, "train.txt"))
        run_train(dataset, hps, os.path.join(FLAGS.logdir, "train"), ps_device="/gpu:0")
    elif FLAGS.mode.startswith("eval"):
        data_dir = os.path.join(FLAGS.datadir, "eval.txt")
        #predict_model = prediction.Model('/dir/ckpt',os.path.join(FLAGS.datadir, "vocabulary.txt"), hps)

        dataset = Dataset(vocab, data_dir, deterministic=True)
        prefix_words = "<brk>".split()
        predict_model = predict.Model(hps, FLAGS.logdir, FLAGS.datadir)
        print ('start input')
        out = predict_model.predictnextkwords(prefix_words, FLAGS.num_sen)
        for row in out:
            print(' '.join(row) + "\n")
        print("len_out: " + str(len(out)))
        #prediction.topkwords(prefix_words, dataset, hps, FLAGS.logdir, FLAGS.mode)
        #sentence_ppl(prefix_words,dataset, hps, FLAGS.logdir, FLAGS.mode)
        #print vocab
        #dataset = Dataset(vocab, os.path.join(FLAGS.datadir, "eval.txt"))
        #run_eval(dataset, hps, FLAGS.logdir, FLAGS.mode, FLAGS.eval_steps)


if __name__ == "__main__":
    tf.app.run()


