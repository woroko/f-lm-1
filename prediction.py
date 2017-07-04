# -*- coding: utf-8 -*-

import tensorflow as tf

import common
from data_utils import Vocabulary
import numpy as np

from common import CheckpointLoader
from language_model import LM
from model_utils import sharded_variable, getdtype
from model_utils import sharded_variable, getdtype, variable_summaries
from common import assign_to_gpu, average_grads, find_trainable_variables
from hparams import HParams
from tensorflow.contrib.rnn import LSTMCell
from factorized_lstm_cells import GLSTMCell, ResidualWrapper, FLSTMCell

class Model:
    def __init__(self, model_path, vocab_path, hps):
        self.vocabulary = Vocabulary.from_file(vocab_path)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config=config)
        saver = tf.train.import_meta_graph('{}.meta'.format(model_path))
        saver.restore(self.session, str(model_path))
        #common.load_from_checkpoint(saver, model_path)

        self.input_xs = tf.get_collection('input_xs')[0]
        self.batch_size = tf.get_collection('batch_size')[0]
        self.softmax = tf.get_collection('softmax')[0]
        self.num_steps = hps.num_steps
        self.hps = hps
  #      self.x = tf.contrib.framework.get_model_variables('x')


    def predict_next_words(self, prefix_words):
        #inputs, targets = process_sentence(prefix_words, self.vocabulary, self.hps)
        prefix_words = prefix_words[-self.num_steps + 1:]
        n = len(prefix_words) + 1
        xs = np.zeros([self.hps.batch_size, self.num_steps])
        xs[0, :n] = ([self.vocabulary.s_id] +
                     list(map(self.vocabulary.get_id, prefix_words)))

        batch_size = 1
        self._init_state(batch_size)
        feed_dict = {self.input_xs: xs, self.batch_size: batch_size}
        print 'Start...'
        return self.session.run(self.softmax[n-1], feed_dict=feed_dict)

    def _init_state(self, batch_size):

        state_size, proj_size = self.hps.state_size, self.hps.projected_size
        #for v in tf.get_collection('initial_state'):

        for v in tf.local_variables():
            x, y = v.get_shape()
            self.session.run(
                tf.assign(v, tf.zeros([x, y]),
                          validate_shape=False))

    def predict_top(self, prefix_words, top=10):
        probs = self.predict_next_words(prefix_words)
        top_indices = argsort_k_largest(probs, top)
        return [(self.vocabulary.get_token(id_), probs[id_]) for id_ in top_indices]


def argsort_k_largest(x, k):
    if k >= len(x):
        return np.argsort(x)[::-1]
    indices = np.argpartition(x, -k)[-k:]
    values = x[indices]
    return indices[np.argsort(-values)]


def sentence_ppl(prefix_words, dataset, hps, logdir, mode):
    inputs, targets = process_sentence(prefix_words, dataset, hps)
    with tf.variable_scope("model"):
        hps.num_sampled = 0  # Always using full softmax at evaluation.
        hps.keep_prob = 1.0
        # model = LM(hps, "eval", "/cpu:0")
        model = LM(hps, "eval", "/gpu:0")

    if hps.average_params:
        print("Averaging parameters for evaluation.")
        saver = tf.train.Saver(model.avg_dict)
    else:
        saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sw = tf.summary.FileWriter(logdir + "/" + mode, sess.graph)
    ckpt_loader = CheckpointLoader(saver, model.global_step, logdir + "/train")

    with sess.as_default():
        while ckpt_loader.load_checkpoint():
            tf.local_variables_initializer().run()
            ppl = sess.run(model.loss, {model.x: inputs, model.y: targets})
            print np.exp(ppl)
            return np.exp(ppl)


def process_sentence(prefix_words, vocab, hps):
    targets = np.zeros([hps.batch_size * hps.num_gpus, hps.num_steps], np.int32)
    inputs = np.zeros([hps.batch_size * hps.num_gpus, hps.num_steps], np.int32)
    token_to_id = vocab
    words = prefix_words.strip().split()
    prefix_char_ids = [token_to_id.get_id(w.decode('utf8')) for w in words]
    prefixId = token_to_id.get_id('<S>')
    inputs[0, 0] = prefixId
    inputs[0, 1:len(prefix_char_ids) + 1] = prefix_char_ids[0:len(prefix_char_ids)]
    targets[0, 0:len(prefix_char_ids)] = prefix_char_ids[0:len(prefix_char_ids)]
    targets[0, len(prefix_char_ids)] = prefixId
    return inputs, targets




