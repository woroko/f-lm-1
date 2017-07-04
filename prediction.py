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
        inputs, targets = process_sentence(prefix_words, self.vocabulary, self.hps)
        n = len(inputs) + 1
        batch_size = 1
        self._init_state(batch_size)
        feed_dict = {self.input_xs: inputs, self.batch_size: batch_size}
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

    def predict_k_top(self, prefix_words, top=10):
        inputs, targets = process_sentence(prefix_words, self.vocabulary, self.hps)
        self._init_state(self.hps.batch_size)
        print type(inputs)
        self.session.run({self.x: inputs})
        inputsforsw = idx_to_inputs(inputs)
        prob = calculate_softmax(inputsforsw)
        top_indices = argsort_k_largest(prob, top)
        return [(self.vocabulary.get_token(id_), prob[id_]) for id_ in top_indices]

def calculate_softmax(prefix_words_id ,hps):
    softmax_w = sharded_variable(
        'softmax_w', [hps.vocab_size, hps.projected_size], hps.num_shards)
    softmax_b = tf.get_variable('softmax_b', [hps.vocab_size])

    full_softmax_w = tf.reshape(
        tf.concat(1, softmax_w), [-1, hps.projected_size])
    full_softmax_w = full_softmax_w[:hps.vocab_size, :]

    logits = (tf.matmul(prefix_words_id, full_softmax_w, transpose_b=True) +
              softmax_b)
    softmax = tf.nn.softmax(logits)
    return softmax[len(prefix_words_id)]

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

def topkwords(prefix_words, dataset, hps, logdir, mode,top=10):
    inputs, targets = process_sentence(prefix_words, dataset._vocab, hps)
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


def idx_to_inputs(x, hps):
    initial_states = []
    inputs = [tf.squeeze(input=tf.cast(v, getdtype(hps, True)), axis=[1]) for v in tf.split(value=x,
                                                                                            num_or_size_splits=hps.num_steps,
                                                                                            axis=1)]
    for i in range(hps.num_layers):
        with tf.variable_scope("lstm_%d" % i) as scope:
            if hps.num_of_groups > 1:
                assert (hps.fact_size is None)
                print("Using G-LSTM")
                print("Using %d groups" % hps.num_of_groups)
                cell = GLSTMCell(num_units=hps.state_size,
                                 num_proj=hps.projected_size,
                                 number_of_groups=hps.num_of_groups)
            else:
                if hps.fact_size:
                    print("Using F-LSTM")
                    print("Using factorization: %d x %d x %d" % (
                    2 * hps.projected_size, int(hps.fact_size), 4 * hps.state_size))
                    cell = FLSTMCell(num_units=hps.state_size,
                                     num_proj=hps.projected_size,
                                     factor_size=int(hps.fact_size))
                else:
                    print("Using LSTMP")
                    cell = LSTMCell(num_units=hps.state_size,
                                    num_proj=hps.projected_size)

            state = tf.contrib.rnn.LSTMStateTuple(initial_states[i][0],
                                                  initial_states[i][1])

            if hps.use_residual:
                cell = ResidualWrapper(cell=cell)

            for t in range(hps.num_steps):
                if t > 0:
                    scope.reuse_variables()
                inputs[t], state = cell(inputs[t], state)
                if hps.keep_prob < 1.0:
                    inputs[t] = tf.nn.dropout(inputs[t], hps.keep_prob)

            with tf.control_dependencies([initial_states[i][0].assign(state[0]),
                                          initial_states[i][1].assign(state[1])]):
                inputs[t] = tf.identity(inputs[t])

                # inputs = tf.reshape(tf.concat(1, inputs), [-1, hps.projected_size])
    inputs = tf.reshape(tf.concat(inputs, 1), [-1, hps.projected_size])
    return inputs

