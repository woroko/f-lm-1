# -*- coding: utf-8 -*-

import tensorflow as tf
import os

import common
from data_utils import Vocabulary
import numpy as np
from common import CheckpointLoader
from language_model import LM


class Model:
    def __init__(self, hps, logdir, datadir, mode='eval'):
        with tf.variable_scope("model"):
            hps.num_sampled = 0
            hps.keep_prob = 1.0
            self.model = LM(hps, "eval", "/gpu:0")
        if hps.average_params:
            print("Averaging parameters for evaluation.")
            saver = tf.train.Saver(self.model.avg_dict)
        else:
            saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        sw = tf.summary.FileWriter(logdir + "/" + mode, self.sess.graph)
        self.hps = hps
        self.num_steps = self.hps.num_steps
        vocab_path = os.path.join(datadir, "vocabulary.txt")
        with self.sess.as_default():
            success = common.load_from_checkpoint(saver, logdir + "/train")
        if not success:
            raise Exception('Loading Checkpoint failed')
        self.vocabulary = Vocabulary.from_file(vocab_path)
    
    def weighted_pick(self, weights):
        #temperature = 0.5
        #scaled_weights = weights / temperature
        #probs = np.exp(scaled_weights - np.max(scaled_weights)) / np.sum(np.exp(scaled_weights - np.max(scaled_weights)))
        
        t = np.cumsum(weights)
        s = np.sum(weights)
        return(int(np.searchsorted(t, np.random.rand(1)*s)))
        #return np.random.choice(range(self.hps.vocab_size), p=probs)
    
    
    # type of prefix_words is list
    def predictnextkwords(self, prefix_words, k):
        
        #prob = self.get_softmax_distrib(x, y, n)
        songs = []
        for songidx in range(k):
            n = len(prefix_words) + 1
            startn = n
            x = np.zeros([self.hps.batch_size, self.hps.num_steps], dtype=np.int32)
            y = np.zeros([self.hps.batch_size, self.hps.num_steps], dtype=np.int32)
            x[0, :n] = ([self.vocabulary.s_id] +
                        list(map(self.vocabulary.get_id, prefix_words)))
            y[0, :n] = (list(map(self.vocabulary.get_id, prefix_words)) +
                        [self.vocabulary.s_id])
            top_indices = []
            
            for i in range(self.hps.num_steps-startn):
                prob = self.get_softmax_distrib(x, y, n)
                #word_idx = self.argsort_k_largest(prob, 5)
                word_idx = self.weighted_pick(prob)
                top_indices.append(word_idx)
                #word_idx = argsort[0]
                #print("word_idx: " + str(word_idx))
                
                x[0, n] = word_idx
                y[0, n-1:n+1] = [word_idx] + [self.vocabulary.s_id]
                n += 1
            
            songs.append(top_indices)
        
        songstxt = []
        for i in range(len(songs)):
            songstxt.append([])
            for id_ in songs[i]:
                try:
                    songstxt[i].append(self.vocabulary.get_token(id_))
                except:
                    songstxt[i].append("<unk>")
        return songstxt
        #return [[self.vocabulary.get_token(id_) for id_ in top_indices_] for top_indices_ in songs]

    def argsort_k_largest(self, prob, k):
        if k >= len(prob):
            return np.argsort(prob)[::-1]
        indices = np.argpartition(prob, -k)[-k:]
        values = prob[indices]
        return indices[np.argsort(-values)]

    def get_softmax_distrib_length(self, x, y, n):
        print('start')
        with self.sess.as_default():
            tf.local_variables_initializer().run()
            print('Start predicting...')
            softmax = self.sess.run(self.model.softmax, {self.model.x: x, self.model.y: y})
            return softmax[n:]
    
    def get_softmax_distrib(self, x, y, n):
        #print('start')
        with self.sess.as_default():
            tf.local_variables_initializer().run()
            #print('Start predicting...')
            softmax = self.sess.run(self.model.softmax, {self.model.x: x, self.model.y: y})
            return softmax[n - 1]

    def getPPL(self, prefix_words):
        n = len(prefix_words) + 1
        x = np.zeros([self.hps.batch_size, self.hps.num_steps], dtype=np.int32)
        y = np.zeros([self.hps.batch_size, self.hps.num_steps], dtype=np.int32)
        x[0, :n] = ([self.vocabulary.s_id] +
                    list(map(self.vocabulary.get_id, prefix_words)))
        y[0, :n] = (list(map(self.vocabulary.get_id, prefix_words)) +
                    [self.vocabulary.s_id])
        with self.sess.as_default():
            tf.local_variables_initializer().run()
            ppl = self.sess.run(self.model.loss, {self.model.x: x, self.model.y: y})
            return [ppl, float(np.exp(ppl))]
