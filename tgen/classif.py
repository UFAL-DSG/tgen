#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classifying trees to determine which DAIs are represented.
"""

from __future__ import unicode_literals
import cPickle as pickle
import time
import datetime
import theano.tensor as T
import numpy as np


from alex.components.nlg.tectotpl.core.util import file_stream

from tgen.rnd import rnd
from tgen.logf import log_debug, log_info, log_warn
from tgen.futil import read_das, read_ttrees, trees_from_doc
from tgen.features import Features
from tgen.ml import DictVectorizer
from tgen.nn import ClassifNN, FeedForward


class TreeClassifier(object):
    """A classifier for trees that decides which DAIs are currently represented
    (to be used in limiting candidate generator and/or re-scoring the trees)."""

    def __init__(self, cfg):
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        self.nn_shape = cfg.get('nn_shape', 'ff')
        self.num_hidden_units = cfg.get('num_hidden_units', 512)
        self.init = cfg.get('initialization', 'uniform_glorot10')
        self.passes = cfg.get('passes', 200)
        self.alpha = cfg.get('alpha', 0.1)
        self.randomize = cfg.get('randomize', True)
        self.batch_size = cfg.get('batch_size', 1)

    @staticmethod
    def load_from_file(fname):
        log_info('Loading model from ' + fname)
        with file_stream(fname, mode='rb', encoding=None) as fh:
            classif = pickle.load(fh)
        return classif

    def save_to_file(self, fname):
        log_info('Saving model to ' + fname)
        with file_stream(fname, mode='wb', encoding=None) as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

    def train(self, das_file, ttree_file, data_portion=1.0):
        """Run training on the given training data."""
        self._init_training(das_file, ttree_file, data_portion)
        for iter_no in xrange(1, self.passes + 1):
            self.train_order = range(len(self.train_trees))
            if self.randomize:
                rnd.shuffle(self.train_order)
            log_info("Train order: " + str(self.train_order))
            self._training_pass(iter_no)

    def _init_training(self, das_file, ttree_file, data_portion):
        """Initialize training.

        Store input data, initialize 1-hot feature representations for input and output and
        transform training data accordingly, initialize the classification neural network."""
        # read input
        log_info('Reading DAs from ' + das_file + '...')
        das = read_das(das_file)
        log_info('Reading t-trees from ' + ttree_file + '...')
        ttree_doc = read_ttrees(ttree_file)
        trees = trees_from_doc(ttree_doc, self.language, self.selector)

        # make training data smaller if necessary
        train_size = int(round(data_portion * len(trees)))
        self.train_trees = trees[:train_size]
        self.train_das = das[:train_size]
        self.train_order = range(len(self.train_trees))
        log_info('Using %d training instances.' % train_size)

        # initialize feature sources
        self.tree_feats = Features(['node: presence t_lemma formeme'])
        self.da_feats = Features(['dat: dat_presence', 'svp: svp_presence'])
        self.X = [self.tree_feats.get_features(tree, {}) for tree in self.train_trees]
        self.y = [self.da_feats.get_features(None, {'da': da}) for da in self.train_das]
        self.tree_vect = DictVectorizer(sparse=False, binarize_numeric=True)
        self.X = self.tree_vect.fit_transform(self.X)
        self.da_vect = DictVectorizer(sparse=False, binarize_numeric=True)
        self.y = self.da_vect.fit_transform(self.y)

        self.num_outputs = len(self.da_vect.get_feature_names())
        self.num_inputs = len(self.tree_vect.get_feature_names())

        # initialize NN classifier
        self._init_neural_network()

    def _init_neural_network(self):
        """Create the neural network for classification, according to the self.nn_shape
        parameter (as set in configuration)."""
        if self.nn_shape.startswith('ff'):
            num_ff_layers = 2
            if self.nn_shape[-1] in ['0', '1', '3', '4']:
                num_ff_layers = int(self.nn_shape[-1])
            layers = self._ff_layers('ff', num_ff_layers)
        self.classif = ClassifNN(layers, [[self.num_inputs]], (T.fmatrix,), normgrad=False)

    def _ff_layers(self, name, num_layers):
        ret = []
        for i in xrange(num_layers):
            ret.append([FeedForward(name + str(i + 1), self.num_hidden_units, T.tanh, self.init)])
        ret.append([FeedForward('output', self.num_outputs, T.nnet.sigmoid, self.init)])
        return ret

    def batches(self):
        for i in xrange(0, len(self.train_order), self.batch_size):
            yield self.train_order[i: i + self.batch_size]

    def _training_pass(self, pass_no):
        """Perform one training pass through the whole training data, print statistics."""

        pass_start_time = time.time()

        log_debug('\n***\nTR %05d:' % pass_no)

        pass_cost = 0
        pass_diff = 0

        for tree_nos in self.batches():

            log_debug('TREE-NOS: ' + str(tree_nos))
            log_debug("\n".join(unicode(self.train_trees[i]) + "\n" + unicode(self.train_das[i])
                                for i in tree_nos))
            log_debug('Y: ' + str(self.y[tree_nos]))

            results = self.classif.classif(self.X[tree_nos])
            cost_gcost = self.classif.update(self.X[tree_nos], self.y[tree_nos], self.alpha)
            bin_result = np.array([[1. if r > 0.5 else 0. for r in result] for result in results])

            log_debug('R: ' + str(bin_result))
            log_debug('COST: %f' % cost_gcost[0])
            log_debug('DIFF: %d' % np.sum(np.abs(self.y[tree_nos] - bin_result)))

            pass_cost += cost_gcost[0]
            pass_diff += np.sum(np.abs(self.y[tree_nos] - bin_result))

        # print and return statistics
        self._print_pass_stats(pass_no, datetime.timedelta(seconds=(time.time() - pass_start_time)),
                               pass_cost, pass_diff)

    def _print_pass_stats(self, pass_no, time, cost, diff):
        log_info('PASS %03d: duration %s, cost %f, diff %d' % (pass_no, str(time), cost, diff))
