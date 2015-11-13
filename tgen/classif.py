#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generating candidate subtrees to enhance the current candidate tree.
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
    """TODO"""

    def __init__(self, cfg):
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        self.nn_shape = cfg.get('nn_shape', 'ff')
        self.num_hidden_units = cfg.get('num_hidden_units', 512)
        self.init = cfg.get('initialization', 'uniform_glorot10')
        self.passes = cfg.get('passes', 200)
        self.randomize = cfg.get('randomize', True)

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
        """TODO"""
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

    def _training_pass(self, pass_no):
        """TODO."""

        pass_start_time = time.time()

        log_debug('\n***\nTR %05d:' % pass_no)

        cost = 0

        for tree_no in self.train_order:

            log_debug('TREE-NO: %d' % tree_no)
            log_debug(str(self.train_trees[tree_no]))
            log_debug(str(self.train_das[tree_no]))
            log_debug('X: ' + str(self.X[tree_no]))
            log_debug('Y: ' + str(self.y[tree_no]))

            result = self.classif.classif([self.X[tree_no]])
            cost_gcost = self.classif.update([self.X[tree_no]], [self.y[tree_no]], 0.1)

            log_debug('R: ' + str(np.array([1 if r > 0.5 else 0 for r in result[0]], dtype=np.float64)))
            log_debug('COST: %f' % cost_gcost[0])

            cost += cost_gcost[0]

        # print and return statistics
        self._print_pass_stats(pass_no, datetime.timedelta(seconds=(time.time() - pass_start_time)), cost)

    def _print_pass_stats(self, pass_no, time, cost):
        log_info('PASS %03d: duration %s, cost %f' % (pass_no, str(time), cost))

