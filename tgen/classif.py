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


from pytreex.core.util import file_stream

from tgen.rnd import rnd
from tgen.logf import log_debug, log_info, log_warn
from tgen.futil import read_das, read_ttrees, trees_from_doc
from tgen.features import Features
from tgen.ml import DictVectorizer
from tgen.nn import ClassifNN, FeedForward, Flatten, Conv1D, Pool1D, Embedding
from tgen.embeddings import TreeEmbeddingExtract
from tgen.tree import TreeData
from tgen.data import DA

class TreeClassifier(object):
    """A classifier for trees that decides which DAIs are currently represented
    (to be used in limiting candidate generator and/or re-scoring the trees)."""

    def __init__(self, cfg):
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        self.tree_embs = cfg.get('nn', '').startswith('emb')
        if self.tree_embs:
            self.tree_embs = TreeEmbeddingExtract(cfg)
            self.emb_size = cfg.get('emb_size', 20)

        self.nn_shape = cfg.get('nn_shape', 'ff')
        self.num_hidden_units = cfg.get('num_hidden_units', 512)
        self.cnn_num_filters = cfg.get('cnn_num_filters', 3)
        self.cnn_filter_length = cfg.get('cnn_filter_length', 3)
        self.init = cfg.get('initialization', 'uniform_glorot10')

        self.passes = cfg.get('passes', 200)
        self.alpha = cfg.get('alpha', 0.1)
        self.randomize = cfg.get('randomize', True)
        self.batch_size = cfg.get('batch_size', 1)

        self.cur_da = None
        self.cur_da_bin = None

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
            self._training_pass(iter_no)

    def classify(self, trees):
        """Classify the tree -- get DA slot-value pairs and DA type to which the tree
        corresponds (as 1/0 array).

        This does not have a lot of practical use here, see is_subset_of_da.
        """
        if self.tree_embs:
            X = np.array([self.tree_embs.get_embeddings(tree) for tree in trees])
        else:
            X = self.tree_vect.transform([self.tree_feats.get_features(tree, {}) for tree in trees])
        # binarize the result
        return np.array([[1. if r > 0.5 else 0. for r in result]
                         for result in self.classif.classif(X)])

    def is_subset_of_da(self, da, trees):
        """Given a DA and an array of trees, this gives a boolean array indicating which
        trees currently cover/describe a subset of the DA.

        @param da: the input DA against which the trees should be tested
        @param trees: the trees to test against the DA
        @return: boolean array, with True where the tree covers/describes a subset of the DA
        """
        # get 1-hot representation of the DA
        da_bin = self.da_vect.transform([self.da_feats.get_features(None, {'da': da})])[0]
        # convert it to array of booleans
        da_bin = da_bin != 0
        # classify the trees
        covered = self.classify(trees)
        # decide whether 1's in their 1-hot vectors are subsets of True's in da_bin
        return [((c != 0) | da_bin == da_bin).all() for c in covered]

    def init_run(self, da):
        """Remember the current DA for subsequent runs of `is_subset_of_cur_da`."""
        self.cur_da = da
        da_bin = self.da_vect.transform([self.da_feats.get_features(None, {'da': da})])[0]
        self.cur_da_bin = da_bin != 0

    def is_subset_of_cur_da(self, trees):
        """Same as `is_subset_of_da`, but using `self.cur_da` set via `init_run`."""
        da_bin = self.cur_da_bin
        covered = self.classify(trees)
        return [((c != 0) | da_bin == da_bin).all() for c in covered]

    def corresponds_to_cur_da(self, trees):
        """Given an array of trees, this gives a boolean array indicating which
        trees currently cover exactly the current DA (set via `init_run`).

        @param trees: the trees to test against the current DA
        @return: boolean array, with True where the tree covers/describes a subset of the current DA
        """
        da_bin = self.cur_da_bin
        covered = self.classify(trees)
        return [((c != 0) == da_bin).all() for c in covered]

    def _init_training(self, das_file, ttree_file, data_portion):
        """Initialize training.

        Store input data, initialize 1-hot feature representations for input and output and
        transform training data accordingly, initialize the classification neural network.
        """
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

        # add empty tree + empty DA to training data
        # (i.e. forbid the network to keep any of its outputs "always-on")
        train_size += 1
        self.train_trees.append(TreeData())
        empty_da = DA.parse('inform()')
        self.train_das.append(empty_da)

        self.train_order = range(len(self.train_trees))
        log_info('Using %d training instances.' % train_size)

        # initialize input features/embeddings
        if self.tree_embs:
            self.dict_size = self.tree_embs.init_dict(self.train_trees)
            self.X = np.array([self.tree_embs.get_embeddings(tree) for tree in self.train_trees])
        else:
            self.tree_feats = Features(['node: presence t_lemma formeme'])
            self.tree_vect = DictVectorizer(sparse=False, binarize_numeric=True)
            self.X = [self.tree_feats.get_features(tree, {}) for tree in self.train_trees]
            self.X = self.tree_vect.fit_transform(self.X)

        # initialize output features
        self.da_feats = Features(['dat: dat_presence', 'svp: svp_presence'])
        self.da_vect = DictVectorizer(sparse=False, binarize_numeric=True)
        self.y = [self.da_feats.get_features(None, {'da': da}) for da in self.train_das]
        self.y = self.da_vect.fit_transform(self.y)

        # initialize I/O shapes
        self.input_shape = [list(self.X[0].shape)]
        self.num_outputs = len(self.da_vect.get_feature_names())

        # initialize NN classifier
        self._init_neural_network()

    def _init_neural_network(self):
        """Create the neural network for classification, according to the self.nn_shape
        parameter (as set in configuration)."""
        layers = []
        if self.tree_embs:
            layers.append([Embedding('emb', self.dict_size, self.emb_size, 'uniform_005')])

        # feedforward networks
        if self.nn_shape.startswith('ff'):
            if self.tree_embs:
                layers.append([Flatten('flat')])
            num_ff_layers = 2
            if self.nn_shape[-1] in ['0', '1', '3', '4']:
                num_ff_layers = int(self.nn_shape[-1])
            layers += self._ff_layers('ff', num_ff_layers)

        # convolutional networks
        elif 'conv' in self.nn_shape or 'pool' in self.nn_shape:
            assert self.tree_embs  # convolution makes no sense without embeddings
            num_conv = 0
            if 'conv' in self.nn_shape:
                num_conv = 1
            if 'conv2' in self.nn_shape:
                num_conv = 2
            pooling = None
            if 'maxpool' in self.nn_shape:
                pooling = T.max
            elif 'avgpool' in self.nn_shape:
                pooling = T.mean
            layers += self._conv_layers('conv', num_conv, pooling)
            layers.append([Flatten('flat')])
            layers += self._ff_layers('ff', 1)

        # input types: integer 3D for tree embeddings (batch + 2D embeddings),
        #              float 2D (matrix) for binary input (batch + features)
        input_types = (T.itensor3,) if self.tree_embs else (T.fmatrix,)

        # create the network, connect layers
        self.classif = ClassifNN(layers, self.input_shape, input_types, normgrad=False)
        log_info("Network shape:\n\n" + str(self.classif))

    def _ff_layers(self, name, num_layers):
        ret = []
        for i in xrange(num_layers):
            ret.append([FeedForward(name + str(i + 1), self.num_hidden_units, T.tanh, self.init)])
        ret.append([FeedForward('output', self.num_outputs, T.nnet.sigmoid, self.init)])
        return ret

    def _conv_layers(self, name, num_layers=1, pooling=None):
        ret = []
        for i in xrange(num_layers):
            ret.append([Conv1D(name + str(i + 1),
                               filter_length=self.cnn_filter_length,
                               num_filters=self.cnn_num_filters,
                               init=self.init, activation=T.tanh)])
        if pooling is not None:
            ret.append([Pool1D(name + str(i + 1) + 'pool', pooling_func=pooling)])
        return ret

    def batches(self):
        for i in xrange(0, len(self.train_order), self.batch_size):
            yield self.train_order[i: i + self.batch_size]

    def _training_pass(self, pass_no):
        """Perform one training pass through the whole training data, print statistics."""

        pass_start_time = time.time()

        log_debug('\n***\nTR %05d:' % pass_no)
        log_debug("Train order: " + str(self.train_order))

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

