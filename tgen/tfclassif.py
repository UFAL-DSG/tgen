#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classifying trees to determine which DAIs are represented.
"""

from __future__ import unicode_literals
import cPickle as pickle
import time
import datetime
import sys
import re
import math
import tempfile
import shutil
import os

import numpy as np
import tensorflow as tf

from pytreex.core.util import file_stream

from tgen.rnd import rnd
from tgen.logf import log_debug, log_info
from tgen.futil import read_das, read_ttrees, trees_from_doc, tokens_from_doc, \
    tagged_lemmas_from_doc
from tgen.features import Features
from tgen.ml import DictVectorizer
from tgen.embeddings import EmbeddingExtract, TokenEmbeddingSeq2SeqExtract, \
    TaggedLemmasEmbeddingSeq2SeqExtract
from tgen.tree import TreeData
from tgen.data import DA
from tgen.tf_ml import TFModel


class TreeEmbeddingClassifExtract(EmbeddingExtract):
    """Extract t-lemma + formeme embeddings in a row, disregarding syntax"""

    VOID = 0
    UNK_T_LEMMA = 1
    UNK_FORMEME = 2

    def __init__(self, cfg):
        super(TreeEmbeddingClassifExtract, self).__init__()

        self.dict_t_lemma = {'UNK_T_LEMMA': self.UNK_T_LEMMA}
        self.dict_formeme = {'UNK_FORMEME': self.UNK_FORMEME}
        self.max_tree_len = cfg.get('max_tree_len', 25)

    def init_dict(self, train_trees, dict_ord=None):
        """Initialize dictionary, given training trees (store t-lemmas and formemes,
        assign them IDs).

        @param train_das: training DAs
        @param dict_ord: lowest ID to be assigned (if None, it is initialized to MIN_VALID)
        @return: the highest ID assigned + 1 (the current lowest available ID)
        """
        if dict_ord is None:
            dict_ord = self.MIN_VALID

        for tree in train_trees:
            for t_lemma, formeme in tree.nodes:
                if t_lemma not in self.dict_t_lemma:
                    self.dict_t_lemma[t_lemma] = dict_ord
                    dict_ord += 1
                if formeme not in self.dict_formeme:
                    self.dict_formeme[formeme] = dict_ord
                    dict_ord += 1

        return dict_ord

    def get_embeddings(self, tree):
        """Get the embeddings of a sentence (list of word form/tag pairs)."""
        embs = []
        for t_lemma, formeme in tree.nodes[:self.max_tree_len]:
            embs.append(self.dict_formeme.get(formeme, self.UNK_FORMEME))
            embs.append(self.dict_t_lemma.get(t_lemma, self.UNK_T_LEMMA))

        if len(embs) < self.max_tree_len * 2:  # left-pad with void
            embs = [self.VOID] * (self.max_tree_len * 2 - len(embs)) + embs

        return embs

    def get_embeddings_shape(self):
        """Return the shape of the embedding matrix (for one object, disregarding batches)."""
        return [self.max_tree_len * 2]


class RerankingClassifier(TFModel):
    """A classifier for trees that decides which DAIs are currently represented
    (to be used in limiting candidate generator and/or re-scoring the trees)."""

    def __init__(self, cfg):

        super(RerankingClassifier, self).__init__(scope_name='rerank-' +
                                                  cfg.get('scope_suffix', ''))
        self.cfg = cfg
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        self.tree_embs = cfg.get('nn', '').startswith('emb')
        if self.tree_embs:
            self.tree_embs = TreeEmbeddingClassifExtract(cfg)
            self.emb_size = cfg.get('emb_size', 50)
        self.mode = cfg.get('mode', 'tokens' if cfg.get('use_tokens') else 'trees')

        self.nn_shape = cfg.get('nn_shape', 'ff')
        self.num_hidden_units = cfg.get('num_hidden_units', 512)

        self.passes = cfg.get('passes', 200)
        self.min_passes = cfg.get('min_passes', 0)
        self.alpha = cfg.get('alpha', 0.1)
        self.randomize = cfg.get('randomize', True)
        self.batch_size = cfg.get('batch_size', 1)

        self.validation_freq = cfg.get('validation_freq', 10)
        self.max_cores = cfg.get('max_cores')
        self.cur_da = None
        self.cur_da_bin = None
        self.checkpoint_path = None

        self.delex_slots = cfg.get('delex_slots', None)
        if self.delex_slots:
            self.delex_slots = set(self.delex_slots.split(','))

        # Train Summaries
        self.train_summary_dir = cfg.get('tb_summary_dir', None)
        if self.train_summary_dir:
            self.loss_summary_reranker = None
            self.train_summary_op = None
            self.train_summary_writer = None

    def save_to_file(self, model_fname):
        """Save the classifier  to a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph will be stored with a \
            different extension
        """
        model_fname = self.tf_check_filename(model_fname)
        log_info("Saving classifier to %s..." % model_fname)
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.get_all_settings(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        if hasattr(self, 'checkpoint_path') and self.checkpoint_path:
            self.restore_checkpoint()
            shutil.rmtree(os.path.dirname(self.checkpoint_path))
        self.saver.save(self.session, tf_session_fname)

    def get_all_settings(self):
        """Get all settings except the trained model parameters (to be stored in a pickle)."""
        data = {'cfg': self.cfg,
                'da_feats': self.da_feats,
                'da_vect': self.da_vect,
                'tree_embs': self.tree_embs,
                'input_shape': self.input_shape,
                'num_outputs': self.num_outputs, }
        if self.tree_embs:
            data['dict_size'] = self.dict_size
        else:
            data['tree_feats'] = self.tree_feats
            data['tree_vect'] = self.tree_vect
        return data

    def _save_checkpoint(self):
        """Save a checkpoint to a temporary path; set `self.checkpoint_path` to the path
        where it is saved; if called repeatedly, will always overwrite the last checkpoint."""
        if not self.checkpoint_path:
            path = tempfile.mkdtemp(suffix="", prefix="tftreecl-")
            self.checkpoint_path = os.path.join(path, "ckpt")
        log_info('Saving checkpoint to %s' % self.checkpoint_path)
        self.saver.save(self.session, self.checkpoint_path)

    def restore_checkpoint(self):
        if not self.checkpoint_path:
            return
        self.saver.restore(self.session, self.checkpoint_path)

    @staticmethod
    def load_from_file(model_fname):
        """Load the reranker from a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph must be stored with a \
            different extension
        """
        log_info("Loading reranker from %s..." % model_fname)
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            data = pickle.load(fh)
            ret = RerankingClassifier(cfg=data['cfg'])
            ret.load_all_settings(data)

        # re-build TF graph and restore the TF session
        tf_session_fname = os.path.abspath(re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname))
        ret._init_neural_network()
        ret.saver.restore(ret.session, tf_session_fname)
        return ret

    def train(self, das, trees, data_portion=1.0, valid_das=None, valid_trees=None):
        """Run training on the given training data.

        @param das: name of source file with training DAs, or list of DAs
        @param trees: name of source file with corresponding trees/sentences, or list of trees
        @param data_portion: portion of the training data to be used (defaults to 1.0)
        @param valid_das: validation data DAs
        @param valid_trees: list of lists of corresponding paraphrases (same length as valid_das)
        """

        log_info('Training reranking classifier...')

        # initialize training
        self._init_training(das, trees, data_portion)
        if self.mode in ['tokens', 'tagged_lemmas'] and valid_trees is not None:
            valid_trees = [self._tokens_to_flat_trees(paraphrases,
                                                      use_tags=self.mode == 'tagged_lemmas')
                           for paraphrases in valid_trees]

        # start training
        top_comb_cost = float('nan')

        for iter_no in xrange(1, self.passes + 1):
            self.train_order = range(len(self.train_trees))
            if self.randomize:
                rnd.shuffle(self.train_order)
            pass_cost, pass_diff = self._training_pass(iter_no)

            if self.validation_freq and iter_no > self.min_passes and iter_no % self.validation_freq == 0:

                valid_diff = 0
                if valid_das:
                    valid_diff = np.sum([np.sum(self.dist_to_da(d, t))
                                         for d, t in zip(valid_das, valid_trees)])

                # cost combining validation and training data performance
                # (+ "real" cost with negligible weight)
                comb_cost = 1000 * valid_diff + 100 * pass_diff + pass_cost
                log_info('Combined validation cost: %8.3f' % comb_cost)

                # if we have the best model so far, save it as a checkpoint (overwrite previous)
                if math.isnan(top_comb_cost) or comb_cost < top_comb_cost:
                    top_comb_cost = comb_cost
                    self._save_checkpoint()

        # restore last checkpoint (best performance on devel data)
        self.restore_checkpoint()

    def classify(self, trees):
        """Classify the tree -- get DA slot-value pairs and DA type to which the tree
        corresponds (as 1/0 array).
        """
        if self.tree_embs:
            inputs = np.array([self.tree_embs.get_embeddings(tree) for tree in trees])
        else:
            inputs = self.tree_vect.transform([self.tree_feats.get_features(tree, {})
                                               for tree in trees])
        fd = {}
        self._add_inputs_to_feed_dict(inputs, fd)
        results = self.session.run(self.outputs, feed_dict=fd)
        # normalize & binarize the result
        return np.array([[1. if r > 0 else 0. for r in result] for result in results])

    def _normalize_da(self, da):
        if isinstance(da, tuple):  # if DA is actually context + DA, ignore context
            da = da[1]
        if self.delex_slots:  # delexicalize the DA if needed
            da = da.get_delexicalized(self.delex_slots)
        return da

    def init_run(self, da):
        """Remember the current DA for subsequent runs of `dist_to_cur_da`."""
        self.cur_da = self._normalize_da(da)
        da_bin = self.da_vect.transform([self.da_feats.get_features(None, {'da': self.cur_da})])[0]
        self.cur_da_bin = da_bin != 0

    def dist_to_da(self, da, trees):
        """Return Hamming distance of given trees to the given DA.

        @param da: the DA as the base of the Hamming distance measure
        @param trees: list of trees to measure the distance
        @return: list of Hamming distances for each tree
        """
        da = self._normalize_da(da)
        da_bin = self.da_vect.transform([self.da_feats.get_features(None, {'da': da})])[0]
        da_bin = da_bin != 0
        covered = self.classify(trees)
        return [sum(abs(c - da_bin)) for c in covered]

    def dist_to_cur_da(self, trees):
        """Return Hamming distance of given trees to the current DA (set in `init_run`).

        @param trees: list of trees to measure the distance
        @return: list of Hamming distances for each tree
        """
        da_bin = self.cur_da_bin
        covered = self.classify(trees)
        return [sum(abs(c - da_bin)) for c in covered]

    def _init_training(self, das, trees, data_portion):
        """Initialize training.

        Store input data, initialize 1-hot feature representations for input and output and
        transform training data accordingly, initialize the classification neural network.

        @param das: name of source file with training DAs, or list of DAs
        @param trees: name of source file with corresponding trees/sentences, or list of trees
        @param data_portion: portion of the training data to be used (0.0-1.0)
        """
        # read input from files or take it directly from parameters
        if not isinstance(das, list):
            log_info('Reading DAs from ' + das + '...')
            das = read_das(das)
        if not isinstance(trees, list):
            log_info('Reading t-trees from ' + trees + '...')
            ttree_doc = read_ttrees(trees)
            if self.mode == 'tokens':
                tokens = tokens_from_doc(ttree_doc, self.language, self.selector)
                trees = self._tokens_to_flat_trees(tokens)
            elif self.mode == 'tagged_lemmas':
                tls = tagged_lemmas_from_doc(ttree_doc, self.language, self.selector)
                trees = self._tokens_to_flat_trees(tls, use_tags=True)
            else:
                trees = trees_from_doc(ttree_doc, self.language, self.selector)
        elif self.mode in ['tokens', 'tagged_lemmas']:
            trees = self._tokens_to_flat_trees(trees, use_tags=self.mode == 'tagged_lemmas')

        # make training data smaller if necessary
        train_size = int(round(data_portion * len(trees)))
        self.train_trees = trees[:train_size]
        self.train_das = das[:train_size]

        # ignore contexts, if they are contained in the DAs
        if isinstance(self.train_das[0], tuple):
            self.train_das = [da for (context, da) in self.train_das]
        # delexicalize if DAs are lexicalized and we don't want that
        if self.delex_slots:
            self.train_das = [da.get_delexicalized(self.delex_slots) for da in self.train_das]

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
        log_info('Number of binary classes: %d.' % len(self.da_vect.get_feature_names()))

        # initialize I/O shapes
        if not self.tree_embs:
            self.input_shape = list(self.X[0].shape)
        else:
            self.input_shape = self.tree_embs.get_embeddings_shape()
        self.num_outputs = len(self.da_vect.get_feature_names())

        # initialize NN classifier
        self._init_neural_network()
        # initialize the NN variables
        self.session.run(tf.global_variables_initializer())

    def _tokens_to_flat_trees(self, sents, use_tags=False):
        """Use sentences (pairs token-tag) read from Treex files and convert them into flat
        trees (each token has a node right under the root, lemma is the token, formeme is 'x').
        Uses TokenEmbeddingSeq2SeqExtract conversion there and back.

        @param sents: sentences to be converted
        @param use_tags: use tags in the embeddings? (only for lemma-tag pairs in training, \
            not testing)
        @return: a list of flat trees
        """
        tree_embs = (TokenEmbeddingSeq2SeqExtract(cfg=self.cfg)
                     if not use_tags
                     else TaggedLemmasEmbeddingSeq2SeqExtract(cfg=self.cfg))
        tree_embs.init_dict(sents)
        # no postprocessing, i.e. keep lowercasing/plural splitting if set in the configuration
        return [tree_embs.ids_to_tree(tree_embs.get_embeddings(sent), postprocess=False)
                for sent in sents]

    def _init_neural_network(self):
        """Create the neural network for classification, according to the self.nn_shape
        parameter (as set in configuration)."""

        # set TensorFlow random seed
        tf.set_random_seed(rnd.randint(-sys.maxint, sys.maxint))

        self.targets = tf.placeholder(tf.float32, [None, self.num_outputs], name='targets')

        with tf.variable_scope(self.scope_name):

            # feedforward networks
            if self.nn_shape.startswith('ff'):
                self.inputs = tf.placeholder(tf.float32, [None] + self.input_shape, name='inputs')
                num_ff_layers = 2
                if self.nn_shape[-1] in ['0', '1', '3', '4']:
                    num_ff_layers = int(self.nn_shape[-1])
                self.outputs = self._ff_layers('ff', num_ff_layers, self.inputs)

            # RNNs
            elif self.nn_shape.startswith('rnn'):
                self.initial_state = tf.placeholder(tf.float32, [None, self.emb_size])
                self.inputs = [tf.placeholder(tf.int32, [None], name=('enc_inp-%d' % i))
                               for i in xrange(self.input_shape[0])]
                self.cell = tf.contrib.rnn.BasicLSTMCell(self.emb_size)
                self.outputs = self._rnn('rnn', self.inputs)

        # the cost as computed by TF actually adds a "fake" sigmoid layer on top
        # (or is computed as if there were a sigmoid layer on top)
        self.cost = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs, labels=self.targets, name='CE'), 1))

        # NB: this would have been the "true" cost function, if there were a "real" sigmoid layer on top.
        # However, it is not numerically stable in practice, so we have to use the TF function.
        # self.cost = tf.reduce_mean(tf.reduce_sum(self.targets * -tf.log(self.outputs)
        #                                          + (1 - self.targets) * -tf.log(1 - self.outputs), 1))

        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.train_func = self.optimizer.minimize(self.cost)

        # Tensorboard summaries
        if self.train_summary_dir:
            self.loss_summary_reranker = tf.summary.scalar("loss_reranker", self.cost)
            self.train_summary_op = tf.summary.merge([self.loss_summary_reranker])

        # initialize session
        session_config = None
        if self.max_cores:
            session_config = tf.ConfigProto(inter_op_parallelism_threads=self.max_cores,
                                            intra_op_parallelism_threads=self.max_cores)
        self.session = tf.Session(config=session_config)

        # this helps us load/save the model
        self.saver = tf.train.Saver(tf.global_variables())
        if self.train_summary_dir:  # Tensorboard summary writer
            self.train_summary_writer = tf.summary.FileWriter(
                os.path.join(self.train_summary_dir, "reranker"), self.session.graph)

    def _ff_layers(self, name, num_layers, X):
        width = [np.prod(self.input_shape)] + (num_layers * [self.num_hidden_units]) + [self.num_outputs]
        # the last layer should be a sigmoid, but TF simulates it for us in cost computation
        # so the output is "unnormalized sigmoids"
        activ = (num_layers * [tf.tanh]) + [tf.identity]
        Y = X
        for i in xrange(num_layers + 1):
            w = tf.get_variable(name + ('-w%d' % i), (width[i], width[i + 1]),
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable(name + ('-b%d' % i), (width[i + 1],),
                                initializer=tf.constant_initializer())
            Y = activ[i](tf.matmul(Y, w) + b)
        return Y

    def _rnn(self, name, enc_inputs):
        encoder_cell = tf.contrib.rnn.EmbeddingWrapper(self.cell, self.dict_size, self.emb_size)
        encoder_outputs, encoder_state = tf.contrib.rnn.static_rnn(encoder_cell, enc_inputs, dtype=tf.float32)

        # TODO for historical reasons, the last layer uses both output and state.
        # try this just with outputs (might work exactly the same)
        if isinstance(self.cell.state_size, tf.contrib.rnn.LSTMStateTuple):
            state_size = self.cell.state_size.c + self.cell.state_size.h
            final_input = tf.concat(axis=1, values=encoder_state)  # concat c + h
        else:
            state_size = self.cell.state_size
            final_input = encoder_state

        w = tf.get_variable(name + '-w', (state_size, self.num_outputs),
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name + 'b', (self.num_outputs,), initializer=tf.constant_initializer())
        return tf.matmul(final_input, w) + b

    def _batches(self):
        """Create batches from the input; use as iterator."""
        for i in xrange(0, len(self.train_order), self.batch_size):
            yield self.train_order[i: i + self.batch_size]

    def _add_inputs_to_feed_dict(self, inputs, fd):

        if self.nn_shape.startswith('rnn'):
            fd[self.initial_state] = np.zeros([inputs.shape[0], self.emb_size])
            sliced_inputs = np.squeeze(np.array(np.split(np.array([ex for ex in inputs
                                                                   if ex is not None]),
                                                         len(inputs[0]), axis=1)), axis=2)
            for input_, slice_ in zip(self.inputs, sliced_inputs):
                fd[input_] = slice_
        else:
            fd[self.inputs] = inputs

    def _training_pass(self, pass_no):
        """Perform one training pass through the whole training data, print statistics."""

        pass_start_time = time.time()

        log_debug('\n***\nTR %05d:' % pass_no)
        log_debug("Train order: " + str(self.train_order))

        pass_cost = 0
        pass_diff = 0

        for tree_nos in self._batches():

            log_debug('TREE-NOS: ' + str(tree_nos))
            log_debug("\n".join(unicode(self.train_trees[i]) + "\n" + unicode(self.train_das[i])
                                for i in tree_nos))
            log_debug('Y: ' + str(self.y[tree_nos]))

            fd = {self.targets: self.y[tree_nos]}
            self._add_inputs_to_feed_dict(self.X[tree_nos], fd)
            if self.train_summary_dir:  # also compute Tensorboard summaries
                results, cost, _, train_summary_op = self.session.run(
                    [self.outputs, self.cost, self.train_func, self.train_summary_op], feed_dict=fd)
            else:
                results, cost, _ = self.session.run([self.outputs, self.cost, self.train_func],
                                                    feed_dict=fd)
            bin_result = np.array([[1. if r > 0 else 0. for r in result] for result in results])

            log_debug('R: ' + str(bin_result))
            log_debug('COST: %f' % cost)
            log_debug('DIFF: %d' % np.sum(np.abs(self.y[tree_nos] - bin_result)))

            pass_cost += cost
            pass_diff += np.sum(np.abs(self.y[tree_nos] - bin_result))

        # print and return statistics
        self._print_pass_stats(pass_no, datetime.timedelta(seconds=(time.time() - pass_start_time)),
                               pass_cost, pass_diff)
        if self.train_summary_dir:  # Tensorboard: iteration summary
            self.train_summary_writer.add_summary(train_summary_op, pass_no)

        return pass_cost, pass_diff

    def _print_pass_stats(self, pass_no, time, cost, diff):
        log_info('PASS %03d: duration %s, cost %f, diff %d' % (pass_no, str(time), cost, diff))

    def evaluate_file(self, das_file, ttree_file):
        """Evaluate the reranking classifier on a given pair of DA/tree files (show the
        total Hamming distance and total number of DAIs)

        @param das_file: DA file path
        @param ttree_file: trees/sentences file path
        @return: a tuple (total DAIs, distance)
        """
        das = read_das(das_file)
        ttree_doc = read_ttrees(ttree_file)
        if self.mode == 'tokens':
            tokens = tokens_from_doc(ttree_doc, self.language, self.selector)
            trees = self._tokens_to_flat_trees(tokens)
        elif self.mode == 'tagged_lemmas':
            tls = tagged_lemmas_from_doc(ttree_doc, self.language, self.selector)
            trees = self._tokens_to_flat_trees(tls)
        else:
            trees = trees_from_doc(ttree_doc, self.language, self.selector)

        da_len = 0
        dist = 0

        for da, tree in zip(das, trees):
            da_len += len(da)
            dist += self.dist_to_da(da, [tree])[0]

        return da_len, dist
