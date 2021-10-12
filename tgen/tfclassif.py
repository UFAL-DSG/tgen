#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classifying trees to determine which DAIs are represented.
"""

from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import zip
from builtins import range
import pickle as pickle
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

from tgen.rnd import rnd
from tgen.logf import log_debug, log_info
from tgen.futil import read_das, file_stream, read_trees_or_tokens
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


class Reranker(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')

        self.mode = cfg.get('mode', 'tokens' if cfg.get('use_tokens') else 'trees')

        self.da_feats = Features(['dat: dat_presence', 'svp: svp_presence'])
        self.da_vect = DictVectorizer(sparse=False, binarize_numeric=True)

        self.cur_da = None
        self.cur_da_bin = None

        self.delex_slots = cfg.get('delex_slots', None)
        if self.delex_slots:
            self.delex_slots = set(self.delex_slots.split(','))

    @staticmethod
    def get_model_type(cfg):
        """Return the correct model class according to the config."""
        if cfg.get('model') == 'e2e_patterns':
            from tgen.e2e.slot_error import E2EPatternClassifier
            return E2EPatternClassifier
        return RerankingClassifier

    @staticmethod
    def load_from_file(reranker_fname):
        """Detect correct model type and start loading."""
        model_type = RerankingClassifier  # default to classifier
        with file_stream(reranker_fname, 'rb', encoding=None) as fh:
            data = pickle.load(fh)
            if isinstance(data, type):
                from tgen.e2e.slot_error import E2EPatternClassifier
                model_type = data
        return model_type.load_from_file(reranker_fname)

    def save_to_file(self, reranker_fname):
        raise NotImplementedError()

    def get_all_settings(self):
        raise NotImplementedError()

    def classify(self, trees):
        raise NotImplementedError()

    def train(self, das, trees, data_portion=1.0, valid_das=None, valid_trees=None):
        raise NotImplementedError()

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

    def dist_to_da(self, da, trees, return_classif=False):
        """Return Hamming distance of given trees to the given DA.

        @param da: the DA as the base of the Hamming distance measure
        @param trees: list of trees to measure the distance
        @return: list of Hamming distances for each tree (+ resulting classification if return_classif)
        """
        self.init_run(da)
        ret = self.dist_to_cur_da(trees, return_classif)
        self.cur_da = None
        self.cur_da_bin = None
        return ret

    def dist_to_cur_da(self, trees, return_classif=False):
        """Return Hamming distance of given trees to the current DA (set in `init_run`).

        @param trees: list of trees to measure the distance
        @return: list of Hamming distances for each tree (+ resulting classification if return_classif)
        """
        da_bin = self.cur_da_bin
        covered = self.classify(trees)
        dist = [sum(abs(c - da_bin)) for c in covered]
        if return_classif:
            return dist, [[f for f, c_ in zip(self.da_vect.feature_names_, c) if c_] for c in covered]
        return dist


class RerankingClassifier(Reranker, TFModel):
    """A classifier for trees that decides which DAIs are currently represented
    (to be used in limiting candidate generator and/or re-scoring the trees)."""

    def __init__(self, cfg):

        Reranker.__init__(self, cfg)
        TFModel.__init__(self, scope_name='rerank-' + cfg.get('scope_suffix', ''))

        self.tree_embs = cfg.get('nn', '').startswith('emb')
        if self.tree_embs:
            self.tree_embs = TreeEmbeddingClassifExtract(cfg)
            self.emb_size = cfg.get('emb_size', 50)

        self.nn_shape = cfg.get('nn_shape', 'ff')
        self.num_hidden_units = cfg.get('num_hidden_units', 512)

        self.passes = cfg.get('passes', 200)
        self.min_passes = cfg.get('min_passes', 0)
        self.alpha = cfg.get('alpha', 0.1)
        self.randomize = cfg.get('randomize', True)
        self.batch_size = cfg.get('batch_size', 1)

        self.validation_freq = cfg.get('validation_freq', 10)
        self.checkpoint_path = None
        self.max_cores = cfg.get('max_cores')

        # Train Summaries
        self.train_summary_dir = cfg.get('tb_summary_dir', None)
        if self.train_summary_dir:
            self.loss_summary_reranker = None
            self.train_summary_op = None
            self.train_summary_writer = None

        # backward compatibility flag -- will be 1 when loading older models
        self.version = 2

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
                'num_outputs': self.num_outputs,
                'version': self.version, }
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
            if 'version' not in data:
                data['version'] = 1
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

        for iter_no in range(1, self.passes + 1):
            self.train_order = list(range(len(self.train_trees)))
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
            log_info('Reading t-trees/tokens from ' + trees + '...')
            trees = read_trees_or_tokens(trees, self.mode, self.language, self.selector)

        if self.mode in ['tokens', 'tagged_lemmas']:
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

        self.train_order = list(range(len(self.train_trees)))
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
        self.session.run(tf.compat.v1.global_variables_initializer())

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
        tf.set_random_seed(rnd.randint(-sys.maxsize, sys.maxsize))

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
            elif self.nn_shape.endswith('rnn'):
                self.initial_state = tf.placeholder(tf.float32, [None, self.emb_size])
                self.inputs = [tf.placeholder(tf.int32, [None], name=('enc_inp-%d' % i))
                               for i in range(self.input_shape[0])]
                self.cell = tf.contrib.rnn.BasicLSTMCell(self.emb_size)
                self.outputs = self._rnn('rnn', self.inputs, bidi=self.nn_shape.startswith('bidi'))

        # older versions of the model put the optimizer into the default scope -- we want them in a separate scope
        # (to be able to swap rerankers with the same main generator), but want to keep loading older models
        # -> version setting decides where the variables will be created
        with tf.variable_scope(self.scope_name if self.version > 1 else tf.get_variable_scope()):
            # the cost as computed by TF actually adds a "fake" sigmoid layer on top
            # (or is computed as if there were a sigmoid layer on top)
            self.cost = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs, labels=self.targets, name='CE'), 1))

            # NB: this would have been the "true" cost function, if there were a "real" sigmoid layer on top.
            # However, it is not numerically stable in practice, so we have to use the TF function.
            # self.cost = tf.reduce_mean(tf.reduce_sum(self.targets * -tf.log(self.outputs)
            #                                          + (1 - self.targets) * -tf.log(1 - self.outputs), 1))

            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.alpha)
            self.train_func = self.optimizer.minimize(self.cost)

        # Tensorboard summaries
        if self.train_summary_dir:
            self.loss_summary_reranker = tf.summary.scalar("loss_reranker", self.cost)
            self.train_summary_op = tf.summary.merge([self.loss_summary_reranker])

        # initialize session
        session_config = None
        if self.max_cores:
            session_config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=self.max_cores,
                                            intra_op_parallelism_threads=self.max_cores)
        self.session = tf.compat.v1.Session(config=session_config)

        # this helps us load/save the model
        self.saver = tf.compat.v1.train.Saver(tf.global_variables())
        if self.train_summary_dir:  # Tensorboard summary writer
            self.train_summary_writer = tf.summary.FileWriter(
                os.path.join(self.train_summary_dir, "reranker"), self.session.graph)

    def _ff_layers(self, name, num_layers, X):
        width = [np.prod(self.input_shape)] + (num_layers * [self.num_hidden_units]) + [self.num_outputs]
        # the last layer should be a sigmoid, but TF simulates it for us in cost computation
        # so the output is "unnormalized sigmoids"
        activ = (num_layers * [tf.tanh]) + [tf.identity]
        Y = X
        for i in range(num_layers + 1):
            w = tf.get_variable(name + ('-w%d' % i), (width[i], width[i + 1]),
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable(name + ('-b%d' % i), (width[i + 1],),
                                initializer=tf.constant_initializer())
            Y = activ[i](tf.matmul(Y, w) + b)
        return Y

    def _rnn(self, name, enc_inputs, bidi=False):
        cell = tf.contrib.rnn.EmbeddingWrapper(self.cell, self.dict_size, self.emb_size)
        if bidi:
            _, state_fw, state_bw = tf.nn.static_bidirectional_rnn(cell, cell, enc_inputs, dtype=tf.float32)
            if isinstance(state_fw, tuple):  # add up LSTM states part-by-part
                enc_state = (state_fw[0] + state_bw[0], state_fw[1] + state_bw[1])
            else:
                enc_state = state_fw + state_bw
        else:
            _, enc_state = tf.nn.static_rnn(cell, enc_inputs, dtype=tf.float32)

        if isinstance(enc_state, tuple):
            state_size = self.cell.state_size.c + self.cell.state_size.h
            enc_state = tf.concat(axis=1, values=enc_state)  # concat c + h
        else:
            state_size = self.cell.state_size

        w = tf.get_variable(name + '-w', (state_size, self.num_outputs),
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name + 'b', (self.num_outputs,), initializer=tf.constant_initializer())
        return tf.matmul(enc_state, w) + b

    def _batches(self):
        """Create batches from the input; use as iterator."""
        for i in range(0, len(self.train_order), self.batch_size):
            yield self.train_order[i: i + self.batch_size]

    def _add_inputs_to_feed_dict(self, inputs, fd):

        if self.nn_shape.endswith('rnn'):
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
            log_debug("\n".join(str(self.train_trees[i]) + "\n" + str(self.train_das[i])
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
        log_info('Reading DAs from ' + das_file + '...')
        das = read_das(das_file)
        log_info('Reading t-trees/tokens from ' + ttree_file + '...')
        trees = read_trees_or_tokens(ttree_file, self.mode, self.language, self.selector)
        if self.mode in ['tokens', 'tagged_lemmas']:
            trees = self._tokens_to_flat_trees(trees, use_tags=self.mode == 'tagged_lemmas')

        tot_len = 0
        tot_dist = 0
        classif_das = []
        for da, tree in zip(das, trees):
            tot_len += len(da)
            dist, classif = self.dist_to_da(da, [tree], return_classif=True)
            tot_dist += dist[0]
            classif_das.append(DA.parse_features(classif[0]))

        return tot_len, tot_dist, classif_das
