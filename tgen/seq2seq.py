#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
import tensorflow as tf
from itertools import izip_longest
from tensorflow.models.rnn.seq2seq import embedding_rnn_seq2seq, sequence_loss

from tensorflow.models.rnn import rnn_cell
from tgen.logf import log_info
from tgen.futil import read_das, read_ttrees, trees_from_doc
from tgen.embeddings import EmbeddingExtract
from tgen.rnd import rnd
from tgen.planner import SentencePlanner


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks, from Python Itertools recipes."
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def cut_batch_into_steps(batch):
    """Take a batch (list of examples, which are lists of steps/words themselves), and slice
    it along the other dimension – return a list of steps/words, each containing a numpy array of
    items for the given step for all examples from the batch.
    """
    return np.squeeze(np.split(np.array([ex for ex in batch if ex is not None]),
                               len(batch[0]), axis=1))


class DAEmbeddingSeq2SeqExtract(EmbeddingExtract):

    UNK_SLOT = 0
    UNK_VALUE = 1
    UNK_ACT = 2

    def __init__(self, cfg={}):
        super(DAEmbeddingSeq2SeqExtract, self).__init__()

        self.dict_act = {'UNK_ACT': self.UNK_ACT}
        self.dict_slot = {'UNK_SLOT': self.UNK_SLOT}
        self.dict_value = {'UNK_VALUE': self.UNK_VALUE}
        self.max_da_len = cfg.get('max_da_len', 10)

    def init_dict(self, train_das, dict_ord=None):
        """"""
        if dict_ord is None:
            dict_ord = self.MIN_VALID

        for da in train_das:
            for dai in da:
                if dai.dat not in self.dict_act:
                    self.dict_act[dai.dat] = dict_ord
                    dict_ord += 1
                if dai.name not in self.dict_slot:
                    self.dict_slot[dai.name] = dict_ord
                    dict_ord += 1
                if dai.value not in self.dict_value:
                    self.dict_value[dai.value] = dict_ord
                    dict_ord += 1

        return dict_ord

    def get_embeddings(self, da):
        """"""
        # list the IDs of act types, slots, values
        da_emb_idxs = []
        for dai in da[:self.max_da_len]:
            da_emb_idxs.append(self.dict_slot.get(dai.dat, self.UNK_ACT))
            da_emb_idxs.append(self.dict_slot.get(dai.name, self.UNK_SLOT))
            da_emb_idxs.append(self.dict_value.get(dai.value, self.UNK_VALUE))
        # left-pad with unknown
        padding = []
        if len(da) < self.max_da_len:
            padding = [self.UNK_ACT, self.UNK_SLOT, self.UNK_VALUE] * (self.max_da_len - len(da))
        return padding + da_emb_idxs

    def get_embeddings_shape(self):
        """Return the shape of the embedding matrix (for one object, disregarding batches)."""
        return [3 * self.max_da_len]


class TreeEmbeddingSeq2SeqExtract(EmbeddingExtract):

    UNK_T_LEMMA = 0
    UNK_FORMEME = 1
    BR_OPEN = 2
    BR_CLOSE = 3
    GO = 4
    STOP = 5
    VOID = 6
    MIN_VALID = 7

    def __init__(self, cfg={}):
        super(TreeEmbeddingSeq2SeqExtract, self).__init__()

        self.dict_t_lemma = {'UNK_T_LEMMA': self.UNK_T_LEMMA}
        self.dict_formeme = {'UNK_FORMEME': self.UNK_FORMEME}
        self.max_tree_len = cfg.get('max_da_len', 25)

    def init_dict(self, train_trees, dict_ord=None):
        """"""
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

    def _get_subtree_embeddings(self, tree, root_idx):
        """Bracketed-style embeddings for a projective tree."""
        embs = [self.BR_OPEN]

        for left_child_idx in tree.children_idxs(root_idx, left_only=True):
            embs.extend(self._get_subtree_embeddings(tree, left_child_idx))

        embs.extend([self.dict_t_lemma.get(tree[root_idx].t_lemma, self.UNK_T_LEMMA),
                     self.dict_formeme.get(tree[root_idx].formeme, self.UNK_FORMEME)])

        for right_child_idx in tree.children_idxs(root_idx, right_only=True):
            embs.extend(self._get_subtree_embeddings(tree, right_child_idx))

        embs.append(self.BR_CLOSE)
        return embs

    def get_embeddings(self, tree):

        # get tree embeddings recursively
        tree_emb_idxs = [self.GO] + self._get_subtree_embeddings(tree, 0) + [self.STOP]

        # right-pad with unknown
        shape = self.get_embeddings_shape()[0]
        padding = [self.VOID] * (shape - len(tree_emb_idxs))

        return tree_emb_idxs + padding

    def get_embeddings_shape(self):
        """Return the shape of the embedding matrix (for one object, disregarding batches)."""
        return [4 * self.max_tree_len + 2]


class Seq2SeqGen(SentencePlanner):

    def __init__(self, cfg):

        super(Seq2SeqGen, self).__init__(cfg)

        # TODO fix configuration
        self.emb_size = cfg.get('emb_size', 50)
        self.batch_size = cfg.get('batch_size', 10)
        self.randomize = True

    def _init_training(self, das_file, ttree_file, data_portion):
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
        log_info('Using %d training instances.' % train_size)

        # initialize embeddings
        self.da_embs = DAEmbeddingSeq2SeqExtract()
        self.tree_embs = TreeEmbeddingSeq2SeqExtract()
        self.da_dict_size = self.da_embs.init_dict(self.train_das)
        self.tree_dict_size = self.tree_embs.init_dict(self.train_trees)

        self.max_tree_len = self.tree_embs.get_embeddings_shape()[0]
        self.max_da_len = self.da_embs.get_embeddings_shape()[0]

        # prepare training batches
        self.train_enc = [cut_batch_into_steps(b)
                          for b in grouper([self.da_embs.get_embeddings(da)
                                            for da in self.train_das],
                                           self.batch_size, None)]
        self.train_dec = [cut_batch_into_steps(b)
                          for b in grouper([self.tree_embs.get_embeddings(tree)
                                            for tree in self.train_trees],
                                           self.batch_size, None)]

        # initialize the NN
        self._init_neural_network()

    def _init_neural_network(self):

        # create placeholders for input & output (always batch-size * 1, list of up to num. steps)
        self.enc_inputs = []
        for i in xrange(self.max_da_len):
            self.enc_inputs.append(tf.placeholder(tf.int32, [None], name=('enc_inp-%d' % i)))
        self.dec_inputs = []
        for i in xrange(self.max_tree_len):
            self.dec_inputs.append(tf.placeholder(tf.int32, [None], name=('dec_inp-%d' % i)))

        # targets are just decoder inputs shifted by one
        self.targets = [self.dec_inputs[i + 1] for i in xrange(len(self.dec_inputs) - 1)]

        # prepare building blocks
        # TODO change dimension when BasicLSTMCell is replaced
        self.initial_state = tf.placeholder(tf.float32, [None, self.emb_size])
        self.cell = rnn_cell.BasicLSTMCell(self.emb_size)

        # build the actual LSTM Seq2Seq network

        # outputs = batch_size * num_decoder_symbols ~ i.e. output logits at each steps
        # states = cell states at each steps
        self.outputs, self.states = embedding_rnn_seq2seq(
            self.enc_inputs, self.dec_inputs, self.cell,
            self.da_dict_size, self.tree_dict_size)

        # weights
        self.cost_weights = tf.ones_like(self.targets, tf.float32, name='cost_weights')

        # cost
        self.cost = sequence_loss(self.outputs, self.targets, self.cost_weights, self.tree_dict_size)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_func = self.optimizer.minimize(self.cost)

        # initialize session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def _training_pass(self, iter_no):

        for batch_no in self.train_order:

            initial_state = np.zeros([self.batch_size, self.emb_size])

            feed_dict = {self.enc_inputs: self.train_enc[batch_no],
                         self.dec_inputs: self.train_dec[batch_no],
                         self.initial_state: initial_state}

            _, cost = self.session.run([self.train_func, self.cost], feed_dict=feed_dict)

            log_info('It %d Batch %d -- cost: %8.5f' % (iter_no, batch_no, cost))

    def train(self, das_file, ttree_file, data_portion=1.0):

        # load and prepare data and initialize the neural network
        self._init_training(das_file, ttree_file, data_portion)

        # do the training passes
        # TODO: better stopping criterion than just # of passes
        # look at the tagger – they're much better -- using performance on development data
        for iter_no in xrange(1, self.passes + 1):

            self.train_order = range(len(self.train_enc))
            if self.randomize:
                rnd.shuffle(self.train_order)

            self._training_pass(iter_no)

    def save_to_file(self, fname):
        raise NotImplementedError()

    def generate_tree(self, inputs):
        raise NotImplementedError()
