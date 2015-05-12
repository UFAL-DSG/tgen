#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Candidate tree rankers (NN).

"""

from __future__ import unicode_literals

import theano.tensor as T
import numpy as np

from tgen.nn import FeedForwardLayer, ConcatLayer, MaxPool1DLayer, Embedding, NN
from tgen.rank import BasePerceptronRanker, FeaturesPerceptronRanker
from tgen.logf import log_debug, log_info


class SimpleNNRanker(FeaturesPerceptronRanker):
    """A simple ranker using a neural network on top of the usual features; using the same
    updates as the original perceptron as far as possible."""

    def __init__(self, cfg):
        super(SimpleNNRanker, self).__init__(cfg)
        self.num_hidden_units = cfg.get('num_hidden_units', 512)
        self.initialization = cfg.get('initialization', 'uniform_glorot10')
        self.net_type = cfg.get('nn', 'linear_perc')

    def _init_training(self, das_file, ttree_file, data_portion):
        # load data, determine number of features etc. etc.
        super(SimpleNNRanker, self)._init_training(das_file, ttree_file, data_portion)

        self._init_neural_network()

        self.w_after_iter = []
        self.update_weights_sum()

        log_debug('\n***\nINIT:')
        log_debug(self._feat_val_str())
        log_info('Training ...')

    def _score(self, cand_feats):
        return self.nn.score(cand_feats)[0]

    def _init_neural_network(self):
        # multi-layer perceptron with tanh + linear layer
        if self.net_type == 'mlp':
            self.nn = NN([[FeedForwardLayer('hidden', self.train_feats.shape[1], self.num_hidden_units,
                                            T.tanh, self.initialization)],
                          [FeedForwardLayer('output', self.num_hidden_units, 1,
                                            None, self.initialization)]])
        # linear perceptron
        else:
            self.nn = NN([[FeedForwardLayer('perc', self.train_feats.shape[1], 1,
                                            None, self.initialization)]])

    def _update_weights(self, da, good_tree, bad_tree, good_feats, bad_feats):
        if self.diffing_trees:
            good_sts, bad_sts = good_tree.diffing_trees(bad_tree, symmetric=True)
            for good_st, bad_st in zip(good_sts, bad_sts):
                good_feats = self._extract_feats(good_st, da)
                bad_feats = self._extract_feats(bad_st, da)
                subtree_w = 1
                if self.diffing_trees.endswith('weighted'):
                    subtree_w = (len(good_st) + len(bad_st)) / float(len(good_tree) + len(bad_tree))
                self.nn.update(bad_feats, good_feats, subtree_w * self.alpha)
        else:
            self.nn.update(bad_feats, good_feats, self.alpha)

    def get_weights(self):
        """Return the current neural net weights."""
        return self.nn.get_param_values()

    def set_weights(self, w):
        """Set new neural network weights."""
        self.nn.set_param_values(w)

    def set_weights_average(self, wss):
        """Set the weights as the average of the given array of weights (used in parallel training)."""
        self.nn.set_param_values(np.average(wss, axis=0))

    def store_iter_weights(self):
        """Remember the current weights to be used for averaged perceptron."""
        self.w_after_iter.append(self.nn.get_param_values())

    def set_weights_iter_average(self):
        """Average the remembered weights."""
        self.nn.set_param_values(np.average(self.w_after_iter, axis=0))

    def get_weights_sum(self):
        """Return the sum of weights (at start of current iteration) to be used to weigh future
        promise."""
        return self.w_sum

    def update_weights_sum(self):
        """Update the current weights sum figure."""
        vals = self.nn.get_param_values()
        # only use the last layer for summation (w, b)
        self.w_sum = np.sum(vals[-2]) + np.sum(vals[-1])

#     def __getstate__(self):
#         state = dict(self.__dict__)
#         w = self.nn.get_param_values()
#         del state['nn']
#         state['w'] = w
#         return state
#
#     def __setstate__(self, state):
#         if 'w' in state:
#             w = state['w']
#             del state['w']
#         self.__dict__.update(state)
#         if 'w' in state:
#             self._init_neural_network()
#             self.set_weights(w)


class EmbNNRanker(BasePerceptronRanker):

    UNK_SLOT = 0
    UNK_VALUE = 1
    UNK_T_LEMMA = 2
    UNK_FORMEME = 3
    MIN_VALID = 4

    def __init__(self, cfg):
        super(EmbNNRanker, self).__init__(cfg)
        self.num_hidden_units = cfg.get('num_hidden_units', 512)
        self.initialization = cfg.get('initialization', 'uniform_glorot10')
        self.emb_size = cfg.get('emb_size', 20)

        self.dict_slot = {'UNK_SLOT': self.UNK_SLOT}
        self.dict_value = {'UNK_VALUE': self.UNK_VALUE}
        self.dict_t_lemma = {'UNK_T_LEMMA': self.UNK_T_LEMMA}
        self.dict_formeme = {'UNK_FORMEME': self.UNK_FORMEME}

        self.max_da_len = cfg.get('max_da_len', 10)
        self.max_tree_len = cfg.get('max_tree_len', 20)

    def _init_training(self, das_file, ttree_file, data_portion):
        super(EmbNNRanker, self)._init_training(das_file, ttree_file, data_portion)
        self._init_dict()
        self._init_neural_network()

        self.train_feats = [self._extract_feats(tree, da)
                            for tree, da in zip(self.train_trees, self.train_das)]

        self.w_after_iter = []
        self.update_weights_sum()

    def _init_dict(self):
        """Initialize word -> integer dictionaries, starting from a minimum
        valid value, always adding a new integer to unknown values to prevent
        clashes among different types of inputs."""
        dict_ord = self.MIN_VALID

        for da in self.train_das:
            for dai in da:
                if dai.name not in self.dict_slot:
                    self.dict_slot[dai.name] = dict_ord
                    dict_ord += 1
                if dai.value not in self.dict_value:
                    self.dict_value[dai.value] = dict_ord
                    dict_ord += 1

        for tree in self.train_trees:
            for t_lemma, formeme in tree.nodes:
                if t_lemma not in self.dict_t_lemma:
                    self.dict_t_lemma[t_lemma] = dict_ord
                    dict_ord += 1
                if formeme not in self.dict_formeme:
                    self.dict_formeme[formeme] = dict_ord
                    dict_ord += 1

        self.dict_size = dict_ord

    def _score(self, cand_embs):
        return self.nn.score(*cand_embs)[0]

    def _extract_feats(self, tree, da):

        # DA embeddings
        da_emb_idxs = []
        for dai in da[:self.max_da_len]:
            da_emb_idxs.append(self.dict_slot.get(dai.name, self.UNK_SLOT))
            da_emb_idxs.append(self.dict_value.get(dai.name, self.UNK_VALUE))

        # pad with "unknown"
        for _ in xrange(len(da_emb_idxs) / 2, self.max_da_len):
            da_emb_idxs.extend([self.UNK_SLOT, self.UNK_VALUE])

        # tree embeddings
        tree_emb_idxs = []
        for parent_ord, (t_lemma, formeme) in zip(tree.parents[1:self.max_tree_len + 1],
                                                  tree.nodes[1:self.max_tree_len + 1]):
            tree_emb_idxs.append(self.dict_t_lemma.get(tree.nodes[parent_ord].t_lemma,
                                                       self.UNK_T_LEMMA))
            tree_emb_idxs.append(self.dict_formeme.get(formeme, self.UNK_FORMEME))
            tree_emb_idxs.append(self.dict_t_lemma.get(t_lemma, self.UNK_T_LEMMA))

        # pad with unknown
        for _ in xrange(len(tree_emb_idxs) / 3, self.max_tree_len):
            tree_emb_idxs.extend([self.UNK_T_LEMMA, self.UNK_FORMEME, self.UNK_T_LEMMA])

        return (da_emb_idxs, tree_emb_idxs)

    def _init_neural_network(self):
        self.nn = NN([[Embedding('emb_das', self.dict_size, self.emb_size, 'uniform_005'),
                       Embedding('emb_trees', self.dict_size, self.emb_size, 'uniform_005')],
                      [MaxPool1DLayer('mp_das', self.max_da_len),
                       MaxPool1DLayer('mp_trees', self.max_tree_len)],
                      [ConcatLayer('concat')],
                      [FeedForwardLayer('ff1', self.emb_size * 2, self.num_hidden_units,
                                        T.tanh, self.initialization)],
                      [FeedForwardLayer('ff2', self.num_hidden_units, self.num_hidden_units,
                                        T.tanh, self.initialization)],
                      [FeedForwardLayer('perc', self.num_hidden_units, 1,
                                        None, self.initialization)]],
                     input_num=2,
                     input_type=T.ivector)

    def update_weights_sum(self):
        """Update the current weights sum figure."""
        vals = self.nn.get_param_values()
        # only use the last layer for summation (w, b)
        self.w_sum = np.sum(vals[-2]) + np.sum(vals[-1])

