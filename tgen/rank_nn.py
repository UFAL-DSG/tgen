#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Candidate tree rankers (NN).

"""

from __future__ import unicode_literals

import theano.tensor as T
import numpy as np

from tgen.nn import FeedForwardLayer, Concat, Flatten, MaxPool1DLayer, Embedding, NN, DotProduct, \
    Conv1DLayer
from tgen.rank import BasePerceptronRanker, FeaturesPerceptronRanker
from tgen.logf import log_debug, log_info


class NNRanker(BasePerceptronRanker):
    """Abstract ancestor of NN rankers."""

    def store_iter_weights(self):
        """Remember the current weights to be used for averaged perceptron."""
        self.w_after_iter.append(self.nn.get_param_values())

    def update_weights_sum(self):
        """Update the current weights sum figure."""
        vals = self.nn.get_param_values()
        # only use the last layer for summation (w, b)
        self.w_sum = np.sum(vals[-2]) + np.sum(vals[-1])

    def get_weights_sum(self):
        """Return the sum of weights (at start of current iteration) to be used to weigh future
        promise."""
        return self.w_sum

    def get_weights(self):
        """Return the current neural net weights."""
        return self.nn.get_param_values()

    def set_weights(self, w):
        """Set new neural network weights."""
        self.nn.set_param_values(w)

    def set_weights_average(self, wss):
        """Set the weights as the average of the given array of weights (used in parallel training)."""
        self.nn.set_param_values(np.average(wss, axis=0))

    def set_weights_iter_average(self):
        """Average the remembered weights."""
        self.nn.set_param_values(np.average(self.w_after_iter, axis=0))

    def _update_weights(self, good_da, bad_da, good_tree, bad_tree, good_feats, bad_feats):
        """Update NN weights, given a DA, a good and a bad tree, and their features."""
        # import ipdb; ipdb.set_trace()
        if self.diffing_trees:
            good_sts, bad_sts = good_tree.diffing_trees(bad_tree, symmetric=True)
            for good_st, bad_st in zip(good_sts, bad_sts):
                good_feats = self._extract_feats(good_st, good_da)
                bad_feats = self._extract_feats(bad_st, bad_da)
                subtree_w = 1
                if self.diffing_trees.endswith('weighted'):
                    subtree_w = (len(good_st) + len(bad_st)) / float(len(good_tree) + len(bad_tree))
                self._update_nn(bad_feats, good_feats, subtree_w * self.alpha)
        else:
            self._update_nn(bad_feats, good_feats, self.alpha)

    def _update_nn(self, bad_feats, good_feats, rate):
        """Direct call to NN weights update."""
        self.nn.update(bad_feats, good_feats, rate)


class SimpleNNRanker(FeaturesPerceptronRanker, NNRanker):
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
                          [FeedForwardLayer('hidden2', self.num_hidden_units, self.num_hidden_units,
                                            T.tanh, self.initialization)],
                          [FeedForwardLayer('hidden3', self.num_hidden_units, self.num_hidden_units,
                                            T.tanh, self.initialization)],
                          [FeedForwardLayer('output', self.num_hidden_units, 1,
                                            None, self.initialization)]])
        # linear perceptron
        else:
            self.nn = NN([[FeedForwardLayer('perc', self.train_feats.shape[1], 1,
                                            None, self.initialization)]])

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


class EmbNNRanker(NNRanker):
    """A ranker using MR and tree embeddings in a NN."""

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

        self.nn_shape = cfg.get('nn_shape', 'ff')
        self.normgrad = cfg.get('normgrad', False)

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
        return self.nn.score([cand_embs[0]], [cand_embs[1]])[0]

    def _extract_feats(self, tree, da):
        """Extract DA and tree embeddings (return as a pair)."""

        # DA embeddings (slot - value; size == 2x self.max_da_len)
        da_emb_idxs = []
        for dai in da[:self.max_da_len]:
            da_emb_idxs.append([self.dict_slot.get(dai.name, self.UNK_SLOT),
                                self.dict_value.get(dai.value, self.UNK_VALUE)])

        # pad with "unknown"
        for _ in xrange(len(da_emb_idxs), self.max_da_len):
            da_emb_idxs.append([self.UNK_SLOT, self.UNK_VALUE])

        # tree embeddings (parent_lemma - formeme - lemma; size == 3x self.max_tree_len)
        tree_emb_idxs = []
        for parent_ord, (t_lemma, formeme) in zip(tree.parents[1:self.max_tree_len + 1],
                                                  tree.nodes[1:self.max_tree_len + 1]):
            tree_emb_idxs.append([self.dict_t_lemma.get(tree.nodes[parent_ord].t_lemma,
                                                        self.UNK_T_LEMMA),
                                  self.dict_formeme.get(formeme, self.UNK_FORMEME),
                                  self.dict_t_lemma.get(t_lemma, self.UNK_T_LEMMA)])

        # pad with unknown
        for _ in xrange(len(tree_emb_idxs), self.max_tree_len):
            tree_emb_idxs.append([self.UNK_T_LEMMA, self.UNK_FORMEME, self.UNK_T_LEMMA])

        return (da_emb_idxs, tree_emb_idxs)

    def _init_neural_network(self):
        layers = [[Embedding('emb_das', self.dict_size, self.emb_size, 'uniform_005'),
                   Embedding('emb_trees', self.dict_size, self.emb_size, 'uniform_005')]]

        if self.nn_shape.startswith('ff'):
            layers += [[Flatten('flat-da'), Flatten('flat-trees')],
                       [Concat('concat')],
                       [FeedForwardLayer('ff1',
                                         self.emb_size * 2 * self.max_da_len +
                                         self.emb_size * 3 * self.max_tree_len,
                                         self.num_hidden_units,
                                         T.tanh, self.initialization)],
                       [FeedForwardLayer('ff2', self.num_hidden_units, self.num_hidden_units,
                                         T.tanh, self.initialization)]]
            if self.nn_shape[-1] in ['3', '4']:
                layers += [[FeedForwardLayer('ff3', self.num_hidden_units, self.num_hidden_units,
                                             T.tanh, self.initialization)]]
            if self.nn_shape[-1] == '4':
                layers += [[FeedForwardLayer('ff4', self.num_hidden_units, self.num_hidden_units,
                                             T.tanh, self.initialization)]]
            layers += [[FeedForwardLayer('perc', self.num_hidden_units, 1,
                                         None, self.initialization)]]

        elif 'maxpool-ff' in self.nn_shape:
            if self.nn_shape.startswith('conv'):
                layers += [[Conv1DLayer('conv_das', n_in=self.max_da_len, filter_length=4, stride=2,
                                        init=self.initialization, activation=T.tanh),
                            Conv1DLayer('conv_trees', n_in=self.max_da_len, filter_length=9, stride=3,
                                        init=self.initialization, activation=T.tanh)]]
            layers += [[MaxPool1DLayer('mp_das'),
                        MaxPool1DLayer('mp_trees')],
                       [Concat('concat')],
                       [Flatten('flatten')],
                       [FeedForwardLayer('ff1', self.emb_size * 5, self.num_hidden_units,
                                         T.tanh, self.initialization)],
                       [FeedForwardLayer('ff2', self.num_hidden_units, self.num_hidden_units,
                                         T.tanh, self.initialization)],
                       [FeedForwardLayer('perc', self.num_hidden_units, 1,
                                         None, self.initialization)]]

        elif self.nn_shape.startswith('dot'):
            layers += [[Flatten('flat-das'), Flatten('flat-trees')],
                       [FeedForwardLayer('ff-das', self.emb_size * 2 * self.max_da_len, self.num_hidden_units,
                                         T.tanh, self.initialization),
                        FeedForwardLayer('ff-trees', self.emb_size * 3 * self.max_tree_len, self.num_hidden_units,
                                         T.tanh, self.initialization)]]
            if self.nn_shape.endswith('2'):
                layers += [[FeedForwardLayer('ff2-das', self.num_hidden_units, self.num_hidden_units,
                                             T.tanh, self.initialization),
                            FeedForwardLayer('ff2-trees', self.num_hidden_units, self.num_hidden_units,
                                             T.tanh, self.initialization)]]
            layers += [[DotProduct('dot')]]

        elif self.nn_shape == 'maxpool-dot':
            layers += [[MaxPool1DLayer('mp_das'),
                        MaxPool1DLayer('mp_trees')],
                       [Flatten('flat-das'), Flatten('flat-trees')],
                       [FeedForwardLayer('ff-das', self.emb_size * 2, self.num_hidden_units,
                                         T.tanh, self.initialization),
                        FeedForwardLayer('ff-trees', self.emb_size * 3, self.num_hidden_units,
                                         T.tanh, self.initialization)],
                       [DotProduct('dot')]]

        elif self.nn_shape == 'avgpool-dot':
            layers += [[MaxPool1DLayer('mp_das', pooling_func=T.mean),
                        MaxPool1DLayer('mp_trees', pooling_func=T.mean)],
                       [FeedForwardLayer('ff-das', self.emb_size * 2, self.num_hidden_units,
                                         T.tanh, self.initialization),
                        FeedForwardLayer('ff-trees', self.emb_size * 3, self.num_hidden_units,
                                         T.tanh, self.initialization)],
                       [DotProduct('dot')]]

        # input: batch * word * sub-embeddings
        self.nn = NN(layers=layers, input_num=2, input_type=T.itensor3, normgrad=self.normgrad)

    def _update_nn(self, bad_feats, good_feats, rate):
        """Changing the NN update call to support arrays of parameters."""
        # TODO: this is just adding another dimension to fit the parallelized scoring
        # (even if updates are not parallelized). Make it nicer.
        bad_feats = ([bad_feats[0]], [bad_feats[1]])
        good_feats = ([good_feats[0]], [good_feats[1]])

        cost_gcost = self.nn.update(*(bad_feats + good_feats + (rate,)))
        log_debug('Cost:' + str(cost_gcost[0]))
        param_vals = [param.get_value() for param in self.nn.params]
        log_debug('Param norms : ' + str(self._l2s(param_vals)))
        log_debug('Gparam norms: ' + str(self._l2s(cost_gcost[1:])))
        l1_params = param_vals[2]
        log_debug('Layer 1 parts :' + str(self._l2s([l1_params[0:100, :], l1_params[100:200, :],
                                                    l1_params[200:350, :], l1_params[350:500, :],
                                                    l1_params[500:, :]])))
        l1_gparams = cost_gcost[3]
        log_debug('Layer 1 gparts:' + str(self._l2s([l1_gparams[0:100, :], l1_gparams[100:200, :],
                                                    l1_gparams[200:350, :], l1_gparams[350:500, :],
                                                    l1_gparams[500:, :]])))

    def _embs_to_str(self):
        out = ""
        da_emb = self.nn.layers[0][0].e.get_value()
        tree_emb = self.nn.layers[0][1].e.get_value()
        for idx, emb in enumerate(da_emb):
            for key, val in self.dict_slot.items():
                if val == idx:
                    out += key + ',' + ','.join([("%f" % d) for d in emb]) + "\n"
            for key, val in self.dict_value.items():
                if val == idx:
                    out += key + ',' + ','.join([("%f" % d) for d in emb]) + "\n"
        for idx, emb in enumerate(tree_emb):
            for key, val in self.dict_t_lemma.items():
                if val == idx:
                    out += str(key) + ',' + ','.join([("%f" % d) for d in emb]) + "\n"
            for key, val in self.dict_formeme.items():
                if val == idx:
                    out += str(key) + ',' + ','.join([("%f" % d) for d in emb]) + "\n"
        return out

    def _l2s(self, params):
        """Compute L2-norm of all members of the given list."""
        return [np.linalg.norm(param) for param in params]

    def store_iter_weights(self):
        """Remember the current weights to be used for averaged perceptron."""
        # fh = open('embs.txt', 'a')
        # print >> fh, '---', self._embs_to_str()
        # fh.close()
        self.w_after_iter.append(self.nn.get_param_values())

    def score_all(self, trees, da):
        cand_embs = [self._extract_feats(tree, da) for tree in trees]
        score = self.nn.score([emb[0] for emb in cand_embs], [emb[1] for emb in cand_embs])
        return np.atleast_1d(score[0])
